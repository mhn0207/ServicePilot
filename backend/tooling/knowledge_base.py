"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务；默认 embedding_function 在客户端生成向量。
        self._write_lock = threading.RLock()
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception as ex:
            if os.getenv("ZHIYING_ENV", "development").lower() == "production":
                raise RuntimeError("生产环境知识库 ChromaDB 不可用，禁止切换到本地库") from ex
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 不传 embedding_function 时使用 ChromaDB 客户端默认模型；
        # Docker 已预下载模型，本地运行首次使用可能触发下载。
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "ServicePilot RAG 知识库"},
        )

        # 如果知识库为空，导入默认文档
        seed_demo = os.getenv("ZHIYING_SEED_DEMO_DOCS", "true").lower() in {"1", "true", "yes"}
        if os.getenv("ZHIYING_ENV", "development").lower() != "production" and seed_demo and self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        # 相同文档身份在同一批次只采用最后一个版本，避免重复 ID。
        prepared = {}
        for doc in documents:
            title, content = doc.get("title", ""), doc.get("content", "")
            if not isinstance(title, str) or not isinstance(content, str):
                raise ValueError("文档 title 和 content 必须是字符串")
            if not title.strip() or not content.strip():
                raise ValueError("文档标题和内容不能为空")
            explicit_id = doc.get("document_id")
            if explicit_id is not None and (not isinstance(explicit_id, str) or not explicit_id.strip()):
                raise ValueError("document_id 必须是非空字符串")
            identity = json.dumps(["explicit", explicit_id] if explicit_id else ["title", title], ensure_ascii=True)
            if not isinstance(identity, str):
                raise ValueError("document_id 必须是字符串")
            prepared[identity] = (title, content, bool(doc.get("document_id")))

        count = 0
        with self._write_lock:
            for identity, (title, content, explicit) in prepared.items():
                document_id = hashlib.sha256(identity.encode()).hexdigest()
                chunks = self._chunk_text(content, chunk_size=500)
                version = hashlib.sha256(content.encode()).hexdigest()
                ids = [f"{document_id}:{i}" for i in range(len(chunks))]
                existing = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
                old_ids = set(existing["ids"])
                # 仅迁移归属可验证的旧身份；旧显式 ID 与 title: 前缀冲突时保留旧记录供核验。
                kind, value = json.loads(identity)
                legacy_identity = value if explicit else f"title:{title}"
                if not explicit or not value.startswith("title:"):
                    legacy_id = hashlib.sha256(legacy_identity.encode()).hexdigest()
                    previous = self._collection.get(where={"document_id": legacy_id}, include=["metadatas"])
                    old_ids.update(i for i, meta in zip(previous["ids"], previous["metadatas"])
                                   if (meta or {}).get("title") == title)
                if not explicit:
                    # 兼容旧版只有 title/chunk_index 元数据的片段。
                    legacy = self._collection.get(where={"title": title}, include=["metadatas"])
                    old_ids.update(i for i, meta in zip(legacy["ids"], legacy["metadatas"])
                                   if not (meta or {}).get("document_id"))
                self._collection.upsert(ids=ids, documents=chunks, metadatas=[
                    {"title": title, "document_id": document_id, "version": version,
                     "chunk_index": i, "total_chunks": len(chunks)} for i in range(len(chunks))
                ])
                stale = old_ids - set(ids)
                if stale:
                    self._collection.delete(ids=list(stale))
                count += len(chunks)
        logger.info("知识库写入 %s 个文档片段", count)
        return count

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        向量与词面混合检索；词面候选最多 200 条，不读取全部向量。
        """
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k 必须是 1–20 的整数")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query 必须是 1–2000 字符的非空文本")
        count = self._collection.count()
        if not count:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(max(20, top_k * 3), count),
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                    "document_id": meta.get("document_id", ""),
                    "version": meta.get("version", ""),
                })

        if len(items) <= 1 and count <= 1:
            return items[:top_k]
        # 独立词面召回，补充未进入向量候选的中文术语和错误码。
        terms = list(dict.fromkeys(self._search_terms(query)))[:64]
        if not terms:
            return items[:top_k]
        try:
            clauses = [{"$contains": term} for term in terms]
            lexical = self._collection.get(
                where_document=clauses[0] if len(clauses) == 1 else {"$or": clauses},
                include=["documents", "metadatas"], limit=200,
            )
        except Exception as ex:
            logger.warning("词面召回不可用，保留向量结果: %s", type(ex).__name__)
            return items[:top_k]
        candidates = {(item["document_id"], item["title"], item["chunk"], item["content"]): item for item in items}
        for text, meta in zip(lexical["documents"], lexical["metadatas"]):
            meta = meta or {}
            key = (meta.get("document_id", ""), meta.get("title", ""), meta.get("chunk_index", 0), text)
            candidates.setdefault(key, {"title": key[1], "content": text, "chunk": key[2],
                                        "document_id": key[0], "version": meta.get("version", "")})
        return self._hybrid_rank(query, list(candidates.values()), items)[:top_k]

    @staticmethod
    def _search_terms(text: str) -> List[str]:
        # 中文双字滑窗无需分词模型；英文单词、数字和错误码保持完整。
        terms = []
        for part in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
            if re.fullmatch(r"[\u3400-\u9fff]+", part):
                terms.extend(part[i:i + 2] for i in range(len(part) - 1))
            else:
                terms.append(part)
        return terms

    @classmethod
    def _hybrid_rank(cls, query, candidates, vector_items):
        """候选集 BM25（标题加权）与向量名次融合；分数不是准确率。"""
        terms = set(cls._search_terms(query))
        tokens = [Counter(cls._search_terms(item["content"]) + cls._search_terms(item["title"]) * 3)
                  for item in candidates]
        average = sum(sum(row.values()) for row in tokens) / max(1, len(tokens)) or 1
        frequencies = {term: sum(term in row for row in tokens) for term in terms}
        lexical_scores = []
        for row in tokens:
            length = sum(row.values())
            lexical_scores.append(sum(
                math.log(1 + (len(tokens) - frequencies[term] + .5) / (frequencies[term] + .5))
                * row[term] * 2.2 / (row[term] + 1.2 * (.25 + .75 * length / average))
                for term in terms if row[term]))
        maximum = max(lexical_scores, default=0)
        if not maximum:
            return vector_items
        def identity(item):
            return (item["document_id"], item["title"], item["chunk"], item["content"])
        vector_ranks = {identity(item): rank for rank, item in enumerate(vector_items, 1)}
        ranked = []
        for item, lexical_score in zip(candidates, lexical_scores):
            rank = vector_ranks.get(identity(item))
            result = dict(item)
            result["vector_score"] = item.get("score")
            result["lexical_score"] = round(lexical_score, 4)
            result["score"] = round(.65 * lexical_score / maximum + (.35 / rank if rank else 0), 6)
            result["retrieval_method"] = "hybrid"
            ranked.append(result)
        return sorted(ranked, key=lambda item: (-item["score"], item["title"], item["chunk"], item["content"]))

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        sentences = [sent[i:i + chunk_size] for sent in sentences
                     for i in range(0, len(sent), chunk_size)]
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入企业服务协同示例知识（仅用于开发演示）。"""
        from pathlib import Path
        corpus = Path(__file__).resolve().parents[1] / "data" / "demo_docs" / "sample_knowledge.json"
        default_docs = json.loads(corpus.read_text(encoding="utf-8"))
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
