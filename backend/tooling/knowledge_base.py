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
import inspect
import json
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

import chromadb
from tooling.knowledge_chunks import (
    LEGACY_CHUNKING, ChunkStrategy, chunk_text, content_version, document_identity, evidence_id,
)
from tooling.lexical_index import LexicalIndex
from tooling.retrieval_ranking import RetrievalRanking
from core.env import env_bool, env_value
from core.storage_health import ChromaHealthProbe
from tooling.knowledge_documents import KnowledgeDocuments

logger = logging.getLogger(__name__)
_UNSET = object()


class KnowledgeBase(RetrievalRanking, KnowledgeDocuments):
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"
    _INDEX_SCAN_BATCH = 1000

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        revision_provider: Optional[Callable[[], Any]] = None,
        lexical_candidate_limit: Optional[int] = None,
    ):
        # 优先连接独立 ChromaDB 服务；默认 embedding_function 在客户端生成向量。
        self._write_lock = threading.RLock()
        self._index_lock = threading.RLock()
        self._revision_provider = revision_provider
        self._lexical_candidate_limit = lexical_candidate_limit
        self._lexical_index: Optional[LexicalIndex] = None
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
            env = (env_value("SERVICEPILOT_ENV", "development") or "development").lower()
            if env == "production":
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
            metadata={"description": "ServicePilot RAG 知识库", "hnsw:space": "cosine"},
        )
        self._health_probe = ChromaHealthProbe(
            self._collection, host=chroma_host if self._use_server else None, port=chroma_port,
            api_path=self._client.get_settings().chroma_server_api_default_path if self._use_server else '/api/v2',
        )

        # 如果知识库为空，导入默认文档
        seed_demo = env_bool("SERVICEPILOT_SEED_DEMO_DOCS", True)
        env = (env_value("SERVICEPILOT_ENV", "development") or "development").lower()
        if env != "production" and seed_demo and self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片至多 500 字），可按文档指定 chunk_strategy。
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
            strategy = doc.get("chunk_strategy", LEGACY_CHUNKING)
            identity = document_identity(title, explicit_id, strategy=strategy)
            prepared[identity] = (title, content, strategy)

        count = 0
        with self._write_lock:
            try:
                for identity, (title, content, strategy) in prepared.items():
                    document_id = content_version(identity)
                    chunks = self._chunk_text(content, strategy=strategy)
                    version = content_version(content)
                    ids = [f"{document_id}:{i}" for i in range(len(chunks))]
                    existing = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
                    old_ids = set(existing["ids"])
                    if strategy == LEGACY_CHUNKING:
                        # 旧片段只归入 legacy 命名空间，并且必须能验证归属。
                        kind, value = json.loads(identity)
                        explicit = kind == "explicit"
                        if not explicit or not value.startswith("title:"):
                            legacy_id = content_version(value if explicit else f"title:{title}")
                            previous = self._collection.get(where={"document_id": legacy_id}, include=["metadatas"])
                            old_ids.update(i for i, meta in zip(previous["ids"], previous["metadatas"])
                                           if meta and meta.get("title") == title
                                           and meta.get("chunk_strategy", LEGACY_CHUNKING) == LEGACY_CHUNKING)
                        if not explicit:
                            # 兼容旧版只有 title/chunk_index 元数据的片段。
                            legacy = self._collection.get(where={"title": title}, include=["metadatas"])
                            old_ids.update(i for i, meta in zip(legacy["ids"], legacy["metadatas"])
                                           if not (meta or {}).get("document_id"))
                    self._collection.upsert(ids=ids, documents=chunks, metadatas=[
                        {"title": title, "document_id": document_id, "version": version,
                         "chunk_strategy": strategy, "chunk_index": i, "total_chunks": len(chunks)}
                        for i in range(len(chunks))
                    ])
                    stale = old_ids - set(ids)
                    if stale:
                        self._collection.delete(ids=list(stale))
                    count += len(chunks)
            finally:
                self._invalidate_lexical_index()
        logger.info("知识库写入 %s 个文档片段", count)
        return count

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def _get_index_lock(self) -> threading.RLock:
        lock = getattr(self, "_index_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._index_lock = lock
        return lock

    def _invalidate_lexical_index(self) -> None:
        with self._get_index_lock():
            self._lexical_index = None

    def _resolve_sync_revision(self) -> Optional[str]:
        provider = getattr(self, "_revision_provider", None)
        if provider is None or inspect.iscoroutinefunction(provider):
            return None
        try:
            value = provider()
        except Exception as ex:
            logger.warning("读取知识库版本失败，回退即时词面检索: %s", type(ex).__name__)
            return None
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            return None
        return value if isinstance(value, str) and value and not value.startswith("dirty:") else None

    async def _resolve_async_revision(self) -> Optional[str]:
        provider = getattr(self, "_revision_provider", None)
        if provider is None:
            return None
        try:
            value = provider()
            if inspect.isawaitable(value):
                value = await value
        except Exception as ex:
            logger.warning("异步读取知识库版本失败，回退即时词面检索: %s", type(ex).__name__)
            return None
        return value if isinstance(value, str) and value and not value.startswith("dirty:") else None

    def _scan_collection_rows(self) -> List[tuple]:
        rows: List[tuple] = []
        offset = 0
        seen_ids: set[str] = set()
        while True:
            try:
                batch = self._collection.get(
                    include=["documents", "metadatas"],
                    limit=self._INDEX_SCAN_BATCH,
                    offset=offset,
                )
            except TypeError:
                if offset > 0:
                    break
                batch = self._collection.get(include=["documents", "metadatas"])
                ids = batch.get("ids") or []
                docs = batch.get("documents") or []
                metas = batch.get("metadatas") or []
                for idx, sid in enumerate(ids):
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        rows.append((sid, docs[idx] if idx < len(docs) else "", metas[idx] if idx < len(metas) else {}))
                break
            ids = batch.get("ids") or []
            docs = batch.get("documents") or []
            metas = batch.get("metadatas") or []
            added = 0
            for idx, sid in enumerate(ids):
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    rows.append((sid, docs[idx] if idx < len(docs) else "", metas[idx] if idx < len(metas) else {}))
                    added += 1
            if len(ids) < self._INDEX_SCAN_BATCH or added == 0:
                break
            offset += len(ids)
        return rows

    def _sync_lexical_index(self, revision: str) -> LexicalIndex:
        with self._get_index_lock():
            current = getattr(self, "_lexical_index", None)
            if current is not None and current.revision == revision:
                return current
            rows = self._scan_collection_rows()
            rebuilt = LexicalIndex.build(revision, rows, previous=current)
            self._lexical_index = rebuilt
            return rebuilt

    def _fetch_by_ids(self, ids: List[str]) -> tuple[List[str], List[Dict[str, Any]]]:
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for start in range(0, len(ids), self._INDEX_SCAN_BATCH):
            batch_ids = ids[start:start + self._INDEX_SCAN_BATCH]
            batch = self._collection.get(ids=batch_ids, include=["documents", "metadatas"])
            documents.extend(batch.get("documents") or [])
            metadatas.extend(batch.get("metadatas") or [])
        return documents, metadatas

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        revision: Optional[str] = _UNSET,  # type: ignore[assignment]
        lexical_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量与词面混合检索；先按相关性计分，再截取 Top-K。
        """
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k 必须是 1–20 的整数")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query 必须是 1–2000 字符的非空文本")
        count = self._collection.count()
        if not count:
            return []
        resolved_revision = self._resolve_sync_revision() if revision is _UNSET else revision
        if not isinstance(resolved_revision, str) or not resolved_revision or resolved_revision.startswith("dirty:"):
            resolved_revision = None
        results = self._collection.query(
            query_texts=[query],
            n_results=min(max(20, top_k * 3), count),
        )

        items = []
        vector_storage_ids: set[str] = set()
        if results["documents"] and results["documents"][0]:
            raw_ids = results.get("ids")
            vec_ids = raw_ids[0] if isinstance(raw_ids, list) and raw_ids else []
            for idx, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                meta = meta or {}
                sid = vec_ids[idx] if idx < len(vec_ids) else None
                if not sid and meta.get("document_id"):
                    sid = f"{meta.get('document_id')}:{meta.get('chunk_index', 0)}"
                if sid:
                    vector_storage_ids.add(sid)
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(max(0.0, min(1.0, 1.0 - float(dist))), 4),  # ChromaDB 返回距离，转为有界相似度
                    "chunk":    meta.get("chunk_index", 0),
                    "document_id": meta.get("document_id", ""),
                    "version": meta.get("version", ""),
                    "chunk_id": evidence_id(meta.get("document_id", ""), meta.get("version", ""), meta.get("chunk_index", 0)),
                })

        if len(items) <= 1 and count <= 1:
            return items[:top_k]
        # 独立词面召回，补充未进入向量候选的中文术语和错误码。
        terms = list(dict.fromkeys(self._search_terms(query)))[:64]
        if not terms:
            return items[:top_k]

        lexical_docs: List[str] = []
        lexical_metas: List[Dict[str, Any]] = []
        if resolved_revision is not None:
            try:
                index = self._sync_lexical_index(resolved_revision)
                default_cap = getattr(self, "_lexical_candidate_limit", None)
                cand_limit = lexical_limit if lexical_limit is not None else (
                    default_cap if default_cap is not None else max(200, top_k * 20)
                )
                lexical_ids = index.query_candidates(query, limit=cand_limit)
                missing_ids = [sid for sid in lexical_ids if sid not in vector_storage_ids]
                if missing_ids:
                    lexical_docs, lexical_metas = self._fetch_by_ids(missing_ids)
            except Exception as ex:
                logger.warning("词面索引或补读不可用，保留向量结果: %s", type(ex).__name__)
                return items[:top_k]
        else:
            try:
                clauses = [{"$contains": term} for term in terms]
                lexical = self._collection.get(
                    where_document=clauses[0] if len(clauses) == 1 else {"$or": clauses},
                    include=["documents", "metadatas"],
                )
                lexical_docs = lexical.get("documents") or []
                lexical_metas = lexical.get("metadatas") or []
            except Exception as ex:
                logger.warning("词面召回不可用，保留向量结果: %s", type(ex).__name__)
                return items[:top_k]

        candidates = {(item["document_id"], item["title"], item["chunk"], item["content"]): item for item in items}
        for text, meta in zip(lexical_docs, lexical_metas):
            meta = meta or {}
            key = (meta.get("document_id", ""), meta.get("title", ""), meta.get("chunk_index", 0), text)
            candidates.setdefault(key, {"title": key[1], "content": text, "chunk": key[2],
                                        "document_id": key[0], "version": meta.get("version", ""),
                                        "chunk_id": evidence_id(key[0], meta.get("version", ""), key[2])})
        return self._hybrid_rank(query, list(candidates.values()), items)[:top_k]

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        before_rev = await self._resolve_async_revision()
        previous_index = getattr(self, "_lexical_index", None)
        result = await asyncio.to_thread(self.search, query, top_k, revision=before_rev)
        if before_rev is not None and previous_index is not getattr(self, "_lexical_index", None):
            after_rev = await self._resolve_async_revision()
            if after_rev != before_rev:
                self._invalidate_lexical_index()
                result = await asyncio.to_thread(self.search, query, top_k, revision=after_rev)
        return result

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    async def check_health(self) -> bool:
        return await self._health_probe.check()

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

    def _chunk_text(self, text: str, chunk_size: int = 500, *,
                    strategy: ChunkStrategy = LEGACY_CHUNKING) -> List[str]:
        return chunk_text(text, chunk_size, strategy=strategy)

    def _load_default_docs(self) -> None:
        """导入企业服务协同示例知识（仅用于开发演示）。"""
        from pathlib import Path
        corpus = Path(__file__).resolve().parents[1] / "data" / "demo_docs" / "sample_knowledge.json"
        default_docs = json.loads(corpus.read_text(encoding="utf-8"))
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
