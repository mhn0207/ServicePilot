import asyncio
import threading
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from core.knowledge_revision import KnowledgeRevision
from tooling.knowledge_base import KnowledgeBase
from tooling.lexical_index import LexicalIndex


class IndexedCollection:
    """支持分页扫描、按 ID 批量读取与 where_document 回退的内存集合替身。"""

    def __init__(self):
        self.rows = {}
        self.get_calls = []

    def upsert(self, ids, documents, metadatas):
        for sid, doc, meta in zip(ids, documents, metadatas):
            self.rows[sid] = (doc, dict(meta))

    def delete(self, ids):
        for sid in ids:
            self.rows.pop(sid, None)

    def count(self):
        return len(self.rows)

    def query(self, query_texts, n_results=5):
        # 故意返回最先插入的无关系条目模拟向量未召回目标词面片段。
        selected = list(self.rows.items())[:min(n_results, len(self.rows))]
        return {
            "ids": [[sid for sid, _ in selected]],
            "documents": [[doc for _, (doc, _) in selected]],
            "metadatas": [[dict(meta) for _, (_, meta) in selected]],
            "distances": [[0.2 + i * 0.01 for i in range(len(selected))]],
        }

    def get(self, ids=None, where=None, where_document=None, include=None, limit=None, offset=None):
        self.get_calls.append({"ids": ids, "where": where, "where_document": where_document, "limit": limit, "offset": offset})
        items = list(self.rows.items())
        if ids is not None:
            id_set = set(ids)
            items = [(sid, val) for sid, val in items if sid in id_set]
        if where is not None:
            items = [(sid, (doc, meta)) for sid, (doc, meta) in items if all(meta.get(k) == v for k, v in where.items())]
        if where_document is not None:
            clauses = where_document.get("$or") if "$or" in where_document else [where_document]
            terms = [c["$contains"] for c in clauses if "$contains" in c]
            items = [(sid, (doc, meta)) for sid, (doc, meta) in items if any(t in doc for t in terms)]
        if offset is not None:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        result = {
            "ids": [sid for sid, _ in items],
            "metadatas": [dict(meta) for _, (_, meta) in items],
        }
        if "documents" in (include or []):
            result["documents"] = [doc for _, (doc, _) in items]
        return result


def make_kb(collection, revision_provider=None, lexical_candidate_limit=None):
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._write_lock = threading.RLock()
    kb._index_lock = threading.RLock()
    kb._collection = collection
    kb._revision_provider = revision_provider
    kb._lexical_candidate_limit = lexical_candidate_limit
    kb._lexical_index = None
    return kb


def test_lexical_index_incremental_reuse_and_bm25_beats_200_cap():
    rows_v1 = [
        (f"d{i}:0", "退款普通说明文本。", {"document_id": f"d{i}", "version": "v1", "title": f"常识{i}", "chunk_index": 0})
        for i in range(250)
    ]
    rows_v1.append(
        ("target:0", "退款错误码 431 需要工单编号处理。", {"document_id": "target", "version": "v1", "title": "错误码431", "chunk_index": 0})
    )
    idx1 = LexicalIndex.build("rev-1", rows_v1)
    candidates = idx1.query_candidates("退款错误码 431", limit=200)
    assert candidates[0] == "target:0"

    # 仅修改 d0 的 version，并删除 d1，其余 249 条应直接复用 LexicalEntry 对象。
    rows_v2 = [
        ("d0:0", "更新后的说明。", {"document_id": "d0", "version": "v2", "title": "常识0", "chunk_index": 0}),
        *rows_v1[2:],
    ]
    idx2 = LexicalIndex.build("rev-2", rows_v2, previous=idx1)
    assert "d1:0" not in idx2.docs
    assert idx2.docs["d0:0"] is not idx1.docs["d0:0"]
    assert idx2.docs["target:0"] is idx1.docs["target:0"]


@pytest.mark.asyncio
async def test_cross_worker_index_sync_pagination_diff_and_delete(monkeypatch):
    collection = IndexedCollection()
    async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
        writer_rev = KnowledgeRevision(redis)
        reader_rev = KnowledgeRevision(redis)

        kb_a = make_kb(collection, revision_provider=writer_rev.current)
        kb_b = make_kb(collection, revision_provider=reader_rev.current)
        # 故意把分页批大小调小，验证多页分页拉取。
        monkeypatch.setattr(kb_b, "_INDEX_SCAN_BATCH", 3)

        async with writer_rev.mutation():
            docs = [
                {"document_id": f"noise-{i}", "title": f"常识{i}", "content": f"常规配送会员发票说明 {i}。"}
                for i in range(25)
            ]
            docs.append({"document_id": "target", "title": "退款审核", "content": "重复扣款431需要人工审核退款资格。"})
            await kb_a.add_documents_async(docs)

        target_doc_meta = next(d for d in kb_a.list_documents()["documents"] if d["title"] == "退款审核")
        target_sid = f"{target_doc_meta['document_id']}:0"

        # Worker B 冷启动检索：分页重建索引 + 仅补读 missing_ids，不走 where_document。
        collection.get_calls.clear()
        hits = await kb_b.search_async("重复扣款431", top_k=2)
        assert hits[0]["title"] == "退款审核"
        assert kb_b._lexical_index is not None
        scan_calls = [c for c in collection.get_calls if c["offset"] is not None]
        assert len(scan_calls) >= 2  # 验证分页扫描确实跨了多页
        assert not any(c["where_document"] is not None for c in collection.get_calls)
        id_reads = [c for c in collection.get_calls if c["ids"] is not None]
        assert len(id_reads) == 1
        assert id_reads[0]["ids"] == [target_sid]

        # 第二次检索命中同 revision 索引：不再全库分页扫描，只按 ID 补读缺失片段。
        collection.get_calls.clear()
        hits2 = await kb_b.search_async("重复扣款431", top_k=2)
        assert hits2[0]["title"] == "退款审核"
        assert all(c["ids"] is not None for c in collection.get_calls)

        # Worker A 更新并删除文档，Worker B 自动感应新 revision。
        async with writer_rev.mutation():
            kb_a.delete_document(target_doc_meta["document_id"], target_doc_meta["version"])
            await kb_a.add_documents_async([
                {"document_id": "new-policy", "title": "新退款政策", "content": "新版431工单自动退款。"}
            ])

        hits3 = await kb_b.search_async("431退款", top_k=2)
        titles = [h["title"] for h in hits3]
        assert "新退款政策" in titles
        assert "退款审核" not in titles


@pytest.mark.asyncio
async def test_dirty_window_and_redis_failure_fallback_to_where_document():
    collection = IndexedCollection()
    async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
        writer_rev = KnowledgeRevision(redis)
        reader_rev = KnowledgeRevision(redis)

        kb_a = make_kb(collection, revision_provider=writer_rev.current)
        kb_b = make_kb(collection, revision_provider=reader_rev.current)

        async with writer_rev.mutation():
            await kb_a.add_documents_async([
                {"document_id": "d1", "title": "配送", "content": "配送物流查询。"},
                {"document_id": "d2", "title": "账单", "content": "旧版401认证故障排查。"},
            ])

        # 先构建 rev1 索引
        assert (await kb_b.search_async("401认证", top_k=1))[0]["title"] == "账单"

        # dirty: 窗口期 revision 为 None，Worker B 不得信任过期索引，必须回退 where_document 读到最新内容。
        async with writer_rev.mutation():
            await kb_a.add_documents_async([
                {"document_id": "d3", "title": "紧急公告", "content": "401认证故障已切换新网关。"},
            ])
            collection.get_calls.clear()
            during = await kb_b.search_async("401认证新网关", top_k=2)
            assert any(h["title"] == "紧急公告" for h in during)
            assert any(c["where_document"] is not None for c in collection.get_calls)

        # Redis 不可用时同样回退 where_document
        kb_broken = make_kb(collection, revision_provider=AsyncMock(side_effect=TimeoutError("redis timeout")))
        collection.get_calls.clear()
        fallback_hits = await kb_broken.search_async("401认证新网关", top_k=2)
        assert any(h["title"] == "紧急公告" for h in fallback_hits)
        assert any(c["where_document"] is not None for c in collection.get_calls)


@pytest.mark.asyncio
async def test_concurrent_searches_rebuild_index_once_and_local_writes_invalidate():
    collection = IndexedCollection()
    kb = make_kb(collection, revision_provider=lambda: "static-rev")
    docs = [
        {"document_id": f"n-{i}", "title": f"普通{i}", "content": f"常规说明 {i}。"}
        for i in range(25)
    ]
    docs.append({"document_id": "err", "title": "故障码502", "content": "网关502需要检查上游端口。"})
    kb.add_documents(docs)

    collection.get_calls.clear()
    results = await asyncio.gather(*[kb.search_async("网关502", top_k=1) for _ in range(6)])
    assert all(r[0]["title"] == "故障码502" for r in results)
    scan_calls = [c for c in collection.get_calls if c["offset"] is not None]
    assert len(scan_calls) == 1  # _index_lock 保证同一 revision 只重建一次

    # 本进程写入与删除会立即清除 _lexical_index
    kb.add_documents([{"document_id": "err2", "title": "故障码504", "content": "网关504超时排查。"}])
    assert kb._lexical_index is None
    assert (await kb.search_async("网关504", top_k=1))[0]["title"] == "故障码504"
    assert kb._lexical_index is not None

    meta = next(d for d in kb.list_documents()["documents"] if d["title"] == "故障码504")
    kb.delete_document(meta["document_id"], meta["version"])
    assert kb._lexical_index is None
    assert all(r["title"] != "故障码504" for r in await kb.search_async("网关504", top_k=3))


@pytest.mark.asyncio
async def test_real_chroma_ephemeral_collection_with_lexical_index():
    import chromadb
    from uuid import uuid4
    from evaluation.experiment_dependencies import SyntheticIndex

    class FakeEmbedding:
        def __call__(self, input):
            return [SyntheticIndex.embed(text) for text in input]

    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    collection_name = f"test_lexical_a1_{uuid4().hex}"
    collection = client.create_collection(
        collection_name,
        embedding_function=FakeEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )
    try:
        async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
            writer_rev = KnowledgeRevision(redis)
            reader_rev = KnowledgeRevision(redis)
            kb_a = make_kb(collection, revision_provider=writer_rev.current)
            kb_b = make_kb(collection, revision_provider=reader_rev.current)

            async with writer_rev.mutation():
                docs = [
                    {"document_id": f"c-{i}", "title": f"常规说明{i}", "content": f"常规服务条款 {i}。"}
                    for i in range(25)
                ]
                docs.append({
                    "document_id": "target-503",
                    "title": "网关503排查",
                    "content": "网关返回 503 错误码时请核查上游健康检查与连接池。",
                })
                await kb_a.add_documents_async(docs)

            hits = await kb_b.search_async("503连接池", top_k=2)
            assert hits[0]["title"] == "网关503排查"
            assert kb_b._lexical_index is not None
            assert len(kb_b._lexical_index.docs) == 26
    finally:
        client.delete_collection(collection_name)
