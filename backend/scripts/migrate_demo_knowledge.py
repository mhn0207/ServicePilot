"""替换本地库中内容完全匹配的旧演示知识；默认只检查，--apply 执行并备份。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fingerprint(collection):
    rows = collection.get(include=["documents", "metadatas", "embeddings"])
    embeddings = rows["embeddings"]
    items = sorted((identity, document, metadata,
                    embeddings[index].tolist() if embeddings is not None else None)
                   for index, (identity, document, metadata) in enumerate(zip(
                       rows["ids"], rows["documents"], rows["metadatas"])))
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def migration_candidates(rows, legacy):
    """标题相同但内容不同、显式导入或无法辨别来源的资料均保留。"""
    expected = {(doc["title"], doc["content"]) for doc in legacy}
    return [identity for identity, document, metadata in zip(
        rows["ids"], rows["documents"], rows["metadatas"])
        if metadata and set(metadata) <= {"title", "chunk_index", "total_chunks"}
        and metadata.get("chunk_index") == 0 and metadata.get("total_chunks") == 1
        and (metadata.get("title"), document) in expected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ROOT / "data/chroma")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    path = args.path.resolve()
    if not (path / "chroma.sqlite3").is_file():
        parser.error("指定目录不是已有 ChromaDB 数据库，禁止自动创建空库")
    legacy = json.loads((ROOT / "scripts/legacy_demo_knowledge.json").read_text(encoding="utf-8"))
    docs = json.loads((ROOT / "data/demo_docs/sample_knowledge.json").read_text(encoding="utf-8"))

    # 本脚本应在使用该本地库的应用停止时执行。
    backup = None
    if args.apply:
        backup = ROOT / "data/backups" / datetime.now().strftime("knowledge-migration-%Y%m%d-%H%M%S-%f")
        shutil.copytree(path, backup / "chroma")

    import chromadb
    from tooling.knowledge_base import KnowledgeBase
    client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.get_collection("knowledge_base")
    rows = collection.get(include=["documents", "metadatas", "embeddings"])
    old_ids = migration_candidates(rows, legacy)
    report = {"database": str(path), "before_chunks": collection.count(),
              "matched_legacy_chunks": len(old_ids), "apply": args.apply}
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    current_pairs = Counter(((metadata or {}).get("title"), document)
                            for document, metadata in zip(rows["documents"], rows["metadatas"]))
    expected_pairs = Counter((doc["title"], chunk) for doc in docs
                             for chunk in KnowledgeBase._chunk_text(None, doc["content"]))
    if not old_ids and all(current_pairs[item] >= count for item, count in expected_pairs.items()):
        report.update(status="already_migrated", after_chunks=collection.count(), backup=str(backup))
        (backup / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    untouched = {c.name: fingerprint(c) for c in client.list_collections() if c.name != "knowledge_base"}
    retained_ids = set(rows["ids"]) - set(old_ids)
    before_retained = {identity: (document, metadata, rows["embeddings"][index].tolist())
                       for index, (identity, document, metadata) in enumerate(zip(
                           rows["ids"], rows["documents"], rows["metadatas"]))
                       if identity in retained_ids}
    # 不让新示例覆盖任何同标题但来源不明的现有资料。
    new_titles = {d["title"] for d in docs}
    if any((metadata or {}).get("title") in new_titles for identity, metadata in zip(
            rows["ids"], rows["metadatas"]) if identity in retained_ids):
        raise RuntimeError("新示例标题与现有资料冲突，停止迁移；请人工核对")

    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = collection
    kb._write_lock = threading.RLock()
    # 先写入并确认新资料完整，再删除已确认的旧示例，模型下载失败不会删除旧资料。
    inserted = kb.add_documents(docs)
    loaded = collection.get(include=["documents", "metadatas"])
    actual = Counter(((meta or {}).get("title"), text) for text, meta in zip(
        loaded["documents"], loaded["metadatas"]))
    expected = Counter((doc["title"], chunk) for doc in docs for chunk in kb._chunk_text(doc["content"]))
    if any(actual[item] < count for item, count in expected.items()):
        raise RuntimeError("新知识写入不完整，保留旧示例；完整数据库备份已保存")
    if old_ids:
        collection.delete(ids=old_ids)
    assert collection.count() == len(retained_ids) + inserted
    assert not set(collection.get()["ids"]) & set(old_ids)
    assert untouched == {c.name: fingerprint(c) for c in client.list_collections() if c.name != "knowledge_base"}
    if retained_ids:
        kept = collection.get(ids=sorted(retained_ids), include=["documents", "metadatas", "embeddings"])
        after_retained = {identity: (document, metadata, kept["embeddings"][index].tolist())
                          for index, (identity, document, metadata) in enumerate(zip(
                              kept["ids"], kept["documents"], kept["metadatas"]))}
        assert before_retained == after_retained
    queries = ["企业订单重复扣款，退款需要审核吗？", "企业服务登录返回 401 应该怎么排查？"]
    report.update(backup=str(backup), inserted_chunks=inserted, after_chunks=collection.count(),
                  preserved_collections=list(untouched),
                  retrieval=[{"query": query, "titles": [hit["title"] for hit in kb.search(query, top_k=3)]}
                             for query in queries])
    (backup / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
