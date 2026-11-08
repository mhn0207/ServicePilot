"""只读比较企业服务固定用例的向量检索与混合检索，不调用聊天模型。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    ("企业订单重复扣款，退款需要审核吗？", "重复扣款与退款审核"),
    ("同一订单扣了两次钱，如何核验？", "重复扣款与退款审核"),
    ("两笔交易金额相同就能确认重复扣款吗？", "重复扣款与退款审核"),
    ("企业服务登录返回 401 应该怎么排查？", "技术可靠性排查"),
    ("接口返回 403，需要检查哪些权限？", "技术可靠性排查"),
    ("服务器500错误，需要提供什么日志信息？", "技术可靠性排查"),
    ("企业发票重开需要财务审核吗？", "企业发票与订阅"),
    ("取消自动续费是否等于当前订单退款？", "企业发票与订阅"),
    ("支付成功但是企业服务未开通，怎么办？", "企业订单与服务开通"),
    ("服务开通异常要提供订单号和支付时间吗？", "企业订单与服务开通"),
    ("需要人工审核时交接哪些证据？", "人工审核与证据交接"),
    ("企业服务平台有哪些Agent角色分工？", "企业服务协同范围"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ROOT / "data/chroma")
    parser.add_argument("--report", type=Path, default=ROOT / "data/eval/hybrid-retrieval-regression.json")
    args = parser.parse_args()
    if not (args.path / "chroma.sqlite3").is_file():
        parser.error("只允许检查已有的本地数据库")
    import chromadb
    from tooling.knowledge_base import KnowledgeBase
    from scripts.migrate_demo_knowledge import fingerprint
    client = chromadb.PersistentClient(path=str(args.path.resolve()), settings=chromadb.Settings(anonymized_telemetry=False))
    before = {c.name: fingerprint(c) for c in client.list_collections()}
    collection = client.get_collection("knowledge_base")
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = collection
    rows = []
    for query, expected in CASES:
        vector = collection.query(query_texts=[query], n_results=min(3, collection.count()))
        vector_titles = [meta["title"] for meta in vector["metadatas"][0]]
        hybrid_titles = [hit["title"] for hit in kb.search(query, top_k=3)]
        rows.append({"query": query, "expected": expected, "vector_top3": vector_titles,
                     "hybrid_top3": hybrid_titles})
    after = {c.name: fingerprint(c) for c in client.list_collections()}
    assert before == after, "检索不应修改知识、记忆或画像"
    report = {"cases": len(rows), "database_unchanged": True,
              "limitations": "固定的12条小样本检查；非独立标注集，不代表生产准确率；未调用LLM重排。",
              "rows": rows}
    for method in ("vector", "hybrid"):
        report[method] = {"hit_at_1": sum(row["expected"] == row[f"{method}_top3"][0] for row in rows),
                          "hit_at_3": sum(row["expected"] in row[f"{method}_top3"] for row in rows)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert rows[0]["hybrid_top3"][0] == rows[0]["expected"], "原始问题必须首条命中"
    assert report["hybrid"]["hit_at_3"] == len(rows), "检查遗漏的业务领域"


if __name__ == "__main__":
    main()
