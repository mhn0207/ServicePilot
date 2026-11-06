"""通过运行中的 /search 评测检索命中；该接口可能调用改写、重排模型。"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import httpx


def score_results(results, expected_title, top_k):
    """以实际返回的片段排名计算命中和倒数排名，排除降级提示。"""
    rank = next((index for index, item in enumerate(results[:top_k], 1)
                 if isinstance(item, dict) and not item.get("fallback")
                 and item.get("title") == expected_title), None)
    return {"hit": rank is not None, "reciprocal_rank": 1 / rank if rank else 0.0}


async def evaluate(base_url, cases, top_k):
    token = os.getenv("ZHIYING_EVAL_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rows = []
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=90) as client:
        for case in cases:
            started = time.monotonic()
            row = {**case, "error": None}
            try:
                response = await client.post("/search", params={"query": case["query"], "top_k": top_k})
                response.raise_for_status()
                data = response.json()
                results = data["results"]
                if not isinstance(results, list):
                    raise ValueError("检索结果不是列表")
                row.update(score_results(results, case["expected_title"], top_k))
                row.update(results=results, cached=data.get("cached"), reranked=data.get("reranked"))
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                # 不写异常正文，避免将认证信息或服务端细节写入报告。
                row.update(hit=False, reciprocal_rank=0.0, error=type(error).__name__)
            row["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
            rows.append(row)
    total = len(rows)
    return {"scope": "search_pipeline", "top_k": top_k, "total": total,
            "errors": sum(row["error"] is not None for row in rows),
            "hit_at_k": sum(row["hit"] for row in rows) / total if total else None,
            "mrr_at_k": sum(row["reciprocal_rank"] for row in rows) / total if total else None,
            "note": "包含错误请求，按未命中计；使用已知示例语料，只验证检索，不评价最终回答或未知问题拒答。",
            "results": rows}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=root / "evaluation" / "rag_cases.json")
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=3)
    parser.add_argument("--output", type=Path, default=root / "data" / "eval" / "rag-retrieval.json")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("用例必须是非空列表")
    report = asyncio.run(evaluate(args.base_url, cases, args.top_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
