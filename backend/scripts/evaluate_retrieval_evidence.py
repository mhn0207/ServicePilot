"""第一批证据级检索 CLI；默认离线，只有 --online 才请求现有 /search。"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

# 同时支持从 backend 执行 python scripts/... 和 python -m scripts...。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from pydantic import ValidationError

from evaluation.retrieval import (
    Dataset, Execution, Fixture, ReturnedChunk, build_snapshot, make_report,
)

OUTPUT_ROOT = ROOT / "data" / "eval" / "rag"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_identity() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                                  text=True, encoding="utf-8", timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain")
    return {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "worktree_dirty": bool(status) if status is not None else None}


async def request_executions(base_url: str, dataset: Dataset, top_k: int) -> list[Execution]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base-url 必须为不含凭据、查询参数或 fragment 的 HTTP(S) 地址")
    token = os.getenv("ZHIYING_EVAL_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rows = []
    async with httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", headers=headers,
                                timeout=90, follow_redirects=False) as client:
        for case in dataset.cases:
            started = time.monotonic()
            try:
                response = await client.post("search", params={"query": case.turns[0], "top_k": top_k})
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                    raise ValueError("invalid response")
                # 只提取协议字段，忽略服务端额外信息，不把私有响应直接写入报告。
                results = []
                for item in data["results"]:
                    if not isinstance(item, dict):
                        raise ValueError("invalid item")
                    results.append(ReturnedChunk.model_validate({
                        key: item[key] for key in ("chunk_id", "title", "content", "fallback") if key in item}))
                row = Execution(case_id=case.id, results=results,
                                cached=data.get("cached"), reranked=data.get("reranked"))
            except httpx.HTTPError:
                row = Execution(case_id=case.id, status="error", error_code="request_error")
            except (ValueError, KeyError, TypeError):
                row = Execution(case_id=case.id, status="error", error_code="invalid_response")
            row.elapsed_ms = round((time.monotonic() - started) * 1000, 3)
            rows.append(row)
    return rows


def write_json(path: Path, model):
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture", type=Path, help="固定结果 JSON；不发请求、不连接数据库")
    mode.add_argument("--online", action="store_true", help="显式请求默认流水线，可能消耗改写/重排额度")
    mode.add_argument("--export-snapshot", action="store_true", help="只导出固定示例快照，供人工标注；不执行用例")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="仅 --online 使用；请指向独立示例环境")
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation" / "rag_evidence_cases.v1.json")
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=3)
    parser.add_argument("--run-id", help="可选唯一目录名，已存在时拒绝覆盖")
    args = parser.parse_args(argv)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12]
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        parser.error("run-id 只能包含 1–80 个字母、数字、下划线或短横线")
    started_at = now()
    try:
        snapshot = build_snapshot(load_json(ROOT / "data" / "demo_docs" / "sample_knowledge.json"))
        output = OUTPUT_ROOT / run_id
        if args.export_snapshot:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "snapshot.json", snapshot)
            print(json.dumps({"knowledge_hash": snapshot.knowledge_hash,
                              "snapshot": str(output / "snapshot.json")}, ensure_ascii=False))
            return 0
        dataset = Dataset.model_validate(load_json(args.cases))
        dataset.validate_snapshot(snapshot)
        if not args.online:
            fixture = Fixture.model_validate(load_json(args.fixture or ROOT / "evaluation" / "fixtures" / "retrieval_results.v1.json"))
            fixture.validate_inputs(dataset, args.top_k)
        # 在请求前拒绝重复运行目录，防止已经调用在线模型后才发现不能写报告。
        output.mkdir(parents=True, exist_ok=False)
        if args.online:
            executions = asyncio.run(request_executions(args.base_url, dataset, args.top_k))
        else:
            executions = fixture.executions
        report = make_report(dataset, snapshot, executions, top_k=args.top_k, run_id=run_id,
                             mode="online" if args.online else "fixture", started_at=started_at, finished_at=now(),
                             code=code_identity(), configuration={
                                 "pipeline": "existing /search: rewrite + hybrid recall + rerank",
                                 "cache_policy": "existing defaults; no bypass in batch 1",
                                 "latency_source": "http_client_wall_time" if args.online else "fixture_recorded",
                                 "model_effectiveness_verified": False,
                                 "knowledge_scope": "public demo snapshot; returned chunks checked, full remote corpus not attested",
                             })
        write_json(output / "snapshot.json", snapshot)
        write_json(output / "dataset.json", dataset)
        # 可重算的安全执行记录；未知正文和异常正文不会保留下来。
        replay = Fixture(protocol_version="rag-fixture-v1", knowledge_hash=snapshot.knowledge_hash,
                         dataset_hash=dataset.dataset_hash, top_k=args.top_k,
                         description="Sanitized replay; fixture metrics do not measure live model quality",
                         executions=[r.execution for r in report.results])
        write_json(output / "executions.json", replay)
        write_json(output / "report.json", report)
    except (OSError, ValueError, TypeError, ValidationError) as error:
        # 不打印数据值、URL、认证头、服务端异常正文。
        print(f"评测输入或报告错误：{type(error).__name__}；请检查协议、哈希、证据标注和唯一 run-id。", file=sys.stderr)
        return 2
    print(json.dumps({"run_id": run_id, "mode": report.mode, "total": report.total,
                      "errors": report.errors, "error_rate": report.error_rate,
                      "main": report.main.model_dump(), "successful": report.successful.model_dump(),
                      "report": str(output / "report.json")}, ensure_ascii=False))
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
