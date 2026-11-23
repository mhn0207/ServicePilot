"""Offline orchestrator evaluation; isolated SDK/intent/index boundaries."""
import asyncio
import copy
import hashlib
import inspect
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import uuid4

from agents.agent_orchestrator import AgentOrchestrator, Request, tool_call_context
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.llm_utils import extract_text_content
from evaluation.agent_scoring import judge_identity, score
from evaluation.experiment_dependencies import SyntheticIndex
from evaluation.provenance import source_identity, write_source_bundle
from evaluation.retrieval import build_snapshot, canonical_hash, checked_execution
from evaluation.strategies import StrategyRunner, preset

BASE = Path(__file__).resolve().parents[1]
REPORT_ROOT = BASE / "data/eval/rag/agents"
PROTOCOL = "agent-evaluation-v1"


def load(path):
    """Load a UTF-8 JSON file (allowing an optional BOM)."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def directory(run_id):
    """Resolve and validate the output directory for an evaluation run ID."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run_id")
    root = Path(REPORT_ROOT)
    if any((parent.is_symlink() or parent.is_junction()) for parent in (root, *root.parents)):
        raise ValueError("symlink root")
    path = root / run_id
    if path.is_symlink() or path.is_junction():
        raise ValueError("symlink/junction run")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("escaped report root")
    return path


def read_agent_run(run_id):
    """Read and validate a stored agent evaluation report."""
    path = directory(run_id) / "report.json"
    if path.is_symlink() or path.is_junction():
        raise ValueError("symlink/junction report")
    report = load(path)
    if report.get("run_id") != run_id:
        raise ValueError("report run identity mismatch")
    if report.get("scope") != "orchestrator" or report.get("protocol_version") != PROTOCOL:
        raise ValueError("unsupported scope/protocol")
    return report


def list_agent_runs():
    """List valid historical agent evaluation summaries sorted newest first."""
    directory("check")
    if not Path(REPORT_ROOT).exists():
        return []
    rows = []
    for entry in Path(REPORT_ROOT).iterdir():
        if (
            entry.is_symlink()
            or entry.is_junction()
            or not entry.is_dir()
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", entry.name)
        ):
            continue
        try:
            report = read_agent_run(entry.name)
            rows.append({
                key: report[key]
                for key in ("run_id", "status", "scope", "mode", "summary", "protocol_version")
            })
        except (ValueError, FileNotFoundError, KeyError):
            continue
    return sorted(rows, key=lambda row: row["run_id"], reverse=True)


def load_suite():
    """Load and verify the frozen agent-dev-v1 suite and knowledge snapshot."""
    cases_path = BASE / "evaluation/corpora/agent.cases.v1.json"
    manifest = load(BASE / "evaluation/corpora/agent.suite.v1.json")
    # 冻结哈希按字节计算，而 .gitattributes 的 text=auto 让同一份标注在 Windows 检出为
    # CRLF、在 Linux/macOS 检出为 LF。两种换行下内容相同即视为同一份冻结标注。
    lf_bytes = cases_path.read_bytes().replace(b"\r\n", b"\n")
    digests = {
        hashlib.sha256(payload).hexdigest()
        for payload in (lf_bytes, lf_bytes.replace(b"\n", b"\r\n"))
    }
    if manifest["cases_sha256"] not in digests:
        raise ValueError("annotation freeze mismatch")

    data = load(cases_path)
    snapshot = build_snapshot(load(BASE / "data/demo_docs/sample_knowledge.json"))
    for case in data["cases"]:
        for turn in case["turns"]:
            for quote in turn["evidence_quotes"]:
                if not any(quote in row.content for row in snapshot.chunks):
                    raise ValueError("invalid evidence quote")

    if manifest["knowledge_hash"] != snapshot.knowledge_hash or manifest["case_set_hash"] != canonical_hash(data):
        raise ValueError("manifest identity mismatch")
    return data, snapshot, manifest


class Intent:
    """Deterministic offline intent recognizer double."""

    def __init__(self):
        self.inputs = []

    async def recognize(self, message, history=None):
        self.inputs.append(copy.deepcopy({"message": message, "history": history}))
        billing = any(keyword in message for keyword in ("扣款", "退款", "续费", "合同", "这笔订单"))
        if billing:
            intent = IntentCategory.BILLING
            intent_group = "billing"
        elif "401" in message:
            intent = IntentCategory.TECHNICAL
            intent_group = "technical"
        else:
            intent = IntentCategory.QUERY
            intent_group = "query"
        return NS(
            intent=intent,
            intent_group=intent_group,
            urgency=UrgencyLevel.LOW,
            confidence=1.0,
            entities={"order_id": re.findall(r"订单\s*(\d{5,})", message)},
        )


class SDK:
    """Offline Anthropic SDK double with payload budget guard and fixture playback."""

    base_url = "https://offline.invalid"

    def __init__(self):
        self.messages = self
        self.inputs = []
        self.calls = []
        self.fixture = {}
        self.req = None
        self.turn = None

    def evaluation_observer(self, stage, kwargs, response):
        text = extract_text_content(response.content)
        self.turn.setdefault("model_steps", []).append({
            "step_id": uuid4().hex,
            "stage": stage,
            "request": copy.deepcopy(kwargs),
            "response": copy.deepcopy(response.content),
        })
        if text and stage.startswith("agent."):
            self.turn.setdefault("role_answers", []).append({
                "role": stage.split(".", 1)[1],
                "answer": text,
            })

    def evaluation_recorder(self, stage, model, started, status, response):
        self.calls.append({
            "stage": stage,
            "model": model,
            "status": status,
            "input_tokens": None,
            "output_tokens": None,
            "kind": "simulated logical SDK call",
        })

    async def create(self, **kwargs):
        payload_chars = len(json.dumps(kwargs, ensure_ascii=False))
        if payload_chars > 64000:
            self.turn.setdefault("budget_events", []).append({
                "status": "rejected",
                "policy": "reject SDK request over 64000 chars",
                "actual_chars": payload_chars,
            })
            raise ValueError("offline_context_budget_exceeded")

        self.inputs.append(copy.deepcopy(kwargs))
        self.turn["generation_inputs"].append(copy.deepcopy(kwargs))
        used_tool = any(
            isinstance(msg["content"], list)
            and any(isinstance(block, dict) and block.get("type") == "tool_result" for block in msg["content"])
            for msg in kwargs["messages"]
        )
        if kwargs.get("tools") and not self.fixture.get("skip_tool") and not used_tool:
            query = self.req.message + " " + " ".join(self.req.entities.get("order_id", []))
            content = [{
                "type": "tool_use",
                "id": "tool-" + uuid4().hex,
                "name": "search_knowledge_base",
                "input": {"query": query, "top_k": 3},
            }]
        else:
            content = [{"type": "text", "text": self.fixture["answer"]}]
        return NS(content=content, stop_reason="end_turn")


def metric(values):
    """Compute pass-rate numerator/denominator ignoring None entries."""
    valid = [x for x in values if x is not None]
    passed = sum(x is True for x in valid)
    return {
        "numerator": passed,
        "denominator": len(valid),
        "value": passed / len(valid) if valid else None,
    }


def summarize(turns):
    """Aggregate per-turn evaluation checks into suite metrics."""
    def eligible(key, turn):
        label = turn.get("annotations", {})
        return {
            "required_call_coverage": label.get("expected_retrieval") == "required",
            "execution_success": bool(turn.get("events")),
            "query_entities": bool(label.get("required_entities")),
            "old_entity_clean": "22222" in label.get("required_entities", []),
            "returned_evidence": bool(label.get("evidence_quotes")),
            "role_coverage": bool(label.get("expected_roles")),
            "required_facts": bool(label.get("required_facts")),
            "forbidden_clear": bool(label.get("forbidden_claims")),
            "judge_validity": True,
        }[key]

    keys = (
        "required_call_coverage",
        "execution_success",
        "query_entities",
        "old_entity_clean",
        "returned_evidence",
        "role_coverage",
        "required_facts",
        "forbidden_clear",
        "judge_validity",
    )
    summary = {}
    for key in keys:
        rows = [turn for turn in turns if eligible(key, turn)]
        values = [turn.get("checks", {}).get(key) for turn in rows]
        valid = [
            turn["case_id"] + ":" + turn["turn_id"]
            for turn in rows
            if turn.get("checks", {}).get(key) is not None
        ]
        passed = sum(val is True for val in values)
        summary[key] = {
            "numerator": passed,
            "denominator": len(rows),
            "value": passed / len(rows) if rows else None,
            "unresolved": sum(val is None for val in values),
            "valid_only_denominator": len(valid),
            "valid_only_ids": valid,
            "policy": "all applicable planned turns; unresolved retained in denominator; value is demonstrated-pass coverage",
        }
    return summary


async def run_agents(progress=None, run_id=None):
    """Execute the fixed agent-dev-v1 suite against the real AgentOrchestrator."""
    run_id = run_id or "agent-" + uuid4().hex
    target = directory(run_id)
    target.mkdir(parents=True, exist_ok=False)
    data = None
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "quality_status": "incomplete; observed dev protocol only",
        "protocol_version": PROTOCOL,
        "scope": "orchestrator",
        "mode": "fixture/synthetic",
        "suite": "agent-dev-v1",
        "run_id": run_id,
        "status": "running",
        "turns": [],
        "summary": {},
        "identities": {},
    }
    try:
        data, snapshot, manifest = load_suite()
        fixtures = load(BASE / "evaluation/fixtures/agent.sdk.v1.json")
        report["identities"] = {
            "annotations": canonical_hash(data),
            "knowledge": snapshot.knowledge_hash,
            "freeze": manifest,
            "fixture": canonical_hash(fixtures),
            "strategy": "experimental B synthetic; not business baseline",
            "lexical": "all-matches-bm25-before-top-k-v2",
            "scoring": "authored-scripted-judge-v1; sentence-units-v2",
            "judge": judge_identity(),
            "context": "reject SDK request >64000 chars; top_k=3; no silent truncation",
            "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        git_head, workspace_dirty = None, None
        try:
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True, check=False)
            status = subprocess.run(["git", "status", "--porcelain"], cwd=BASE, capture_output=True, text=True, check=False)
            if git.returncode == 0:
                git_head = git.stdout.strip()
            if status.returncode == 0:
                workspace_dirty = bool(status.stdout)
        except OSError:
            pass  # Runtime images need not include Git; source module hashes remain available.

        tracked_modules = (
            BASE / "agents/agent_orchestrator.py",
            BASE / "core/llm_utils.py",
            BASE / "evaluation/agent_scoring.py",
            BASE / "evaluation/answer_quality.py",
        )
        report["identities"].update(
            git_head=git_head,
            workspace_dirty=workspace_dirty,
            source=source_identity(BASE),
            code_modules={
                path.relative_to(BASE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in tracked_modules
            },
        )
        write_source_bundle(BASE, target, report["identities"]["source"])
        artifacts = [
            ("snapshot.json", snapshot.model_dump()),
            ("cases.json", data),
            ("sdk.json", fixtures),
            ("judge.json", load(BASE / "evaluation/fixtures/agent.judge.v1.json")),
        ]
        for filename, value in artifacts:
            (target / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

        total = sum(len(case["turns"]) for case in data["cases"])
        for case in data["cases"]:
            sdk = SDK()
            intent = Intent()
            orchestrator = AgentOrchestrator(api_key="offline", client=sdk, intent_recognizer=intent)
            runner = StrategyRunner(snapshot, SyntheticIndex(snapshot))
            state = {}
            history = []
            for idx, label in enumerate(case["turns"]):
                sdk.fixture = fixtures[case["id"]][idx]
                req = Request(
                    message=label["question"],
                    user_id="offline-user",
                    conv_id=case["id"],
                    history=copy.deepcopy(history),
                    conversation_state=state,
                )
                sdk.req = req
                turn = {
                    "case_id": case["id"],
                    "turn_id": str(idx + 1),
                    "question": req.message,
                    "request_id": req.request_id,
                    "roles": [],
                    "queries": [],
                    "events": [],
                    "answer": "",
                    "checks": {},
                    "generation_inputs": [],
                    "annotations": label,
                }
                report["turns"].append(turn)
                sdk.turn = turn

                async def search_with_rewrite(tool_name, query, top_k=5):
                    if tool_name != "knowledge_search" or top_k != 3:
                        raise ValueError("offline business retrieval contract mismatch")
                    args = {"query": query, "top_k": top_k}
                    event = {
                        "run_id": run_id,
                        "case_id": case["id"],
                        "turn_id": str(idx + 1),
                        "request_id": req.request_id,
                        "query": args["query"],
                        "parameters": copy.deepcopy(args),
                        "status": "running",
                        "shown_chunks": [],
                        "cached": False,
                        "degraded": False,
                        "source": "StrategyRunner B synthetic",
                    }
                    event.update(tool_call_context.get() or {})
                    turn["events"].append(event)
                    if sdk.fixture.get("tool_fault"):
                        event.update(status="error", error_code="fixture_timeout")
                        raise TimeoutError("offline fixture timeout")
                    execution, trace = await runner.run(case["id"], args["query"], preset("B"))
                    execution = checked_execution(execution, snapshot)
                    rows = [row.model_dump() for row in execution.results]
                    event.update(
                        status=execution.status,
                        shown_chunks=rows,
                        chunks_hash=canonical_hash(rows),
                        strategy_trace=trace.model_dump(),
                    )
                    return NS(
                        success=execution.status == "success",
                        data=rows,
                        cached=False,
                        reranked=False,
                        error=execution.error_code,
                    )

                orchestrator.set_shared_tools(
                    build_shared_rag_tools(NS(search_with_rewrite=search_with_rewrite))
                )

                def observe(trace):
                    event = next(
                        (item for item in turn["events"] if item.get("tool_use_id") == trace.get("tool_use_id")),
                        None,
                    )
                    if event is not None:
                        event["actual_tool_result"] = trace.get("result")

                req.tool_observer = observe
                start = len(sdk.inputs)
                calls = len(sdk.calls)
                result = await orchestrator.run(req)
                turn.update(
                    answer=result.response,
                    roles=[role.value for role in result.agent_types],
                    queries=[event["query"] for event in turn["events"]],
                    generation_inputs=sdk.inputs[start:],
                    intent_inputs=copy.deepcopy(intent.inputs[-1:]),
                    model_calls=sdk.calls[calls:],
                    entities=copy.deepcopy(req.entities),
                    tool_traces=result.tool_traces,
                    status=result.status,
                )
                # tool_use_id maps by execution position; same query remains separate per role.
                turn["checks"] = score(label, turn)
                history += [
                    {"role": "user", "content": req.message},
                    {"role": "assistant", "content": result.response},
                ]
                state = req.conversation_state
                if progress:
                    pending = progress(len(report["turns"]), total)
                    if inspect.isawaitable(pending):
                        await pending
                await asyncio.sleep(0)
        report["status"] = "completed"
    except asyncio.CancelledError:
        report["status"] = "cancelled"
        raise
    except Exception:
        report.update(status="failed", error_code="offline_execution_failed")
        raise
    finally:
        if data is not None:
            observed = {(turn["case_id"], turn["turn_id"]) for turn in report["turns"]}
            for case in data["cases"]:
                for idx, label in enumerate(case["turns"]):
                    if (case["id"], str(idx + 1)) not in observed:
                        report["turns"].append({
                            "case_id": case["id"],
                            "turn_id": str(idx + 1),
                            "question": label["question"],
                            "roles": [],
                            "queries": [],
                            "events": [],
                            "answer": "",
                            "annotations": label,
                            "status": "not_executed",
                            "checks": {
                                "judge_validity": False,
                                "required_call_coverage": False if label["expected_retrieval"] == "required" else None,
                                "role_coverage": False,
                                "returned_evidence": False if label["evidence_quotes"] else None,
                            },
                        })
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["summary"] = summarize(report["turns"])
        pending_path = target / "report.pending.json"
        pending_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pending_path.replace(target / "report.json")
    return report


def replay_agent_run(run_id):
    """Re-score a stored agent evaluation report when judge and module hashes match."""
    report = copy.deepcopy(read_agent_run(run_id))
    identity = report.get("identities", {})
    actual = hashlib.sha256((BASE / "evaluation/agent_scoring.py").read_bytes()).hexdigest()
    quality_hash = hashlib.sha256((BASE / "evaluation/answer_quality.py").read_bytes()).hexdigest()
    if (
        identity.get("judge") != judge_identity()
        or identity.get("code_modules", {}).get("evaluation/agent_scoring.py") != actual
        or identity.get("code_modules", {}).get("evaluation/answer_quality.py") != quality_hash
    ):
        raise ValueError("incompatible Judge/scoring identity; historical report remains readable")
    for turn in report["turns"]:
        if turn.get("answer"):
            turn["checks"] = score(turn["annotations"], turn)
    report["summary"] = summarize(report["turns"])
    return report
