"""Run paired Chinese cases against two real, isolated Docker HTTP services."""

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import httpx
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "evaluation/corpora/chat-paired.zh.v2.json"
FAULT_FILE = "/tmp/servicepilot-ledger-unavailable"
ARMS = ("single", "routed")


def docker(*args, env=None, check=True):
    result = subprocess.run(["docker", *args], env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", check=False)
    if check and result.returncode:
        # Docker arguments can include runtime settings. Surface the failing operation only.
        raise RuntimeError(f"docker {args[0]} failed with code {result.returncode}")
    return result.stdout.strip()


def run_container(stack, network, name, image, *options, command=(), env=None):
    docker("run", "-d", "--name", name, "--network", network, *options, image, *command, env=env)
    stack.callback(docker, "rm", "-f", "-v", name, check=False)


def wait_http(url, container, timeout=120):
    deadline = time.monotonic() + timeout
    with httpx.Client(trust_env=False, timeout=3) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(url)
                if response.is_success:
                    return
            except (httpx.RequestError, ValueError):
                pass
            if docker("inspect", "--format", "{{.State.Running}}", container) != "true":
                raise RuntimeError(f"Container exited before readiness: {container}")
            time.sleep(2)
    raise TimeoutError(f"Container did not become ready: {container}")


def endpoint(container):
    address = docker("port", container, "8000/tcp")
    return f"http://{address}"


def start_app(stack, network, app, image, provider, token, arm, redis_url, chroma):
    settings = {
        "SERVICEPILOT_CHAT_PAIR_ARM": arm,
        "SERVICEPILOT_ENV": "development",
        "SERVICEPILOT_API_TOKENS": json.dumps({token: {"user_id": "u1001", "role": "admin"}}),
        "SERVICEPILOT_SANDBOX_ENABLED": "true",
        "SERVICEPILOT_SEED_DEMO_DOCS": "true",
        "INTENT_EMBEDDING_ENABLED": "false",
        "MCP_EXTERNAL_ENABLED": "false",
        "ANONYMIZED_TELEMETRY": "FALSE",
        "REDIS_URL": redis_url,
        "CHROMA_HOST": chroma,
        "CHROMA_PORT": "8000",
    }
    env = {**os.environ, **provider, **settings}
    variables = tuple(arg for key in (*provider, *settings) for arg in ("-e", key))
    run_container(stack, network, app, image, "-p", "127.0.0.1::8000", *variables, env=env,
                  command=("python", "-m", "uvicorn", "evaluation.chat_pair_server:app",
                           "--host", "0.0.0.0", "--port", "8000"))
    url = endpoint(app)
    wait_http(f"{url}/health", app)
    return app, url


def start_services(stack, image, provider, token, prefix, verify_shared=False):
    network = docker("network", "create", prefix)
    stack.callback(docker, "network", "rm", network, check=False)
    redis = f"{prefix}-redis"
    run_container(stack, network, redis, "redis:7-alpine")
    apps = {}
    for index, arm in enumerate(ARMS):
        chroma = f"{prefix}-chroma-{arm}"
        app = f"{prefix}-{arm}"
        run_container(stack, network, chroma, "chromadb/chroma:0.5.23", "-p", "127.0.0.1::8000",
                      "-e", "ANONYMIZED_TELEMETRY=FALSE")
        wait_http(f"{endpoint(chroma)}/api/v1/heartbeat", chroma)
        apps[arm] = start_app(stack, network, app, image, provider, token, arm,
                              f"redis://{redis}:6379/{index}", chroma)
    if verify_shared:
        apps["replica"] = start_app(stack, network, f"{prefix}-replica", image, provider, token, "routed",
                                    f"redis://{redis}:6379/1", f"{prefix}-chroma-routed")
    return apps


def verify_shared_knowledge(client, writer, reader):
    title = "SPCACHE6013 跨进程知识修订"
    before = "SPCACHE6013 核验版本 A：按旧规则处理。"
    after = "SPCACHE6013 核验版本 B：请以新版规则为准。"

    def publish(content):
        client.post(f"{writer}/knowledge/add", json={"documents": [
            {"title": title, "content": content},
        ]}).raise_for_status()

    def search():
        response = client.post(f"{reader}/search", params={"query": "SPCACHE6013", "top_k": 10})
        response.raise_for_status()
        result = response.json()
        if not result.get("success"):
            raise RuntimeError("Shared knowledge search failed")
        return result

    publish(before)
    warmup = search()
    warmed = search()
    publish(after)
    updated = search()
    documents = client.get(f"{writer}/knowledge/documents")
    documents.raise_for_status()
    document = next(doc for doc in documents.json()["documents"] if doc["title"] == title)
    client.delete(f"{writer}/knowledge/documents/{document['document_id']}",
                  params={"expected_version": document["version"]}).raise_for_status()
    deleted = search()
    return {
        "old_result_present": any(hit["content"] == before for hit in warmup["results"]),
        "other_worker_cache_warmed": warmed["cached"],
        "updated_result_present": any(hit["content"] == after for hit in updated["results"]),
        "old_result_evicted": not any(hit["content"] == before for hit in updated["results"]),
        "deleted_result_evicted": not any(hit["title"] == title for hit in deleted["results"]),
    }


def tool_matches(expected, actual):
    if "any_of" in expected:
        return any(tool_matches(option, actual) for option in expected["any_of"])
    effective_success = actual.get("success") is True and actual.get("result_success") is not False
    arguments = actual.get("input", {})
    return (actual.get("tool_name") == expected["name"]
            and all(arguments.get(key) == value for key, value in expected["args"].items())
            and all(re.search(pattern, str(arguments.get(key, "")), re.I)
                    for key, pattern in expected.get("arg_patterns", {}).items())
            and ("success" not in expected or effective_success == expected["success"]))


def fact_matches(rule, answer):
    alternatives = [[rule]] if isinstance(rule, str) else rule["any_of"]
    return any(all(re.search(pattern, answer, re.I) for pattern in clauses) for clauses in alternatives)


def score(turn, reply, calls, status):
    answer = reply.get("response", "")
    facts = {key: fact_matches(rule, answer) for key, rule in turn["facts"].items()}
    violations = [key for key, pattern in turn.get("forbidden", {}).items() if re.search(pattern, answer, re.I)]
    tools = [any(tool_matches(expected, call) for call in calls) for expected in turn["tools"]]
    forbidden_tools = [expected for expected in turn.get("forbidden_tools", [])
                       if any(tool_matches(expected, call) for call in calls)]
    return {
        "facts": facts, "violations": violations, "tools": tools, "forbidden_tools": forbidden_tools,
        "passed": (status == 200 and reply.get("status") == "success" and all(facts.values()) and all(tools)
                   and not violations and not forbidden_tools),
    }


def run_turn(client, containers, arm, case, turn, turn_id, run_id):
    container, url = containers[arm]
    fault = turn.get("fault") == "ledger_unavailable"
    if fault:
        docker("exec", container, "touch", FAULT_FILE)
    payload = {
        "message": turn["question"], "user_id": "u1001", "conv_id": f"{run_id}-{case['id']}",
        "client_request_id": str(turn_id),
    }
    try:
        started = time.perf_counter()
        response = client.post(f"{url}/chat", json=payload)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        try:
            reply = response.json()
        except ValueError:
            reply = {}
    finally:
        if fault:
            docker("exec", container, "rm", "-f", FAULT_FILE)
    calls = []
    if reply.get("request_id"):
        trace = client.get(f"{url}/trace/tool/{reply['request_id']}")
        trace.raise_for_status()
        calls = trace.json().get("trace", {}).get("tool_calls", [])
    llm_calls = reply.get("performance", {}).get("llm_calls", [])
    usage = reply.get("performance", {}).get("llm")
    replay = None
    if case["id"] == "login-and-ledger" and reply.get("request_id") and response.is_success:
        repeated = client.post(f"{url}/chat", json=payload)
        history = client.get(f"{url}/conversation/history", params={"conv_id": payload["conv_id"]})
        replay = {
            "same_receipt": repeated.is_success and repeated.json() == reply,
            "one_committed_turn": history.is_success and [
                (message["role"], message["content"]) for message in history.json()["messages"]
            ] == [("user", turn["question"]), ("assistant", reply["response"])],
        }
    return {
        "case": case["id"], "group": case["group"], "turn": turn_id, "arm": arm,
        "question": turn["question"], "http": response.status_code,
        "status": reply.get("status"), "request_id": reply.get("request_id"),
        "roles": reply.get("agent_types", []), "answer": reply.get("response", ""),
        "error": reply.get("error_code") or reply.get("detail"),
        "latency_ms": elapsed, "usage": usage, "llm_calls": llm_calls, "tool_calls": calls,
        "replay": replay,
        "fact_rules": turn["facts"],
        "checks": score(turn, reply, calls, response.status_code),
    }


def summarize(rows):
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            continue
        latency = sorted(row["latency_ms"] for row in selected)
        facts = [covered for row in selected for covered in row["checks"]["facts"].values()]
        tools = [matched for row in selected for matched in row["checks"]["tools"]]
        usage = [row["usage"] for row in selected if row["usage"] is not None]
        tokens = [value for value in usage if value["usage_reported_calls"] > 0]
        summary[arm] = {
            "turns_passed": sum(row["checks"]["passed"] for row in selected), "turns": len(selected),
            "facts_covered": sum(facts), "facts": len(facts),
            "tools_matched": sum(tools), "tools_expected": len(tools),
            "p50_ms": latency[math.ceil(len(latency) * .50) - 1],
            "p95_ms": latency[math.ceil(len(latency) * .95) - 1],
            "llm_calls": sum(value["calls"] for value in usage),
            "usage_reported_calls": sum(value["usage_reported_calls"] for value in usage),
            **{key: sum(value[key] for value in tokens) if tokens else None for key in
               ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")},
        }
    return summary


def review_report(report, reviews):
    rows = {(row["arm"], row["case"], row["turn"]): row for row in report["rows"]}
    seen = set()
    for review in reviews:
        key = (review["arm"], review["case"], review["turn"])
        row = rows[key]
        fact = review["fact"]
        identity = (*key, fact)
        if (identity in seen or fact not in row["checks"]["facts"] or type(review["covered"]) is not bool
                or not review["reviewer"].strip() or not review["reason"].strip()
                or not review["quote"].strip() or review["quote"] not in row["answer"]):
            raise ValueError(f"Invalid review for {identity}")
        seen.add(identity)
        row.setdefault("reviews", []).append(review)
    for row in report["rows"]:
        if "reviews" not in row:
            continue
        checks = row["checks"]
        facts = dict(checks["facts"])
        for review in row["reviews"]:
            facts[review["fact"]] = review["covered"]
        row["automated_checks"] = checks
        row["checks"] = {
            **checks, "facts": facts,
            "passed": (row["http"] == 200 and row["status"] == "success" and all(facts.values())
                       and all(checks["tools"]) and not checks["violations"] and not checks["forbidden_tools"]),
        }
    report["automated_summary"] = report["summary"]
    report["summary"] = summarize(report["rows"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Repository Docker production image built with the current backend")
    parser.add_argument("--provider-env", type=Path)
    parser.add_argument("--review-report", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--verify-shared-knowledge", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=[case["id"] for case in json.loads(SUITE.read_bytes())["cases"]])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    if args.review_report or args.reviews:
        if not (args.review_report and args.reviews) or args.image or args.provider_env or args.case or args.verify_shared_knowledge:
            parser.error("pass both --review-report and --reviews for an offline review")
        report = json.loads(args.review_report.read_bytes())
        reviews = json.loads(args.reviews.read_bytes())
        if report["run_id"] != reviews["run_id"] or report["suite_sha256"] != reviews["suite_sha256"]:
            parser.error("review identity does not match the report")
        if report["status"] != "completed" or "automated_summary" in report:
            parser.error("review requires an original completed report")
        reviewed = review_report(report, reviews["reviews"])
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(reviewed, output, ensure_ascii=False, indent=2)
            output.write("\n")
        return
    if not (args.image and args.provider_env):
        parser.error("live mode requires --image and --provider-env")
    config = dotenv_values(args.provider_env, interpolate=False)
    if not config.get("ANTHROPIC_API_KEY") or not config.get("ANTHROPIC_MODEL"):
        parser.error("provider config requires ANTHROPIC_API_KEY and ANTHROPIC_MODEL")
    provider = {key: config.get(key) or "" for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL")}
    suite = json.loads(SUITE.read_bytes())
    run_id = f"servicepilot-pair-{secrets.token_hex(4)}"
    token = secrets.token_urlsafe(32)
    rows = []
    report = {
        "suite": suite["suite"], "suite_sha256": hashlib.sha256(SUITE.read_bytes()).hexdigest(),
        "knowledge_sha256": hashlib.sha256((ROOT / "data/demo_docs/sample_knowledge.json").read_bytes()).hexdigest(),
        "image_id": docker("image", "inspect", "--format", "{{.Id}}", args.image),
        "model": provider["ANTHROPIC_MODEL"], "mode": "real-provider/docker-http", "run_id": run_id,
        "scoring": "fact alternatives across spans plus exact tool checks; attach manual reviews for semantic exceptions",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        try:
            with ExitStack() as stack, httpx.Client(
                timeout=65, trust_env=False, headers={"Authorization": f"Bearer {token}"},
            ) as client:
                print(f"Starting isolated Docker run {run_id}", flush=True)
                containers = start_services(stack, args.image, provider, token, run_id, args.verify_shared_knowledge)
                for index, case in enumerate(suite["cases"]):
                    if args.case and case["id"] != args.case:
                        continue
                    for arm in ARMS if index % 2 == 0 else reversed(ARMS):
                        for turn_id, turn in enumerate(case["turns"], 1):
                            row = run_turn(client, containers, arm, case, turn, turn_id, run_id)
                            rows.append(row)
                            print(arm, case["id"], turn_id, row["http"], row["checks"]["passed"], flush=True)
                if args.verify_shared_knowledge:
                    result = verify_shared_knowledge(client, containers["routed"][1], containers["replica"][1])
                    report["shared_knowledge"] = result
                    print("Shared knowledge verification:", result, flush=True)
                    if not all(result.values()):
                        raise RuntimeError("Shared knowledge cache verification failed")
            report["status"] = "completed"
        except BaseException as exc:
            report.update(status="interrupted", error_type=type(exc).__name__)
            raise
        finally:
            report["summary"] = summarize(rows)
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
