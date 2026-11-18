"""HTTP 链路回归：使用 Redis 协议模拟器验证锁、事务、记忆和幂等行为。"""
import asyncio
import json
import sqlite3
import sys
import time
import types
from collections import deque
from contextlib import closing
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio
from prometheus_client import CollectorRegistry
from redis.exceptions import ConnectionError as RedisConnectionError

try:
    import chromadb
except ModuleNotFoundError:
    sys.modules["chromadb"] = types.ModuleType("chromadb")

import api.main as api
from agents.agent_orchestrator import (
    AgentOrchestrator, AgentType, BillingAgent, GeneralAgent, TechnicalAgent,
    ResponseComposer,
)
from agents.tools import AgentToolSpec
from evaluation.evaluator import EndToEndEvaluator, QualityScores
from core.concurrency import KeyedLocks
from core.intent_recognizer import IntentCategory, IntentRecognizer, IntentResult, UrgencyLevel
from memory.conversation_memory import MemoryManager
from monitor.performance_monitor import PerformanceMonitor
from tooling.sandbox_store import DEMO_ORDER_ID, DEMO_USER_ID, SandboxStore


def text_response(text="回复完成", stop="end_turn"):
    return SimpleNamespace(content=[{"type": "text", "text": text}], stop_reason=stop)


@pytest_asyncio.fixture
async def services(monkeypatch):
    monkeypatch.setenv("ZHIYING_ENV", "development")
    monkeypatch.delenv("ZHIYING_API_TOKENS", raising=False)
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    memory._request_locks = KeyedLocks()
    memory._query_episodic = AsyncMock(return_value={})
    memory._get_profile = AsyncMock(return_value={})
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=text_response())))
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pool = {
        kind: [cls(client, "test-model")]
        for kind, cls in [(AgentType.GENERAL, GeneralAgent), (AgentType.BILLING, BillingAgent),
                          (AgentType.TECHNICAL, TechnicalAgent)]
    }
    orch._composer = ResponseComposer(client, "test-model")
    orch._recent_tool_traces = deque(maxlen=200)
    recognizer = IntentRecognizer.__new__(IntentRecognizer)

    async def recognize(message, history=None):
        return IntentResult(IntentCategory.REFUND, .95, UrgencyLevel.LOW, "billing",
                            recognizer._extract_entities(message), "test", 0)

    orch.recognize_intent = AsyncMock(side_effect=recognize)
    monkeypatch.setattr(api, "_memory", memory)
    monkeypatch.setattr(api, "_orchestrator", orch)
    monkeypatch.setattr(api, "_profile_updater", None)
    monkeypatch.setattr(api, "_monitor", None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
        yield memory, orch, client, http
    await memory._redis.aclose()


def body(message="退款规则", request_id="one", conv_id="c"):
    return {"message": message, "user_id": "u", "conv_id": conv_id, "client_request_id": request_id}


@pytest.mark.asyncio
async def test_chat_intent_shortcut_only_on_first_turn(services, monkeypatch):
    _, orchestrator, agent_model, http = services
    monkeypatch.setenv("SERVICEPILOT_INTENT_FAST_PATH", "true")
    monkeypatch.setenv("INTENT_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    intent_model = AsyncMock(return_value=text_response('{"intent":"refund","confidence":0.93}'))
    monkeypatch.setattr("core.intent_recognizer.AsyncAnthropic",
                        Mock(return_value=SimpleNamespace(messages=SimpleNamespace(create=intent_model))))
    orchestrator._intent_recognizer = IntentRecognizer("offline")
    monkeypatch.delattr(orchestrator, "recognize_intent")

    first = await http.post("/chat", json=body("我要申请退款", "first"))
    assert first.status_code == 200
    assert first.json()["primary_agent"] == "billing" and first.json()["supporting_agents"] == []
    assert first.json()["intent_source_scores"]["llm"] == 0
    intent_model.assert_not_awaited()

    second = await http.post("/chat", json=body("我要申请退款", "second"))
    assert second.status_code == 200
    assert second.json()["primary_agent"] == "billing" and second.json()["intent"] == "refund"
    assert second.json()["intent_source_scores"]["llm"] == .93
    assert "最近对话" in intent_model.call_args.kwargs["messages"][0]["content"]
    intent_model.assert_awaited_once()
    assert agent_model.messages.create.await_count == 2


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    store = SandboxStore(tmp_path / "sandbox.sqlite3")
    store.initialize()
    owner_token = "sandbox-owner-test-token-012345"
    other_token = "sandbox-other-test-token-012345"
    monkeypatch.setattr(api, "_sandbox_store", store)
    monkeypatch.setenv("SERVICEPILOT_API_TOKENS", json.dumps({
        owner_token: {"user_id": DEMO_USER_ID, "role": "user"},
        other_token: {"user_id": "other", "role": "user"},
    }))
    return store, {"Authorization": f"Bearer {owner_token}"}, {"Authorization": f"Bearer {other_token}"}


def sandbox_review_ids(store):
    with closing(sqlite3.connect(store.path)) as db:
        return [row[0] for row in db.execute("SELECT review_id FROM refund_reviews")]


@pytest.mark.asyncio
async def test_sandbox_requires_configured_authentication_at_startup(monkeypatch):
    monkeypatch.setenv("SERVICEPILOT_SANDBOX_ENABLED", "true")
    monkeypatch.setenv("SERVICEPILOT_ENV", "development")
    monkeypatch.delenv("SERVICEPILOT_API_TOKENS", raising=False)
    monkeypatch.delenv("ZHIYING_API_TOKENS", raising=False)
    monkeypatch.setattr(api, "_sandbox_store", None)
    with pytest.raises(RuntimeError, match="启用业务沙盒需要配置"):
        async with api.lifespan(api.app):
            pytest.fail("沙盒不应在未配置鉴权时启动")


@pytest.mark.asyncio
async def test_sandbox_command_is_unavailable_when_disabled(services, monkeypatch):
    _, _, model, http = services
    monkeypatch.setattr(api, "_sandbox_store", None)
    result = await http.post("/chat", json=body(f"沙盒查账 {DEMO_ORDER_ID}"))
    assert result.status_code == 404
    model.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_sandbox_chat_checks_owner_and_deduplicates_review_across_conversations(services, sandbox):
    _, _, model, http = services
    store, owner, other = sandbox
    lookup = f"沙盒查账 {DEMO_ORDER_ID}"
    confirm = f"确认提交退款核验 {DEMO_ORDER_ID}"
    assert (await http.post("/chat", json=body(lookup))).status_code == 401

    # 传入订单所有者的 user_id 不能覆盖 Token 中的身份。
    for index, command in enumerate((lookup, confirm)):
        denied = await http.post("/chat", json={**body(command, f"denied-{index}"), "user_id": DEMO_USER_ID}, headers=other)
        assert "未找到当前账号" in denied.json()["response"]

    ledger = await http.post("/chat", json=body(lookup), headers=owner)
    assert ledger.status_code == 200
    assert "199.00 元，已入账" in ledger.json()["response"]
    assert "199.00 元，预授权冻结（未入账）" in ledger.json()["response"]
    incomplete = await http.post("/chat", json=body("确认提交退款核验", "incomplete"), headers=owner)
    assert "请输入完整编号" in incomplete.json()["response"]
    assert sandbox_review_ids(store) == []

    first, concurrent = await asyncio.gather(
        http.post("/chat", json=body(confirm, "confirm"), headers=owner),
        http.post("/chat", json=body(confirm, "another", "other-conversation"), headers=owner),
    )
    assert first.status_code == concurrent.status_code == 200
    assert first.json()["response"] == concurrent.json()["response"]
    review_id, = sandbox_review_ids(store)
    assert review_id in first.json()["response"]
    replay = await http.post("/chat", json=body(confirm, "confirm"), headers=owner)
    assert replay.json() == first.json()
    assert (await http.post("/chat", json=body(lookup, "confirm"), headers=owner)).status_code == 409

    # 用重新初始化的存储验证持久性和 seed 幂等。
    reopened = SandboxStore(store.path)
    await asyncio.to_thread(reopened.initialize)
    assert len(reopened.order_ledger(DEMO_USER_ID, DEMO_ORDER_ID)["entries"]) == 2
    query = f"查询退款核验单 {review_id}"
    found = await http.post("/chat", json=body(query, "query"), headers=owner)
    assert "待人工核验" in found.json()["response"]
    invisible = await http.post("/chat", json=body(query, "query"), headers=other)
    assert "未找到当前账号" in invisible.json()["response"]
    model.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_sandbox_recovers_review_if_chat_receipt_fails_after_sqlite_commit(services, sandbox, monkeypatch):
    memory, _, model, http = services
    store, owner, _ = sandbox
    original_commit = memory.commit_turn
    saved_id = None

    async def fail_first_commit(*args, **kwargs):
        nonlocal saved_id
        if saved_id is None:
            saved_id, = sandbox_review_ids(store)
            raise RedisConnectionError("simulated lost Redis connection")
        await original_commit(*args, **kwargs)

    monkeypatch.setattr(memory, "commit_turn", fail_first_commit)
    payload = body(f"确认提交退款核验 {DEMO_ORDER_ID}", "confirm")
    assert (await http.post("/chat", json=payload, headers=owner)).status_code == 503
    retry = await http.post("/chat", json=payload, headers=owner)
    assert retry.status_code == 200
    assert saved_id in retry.json()["response"]
    assert sandbox_review_ids(store) == [saved_id]
    model.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_billing_agent_reads_sandbox_ledger_but_cannot_authorize_write(services, sandbox, monkeypatch):
    _, orch, model, http = services
    store, owner, _ = sandbox
    monkeypatch.setattr(api, "_tool_manager", SimpleNamespace())
    monkeypatch.setattr(api, "_mcp_client", None)
    monkeypatch.setattr(api, "_eval_orchestrator", None)
    api._refresh_agent_tools()
    read_tool = "query_sandbox_order_ledger"
    assert read_tool in orch._pool[AgentType.BILLING][0].get_tools()
    assert read_tool not in orch._pool[AgentType.TECHNICAL][0].get_tools()
    model.messages.create.side_effect = [
        SimpleNamespace(content=[
            {"type": "tool_use", "id": "read", "name": read_tool, "input": {"order_id": DEMO_ORDER_ID}},
            {"type": "tool_use", "id": "write", "name": "submit_sandbox_refund_review", "input": {"order_id": DEMO_ORDER_ID}},
        ]),
        text_response("已入账与预授权冻结是不同的状态。"),
    ]
    reply = await http.post("/chat", json=body(f"请帮我核对订单 {DEMO_ORDER_ID} 的扣款"), headers=owner)
    assert reply.status_code == 200
    assert reply.json()["tools_used"] == [read_tool]
    tool_results = model.messages.create.await_args.kwargs["messages"][-1]["content"]
    assert "authorization_hold" in tool_results[0]["content"]
    assert tool_results[1]["is_error"] is True
    assert sandbox_review_ids(store) == []


def stream_events(text):
    events = []
    for frame in text.split("\n\n"):
        if frame.startswith("event: "):
            event, data = frame.splitlines()
            events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("billing_text, expected_status", [
    ("两笔都已入账。", "success"),
    ("", "partial"),
])
async def test_cross_domain_chat_preserves_verified_ledger(services, sandbox, billing_text, expected_status):
    _, orch, model, http = services
    store, owner, _ = sandbox
    from agents.sandbox_tools import ORDER_LOOKUP, build_sandbox_tools

    orch.set_shared_tools(build_sandbox_tools(store))

    async def reply(**kwargs):
        tools = {tool["name"] for tool in kwargs.get("tools", [])}
        if "inspect_request_context" in tools:
            return text_response("")
        if ORDER_LOOKUP not in tools:
            return text_response("401 可先检查 Token 是否过期。")
        if isinstance(kwargs["messages"][-1]["content"], list):
            return text_response(billing_text)
        return SimpleNamespace(content=[{
            "type": "tool_use", "id": "ledger", "name": ORDER_LOOKUP,
            "input": {"order_id": DEMO_ORDER_ID},
        }])

    model.messages.create.side_effect = reply
    message = f"登录时报 401，同时订单 {DEMO_ORDER_ID} 疑似重复扣款，请分别说明如何处理"
    response = await http.post("/chat", json=body(message), headers=owner)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == expected_status
    assert ORDER_LOOKUP in result["tools_used"]
    assert "401 可先检查 Token 是否过期" in result["response"]
    assert "CHG-DEMO-01：199.00 元，已入账" in result["response"]
    assert "CHG-DEMO-02：199.00 元，预授权冻结（未入账）" in result["response"]
    assert "两笔都已入账" not in result["response"]
    assert all("tools" in call.kwargs for call in model.messages.create.await_args_list)
    assert sandbox_review_ids(store) == []


@pytest.mark.asyncio
async def test_parallel_branches_reserve_same_write_once(services):
    _, orch, model, http = services
    attempted = []

    async def write(req, args):
        attempted.append(args["case_id"])
        await asyncio.sleep(.01)
        return {"success": True}

    orch.set_shared_tools({"write_case": AgentToolSpec(
        "write_case", "test operation",
        {"type": "object", "properties": {"case_id": {"type": "string"}}, "required": ["case_id"]},
        write, retry_safe=False,
    )})

    async def reply(**kwargs):
        if not kwargs.get("tools") or isinstance(kwargs["messages"][-1]["content"], list):
            return text_response("已处理该分支。")
        return SimpleNamespace(content=[{
            "type": "tool_use", "id": "write", "name": "write_case", "input": {"case_id": "same"},
        }])

    model.messages.create.side_effect = reply
    response = await http.post("/chat", json=body("登录报 401，同时这笔订单重复扣款，分别处理"))
    assert response.status_code == 200
    trace = orch.get_tool_trace(response.json()["request_id"])
    writes = [call for call in trace["tool_calls"] if call["tool_name"] == "write_case"]
    assert len(writes) == 2
    assert sum(call["result_success"] is True for call in writes) == 1
    assert attempted == ["same"]


@pytest.mark.asyncio
async def test_chat_stream_observes_actual_billing_steps_and_can_replay_via_json(services, sandbox, monkeypatch):
    memory, _, model, http = services
    _, owner, _ = sandbox
    monkeypatch.setattr(api, "_tool_manager", SimpleNamespace())
    monkeypatch.setattr(api, "_mcp_client", None)
    monkeypatch.setattr(api, "_eval_orchestrator", None)
    api._refresh_agent_tools()
    tool = "query_sandbox_order_ledger"
    model.messages.create.side_effect = [
        SimpleNamespace(content=[{
            "type": "tool_use", "id": "read", "name": tool, "input": {"order_id": DEMO_ORDER_ID},
        }]),
        text_response(),
    ]
    payload = body(f"请核对 {DEMO_ORDER_ID} 的扣款")
    assert (await http.post("/chat/stream", json=payload)).status_code == 401
    result = await http.post("/chat/stream", json=payload, headers=owner)
    assert result.headers["content-type"].startswith("text/event-stream")
    assert result.headers["x-accel-buffering"] == "no"
    events = stream_events(result.text)
    assert [data["step"] for kind, data in events if kind == "stage"] == [
        "lease_acquired", "intent_recognized", "agent_routing", "tool_call", "committing",
    ]
    assert events[-3][1] == {"step": "tool_call", "tool": tool, "agent": "billing"}
    final_kind, final = events[-1]
    assert final_kind == "final" and final["tools_used"] == [tool]
    assert "agent_execution" in final["performance"]["stages_ms"]
    assert (await http.post("/chat", json=payload, headers=owner)).json() == final
    assert model.messages.create.await_count == 2
    assert len(await memory.get_history(DEMO_USER_ID, payload["conv_id"])) == 2
    conflict = await http.post("/chat/stream", json={**payload, "message": "不同问题"}, headers=owner)
    assert stream_events(conflict.text)[-1][1]["status"] == 409


@pytest.mark.asyncio
async def test_chat_stream_disconnect_during_sandbox_commit_replays_one_turn(services, sandbox, monkeypatch):
    memory, _, _, http = services
    store, owner, _ = sandbox
    payload = body(f"确认提交退款核验 {DEMO_ORDER_ID}")
    disconnected, resuming = asyncio.Event(), asyncio.Event()
    original_commit = memory.commit_turn
    commit_started = asyncio.Event()
    sent = []

    async def paused_commit(*args, **kwargs):
        commit_started.set()
        await resuming.wait()
        return await original_commit(*args, **kwargs)

    monkeypatch.setattr(memory, "commit_turn", paused_commit)
    inbound = asyncio.Queue()
    inbound.put_nowait({"type": "http.request", "body": json.dumps(payload).encode()})

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and b'"step": "committing"' in message.get("body", b""):
            inbound.put_nowait({"type": "http.disconnect"})
            disconnected.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/chat/stream", "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"authorization", owner["Authorization"].encode())],
    }
    connection = asyncio.create_task(api.app(scope, inbound.get, send))
    retry = None
    try:
        await asyncio.wait_for(disconnected.wait(), 2)
        await asyncio.wait_for(commit_started.wait(), 2)
        await asyncio.wait_for(connection, 2)
        assert not any(b"event: final" in message.get("body", b"") for message in sent)
        review_id, = sandbox_review_ids(store)
        assert await memory.get_history(DEMO_USER_ID, payload["conv_id"]) == []
        retry = asyncio.create_task(http.post("/chat/stream", json=payload, headers=owner))
        resuming.set()
        events = stream_events((await asyncio.wait_for(retry, 2)).text)
        assert [data["step"] for kind, data in events if kind == "stage"] == ["lease_acquired", "replayed"]
        assert events[-1][0] == "final" and review_id in events[-1][1]["response"]
        assert sandbox_review_ids(store) == [review_id]
        assert len(await memory.get_history(DEMO_USER_ID, payload["conv_id"])) == 2
    finally:
        resuming.set()
        await asyncio.gather(connection, retry or connection, *api._stream_turns, return_exceptions=True)


@pytest.mark.asyncio
async def test_chat_reports_stage_timings_and_provider_usage_without_changing_replay(services):
    memory, orch, client, http = services
    client.messages.create.return_value = SimpleNamespace(content=[{"type":"text", "text":"回复完成"}],
        stop_reason="end_turn", usage=SimpleNamespace(input_tokens=15, output_tokens=8))
    first = await http.post("/chat", json=body())
    perf = first.json()["performance"]
    assert {"memory_read", "intent", "agent_execution"} <= set(perf["stages_ms"])
    assert perf["llm"]["calls"] == 1 and perf["llm"]["output_tokens"] == 8
    assert first.json() == (await http.post("/chat", json=body())).json()


@pytest.mark.asyncio
async def test_failed_turn_can_retry_after_recovery_without_duplicate_history(services):
    memory, orch, model, http = services
    model.messages.create.side_effect = RuntimeError("provider down")
    first = await http.post("/chat", json=body())
    assert first.status_code == 503
    assert first.json()["status"] == "failed" and first.json()["retryable"]
    assert await memory._redis.llen(memory._wm_key("u", "c")) == 0
    assert not await memory._redis.exists(memory._state_key("u", "c"))
    assert (await http.post("/chat", json=body("另一条消息"))).status_code == 409
    model.messages.create.side_effect = None
    recovered = await http.post("/chat", json=body())
    repeat = await http.post("/chat", json=body())
    assert recovered.status_code == repeat.status_code == 200
    assert recovered.json() == repeat.json()
    assert recovered.json()["status"] == "success"
    assert model.messages.create.await_count == 3  # 两次失败尝试 + 一次成功
    assert await memory._redis.llen(memory._wm_key("u", "c")) == 2
    assert await memory.get_trace(recovered.json()["request_id"])


@pytest.mark.asyncio
@pytest.mark.parametrize("text,stop,status,http_status", [
    ("", "end_turn", "failed", 503),
    ("退款需要先", "max_tokens", "partial", 200),
])
async def test_empty_and_truncated_answers_are_not_recorded_as_complete(services, text, stop, status, http_status):
    memory, orch, model, http = services
    model.messages.create.return_value = text_response(text, stop)
    response = await http.post("/chat", json=body())
    assert response.status_code == http_status and response.json()["status"] == status
    assert orch._pool[AgentType.BILLING][0].stats.success == 0
    if status == "partial":
        repeat = await http.post("/chat", json=body())
        assert repeat.json() == response.json()
        assert model.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_unsafe_tool_is_recorded_before_execution_and_not_replayed(services):
    memory, orch, model, http = services
    writes = []

    async def write(req, args):
        marker = await memory.get_turn_receipt("u", "c", "one")
        assert marker["response"]["error_code"] == "tool_outcome_unknown"
        assert not marker["response"]["retryable"]
        writes.append(args)
        return {"success": True}

    orch.set_shared_tools({"write": AgentToolSpec("write", "test", {"type": "object"}, write, retry_safe=False)})
    model.messages.create.side_effect = [
        SimpleNamespace(content=[{"type": "tool_use", "id": "w", "name": "write", "input": {}}]),
        RuntimeError("lost model response"), RuntimeError("fallback down"),
    ]
    first = await http.post("/chat", json=body())
    assert first.status_code == 503 and not first.json()["retryable"]
    model.messages.create.side_effect = None
    repeat = await http.post("/chat", json=body())
    assert repeat.json() == first.json()
    assert len(writes) == 1 and model.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_repeated_write_within_one_model_tool_loop_executes_once(services):
    memory, orch, model, http = services
    write = AsyncMock(return_value={"success": True})
    orch.set_shared_tools({"write": AgentToolSpec("write", "test", {"type": "object"}, write, retry_safe=False)})
    model.messages.create.side_effect = [
        SimpleNamespace(content=[{"type": "tool_use", "id": "w1", "name": "write", "input": {}}]),
        SimpleNamespace(content=[{"type": "tool_use", "id": "w2", "name": "write", "input": {}}]),
        text_response("已处理"),
    ]
    response = await http.post("/chat", json=body())
    assert response.status_code == 200 and write.await_count == 1
    trace = await memory.get_trace(response.json()["request_id"])
    assert trace["tool_calls"][1]["success"] is False


@pytest.mark.asyncio
async def test_unsafe_tool_timeout_preserves_non_replayable_receipt(services, monkeypatch):
    memory, orch, model, http = services
    calls = 0

    async def generate(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(content=[{"type": "tool_use", "id": "w", "name": "write", "input": {}}])
        await asyncio.Event().wait()

    write = AsyncMock(return_value={"success": True})
    orch.set_shared_tools({"write": AgentToolSpec("write", "test", {"type": "object"}, write, retry_safe=False)})
    model.messages.create.side_effect = generate
    original_timeout = asyncio.timeout
    monkeypatch.setattr(api.asyncio, "timeout", lambda seconds: original_timeout(.2))
    first = await http.post("/chat", json=body())
    assert first.status_code == 504
    repeat = await http.post("/chat", json=body())
    assert repeat.status_code == 503 and not repeat.json()["retryable"]
    assert write.await_count == 1 and calls == 2
    assert await memory._redis.llen(memory._wm_key("u", "c")) == 0


@pytest.mark.asyncio
async def test_concurrent_retries_execute_once_and_bind_authenticated_identity(services, monkeypatch):
    memory, orch, model, http = services
    token = "test-token-with-at-least-24-characters"
    monkeypatch.setenv("ZHIYING_API_TOKENS", json.dumps({token: {"user_id": "real", "role": "user"}}))
    headers = {"Authorization": "Bearer " + token}
    first, second = await asyncio.gather(*(http.post("/chat", json=body(), headers=headers) for _ in range(2)))
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() and model.messages.create.await_count == 1
    assert await memory._redis.llen(memory._wm_key("real", "c")) == 2
    assert not await memory._redis.exists(memory._wm_key("u", "c"))
    assert (await http.get("/monitor", headers=headers)).status_code == 403
    assert (await http.post("/chat", json=body())).status_code == 401


@pytest.mark.asyncio
async def test_slots_survive_turns_but_reset_for_another_order_or_conversation(services):
    memory, orch, model, http = services
    await http.post("/chat", json=body("订单号 #12345，昨天支付99元", "first"))
    second = await http.post("/chat", json=body("退款多久到？", "second"))
    assert second.json()["entities"]["order_id"] == ["12345"]
    assert second.json()["entities"]["amount"] == ["99元"]
    assert second.json()["entities"]["date"] == ["昨天"]
    third = await http.post("/chat", json=body("另一笔订单 #67890", "third"))
    assert third.json()["entities"]["order_id"] == ["67890"]
    assert not third.json()["entities"]["amount"] and not third.json()["entities"]["date"]
    other = await http.post("/chat", json=body("退款多久到？", "fourth", "other"))
    assert not other.json()["entities"]["order_id"]
    state = json.loads(await memory._redis.get(memory._state_key("u", "c")))
    assert state["sources"]["order_id"] == third.json()["request_id"]


@pytest.mark.asyncio
async def test_dialog_evaluation_uses_same_slot_merge_as_chat(services):
    memory, orch, model, http = services
    orch._intent_recognizer = SimpleNamespace(recognize=orch.recognize_intent)
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._orchestrator = orch
    evaluator._judge = SimpleNamespace(judge=AsyncMock(return_value=QualityScores(1, 1, 1, 1)))
    results = await evaluator._evaluate_dialog_case({
        "turns": ["订单号 #12345，昨天支付99元", "退款多久到？"],
        "expected_entities": {"order_id": ["12345"], "amount": ["99元"]},
    }, 0)
    assert results[-1].passed and results[-1].metadata["deterministic_checks"]["entities"]
    assert results[-1].metadata["entities"]["date"] == ["昨天"]


def raw_message(index):
    return json.dumps({"role": "user", "content": f"old-{index}",
                       "ts": datetime.now().isoformat(), "metadata": {}})


@pytest.mark.asyncio
async def test_failed_episodic_archive_is_retried_after_trimming(services):
    memory, *_ = services
    summarize = AsyncMock(return_value=text_response("本次归档摘要"))
    memory._client = SimpleNamespace(messages=SimpleNamespace(create=summarize))
    memory._model = "fake"
    upsert = Mock(side_effect=[RuntimeError("Chroma unavailable"), None])
    memory._episodic = SimpleNamespace(upsert=upsert)
    memory.update_profile = AsyncMock()
    key = memory._wm_key("u", "c")
    await memory._redis.rpush(key, *(raw_message(i) for i in reversed(range(15))))

    await memory.maintain("u", "c")
    assert await memory._redis.llen(key) == 5
    assert await memory._redis.get(memory._summary_key("u", "c")) == "本次归档摘要"
    upsert.assert_called_once()

    # Simulate another worker handling the next maintenance after the original exits.
    restarted = MemoryManager.__new__(MemoryManager)
    restarted._redis = memory._redis
    restarted._episodic = memory._episodic
    restarted.update_profile = AsyncMock()
    await restarted.maintain("u", "c")
    assert upsert.call_count == 2
    first, retry = (call.kwargs for call in upsert.call_args_list)
    assert retry["ids"] == first["ids"]
    assert retry["documents"] == ["本次归档摘要"]
    assert retry["metadatas"][0]["full_text"] == "\n".join(f"user: old-{i}" for i in range(10))
    summarize.assert_awaited_once()
    await restarted.maintain("u", "c")
    assert upsert.call_count == 2


@pytest.mark.asyncio
async def test_episodic_archive_survives_lost_compression_commit_response(services, monkeypatch):
    memory, *_ = services
    summarize = AsyncMock(return_value=text_response("已提交的摘要"))
    memory._client = SimpleNamespace(messages=SimpleNamespace(create=summarize))
    memory._model = "fake"
    memory._episodic = SimpleNamespace(upsert=Mock())
    memory.update_profile = AsyncMock()
    key = memory._wm_key("u", "c")
    await memory._redis.rpush(key, *(raw_message(i) for i in reversed(range(15))))
    pipeline = type(memory._redis.pipeline())
    original_execute = pipeline.execute

    async def lose_commit_response(pipe, *args, **kwargs):
        is_watched_commit = pipe.watching
        result = await original_execute(pipe, *args, **kwargs)
        if is_watched_commit:
            raise RedisConnectionError("lost EXEC response after commit")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "execute", lose_commit_response)
        with pytest.raises(RedisConnectionError, match="lost EXEC response"):
            await memory.maintain("u", "c")
    assert await memory._redis.llen(key) == 5
    memory._episodic.upsert.assert_not_called()

    await memory.maintain("u", "c")
    memory._episodic.upsert.assert_called_once()
    assert memory._episodic.upsert.call_args.kwargs["documents"] == ["已提交的摘要"]
    summarize.assert_awaited_once()


@pytest.mark.asyncio
async def test_episodic_archive_can_replay_after_cancellation_before_ack(services, monkeypatch):
    memory, *_ = services
    memory._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(
        return_value=text_response("已写入的摘要"))))
    memory._model = "fake"
    memory.update_profile = AsyncMock()
    stored_documents = {}

    def upsert(*, ids, documents, metadatas):
        stored_documents[ids[0]] = documents[0]

    memory._episodic = SimpleNamespace(upsert=Mock(side_effect=upsert))
    await memory._redis.rpush(memory._wm_key("u", "c"),
                             *(raw_message(i) for i in reversed(range(15))))
    before_ack = asyncio.Event()
    original_lrem = memory._redis.lrem

    async def block_ack(*args):
        before_ack.set()
        await asyncio.Future()

    monkeypatch.setattr(memory._redis, "lrem", block_ack)
    first = asyncio.create_task(memory.maintain("u", "c"))
    try:
        await asyncio.wait_for(before_ack.wait(), 1)
        assert list(stored_documents.values()) == ["已写入的摘要"]
        # A competing maintenance cannot replay while the first writer owns the archive lease.
        await memory.maintain("u", "c")
        memory._episodic.upsert.assert_called_once()
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        monkeypatch.setattr(memory._redis, "lrem", original_lrem)

    await memory.maintain("u", "c")
    assert memory._episodic.upsert.call_count == 2
    assert list(stored_documents.values()) == ["已写入的摘要"]
    await memory.maintain("u", "c")
    assert memory._episodic.upsert.call_count == 2


@pytest.mark.asyncio
async def test_background_compression_does_not_block_chat_or_drop_new_messages(services):
    memory, orch, model, http = services
    started, finish = asyncio.Event(), asyncio.Event()

    async def summarize(**kwargs):
        started.set()
        await finish.wait()
        return text_response("有效摘要")

    memory._client = SimpleNamespace(messages=SimpleNamespace(create=summarize))
    memory._model = "fake"
    memory._merge_summary = AsyncMock(return_value="有效摘要")
    memory._store_episodic = AsyncMock()
    memory.update_profile = AsyncMock()
    key = memory._wm_key("u", "c")
    await memory._redis.rpush(key, *(raw_message(i) for i in reversed(range(25))))
    maintain = asyncio.create_task(memory.maintain("u", "c"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        response = await asyncio.wait_for(http.post("/chat", json=body()), 1)
        assert response.status_code == 200 and not maintain.done()
        finish.set()
        await maintain
        rows = [json.loads(row) for row in await memory._redis.lrange(key, 0, -1)]
        assert len(rows) == 7 and rows[0]["role"] == "assistant" and rows[1]["content"] == "退款规则"
        assert await memory._redis.get(memory._summary_key("u", "c")) == "有效摘要"
        memory._store_episodic.assert_awaited_once()
    finally:
        finish.set()
        await asyncio.gather(maintain, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_compression_commits_only_one_snapshot(services):
    memory, *_ = services
    count = 0
    started, finish = asyncio.Event(), asyncio.Event()

    async def summarize(**kwargs):
        nonlocal count
        count += 1
        attempt = count
        if attempt == 2:
            started.set()
        await finish.wait()
        return text_response(f"快照摘要 {attempt}")

    memory._client = SimpleNamespace(messages=SimpleNamespace(create=summarize))
    memory._model = "fake"
    memory._store_episodic = AsyncMock()
    key = memory._wm_key("u", "c")
    await memory._redis.rpush(key, *(raw_message(i) for i in reversed(range(25))))
    tasks = [asyncio.create_task(memory._compress("u", "c")) for _ in range(2)]
    try:
        await asyncio.wait_for(started.wait(), 1)
        finish.set()
        await asyncio.gather(*tasks)
        assert await memory._redis.llen(key) == 5
        memory._store_episodic.assert_awaited_once()
        committed_summary = await memory._redis.get(memory._summary_key("u", "c"))
        assert committed_summary in {"快照摘要 1", "快照摘要 2"}
        assert memory._store_episodic.await_args.args[-1] == committed_summary
    finally:
        finish.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_http_timing_includes_memory_commit_and_metrics_work_without_extra_port(services, monkeypatch):
    memory, orch, model, http = services
    registry = CollectorRegistry()
    monitor = PerformanceMonitor(orch, SimpleNamespace(), registry=registry)
    monkeypatch.setattr(api, "_monitor", monitor)
    original_commit = memory.commit_turn
    commit_delay_ms = 0.0

    async def delayed_commit(*args, **kwargs):
        nonlocal commit_delay_ms
        started = time.monotonic()
        await asyncio.sleep(.05)
        commit_delay_ms = (time.monotonic() - started) * 1000
        return await original_commit(*args, **kwargs)

    monkeypatch.setattr(memory, "commit_turn", delayed_commit)
    response = await http.post("/chat", json=body())
    total = float(response.headers["Server-Timing"].split("=")[1])
    assert commit_delay_ms > 0
    assert total >= response.json()["latency_ms"] + commit_delay_ms - .2
    samples = [s for metric in registry.collect() for s in metric.samples]
    assert next(s.value for s in samples if s.name == "requests_total") == 1
    assert next(s.value for s in samples if s.name == "chat_latency_ms_count") == 1
    histogram = monitor._prom["agent_latency_ms"].labels(agent="bucket-check")
    histogram.observe(3000)
    assert next(s.value for s in histogram.collect()[0].samples
                if s.name.endswith("_bucket") and s.labels["le"] == "3000.0") == 1

@pytest.mark.asyncio
async def test_batch4_http_chat_fixture_receipt_and_current_order(services, tmp_path):
    """scope=http_chat_fixture: real route/transactions, fixture model and fake Redis."""
    memory, orch, model, http = services
    responses=[]
    for index, message in enumerate(['订单号 #11111，支付99元', '改为订单号 #22222', '这笔订单还缺什么资料']):
        payload=body(message, f'batch4-{index}', 'batch4')
        first=await http.post('/chat',json=payload)
        assert first.status_code == 200
        calls=model.messages.create.await_count
        retry=await http.post('/chat',json=payload)
        assert retry.json() == first.json()
        assert model.messages.create.await_count == calls
        responses.append(first.json())
    assert responses[2]['entities']['order_id'] == ['22222']
    assert not responses[2]['entities']['amount']
    other=await http.post('/chat',json={**body('这笔订单还缺什么资料','isolated','batch4'),'user_id':'another-user'})
    assert not other.json()['entities']['order_id']
    assert model.messages.create.await_args.kwargs['messages'][-1]['content']
    from pathlib import Path
    from uuid import uuid4
    destination=tmp_path/'http_chat_fixture'/uuid4().hex
    destination.mkdir(parents=True)
    (destination/'report.json').write_text(json.dumps({'scope':'http_chat_fixture','mode':'fixture', 'status':'completed', 'checks':{'receipt_retries':3,'current_order':'22222','user_isolation':True},'requests':[r['request_id'] for r in responses]},ensure_ascii=False),encoding='utf-8')

@pytest.mark.asyncio
async def test_budget_failure_keeps_retry_receipt_without_committing_history(services,monkeypatch):
    from core.request_budget import chat_budget
    memory,orch,model,http=services
    async def slow(**kwargs):await asyncio.Event().wait()
    model.messages.create.side_effect=slow
    monkeypatch.setattr(api,'chat_budget',lambda seconds:chat_budget(.01))
    payload=body('退款规则','budget-failure','budget-test')
    response=await http.post('/chat',json=payload)
    assert response.status_code==503 and response.json()['retryable']
    assert await memory._redis.llen(memory._wm_key('u','budget-test'))==0
    assert any(event['status'] in ('timeout','not_called') for event in response.json()['performance']['budget_events'])
    monkeypatch.setattr(api,'chat_budget',chat_budget)
    model.messages.create.side_effect=None
    assert (await http.post('/chat',json=payload)).status_code==200
    assert await memory._redis.llen(memory._wm_key('u','budget-test'))==2
