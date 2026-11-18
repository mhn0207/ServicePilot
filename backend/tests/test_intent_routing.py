"""Cold-cache intent and routing cases with fixed LLM responses and real local n-grams."""
import asyncio
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.agent_orchestrator import AgentOrchestrator, AgentType, Request
from core.conversation_state import merge_state
from core.intent_recognizer import IntentRecognizer


@dataclass(frozen=True)
class RouteCase:
    id: str
    message: str
    llm_intent: str
    primary: str
    supporting: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    amounts: tuple[str, ...] = ()
    order_ids: tuple[str, ...] = ()
    history: tuple[tuple[str, str], ...] = ()
    previous_amount: str | None = None
    llm_confidence: float = .93
    clarification: bool = False
    will_shortcut: bool = False


CASES = (
    RouteCase("login", "登录一直报401", "technical_login", "technical", error_codes=("401",), will_shortcut=True),
    RouteCase("login-code", "验证码收不到", "technical_login", "technical", will_shortcut=True),
    RouteCase("crash", "页面报500错误", "technical_crash", "technical", error_codes=("500",), will_shortcut=True),
    RouteCase("refund", "我要申请退款", "refund", "billing", will_shortcut=True),
    RouteCase("refund-timing", "退款多久到账？", "refund", "billing", will_shortcut=True),
    RouteCase("invoice", "帮我开发票", "invoice", "billing", will_shortcut=True),
    RouteCase("payment", "为什么重复扣款？", "payment_issue", "billing", will_shortcut=True),
    RouteCase("handoff", "我要找人工", "human_handoff", "escalation", will_shortcut=True),
    RouteCase("composite", "登录报401，而且同一订单被重复扣款了", "technical_login", "technical",
              supporting=("billing",), error_codes=("401",)),
    RouteCase("amount-500", "这笔订单被重复扣款了 500 元", "payment_issue", "billing", amounts=("500 元",)),
    RouteCase("amount-401", "我充值 401 元没到账", "billing", "billing", amounts=("401 元",)),
    RouteCase("hotline", "400 客服电话打不通", "request", "general"),
    RouteCase("context-correction", "不是 500 元，是 100 元", "refund", "billing", amounts=("100 元",),
              history=(("user", "我想申请退款 500 元"), ("assistant", "请确认退款金额")),
              previous_amount="500元"),
    RouteCase("same-text-with-history", "我要申请退款", "refund", "billing",
              history=(("user", "之前的问题已解决"),)),
    RouteCase("clarify", "有点不对，帮我看一下", "other", "general", llm_confidence=.22, clarification=True),
    RouteCase("bare-number", "401", "other", "general", llm_confidence=.32, clarification=True),
    RouteCase("order", "请查订单号 ORD-20261024 的处理状态", "order_status", "general",
              order_ids=("ORD-20261024",)),
    RouteCase("negated-handoff", "不是要转人工，只是想问发票怎么开", "invoice", "billing"),
    RouteCase("technical-variant", "登录失败，接口返回401", "technical_login", "technical", error_codes=("401",)),
)


async def run_case(case, *, artificial_delay=0):
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        if artificial_delay:
            await asyncio.sleep(artificial_delay)
        return SimpleNamespace(content=[{"type": "text", "text": json.dumps({
            "intent": case.llm_intent, "confidence": case.llm_confidence,
        })}])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    with patch("core.intent_recognizer.AsyncAnthropic", return_value=client):
        recognizer = IntentRecognizer("offline-test")
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._intent_recognizer = recognizer
    orchestrator._pool = {kind: [SimpleNamespace(stats=SimpleNamespace(available=lambda: True))]
                          for kind in AgentType}
    history = [{"role": role, "content": content} for role, content in case.history]
    started = time.perf_counter()
    result = await orchestrator.recognize_intent(case.message, history)
    elapsed_ms = (time.perf_counter() - started) * 1000
    req = Request(message=case.message, user_id="test", conv_id="test", intent=result.intent,
                  intent_group=result.intent_group, urgency=result.urgency,
                  intent_confidence=result.confidence, entities=result.entities, history=history)
    if case.previous_amount:
        req.entities = merge_state({"domain": "commerce", "entities": {"amount": [case.previous_amount]}},
                                   req.entities, req.intent_group, req.message, req.request_id)["entities"]
    clarify = orchestrator._needs_clarification(req)
    decision = orchestrator._route_decision(req)
    actual = dict(intent=result.intent.value, primary="general" if clarify else decision.primary_agent.value,
                  supporting=[] if clarify else [agent.value for agent in decision.supporting_agents],
                  error_codes=req.entities["error_code"], amounts=req.entities["amount"],
                  order_ids=req.entities["order_id"], clarification=clarify)
    expected = dict(intent=case.llm_intent, primary=case.primary, supporting=list(case.supporting),
                    error_codes=list(case.error_codes), amounts=list(case.amounts),
                    order_ids=list(case.order_ids), clarification=case.clarification)
    return dict(id=case.id, actual=actual, expected=expected, calls=calls, confidence=result.confidence,
                source_scores=result.source_scores, latency_ms=elapsed_ms)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
async def test_intent_and_routes_with_fast_path(case, monkeypatch):
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("INTENT_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("SERVICEPILOT_INTENT_FAST_PATH", "true")
    row = await run_case(case)
    assert row["actual"] == row["expected"]
    assert row["calls"] == (0 if case.will_shortcut else 1)
    assert 0 <= row["confidence"] <= 1
