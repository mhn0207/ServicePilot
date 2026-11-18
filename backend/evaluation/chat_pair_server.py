"""Docker-only entry point for the isolated, real HTTP chat comparison."""

import os
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from agents.agent_orchestrator import AgentOrchestrator, AgentType, BaseAgent, GeneralAgent, RoutingDecision
from api import main


LEDGER_FAULT = Path("/tmp/servicepilot-ledger-unavailable")


class SingleServiceAgent(GeneralAgent):
    system_prompt = (
        "你是 ServicePilot 的企业客服。独立处理技术排障、账单、订阅和政策问题。"
        "复合请求逐项回答。用可用的只读工具核实订单、账单和技术事实。"
        "严格区分工具核实的事实、建议和未知结果。无法核实时说明限制，不承诺或执行退款。"
    )
    profile = replace(
        GeneralAgent.profile,
        role="企业客服 Agent",
        mission="独立完成技术排障、账单核对和政策咨询，覆盖用户的每项诉求。",
        workflow=("识别所有诉求", "按需核对工具和知识库", "逐项回答并说明证据与限制"),
        tool_scope=(),
    )

    def _build_role_packet(self, req):
        return BaseAgent._build_role_packet(self, req)


class SingleAgentOrchestrator(AgentOrchestrator):
    def _route_decision(self, req):
        return RoutingDecision(primary_agent=AgentType.GENERAL, reason="single-agent baseline", confidence=1.0)


@asynccontextmanager
async def paired_lifespan(app):
    arm = os.environ["SERVICEPILOT_CHAT_PAIR_ARM"]
    if arm not in {"single", "routed"}:
        raise ValueError("SERVICEPILOT_CHAT_PAIR_ARM must be single or routed")
    async with main.lifespan(app):
        if main._sandbox_store is None:
            raise RuntimeError("The chat comparison requires an isolated billing sandbox")

        order_ledger = main._sandbox_store.order_ledger

        def read_ledger(user_id, order_id):
            if LEDGER_FAULT.exists():
                raise sqlite3.OperationalError("evaluation: ledger temporarily unavailable")
            return order_ledger(user_id, order_id)

        main._sandbox_store.order_ledger = read_ledger
        if arm == "single":
            routed = main._orchestrator
            original = routed._pool[AgentType.GENERAL][0]
            cfg = main._anthropic_cfg()
            baseline = SingleAgentOrchestrator(
                **cfg, client=original._client, intent_recognizer=routed._intent_recognizer,
                skill_manager=main._skill_manager,
            )
            baseline_agent = SingleServiceAgent(original._client, cfg["model"], main._skill_manager)
            tools = {
                name: replace(spec, allowed_agents=None)
                for agents in routed._pool.values() for agent in agents
                for name, spec in agent.get_tools().items() if spec.retry_safe
            }
            baseline._pool = {AgentType.GENERAL: [baseline_agent]}
            baseline.set_shared_tools(tools)
            main._orchestrator = baseline
        yield


app = main.app
app.router.lifespan_context = paired_lifespan
