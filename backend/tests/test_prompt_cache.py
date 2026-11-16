"""system 参数的缓存布局：默认保持字符串，显式断点只在开启后出现。"""
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import BillingAgent, Request


class FakeClient:
    def __init__(self, response):
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                return response

        self.messages = Messages()


def make_request(message):
    return Request(message=message, user_id="u1", conv_id="c1")


class Skills:
    def prompt_for(self, message, agent_type=None):
        return f"skill for {agent_type}"


def text_response():
    return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn")


async def system_sent(skill_manager):
    client = FakeClient(text_response())
    agent = BillingAgent(client, "test", skill_manager=skill_manager)
    await agent._call_llm(make_request(message="你好"))
    return agent, client.calls[0]["system"]


@pytest.mark.asyncio
async def test_default_system_is_the_unchanged_string(monkeypatch):
    monkeypatch.delenv("ZHIYING_PROMPT_CACHE_BREAKPOINTS", raising=False)
    agent, system = await system_sent(Skills())
    assert isinstance(system, str)
    assert system == agent._build_system_prompt(make_request(message="你好"))
    assert system.endswith("\n\n[动态 Skills]\nskill for billing")


@pytest.mark.asyncio
async def test_breakpoints_split_static_contract_from_skills(monkeypatch):
    monkeypatch.setenv("ZHIYING_PROMPT_CACHE_BREAKPOINTS", "1")
    agent, system = await system_sent(Skills())
    assert [block["cache_control"] for block in system] == [{"type": "ephemeral"}] * 2
    # 第一段不含任何随请求变化的内容，第二段才是 Skills；拼起来与字符串形式一致。
    assert system[0]["text"] == agent._system_parts(make_request(message="完全不同的另一条消息"))[0]
    assert "".join(block["text"] for block in system) == agent._build_system_prompt(make_request(message="你好"))


@pytest.mark.asyncio
async def test_breakpoints_without_skills_send_one_block(monkeypatch):
    monkeypatch.setenv("ZHIYING_PROMPT_CACHE_BREAKPOINTS", "1")
    _, system = await system_sent(None)
    assert len(system) == 1 and system[0]["cache_control"] == {"type": "ephemeral"}
