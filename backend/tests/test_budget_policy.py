"""预算参数集中在 BudgetPolicy；任何一项改动都必须保持阶段之和不超过请求预算。"""
from dataclasses import replace

import pytest

from core.request_budget import POLICY, BudgetPolicy, model_timeout, request_budget


def test_default_policy_is_consistent():
    assert POLICY.validate() is POLICY
    assert POLICY.intent_model_s + POLICY.primary_s + POLICY.fallback_min_start_s <= POLICY.request_s


@pytest.mark.parametrize('change', [
    dict(hard_timeout_s=45),            # 外层超时不再大于请求预算
    dict(primary_s=30),                 # 意图 + 主 Agent + 兜底 超出 45 秒
    dict(primary_reserve_s=4),          # 留给汇总的时间不够一次调用
    dict(fallback_s=9),                 # 兜底阶段超出其启动门槛
    dict(fallback_reserve_s=1),         # 兜底之后不够启动汇总
    dict(rag_optional_min_start_s=17),  # 重排之后不够一次 Agent 调用
    dict(agent_model_s=16),             # 预算内上限超过无预算默认超时
])
def test_inconsistent_policy_is_rejected(change):
    with pytest.raises(ValueError, match='预算参数不一致'):
        replace(POLICY, **change).validate()


def test_model_caps_come_from_policy():
    with request_budget(POLICY.request_s, clock=lambda: 0.0):
        assert model_timeout('intent') == POLICY.intent_model_s
        assert model_timeout('rag.rewrite') == POLICY.rewrite_model_s
        assert model_timeout('rag.rerank') == POLICY.rerank_model_s
        assert model_timeout('composer') == POLICY.composer_model_s
        assert model_timeout('agent.billing') == POLICY.agent_model_s
    assert BudgetPolicy().model_cap('unknown') == POLICY.agent_model_s
