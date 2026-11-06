"""评测约束：高 Judge 分不能掩盖漏掉的协作角色，检索故障不能算命中。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from agents.agent_orchestrator import AgentType
from api.main import EvalRunInput
from evaluation.evaluator import EndToEndEvaluator, QualityScores
from scripts import evaluate_retrieval


@pytest.mark.asyncio
@pytest.mark.parametrize("agents,passed", [
    ([AgentType.TECHNICAL], False),
    ([AgentType.TECHNICAL, AgentType.BILLING], True),
    ([], False),
])
async def test_high_judge_score_cannot_hide_missing_collaboration(agents, passed):
    # 经真实 API 输入模型解析，确保字段不会在入口被丢弃。
    body = EvalRunInput(dialog_cases=[{
        "question": "登录401且重复扣款", "expected_agents": ["technical", "billing"],
    }])
    case = body.dialog_cases[0].model_dump(exclude_none=True)
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._orchestrator = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(
        response="回复", agent_type=AgentType.TECHNICAL, agent_types=agents, intent=None,
        status="success", tools_used=[])))
    evaluator._judge = SimpleNamespace(judge=AsyncMock(return_value=QualityScores(1, 1, 1, 1)))
    result, = await evaluator._evaluate_dialog_case(case, 0)
    assert result.passed is passed
    assert result.metadata["deterministic_checks"]["agents"] is passed
    assert result.metadata["agent_types"] == [agent.value for agent in agents]


def test_retrieval_checks_rank_and_excludes_fallback_and_out_of_budget_results():
    rows = [{"title": "目标", "fallback": True}, {"title": "其他"}, {"title": "目标"}]
    assert evaluate_retrieval.score_results(rows, "目标", 2) == {"hit": False, "reciprocal_rank": 0}
    assert evaluate_retrieval.score_results(rows, "目标", 3) == {"hit": True, "reciprocal_rank": 1 / 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_multiturn_judge_never_sees_future_order_reference(explicit):
    case = {"turns": ["订单11111想退款", "换成订单22222"],
            "reference_answer": "最终应使用订单22222", "evidence": "最终订单22222"}
    if explicit:
        case["turn_references"] = [
            {"reference_answer": "当前仅有订单11111", "evidence": "请补齐支付信息"},
            {"reference_answer": "当前订单22222", "evidence": "旧订单已替换"}]
    body = EvalRunInput(dialog_cases=[case])
    parsed = body.dialog_cases[0].model_dump(exclude_none=True)
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._orchestrator = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(
        response="需要核验信息", agent_type=AgentType.BILLING, intent=None, status="success")))
    judge = AsyncMock(return_value=QualityScores(.9, .9, .8, .9))
    evaluator._judge = SimpleNamespace(judge=judge)
    results = await evaluator._evaluate_dialog_case(parsed, 0)
    early, final = [call.kwargs["context"] for call in judge.call_args_list]
    assert "22222" not in early
    assert "22222" in final
    assert results[0].metadata["reference_scope"] == ("turn" if explicit else "current_only")
    assert results[0].metadata["ground_truth_provided"] is explicit


def test_api_rejects_misaligned_turn_references():
    with pytest.raises(ValueError, match="turn_references"):
        EvalRunInput(dialog_cases=[{"turns": ["第一轮", "第二轮"],
                                   "turn_references": [{"reference_answer": "只一轮"}]}])


@pytest.mark.asyncio
async def test_retrieval_http_errors_remain_in_metric_denominator(monkeypatch):
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-token"
        if request.url.params["query"] == "失败":
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(200, json={"results": [{"title": "其他"}, {"title": "目标"}],
                                         "cached": False, "reranked": True})

    client_class = httpx.AsyncClient
    monkeypatch.setenv("ZHIYING_EVAL_TOKEN", "test-token")
    monkeypatch.setattr(evaluate_retrieval.httpx, "AsyncClient", lambda **kwargs:
                        client_class(**kwargs, transport=httpx.MockTransport(handler)))
    cases = [{"id": "a", "query": "成功", "expected_title": "目标"},
             {"id": "b", "query": "失败", "expected_title": "目标"}]
    report = await evaluate_retrieval.evaluate("http://test", cases, 3)
    assert report["total"] == 2 and report["errors"] == 1
    assert report["hit_at_k"] == .5 and report["mrr_at_k"] == .25
    assert report["results"][1]["error"] == "HTTPStatusError"
