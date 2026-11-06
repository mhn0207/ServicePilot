import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from core.request_budget import request_budget, phase_budget, remaining, model_timeout
from core.llm_utils import create_message
from agents.agent_orchestrator import AgentOrchestrator, AgentType, AgentResponse, Request


def test_shared_deadline_limits_phase_and_preserves_parent_after_phase():
    now=[10.0]
    with request_budget(45,clock=lambda:now[0]):
        now[0]=15
        assert model_timeout('intent')==5
        with phase_budget(28,reserve=5):
            now[0]=40
            assert model_timeout('agent.billing')==3
        assert remaining()==15
    assert remaining() is None

@pytest.mark.asyncio
async def test_exhausted_budget_does_not_call_sdk_or_online_counters():
    client=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))
    with request_budget(0):
        with pytest.raises(TimeoutError): await create_message(client,stage='agent.billing',model='test',messages=[])
    client.messages.create.assert_not_called()

@pytest.mark.asyncio
async def test_not_enough_time_does_not_start_fallback():
    primary=SimpleNamespace(handle=AsyncMock(return_value=AgentResponse(agent_type=AgentType.BILLING,content='',success=False,status='failed')))
    fallback=SimpleNamespace(handle=AsyncMock())
    orchestrator=AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._best_agent=lambda kind:primary if kind==AgentType.BILLING else fallback
    with request_budget(6):
        result=await orchestrator._execute(Request(message='退款',user_id='u',conv_id='c'),AgentType.BILLING)
    assert result.status=='failed'
    fallback.handle.assert_not_called()

@pytest.mark.asyncio
async def test_parallel_requests_do_not_share_deadline():
    async def run(seconds):
        with request_budget(seconds):
            await asyncio.sleep(0)
            return remaining()
    short,long=await asyncio.gather(run(1),run(20))
    assert short<2 and long>19

@pytest.mark.asyncio
async def test_optional_retrieval_models_are_skipped_when_time_is_short():
    from tooling.tool_manager import MCPToolManager
    manager=MCPToolManager.__new__(MCPToolManager)
    manager._client=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))
    with request_budget(10):
        assert await manager.rewrite_query('退款')==['退款']
        assert await manager._rerank_with_status('退款',[{'content':'one'},{'content':'two'}],1)==([{'content':'one'}],False)
    manager._client.messages.create.assert_not_called()

@pytest.mark.asyncio
async def test_slow_sdk_is_cancelled_before_request_deadline():
    cancelled=[]
    async def slow(**kwargs):
        try:await asyncio.Event().wait()
        finally:cancelled.append(True)
    client=SimpleNamespace(messages=SimpleNamespace(create=slow))
    with request_budget(.01):
        with pytest.raises(TimeoutError):await create_message(client,stage='agent.billing',model='test',messages=[])
    assert cancelled==[True]

@pytest.mark.asyncio
@pytest.mark.parametrize('same_handler',[True,False])
async def test_fallback_reuses_only_same_authorized_retrieval_handler(same_handler):
    from agents.agent_orchestrator import BillingAgent, GeneralAgent
    from agents.tools import AgentToolSpec
    calls=[]; fallback_inputs=[]
    async def search(req,args):calls.append(args);return {'success':True,'documents':[{'content':'public-evidence'}]}
    async def other(req,args):return {'success':True,'documents':[]}
    results=[SimpleNamespace(content=[{'type':'tool_use','id':'one','name':'search_knowledge_base','input':{'query':'退款'}}],stop_reason='tool_use'),RuntimeError('generation failed')]
    async def primary_create(**kwargs):
        result=results.pop(0)
        if isinstance(result,Exception):raise result
        return result
    async def fallback_create(**kwargs):
        fallback_inputs.append(kwargs)
        return SimpleNamespace(content=[{'type':'text','text':'需要人工核验'}],stop_reason='end_turn')
    primary=BillingAgent(SimpleNamespace(messages=SimpleNamespace(create=primary_create)),'test')
    fallback=GeneralAgent(SimpleNamespace(messages=SimpleNamespace(create=fallback_create)),'test')
    primary.set_shared_tools({'search_knowledge_base':AgentToolSpec('search_knowledge_base','search',{'type':'object'},search)})
    fallback.set_shared_tools({'search_knowledge_base':AgentToolSpec('search_knowledge_base','search',{'type':'object'},search if same_handler else other)})
    orchestrator=AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._best_agent=lambda kind:primary if kind==AgentType.BILLING else fallback
    request=Request(message='退款规则',user_id='u',conv_id='c')
    with request_budget(45):result=await orchestrator._execute(request,AgentType.BILLING)
    assert result.success
    assert len(calls)==1
    shown=str(fallback_inputs[0]['messages'])
    assert ('public-evidence' in shown)==same_handler
    assert any(t.get('event')=='retrieval_reuse' for t in result.tool_traces)==same_handler
    assert not Request(message='退款规则',user_id='another',conv_id='c').retrieval_results

@pytest.mark.asyncio
async def test_budget_degraded_search_does_not_fill_shared_cache():
    from tooling.tool_manager import MCPToolManager, Tool
    manager=MCPToolManager(api_key='offline')
    async def search(params,context):return [{'content':'first'},{'content':'second'}]
    manager.register(Tool(name='search',description='test',handler=search,schema={'type':'object'},cache_ttl=30))
    manager._client=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[SimpleNamespace(content=[{'type':'text','text':'[]'}]),SimpleNamespace(content=[{'type':'text','text':'[1,0]'}])])) )
    with request_budget(10):first=await manager.search_with_rewrite('search','query',top_k=1)
    assert first.data==[{'content':'first'}]
    with request_budget(45):second=await manager.search_with_rewrite('search','query',top_k=1)
    assert second.data==[{'content':'second'}]
    assert second.reranked and not second.cached

@pytest.mark.asyncio
async def test_short_request_budgets_do_not_open_healthy_tool_breaker():
    from tooling.tool_manager import MCPToolManager, Tool, CircuitState
    manager=MCPToolManager(api_key='offline')
    async def slow(params,context):await asyncio.sleep(.02);return []
    tool=Tool(name='search',description='test',handler=slow,schema={'type':'object'},timeout_s=30)
    manager.register(tool)
    for _ in range(5):
        with request_budget(.003):result=await manager.call('search',{},use_cache=False)
        assert not result.success
    assert tool.breaker.state==CircuitState.CLOSED
    assert tool.breaker.fail_count==0
    assert tool.stats.consecutive_fails==0

@pytest.mark.asyncio
async def test_budget_cancel_releases_half_open_probe_without_recording_failure():
    import time
    from tooling.tool_manager import MCPToolManager, Tool, CircuitState
    manager=MCPToolManager(api_key='offline')
    async def slow(params,context):await asyncio.sleep(.02);return []
    tool=Tool(name='search',description='test',handler=slow,schema={'type':'object'})
    manager.register(tool)
    tool.breaker.state=CircuitState.OPEN;tool.breaker.opened_at=time.monotonic()-61;tool.breaker.fail_count=1
    with request_budget(.003):result=await manager.call('search',{},use_cache=False)
    assert not result.success and tool.breaker.fail_count==1
    with request_budget(1):result=await manager.call('search',{},use_cache=False)
    assert result.success and tool.breaker.state==CircuitState.CLOSED

def test_trace_distinguishes_phase_budget_from_whole_request():
    from core.performance import collect_performance,performance_snapshot
    from core.request_budget import observe
    now=[10.0]
    with collect_performance(),request_budget(45,clock=lambda:now[0]):
        now[0]=15
        with phase_budget(28,reserve=5):
            now[0]=40;observe('agent.billing','started')
        row=performance_snapshot()['budget_events'][0]
        assert row['phase_remaining_s']==3
        assert row['request_remaining_s']==15
