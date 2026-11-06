import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from core.intent_recognizer import IntentRecognizer
from core.conversation_state import merge_state
from agents.agent_orchestrator import BillingAgent, Request
from agents.tools import AgentToolSpec

@pytest.mark.parametrize('message,expected',[
 ('纠正：现在改为订单22222，之前的99元不属于这笔订单。', []),
 ('金额不是99元，是199元。', ['199元']),
 ('不是99元', []),
 ('99元不是这笔订单的金额', []),
 ('没有收到99元退款', ['99元']),
 ('订单22222支付99元，需要退款', ['99元']),
 ('99元不属于重复扣款，是正常续费', ['99元']),
])
def test_amount_extraction_respects_explicit_retraction(message, expected):
    result=IntentRecognizer.__new__(IntentRecognizer)._extract_entities(message)
    assert result['amount']==expected

@pytest.mark.parametrize('message', ['99元不属于这笔订单', '金额不是99元'])
def test_retraction_clears_same_order_amount_even_when_incoming_is_empty(message):
    previous={'domain':'commerce','entities':{'order_id':['22222'],'amount':['99元']},'sources':{'amount':'old'}}
    entities=IntentRecognizer.__new__(IntentRecognizer)._extract_entities(message)
    result=merge_state(previous,entities,'billing',message,'correction')
    assert result['entities']['order_id']==['22222']
    assert result['entities']['amount']==[]
    assert 'amount' not in result['sources']

@pytest.mark.parametrize('message,required', [('取消自动续费是否自动退款？',True),('重复扣款需要什么资料？',True),('退款多久到账',True),('请介绍退款政策',True),('退款金额需要改为99元',False),('你好',False),('订单号改为22222',False)])
def test_policy_questions_require_actual_knowledge_tool_before_answer(message,required):
    calls=[]; returns=[]
    async def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get('tool_choice'):
            return SimpleNamespace(content=[{'type':'tool_use','id':'search-one','name':'search_knowledge_base','input':{'query':message}}],stop_reason='tool_use')
        return SimpleNamespace(content=[{'type':'text','text':'已根据资料回答'}],stop_reason='end_turn')
    async def search(req,args):
        returns.append(args);return {'success':True,'documents':[{'title':'订阅','content':'取消不等于退款'}]}
    client=SimpleNamespace(messages=SimpleNamespace(create=create))
    agent=BillingAgent(client,'offline-test')
    agent.set_shared_tools({'search_knowledge_base':AgentToolSpec('search_knowledge_base','search',{'type':'object'},search)})
    result=asyncio.run(agent._call_llm(Request(message=message,user_id='test',conv_id='test')))
    assert bool(returns)==required
    if required:
        assert len(calls)==2
        assert any(b.get('type')=='tool_result' for m in calls[1]['messages'] if isinstance(m['content'],list) for b in m['content'])
