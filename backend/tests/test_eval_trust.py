"""Regression checks for false confidence at evaluation boundaries; no providers."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest

from evaluation.answer_quality import check_grounding
from evaluation.evaluator import EndToEndEvaluator, EvalReport
from evaluation.retrieval import Dataset
from scripts import evaluate_retrieval_evidence as evidence_cli


def test_complete_extraction_cannot_hide_second_assertion_in_same_sentence():
    answer = 'Renewal stops but refunds are guaranteed tomorrow.'
    value = dict(extraction='complete', claims=[dict(claim_id='one', unit_ids=[0],
        kind='factual', status='supported', answer_span=dict(start=0, end=13, quote=answer[:13]),
        evidence=[dict(chunk_id='c', quote='Renewal stops')], rationale='First clause only')])
    with pytest.raises(ValueError, match='uncovered answer text'):
        check_grounding(value, answer, [dict(chunk_id='c', content='Renewal stops')])


def test_multiple_spans_can_cover_one_sentence_without_hiding_text():
    answer = 'A stops; B remains.'
    claims = [dict(claim_id=str(i), unit_ids=[0], kind='factual', status='unsupported',
        answer_span=dict(start=a, end=b, quote=answer[a:b]), evidence=[], rationale='No evidence')
        for i, (a, b) in enumerate([(0, 8), (9, 19)])]
    assert len(check_grounding(dict(extraction='complete', claims=claims), answer, []).claims) == 2


def test_small_degradations_compare_to_fixed_reference_not_previous_run():
    ev = EndToEndEvaluator.__new__(EndToEndEvaluator)
    def report(value):
        return EvalReport(timestamp='test', total=1, passed=1, pass_rate=value,
            avg_scores={'accuracy': value}, regressions=[], recommendations=[], results=[], suite_id='s')
    # A real valid result is needed for status=complete.
    from evaluation.evaluator import EvalResult
    first = report(1.0)
    first.results = [EvalResult(test_id='one', passed=True, scores={}, detail='', metadata={})]
    ev._baseline, ev._history = first, [first]
    later = report(.96); later.results = first.results
    ev._history.append(later)
    assert ev._detect_regressions({'accuracy': .9216}, 's')


def test_search_exposes_internal_execution_failure(monkeypatch):
    from api import main
    monkeypatch.setattr(main, '_tool_manager', NS(search_with_rewrite=AsyncMock(
        return_value=NS(success=False, data=[], cached=False, reranked=False))))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
            return (await client.post('/search', params={'query':'refund', 'top_k':3})).json()
    monkeypatch.setenv('ZHIYING_ENV', 'development')
    monkeypatch.delenv('ZHIYING_API_TOKENS', raising=False)
    assert asyncio.run(run())['success'] is False


@pytest.mark.parametrize('payload', [{'results': [], 'success': False}, {'results': []}])
def test_online_evidence_does_not_infer_success_from_http_200(monkeypatch, payload):
    from evaluation.retrieval import build_snapshot, RetrievalCase
    snapshot = build_snapshot([dict(title='one', content='Known fact')])
    case = RetrievalCase(id='c', group_id='g', category='unanswerable', split='dev', turns=['unknown'],
        answerability='unanswerable', expected_retrieval='not_applicable', evidence_groups=[])
    data = Dataset(protocol_version='rag-evidence-v1', knowledge_hash=snapshot.knowledge_hash, cases=[case])
    client = httpx.AsyncClient
    monkeypatch.setattr(evidence_cli.httpx, 'AsyncClient', lambda **kw: client(**kw,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))))
    rows = asyncio.run(evidence_cli.request_executions('http://test', data, 3))
    assert rows[0].status == 'error'


@pytest.mark.asyncio
async def test_agent_suite_exercises_business_tool_choice_and_result_contract(tmp_path, monkeypatch):
    from evaluation import agent_evaluation as agent
    monkeypatch.setattr(agent, 'REPORT_ROOT', tmp_path)
    report = await agent.run_agents(run_id='contract')
    turn = next(t for t in report['turns'] if t['case_id']=='subscription')
    first = turn['generation_inputs'][0]
    assert first.get('tool_choice') == {'type':'tool', 'name':'search_knowledge_base'}
    assert turn['events'][0]['parameters']['top_k'] == 3
    assert turn['events'][0]['actual_tool_result']['results'] == turn['events'][0]['shown_chunks']


def test_source_bundle_recovers_dirty_source_and_excludes_local_secrets(tmp_path):
    import zipfile
    from evaluation.provenance import source_identity, write_source_bundle
    root = tmp_path/'backend'; (root/'evaluation').mkdir(parents=True)
    (root/'evaluation'/'score.py').write_bytes(b'VERSION = 2\n')
    (root/'.env').write_text('DO_NOT_ARCHIVE', encoding='utf-8')
    output = tmp_path/'report'; output.mkdir()
    code = source_identity(root)
    write_source_bundle(root, output, code)
    with zipfile.ZipFile(output/'source.zip') as bundle:
        assert bundle.namelist() == ['evaluation/score.py']
        assert bundle.read('evaluation/score.py') == b'VERSION = 2\n'
    (root/'evaluation'/'score.py').write_text('VERSION = 3\n', encoding='utf-8')
    assert source_identity(root)['source_hash'] != code['source_hash']
    with pytest.raises(ValueError, match='source changed'):
        write_source_bundle(root, output, code)


@pytest.mark.asyncio
async def test_rerank_trace_retains_ordered_input_and_raw_model_indices():
    from evaluation.retrieval import build_snapshot
    from evaluation.experiment_dependencies import SyntheticIndex
    from evaluation.strategies import StrategyRunner, preset
    snapshot = build_snapshot([dict(title=str(i),content=f'payment rule {i}') for i in range(6)])
    class Model:
        identity='fixed-test'
        async def call(self,stage,query,candidates,n,parameters):
            return {'value': ['payment'] if stage=='rewrite' else [4, 1, 4]}
    ex, trace = await StrategyRunner(snapshot,SyntheticIndex(snapshot),Model()).run('c','payment',preset('D'))
    call = next(c for c in trace.model_calls if c.stage=='rerank')
    assert call.response_value == [4, 1, 4]
    assert len(call.request_data['candidates']) == 6
    assert ex.results[0].chunk_id == call.request_data['candidates'][4]['chunk_id']


def test_baseline_promotion_is_explicit_and_rejects_partial_report(tmp_path):
    from evaluation.evaluator import EvalResult
    ev = EndToEndEvaluator.__new__(EndToEndEvaluator)
    ev._run_lock = asyncio.Lock(); ev._baseline = None; ev._baseline_path = tmp_path/'baseline.json'
    report = EvalReport(timestamp='one', total=1, passed=1, pass_rate=1.0, avg_scores={'accuracy':1.0},
        regressions=[], recommendations=[], results=[EvalResult(test_id='one',passed=True,scores={})],suite_id='s')
    ev._history = [report]
    ev.promote_baseline(report)
    original = ev._baseline_path.read_bytes()
    report.results[0].metadata['judge_failed'] = True
    with pytest.raises(ValueError): ev.promote_baseline(report)
    assert ev._baseline_path.read_bytes() == original


def test_baseline_endpoint_selects_server_report_and_requires_admin(tmp_path, monkeypatch):
    from api import main
    from evaluation.evaluator import EvalResult
    ev = EndToEndEvaluator.__new__(EndToEndEvaluator)
    ev._run_lock=asyncio.Lock(); ev._baseline=None; ev._baseline_path=tmp_path/'baseline.json'
    ev._history=[EvalReport(timestamp='one',total=1,passed=1,pass_rate=1.0,avg_scores={},regressions=[],
        recommendations=[],results=[EvalResult(test_id='one',passed=True,scores={})],suite_id='s')]
    monkeypatch.setattr(main,'_evaluator',ev)
    import json
    monkeypatch.setenv('ZHIYING_API_TOKENS',json.dumps({'a'*24:{'user_id':'a','role':'admin'},'u'*24:{'user_id':'u','role':'user'}}))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://test') as client:
            assert (await client.post('/eval/baseline',json={'timestamp':'one'},headers={'Authorization':'Bearer '+'u'*24})).status_code==403
            result=await client.post('/eval/baseline',json={'timestamp':'one'},headers={'Authorization':'Bearer '+'a'*24})
            assert result.status_code==200
            assert result.json()['suite_id']=='s'
    asyncio.run(run())
    assert ev._load_baseline().timestamp=='one'
