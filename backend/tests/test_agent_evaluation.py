import pytest
from evaluation.agent_evaluation import run_agents, read_agent_run, list_agent_runs

@pytest.mark.asyncio
async def test_agent_fixture_real_loop(tmp_path, monkeypatch):
    monkeypatch.setattr('evaluation.agent_evaluation.REPORT_ROOT', tmp_path)
    report = await run_agents(run_id='offline-test')
    assert report['status'] == 'completed'
    assert len(report['turns']) >= 9
    assert report['summary']['required_call_coverage']['value'] < 1
    assert any(t['generation_inputs'] for t in report['turns'])
    assert any(t['events'] for t in report['turns'])
    assert read_agent_run('offline-test') == report
    assert list_agent_runs()[0]['run_id'] == 'offline-test'
    with pytest.raises(FileExistsError):
        await run_agents(run_id='offline-test')

@pytest.mark.parametrize('identity', ['../secret', '/', 'a.b', '', 'x'*81])
def test_invalid_agent_run(identity):
    with pytest.raises(ValueError):
        read_agent_run(identity)

@pytest.mark.asyncio
async def test_actual_tool_result_inputs_and_switch(tmp_path, monkeypatch):
    monkeypatch.setattr('evaluation.agent_evaluation.REPORT_ROOT', tmp_path)
    r=await run_agents(run_id='inputs')
    turns={(t['case_id'],t['turn_id']):t for t in r['turns']}
    compound=turns['compound','1']
    assert set(compound['roles']) == {'technical','billing'}
    assert {e['role'] for e in compound['events']} == {'technical','billing'}
    assert len({e['tool_use_id'] for e in compound['events']}) == 2
    import json
    actual=json.dumps(compound['generation_inputs'],ensure_ascii=False)
    assert 'tool_result' in actual and 'chunk_id' in actual
    final=turns['order-switch','3']
    assert final['entities']['order_id']==['22222']
    assert all('22222' in q and '11111' not in q for q in final['queries'])
    first=json.dumps(turns['order-switch','1']['generation_inputs'],ensure_ascii=False)
    assert '22222' not in first
    assert 'required_facts' not in actual and 'expected_roles' not in actual

@pytest.mark.asyncio
async def test_cancel_persists_partial_and_replay_no_calls(tmp_path, monkeypatch):
    import asyncio
    from evaluation.agent_evaluation import replay_agent_run, SDK
    monkeypatch.setattr('evaluation.agent_evaluation.REPORT_ROOT', tmp_path)
    def progress(completed,total): raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError): await run_agents(progress,run_id='cancel')
    assert read_agent_run('cancel')['status']=='cancelled'
    r=await run_agents(run_id='replay')
    async def forbidden(**kw): raise AssertionError('replay called SDK')
    monkeypatch.setattr(SDK,'create',forbidden)
    assert replay_agent_run('replay')['summary']==r['summary']

@pytest.mark.asyncio
async def test_statistics_untouched(tmp_path,monkeypatch):
    import copy
    import core.performance as performance
    monkeypatch.setattr('evaluation.agent_evaluation.REPORT_ROOT',tmp_path)
    before=copy.deepcopy(performance._totals)
    await run_agents(run_id='isolated')
    assert performance._totals==before

@pytest.mark.asyncio
async def test_offline_boundaries_and_role_answers(tmp_path,monkeypatch):
    import socket
    import dotenv
    from agents.agent_orchestrator import AgentOrchestrator
    def deny(*args,**kwargs): raise AssertionError('external dependency reached')
    monkeypatch.setattr('evaluation.agent_evaluation.REPORT_ROOT',tmp_path)
    monkeypatch.setattr(dotenv,'load_dotenv',deny)
    monkeypatch.setattr(socket.socket,'connect',deny)
    r=await run_agents(run_id='bounded')
    c=next(t for t in r['turns'] if t['case_id']=='compound')
    assert {x['role'] for x in c['role_answers']}=={'technical','billing'}
    assert any(x['stage']=='composer' for x in c['model_steps'])
    for e in c['events']:
        assert e['tool_use_id'] and e['actual_tool_result']['chunks']==e['shown_chunks']
    assert r['summary']['execution_success']['value']<1


def test_read_rejects_symlink(tmp_path,monkeypatch):
    from pathlib import Path
    monkeypatch.setattr('evaluation.agent_evaluation.REPORT_ROOT',tmp_path)
    outside=tmp_path/'outside'; outside.mkdir()
    linked=tmp_path/'linked'
    try: linked.symlink_to(outside,target_is_directory=True)
    except OSError: pytest.skip('Windows symlink privilege unavailable')
    with pytest.raises(ValueError): read_agent_run('linked')

@pytest.mark.asyncio
async def test_cancel_keeps_main_denominators(tmp_path,monkeypatch):
    import asyncio
    monkeypatch.setattr('evaluation.agent_evaluation.REPORT_ROOT',tmp_path)
    def cancel(a,b): raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError): await run_agents(cancel,run_id='denominator')
    r=read_agent_run('denominator')
    assert len(r['turns'])==9
    assert r['summary']['judge_validity']['denominator']==9
    assert r['summary']['forbidden_clear']['denominator']==9
    assert r['summary']['forbidden_clear']['unresolved']>=8
    assert r['summary']['required_call_coverage']['denominator']==7

@pytest.mark.asyncio
async def test_replay_refuses_changed_identity(tmp_path,monkeypatch):
    from evaluation.agent_evaluation import replay_agent_run
    monkeypatch.setattr('evaluation.agent_evaluation.REPORT_ROOT',tmp_path)
    await run_agents(run_id='identity')
    monkeypatch.setattr('evaluation.agent_evaluation.judge_identity',lambda:'new-judge')
    with pytest.raises(ValueError,match='incompatible Judge'): replay_agent_run('identity')
    assert read_agent_run('identity')['run_id']=='identity'
