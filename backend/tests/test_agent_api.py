import asyncio
import json
import httpx
import sys
from types import SimpleNamespace
from api import main
from evaluation.jobs import EvaluationJobs


def test_rag_api_fixed_inputs_admin_and_jobs(monkeypatch):
    monkeypatch.setenv('ZHIYING_API_TOKENS', json.dumps({'a'*24: {'user_id':'admin','role':'admin'}, 'u'*24:{'user_id':'user','role':'user'}}))
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
            assert (await client.get('/eval/rag/runs')).status_code == 401
            assert (await client.get('/eval/rag/runs', headers={'Authorization':'Bearer '+'u'*24})).status_code == 403
            headers={'Authorization':'Bearer '+'a'*24}
            assert (await client.post('/eval/rag/run', json={'suite':'arbitrary','path':'secret'},headers=headers)).status_code == 422
            assert (await client.get('/eval/rag/runs/invalid.id',headers=headers)).status_code == 400
    asyncio.run(check())


def test_rag_api_reuses_queue_and_run_identity(monkeypatch):
    monkeypatch.delenv('ZHIYING_API_TOKENS', raising=False)
    monkeypatch.setenv('ZHIYING_ENV', 'development')
    saved = {}
    async def run_agents(progress=None, run_id=None):
        progress(1, 1)
        saved[run_id] = {'run_id':run_id, 'status':'completed', 'scope':'orchestrator'}
        return saved[run_id]
    monkeypatch.setitem(sys.modules, 'evaluation.agent_evaluation', SimpleNamespace(
        run_agents=run_agents, list_agent_runs=lambda:list(saved.values()), read_agent_run=lambda key:saved[key]))
    async def check():
        jobs = EvaluationJobs(capacity=1)
        monkeypatch.setattr(main, '_eval_jobs', jobs)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://test') as client:
            response = await client.post('/eval/rag/run',json={'suite':'agent-dev-v1'})
            assert response.status_code == 202
            initial=response.json()
            await asyncio.gather(*jobs.tasks)
            job=(await client.get('/eval/jobs/'+initial['job_id'])).json()
            assert job['run_id'] == job['result']['run_id'] == initial['run_id']
            assert (await client.get('/eval/rag/runs/'+initial['run_id'])).json() == job['result']
            assert (await client.get('/eval/rag/runs')).json() == [job['result']]
            jobs.capacity = 0
            assert (await client.post('/eval/rag/run')).status_code == 429
            jobs.jobs.clear()
            assert (await client.get('/eval/jobs/'+initial['job_id'])).status_code == 404
            assert (await client.get('/eval/rag/runs/'+initial['run_id'])).json() == job['result']
        await jobs.close()
    asyncio.run(check())
