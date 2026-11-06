import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.retrieval import build_snapshot, Dataset, ReturnedChunk, score_evidence
from evaluation.strategies import Strategy, StrategyRunner, preset, fuse
from evaluation.experiment_dependencies import SyntheticIndex, SyntheticModel, ChromaIndex, public_chunk
from evaluation.experiments import load_suite, make_experiment, compare, validate_report
from scripts import evaluate_retrieval_strategies as cli

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def suite():
    return load_suite(cli.DEFAULT_SUITE)


def small_snapshot():
    return build_snapshot([{'title':f'policy-{i}','content':f'rule payment {i}'} for i in range(6)])


class ObservedIndex(SyntheticIndex):
    def __init__(self,snapshot):
        super().__init__(snapshot)
        self.calls=[]

    def vector(self,q,k):
        self.calls.append(('vector',q,k))
        return super().vector(q,k)

    def lexical(self,q,k):
        self.calls.append(('lexical',q,k))
        return super().lexical(q,k)


def run(snapshot,index,model,name='A',**kwargs):
    runner=StrategyRunner(snapshot,index,model)
    return asyncio.run(runner.run('c','payment rule',preset(name,**kwargs)))


@pytest.mark.parametrize('name,branches,stages',[('A',{'vector'},[]),('B',{'vector','lexical'},[]),
                                              ('C',{'vector','lexical'},['rewrite']),('D',{'vector','lexical'},['rewrite','rerank'])])
def test_actual_strategy_switches_and_budgets(name,branches,stages):
    s=small_snapshot()
    index=ObservedIndex(s)
    e,t=run(s,index,SyntheticModel(),name,candidate_k=5,vector_k=6,lexical_k=7,top_k=2)
    assert {c[0] for c in index.calls}==branches
    assert all(c[2]==(6 if c[0]=='vector' else 7) for c in index.calls)
    assert [c.stage for c in t.model_calls]==stages
    assert len(e.results)==2 and e.status=='success'
    assert t.counts['pre_rerank']>=len(e.results)
    assert t.effective_steps['rewrite'].status==('executed' if name in 'CD' else 'disabled')
    assert t.effective_steps['rerank'].status==('executed' if name=='D' else 'disabled')
    if name in 'CD':
        assert t.counts['queries']>1 and t.counts['deduplicated']>0
    else:
        assert t.counts['queries']==1


def test_candidate_budget_changes_actual_pool_and_branch_budget_changes_calls():
    s=small_snapshot()
    for budget in (3,5):
        index=ObservedIndex(s)
        e,t=run(s,index,None,'B',candidate_k=budget,vector_k=budget,lexical_k=budget)
        assert t.candidate_counts[0]['returned']<=budget
        assert t.candidate_counts[0]['vector']==budget
        assert {c[2] for c in index.calls}=={budget}
        assert len(e.results)==3


@pytest.mark.parametrize('changes',[{'top_k':21},{'top_k':0},{'candidate_k':2},
    {'candidate_k':21},{'lexical_k':1},{'rewrite_count':0},{'rerank':True}, {'rewrite':True,'retriever':'vector'}, {'top_k':True}])
def test_invalid_budgets_and_switch_relations(changes):
    with pytest.raises(ValueError):
        Strategy.model_validate({'retriever':'hybrid',**changes})


def test_bypass_and_both_cache_layers_config_knowledge_and_context_identity():
    s=small_snapshot()
    index=ObservedIndex(s)
    runner=StrategyRunner(s,index,SyntheticModel())
    async def check():
        for _ in range(2):
            _,t=await runner.run('c','payment rule',preset('D'))
            assert all(c.reads==c.writes==c.hits==0 for c in t.caches.values())
        assert not runner.pipeline_cache and not runner.tool_cache
        config=preset('C',cache_mode='use')
        _,cold=await runner.run('c','payment rule',config)
        _,warm=await runner.run('c','payment rule',config)
        assert cold.caches['pipeline'].writes==1 and warm.caches['pipeline'].hits==1
        assert warm.model_calls==[] and warm.embedding_calls==0
        runner.pipeline_cache.clear()
        _,tool=await runner.run('c','payment rule',config)
        assert tool.caches['tool'].hits==len(tool.rewrite_query_list)
        _,different=await runner.run('c','payment rule',preset('B',cache_mode='use'))
        assert different.caches['tool'].hits==different.caches['pipeline'].hits==0
        _,context=await runner.run('c','payment rule',config,context={'tenant':'fictional-other'})
        assert context.caches['tool'].hits==context.caches['pipeline'].hits==0
        # 同实例只为验证身份；真实工厂每个策略使用独立实例。
        runner.snapshot=s.model_copy(update={'knowledge_hash':'different'})
        _,knowledge=await runner.run('c','payment rule',config)
        assert knowledge.caches['tool'].hits==knowledge.caches['pipeline'].hits==0
    asyncio.run(check())


class FailedModel(SyntheticModel):
    async def call(self,*args):
        raise RuntimeError('unknown private response body must not leak')


def test_model_failure_degrades_but_keeps_evidence_and_unknown_usage():
    s=small_snapshot()
    e,t=run(s,SyntheticIndex(s),FailedModel(),'D')
    assert e.status=='success' and len(e.results)==3
    assert t.rewrite_query_list==['payment rule']
    assert t.effective_steps['rewrite'].status==t.effective_steps['rerank'].status=='degraded'
    assert all(c.input_tokens is None for c in t.model_calls)
    assert 'private response' not in t.model_dump_json()


@pytest.mark.parametrize('failure',['none','both','vector','bad-identity'])
def test_empty_recall_and_request_error_are_distinct(failure):
    s=small_snapshot()
    class Index(SyntheticIndex):
        def vector(self,q,k):
            if failure in ('both','vector'):
                raise RuntimeError('private error')
            if failure=='bad-identity':
                return [ReturnedChunk(chunk_id='unknown',title='private',content='private')]
            return []
        def lexical(self,q,k):
            if failure=='both':
                raise RuntimeError('private error')
            return self.rows[:1] if failure=='vector' else []
    e,t=run(s,Index(s),SyntheticModel(),'D')
    if failure in ('both','bad-identity'):
        assert e.status=='error' and t.effective_steps['retrieval'].status=='failed'
    else:
        assert e.status=='success'
        assert t.effective_steps['rerank'].status=='skipped'
        assert t.effective_steps['retrieval'].status==('degraded' if failure=='vector' else 'executed')
    assert 'private' not in t.model_dump_json()


@pytest.mark.parametrize('value,truncated,status',[
    ([0,0],False,'executed'),([999],False,'degraded'),([-1],False,'degraded'),
    ([True],False,'degraded'),([],False,'degraded'),([0],True,'degraded'),('bad',False,'degraded')])
def test_rerank_duplicates_omissions_out_of_range_and_truncation(value,truncated,status):
    s=small_snapshot()
    class Model(SyntheticModel):
        async def call(self,stage,*args):
            if stage=='rewrite':
                return {'value':['payment rule']}
            return {'value':value,'truncated':truncated,'input_tokens':0,'output_tokens':7}
    e,t=run(s,SyntheticIndex(s),Model(),'D')
    assert t.effective_steps['rerank'].status==status
    assert len({r.chunk_id for r in e.results})==3
    assert t.model_calls[-1].input_tokens==0
    if status=='executed':
        assert t.counts['rerank_duplicate_indices']==1
        assert t.counts['rerank_omitted_indices']==5


def test_rrf_is_deterministic_under_equal_scores_and_cross_query_duplicates():
    s=small_snapshot()
    rows=[public_chunk(c) for c in s.chunks]
    a,b=rows[:2]
    assert fuse([[a,b],[b,a]])==fuse([[b,a],[a,b]])
    assert [r.chunk_id for r in fuse([[a,b],[b,a]])]==sorted([a.chunk_id,b.chunk_id])
    assert len(fuse([[a,a,b],[a,b]]))==2


def test_corpus_annotations_freeze_boundaries_and_wrong_same_title_chunk(suite):
    manifest,snapshot,source,dev=suite
    assert len(snapshot.documents)==24 and len(snapshot.chunks)==72
    assert len(dev.cases)==40
    _,_,_,validation=load_suite(cli.DEFAULT_SUITE,'validation')
    assert len(validation.cases)==20
    assert not {c.group_id for c in dev.cases}&{c.group_id for c in validation.cases}
    assert build_snapshot([d.model_dump() for d in snapshot.documents])==snapshot
    for document in snapshot.documents:
        parts=[c for c in snapshot.chunks if c.title==document.title]
        assert len(parts)==3 and all(len(c.content)<=500 for c in parts)
    case=next(c for c in dev.cases if c.evidence_groups)
    good=next(c for c in snapshot.chunks if c.chunk_id==case.evidence_groups[0].acceptable_chunk_ids[0])
    wrong=next(c for c in snapshot.chunks if c.title==good.title and c.chunk_id!=good.chunk_id)
    from evaluation.retrieval import Execution
    assert score_evidence(case,Execution(case_id=case.id,results=[public_chunk(wrong)]),3).hit_at_k==0


@pytest.mark.parametrize('change',['quote','id','empty','split','freeze','corpus'])
def test_freeze_rejects_changed_annotations_or_corpus(tmp_path,change):
    import shutil
    for p in cli.DEFAULT_SUITE.parent.glob('*.json'):
        shutil.copy(p,tmp_path/p.name)
    file='cases.v2.json' if change in ('quote','id','empty','split') else 'suite.v2.json' if change=='freeze' else 'service_rules.v2.json'
    data=json.loads((tmp_path/file).read_text(encoding='utf-8'))
    if change=='quote': data['cases'][0]['evidence_groups'][0]['source_quote']='invented'
    elif change=='id': data['cases'][0]['evidence_groups'][0]['acceptable_chunk_ids']=['expired']
    elif change=='empty': data['cases'][0]['evidence_groups']=[]
    elif change=='split': data['cases'][0]['split']='validation'
    elif change=='freeze': data['splits']['validation']['case_ids'].pop()
    else: data[0]['content']+='changed'
    (tmp_path/file).write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError):load_suite(tmp_path/'suite.v2.json')


def build_report(suite,name='A',mode='synthetic',model=None):
    manifest,snapshot,source,dataset=suite
    config=preset(name)
    runner=StrategyRunner(snapshot,SyntheticIndex(snapshot),model or (SyntheticModel() if name in 'CD' else None))
    async def collect():
        return [await runner.run(c.id,c.turns[0],config) for c in dataset.cases]
    rows=asyncio.run(collect())
    return make_experiment(dataset=dataset,source=source,snapshot=snapshot,manifest=manifest,split=dataset.cases[0].split,
        config=config,dependencies=runner.identity(),executions=[r[0] for r in rows],traces=[r[1] for r in rows],
        run_id=name,mode=mode,code={'test':True},started_at='test',finished_at='test',setup={})


def test_end_to_end_evidence_report_compare_and_usage(suite):
    a=build_report(suite)
    b=build_report(suite,'B')
    comparison=compare(a,b)
    assert a.scoring.total==40 and a.scoring.main.count==36 and a.scoring.errors==0
    assert sum(comparison['counts'].values())==40
    assert len(comparison['common_success_case_ids'])==40
    assert a.statistics['usage']['input_tokens']==0
    d=build_report(suite,'D')
    assert d.statistics['usage']['input_tokens'] is None
    assert d.statistics['model_calls']>0 and d.statistics['complete_count']==40
    failed=build_report(suite,'D',model=FailedModel())
    assert failed.statistics['degraded_count']==40 and failed.statistics['complete_count']==0
    assert failed.scoring.errors==0
    assert compare(a,failed)['eligible_denominator']==36


@pytest.mark.parametrize('field',['mode','split','knowledge_hash','evaluated_dataset_hash','source_dataset_hash',
    'case_set','freeze_hash','completed','config','dependencies','missing_result','changed_label','changed_score','protocol'])
def test_comparison_rejects_incompatible_and_incomplete_reports(suite,field):
    a=build_report(suite)
    b=a.model_copy(deep=True)
    if field=='completed': b.completed=False
    elif field=='config': b.config=b.config.model_copy(update={'candidate_k':19})
    elif field=='dependencies': b.dependencies['embedding']='different'
    elif field=='missing_result': b.scoring.results.pop()
    elif field=='changed_label': b.scoring.results[0].case.turns=['different source question']
    elif field=='changed_score': b.scoring.results[0].scores.hit_at_k=42
    elif field=='protocol':
        raw=b.model_dump(); raw['scoring']['protocol_version']='other'
        with pytest.raises(ValueError):type(b).model_validate(raw)
        return
    elif field=='case_set': b.case_set.pop()
    else: setattr(b,field,'real' if field=='mode' else 'validation' if field=='split' else 'different')
    with pytest.raises(ValueError):compare(a,b)


def test_chroma_adapter_full_store_verification_and_actual_budget():
    s=small_snapshot()
    class Collection:
        def __init__(self):self.rows={};self.query_budget=None
        def count(self):return len(self.rows)
        def add(self,ids,documents,metadatas):self.rows=dict(zip(ids,zip(documents,metadatas)))
        def get(self,include,**kwargs):
            ids=sorted(self.rows)[:kwargs.get('limit',len(self.rows))]
            return {'ids':ids,'documents':[self.rows[i][0] for i in ids],'metadatas':[self.rows[i][1] for i in ids]}
        def query(self,query_texts,n_results,include):
            self.query_budget=n_results
            data=self.get(include,limit=n_results)
            return {**{k:[v] for k,v in data.items()},'distances':[[.1]*n_results]}
    collection=Collection()
    index=ChromaIndex(collection,s,'fake-same-embedding')
    assert len(index.vector('x',4))==4 and collection.query_budget==4
    key=next(iter(collection.rows))
    original=collection.rows[key]
    collection.rows[key]=('wrong',original[1])
    with pytest.raises(ValueError):index.verify(s)
    collection.rows[key]=original
    collection.rows['extra']=original
    with pytest.raises(ValueError):index.verify(s)


def test_cli_matrix_replay_compare_validation_guard_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(cli,'code_identity',lambda:{'test':True})
    assert cli.main(['--matrix','--run-id','matrix'])==0
    original=(tmp_path/'matrix/A/report.json').read_bytes()
    assert cli.main(['--matrix','--run-id','matrix'])==2
    assert (tmp_path/'matrix/A/report.json').read_bytes()==original
    assert cli.main(['--fixture',str(tmp_path/'matrix/D'),'--run-id','replay'])==0
    report=json.loads((tmp_path/'replay/report.json').read_text(encoding='utf-8'))
    assert report['observation']=='replayed' and report['mode']=='synthetic'
    assert cli.main(['--compare',str(tmp_path/'matrix/A/report.json'),str(tmp_path/'matrix/B/report.json'),'--run-id','compare'])==0
    assert cli.main(['--matrix','--split','validation'])==2
    assert cli.main(['--strategy','A','--split','validation'])==2
    assert cli.main(['--split','validation','--candidate','D','--with-baselines','--run-id','validation'])==0
    assert cli.main(['--mode','real','--strategy','D'])==2
    assert cli.main(['--top-k','0'])==2
    assert cli.main(['--validate-suite'])==0
    assert cli.main(['--export-snapshot','--run-id','snapshot'])==0


def test_default_offline_process_has_no_online_side_effects(tmp_path):
    program='''
import sys,pathlib,inspect
sys.path.insert(0,sys.argv[1])
def guard(event,args):
    if event=='socket.connect':
        # Windows asyncio 的 socketpair 使用本地自连接；只允许标准库这一条路径。
        if not any(f.function=='socketpair' and f.filename.endswith('socket.py') for f in inspect.stack()):
            raise AssertionError('offline external network')
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]).replace(chr(92),'/').lower()
        if name.endswith('/.env') or '.sqlite' in name or '/data/chroma/' in name:raise AssertionError('private storage')
sys.addaudithook(guard)
from scripts import evaluate_retrieval_strategies as cli
cli.OUTPUT_ROOT=pathlib.Path(sys.argv[2])
assert cli.main(['--matrix','--run-id','offline'])==0
for name in ('api.main','tooling.knowledge_base','tooling.tool_manager','core.performance','memory.conversation_memory','chromadb','anthropic'):
    assert name not in sys.modules,name
'''
    process=subprocess.run([sys.executable,'-X','utf8','-c',program,str(ROOT),str(tmp_path)],cwd=tmp_path,
                           capture_output=True,text=True,encoding='utf-8')
    assert process.returncode==0,process.stderr


def test_existing_online_statistics_and_tool_state_unchanged(suite):
    from core.performance import llm_usage_summary
    from tooling.tool_manager import MCPToolManager, Tool
    manager=MCPToolManager('unit-test-placeholder')
    manager.register(Tool('test','test',lambda p,c:[],{}))
    manager._cache['sentinel']=('value',100,False)
    before=copy.deepcopy((llm_usage_summary(),manager._cache,manager.get_stats()))
    build_report(suite,'D')
    assert (llm_usage_summary(),manager._cache,manager.get_stats())==before


def test_real_chroma_ephemeral_index_with_fake_embedding_only(suite):
    # 真实 Chroma 引擎 + 注入假向量，不加载模型/连接服务器/用户持久库。
    import chromadb
    from uuid import uuid4
    class Embedding:
        def __call__(self,input):
            return [SyntheticIndex.embed(text) for text in input]
    s=suite[1]
    async def check():
        def create():
            client=chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
            collection=client.create_collection('test_eval_'+uuid4().hex,embedding_function=Embedding(),
                                                metadata={'hnsw:space':'cosine'})
            return ChromaIndex(collection,s,'fake-64-cosine',client)
        index=await asyncio.to_thread(create)
        try:
            e,t=await StrategyRunner(s,index,SyntheticModel()).run('c','payment capture',preset('D'))
            assert e.status=='success' and len(e.results)==3
            assert all(c['vector']==20 for c in t.candidate_counts)
            await asyncio.to_thread(index.verify,s)
        finally:
            await asyncio.to_thread(index.client.delete_collection,index.collection.name)
    asyncio.run(check())


def test_provider_adapter_preserves_usage_on_invalid_json_without_global_statistics(monkeypatch):
    from evaluation.experiment_dependencies import ProviderModel
    from core.performance import llm_usage_summary
    import anthropic
    observed=[]
    async def create(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type='text',text='invalid JSON private response')],
            stop_reason='max_tokens',usage=SimpleNamespace(input_tokens=23,output_tokens=9))
    monkeypatch.setattr(anthropic,'AsyncAnthropic',lambda **kw:SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setenv('SERVICEPILOT_EVAL_MODEL','test-model')
    monkeypatch.setenv('SERVICEPILOT_EVAL_API_KEY','secret-placeholder')
    monkeypatch.setenv('SERVICEPILOT_EVAL_BASE_URL','https://provider.invalid')
    before=copy.deepcopy(llm_usage_summary())
    model=ProviderModel()
    result=asyncio.run(model.call('rerank','question',[public_chunk(c) for c in small_snapshot().chunks],3,
                                  {'temperature':0.0,'max_tokens':2048}))
    assert result['value'] is None and result['truncated']
    assert result['input_tokens']==23 and result['cache_read_input_tokens'] is None
    assert llm_usage_summary()==before
    prompt=observed[0]['messages'][0]['content']
    assert 'secret-placeholder' not in prompt and 'acceptable_chunk_ids' not in prompt
    assert 'chunk_id' not in prompt and 'rule payment' in prompt


def test_real_factory_missing_model_cannot_download_or_initialize_default_client(tmp_path,monkeypatch):
    import chromadb
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    from evaluation.experiment_dependencies import real_index
    def forbidden(*a,**kw):
        raise AssertionError('unexpected client or download')
    monkeypatch.setattr(chromadb,'HttpClient',forbidden)
    monkeypatch.setattr(chromadb,'PersistentClient',forbidden)
    monkeypatch.setattr(chromadb,'EphemeralClient',forbidden)
    monkeypatch.setattr(ONNXMiniLM_L6_V2,'_download',forbidden)
    with pytest.raises(ValueError,match='不会自动下载'):
        real_index(small_snapshot(),tmp_path/'missing-model')


@pytest.mark.parametrize('variable',['top_k','candidate_k','vector_k','lexical_k','cache_mode','embedding','mode'])
def test_comparison_rejects_valid_reports_with_different_control_variables(suite,variable):
    # 重新签身份以保证拒绝来自比较条件，而非仅来自旧 hash。
    from evaluation.experiments import run_identity,execution_statistics
    from evaluation.retrieval import canonical_hash
    a=build_report(suite)
    b=a.model_copy(deep=True)
    if variable in ('embedding','mode'):
        if variable=='mode':
            b.mode='real'
            b.statistics=execution_statistics([r.execution for r in b.scoring.results],b.traces,b.mode)
        else:
            b.dependencies['embedding']='other-valid-embedding'
            for t in b.traces:t.embedding_identity='other-valid-embedding'
    else:
        change={'top_k':4,'candidate_k':19,'vector_k':21,'lexical_k':21,'cache_mode':'use'}[variable]
        b.config=Strategy.model_validate({**b.config.model_dump(),variable:change})
        b.config_id=canonical_hash(b.config.model_dump())
        for t in b.traces:t.requested_config=b.config
        if variable=='top_k':
            from evaluation.retrieval import score_evidence,summarize
            for r in b.scoring.results:r.scores=score_evidence(r.case,r.execution,4)
            b.scoring.main=summarize(b.scoring.results)
            b.scoring.successful=summarize(b.scoring.results)
    b.run_identity=run_identity(b)
    validate_report(b)
    with pytest.raises(ValueError):compare(a,b)
