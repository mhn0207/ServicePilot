"""Offline orchestrator evaluation; isolated SDK/intent/index boundaries."""
import asyncio
import copy
import hashlib
import inspect
import subprocess
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import uuid4
from agents.agent_orchestrator import AgentOrchestrator, Request, tool_call_context
from agents.tools import build_shared_rag_tools
from core.llm_utils import extract_text_content
from core.intent_recognizer import IntentCategory, UrgencyLevel
from evaluation.experiment_dependencies import SyntheticIndex
from evaluation.retrieval import build_snapshot, canonical_hash, checked_execution
from evaluation.strategies import StrategyRunner, preset
from evaluation.answer_quality import check_correctness, check_grounding, answer_units
from evaluation.provenance import source_identity, write_source_bundle
BASE=Path(__file__).resolve().parents[1]
REPORT_ROOT=BASE/'data/eval/rag/agents'
PROTOCOL='agent-evaluation-v1'
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def directory(run_id):
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',run_id): raise ValueError('invalid run_id')
    root=Path(REPORT_ROOT)
    if any((p.is_symlink() or p.is_junction()) for p in (root,*root.parents)): raise ValueError('symlink root')
    p=root/run_id
    if p.is_symlink() or p.is_junction(): raise ValueError('symlink/junction run')
    if not p.resolve().is_relative_to(root.resolve()): raise ValueError('escaped report root')
    return p
def read_agent_run(run_id):
    p=directory(run_id)/'report.json'
    if p.is_symlink() or p.is_junction(): raise ValueError('symlink/junction report')
    r=load(p)
    if r.get('run_id') != run_id: raise ValueError('report run identity mismatch')
    if r.get('scope')!='orchestrator' or r.get('protocol_version')!=PROTOCOL: raise ValueError('unsupported scope/protocol')
    return r
def list_agent_runs():
    directory('check')
    if not Path(REPORT_ROOT).exists(): return []
    rows=[]
    for p in Path(REPORT_ROOT).iterdir():
        if p.is_symlink() or p.is_junction() or not p.is_dir() or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',p.name): continue
        try:
            r=read_agent_run(p.name); rows.append({k:r[k] for k in ('run_id','status','scope','mode','summary','protocol_version')})
        except (ValueError,FileNotFoundError,KeyError): continue
    return sorted(rows,key=lambda r:r['run_id'],reverse=True)
def load_suite():
    p=BASE/'evaluation/corpora/agent.cases.v1.json'
    manifest=load(BASE/'evaluation/corpora/agent.suite.v1.json')
    # 冻结哈希按字节计算，而 .gitattributes 的 text=auto 让同一份标注在 Windows 检出为
    # CRLF、在 Linux/macOS 检出为 LF。两种换行下内容相同即视为同一份冻结标注。
    lf=p.read_bytes().replace(b'\r\n',b'\n')
    digests={hashlib.sha256(data).hexdigest() for data in (lf,lf.replace(b'\n',b'\r\n'))}
    if manifest['cases_sha256'] not in digests: raise ValueError('annotation freeze mismatch')
    data=load(p); snapshot=build_snapshot(load(BASE/'data/demo_docs/sample_knowledge.json'))
    for c in data['cases']:
        for t in c['turns']:
            for q in t['evidence_quotes']:
                if not any(q in row.content for row in snapshot.chunks): raise ValueError('invalid evidence quote')
    if manifest['knowledge_hash'] != snapshot.knowledge_hash or manifest['case_set_hash'] != canonical_hash(data): raise ValueError('manifest identity mismatch')
    return data,snapshot,manifest
class Intent:
    def __init__(self): self.inputs=[]
    async def recognize(self,message,history=None):
        self.inputs.append(copy.deepcopy(dict(message=message,history=history)))
        billing=any(k in message for k in ('扣款','退款','续费','合同','这笔订单'))
        intent=IntentCategory.BILLING if billing else IntentCategory.TECHNICAL if '401' in message else IntentCategory.QUERY
        return NS(intent=intent,intent_group='billing' if billing else 'technical' if '401' in message else 'query',urgency=UrgencyLevel.LOW,confidence=1.0,entities={'order_id':re.findall(r'订单\s*(\d{5,})',message)})
class SDK:
    base_url='https://offline.invalid'
    def __init__(self): self.messages=self; self.inputs=[]; self.calls=[]; self.fixture={}; self.req=None
    def evaluation_observer(self,stage,kwargs,response):
        text=extract_text_content(response.content)
        self.turn.setdefault('model_steps',[]).append(dict(step_id=uuid4().hex,stage=stage,request=copy.deepcopy(kwargs),response=copy.deepcopy(response.content)))
        if text and stage.startswith('agent.'):
            self.turn.setdefault('role_answers',[]).append(dict(role=stage.split('.',1)[1],answer=text))
    def evaluation_recorder(self,stage,model,started,status,response):
        self.calls.append(dict(stage=stage,model=model,status=status,input_tokens=None,output_tokens=None,kind='simulated logical SDK call'))
    async def create(self,**kwargs):
        payload_chars=len(json.dumps(kwargs,ensure_ascii=False))
        if payload_chars>64000:
            self.turn.setdefault('budget_events',[]).append(dict(status='rejected',policy='reject SDK request over 64000 chars',actual_chars=payload_chars))
            raise ValueError('offline_context_budget_exceeded')
        self.inputs.append(copy.deepcopy(kwargs))
        self.turn["generation_inputs"].append(copy.deepcopy(kwargs))
        used=any(isinstance(m['content'],list) and any(isinstance(b,dict) and b.get('type')=='tool_result' for b in m['content']) for m in kwargs['messages'])
        if kwargs.get('tools') and not self.fixture.get('skip_tool') and not used:
            query=self.req.message+' '+' '.join(self.req.entities.get('order_id',[]))
            content=[dict(type='tool_use',id='tool-'+uuid4().hex,name='search_knowledge_base',input={'query':query,'top_k':3})]
        else: content=[dict(type='text',text=self.fixture['answer'])]
        return NS(content=content,stop_reason='end_turn')
def metric(values):
    v=[x for x in values if x is not None]
    return dict(numerator=sum(x is True for x in v),denominator=len(v),value=sum(x is True for x in v)/len(v) if v else None)
def summarize(turns):
    def eligible(key,t):
        label=t.get('annotations',{})
        return {'required_call_coverage':label.get('expected_retrieval')=='required',
            'execution_success':bool(t.get('events')), 'query_entities':bool(label.get('required_entities')),
            'old_entity_clean':'22222' in label.get('required_entities',[]),
            'returned_evidence':bool(label.get('evidence_quotes')), 'role_coverage':bool(label.get('expected_roles')),
            'required_facts':bool(label.get('required_facts')), 'forbidden_clear':bool(label.get('forbidden_claims')),
            'judge_validity':True}[key]
    keys=('required_call_coverage','execution_success','query_entities','old_entity_clean','returned_evidence','role_coverage','required_facts','forbidden_clear','judge_validity')
    summary={}
    for key in keys:
        rows=[t for t in turns if eligible(key,t)]
        values=[t.get('checks',{}).get(key) for t in rows]
        valid=[t['case_id']+':'+t['turn_id'] for t in rows if t.get('checks',{}).get(key) is not None]
        summary[key]=dict(numerator=sum(v is True for v in values),denominator=len(rows),
            value=sum(v is True for v in values)/len(rows) if rows else None,
            unresolved=sum(v is None for v in values),valid_only_denominator=len(valid),valid_only_ids=valid,
            policy='all applicable planned turns; unresolved retained in denominator; value is demonstrated-pass coverage')
    return summary
from evaluation.agent_scoring import score, judge_identity

async def run_agents(progress=None,run_id=None):
    run_id=run_id or 'agent-'+uuid4().hex
    target=directory(run_id); target.mkdir(parents=True,exist_ok=False)
    data=None
    report=dict(started_at=datetime.now(timezone.utc).isoformat(),quality_status='incomplete; observed dev protocol only',protocol_version=PROTOCOL,scope='orchestrator',mode='fixture/synthetic',suite='agent-dev-v1',run_id=run_id,status='running',turns=[],summary={},identities={})
    try:
        data,snapshot,manifest=load_suite(); fixtures=load(BASE/'evaluation/fixtures/agent.sdk.v1.json')
        report['identities']=dict(annotations=canonical_hash(data),knowledge=snapshot.knowledge_hash,freeze=manifest,fixture=canonical_hash(fixtures),strategy='experimental B synthetic; not business baseline',lexical='all-matches-bm25-before-top-k-v2',scoring='authored-scripted-judge-v1; sentence-units-v2',judge=judge_identity(),context='reject SDK request >64000 chars; top_k=3; no silent truncation',code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        git_head, workspace_dirty = None, None
        try:
            git=subprocess.run(['git','rev-parse','HEAD'],cwd=BASE,capture_output=True,text=True,check=False)
            status=subprocess.run(['git','status','--porcelain'],cwd=BASE,capture_output=True,text=True,check=False)
            if git.returncode == 0:
                git_head = git.stdout.strip()
            if status.returncode == 0:
                workspace_dirty = bool(status.stdout)
        except OSError:
            pass  # Runtime images need not include Git; source module hashes remain available.
        report['identities'].update(git_head=git_head,workspace_dirty=workspace_dirty,
            source=source_identity(BASE),
            code_modules={p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'agents/agent_orchestrator.py',BASE/'core/llm_utils.py',BASE/'evaluation/agent_scoring.py',BASE/'evaluation/answer_quality.py')})
        write_source_bundle(BASE,target,report['identities']['source'])
        for name,value in [('snapshot.json',snapshot.model_dump()),('cases.json',data),('sdk.json',fixtures),('judge.json',load(BASE/'evaluation/fixtures/agent.judge.v1.json'))]:
            (target/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        total=sum(len(c['turns']) for c in data['cases'])
        for case in data['cases']:
            sdk=SDK(); intent=Intent(); o=AgentOrchestrator(api_key='offline',client=sdk,intent_recognizer=intent)
            runner=StrategyRunner(snapshot,SyntheticIndex(snapshot)); state={}; history=[]
            for i,label in enumerate(case['turns']):
                sdk.fixture=fixtures[case['id']][i]
                req=Request(message=label['question'],user_id='offline-user',conv_id=case['id'],history=copy.deepcopy(history),conversation_state=state)
                sdk.req=req
                t=dict(case_id=case['id'],turn_id=str(i+1),question=req.message,request_id=req.request_id,roles=[],queries=[],events=[],answer='',checks={},generation_inputs=[],annotations=label)
                report['turns'].append(t)
                sdk.turn=t
                async def search_with_rewrite(tool_name,query,top_k=5):
                    if tool_name != 'knowledge_search' or top_k != 3:
                        raise ValueError('offline business retrieval contract mismatch')
                    args=dict(query=query,top_k=top_k)
                    e=dict(run_id=run_id,case_id=case['id'],turn_id=str(i+1),request_id=req.request_id,query=args['query'],parameters=copy.deepcopy(args),status='running',shown_chunks=[],cached=False,degraded=False,source='StrategyRunner B synthetic')
                    e.update(tool_call_context.get() or {})
                    t['events'].append(e)
                    if sdk.fixture.get('tool_fault'):
                        e.update(status='error',error_code='fixture_timeout'); raise TimeoutError('offline fixture timeout')
                    ex,tr=await runner.run(case['id'],args['query'],preset('B'))
                    ex=checked_execution(ex,snapshot); rows=[r.model_dump() for r in ex.results]
                    e.update(status=ex.status,shown_chunks=rows,chunks_hash=canonical_hash(rows),strategy_trace=tr.model_dump())
                    return NS(success=ex.status=='success',data=rows,cached=False,reranked=False,error=ex.error_code)
                o.set_shared_tools(build_shared_rag_tools(NS(search_with_rewrite=search_with_rewrite)))
                def observe(trace):
                    event = next((e for e in t['events'] if e.get('tool_use_id') == trace.get('tool_use_id')), None)
                    if event is not None:
                        event['actual_tool_result']=trace.get('result')
                req.tool_observer=observe
                start=len(sdk.inputs); calls=len(sdk.calls)
                result=await o.run(req)
                t.update(answer=result.response,roles=[r.value for r in result.agent_types],queries=[e['query'] for e in t['events']],generation_inputs=sdk.inputs[start:],intent_inputs=copy.deepcopy(intent.inputs[-1:]),model_calls=sdk.calls[calls:],entities=copy.deepcopy(req.entities),tool_traces=result.tool_traces,status=result.status)
                # tool_use_id maps by execution position; same query remains separate per role.

                t['checks']=score(label,t); history += [dict(role='user',content=req.message),dict(role='assistant',content=result.response)]; state=req.conversation_state
                if progress:
                    p=progress(len(report['turns']),total)
                    if inspect.isawaitable(p): await p
                await asyncio.sleep(0)
        report['status']='completed'
    except asyncio.CancelledError:
        report['status']='cancelled'; raise
    except Exception:
        report.update(status='failed',error_code='offline_execution_failed'); raise
    finally:
        if data is not None:
            observed={(t['case_id'],t['turn_id']) for t in report['turns']}
            for c in data['cases']:
                for i,label in enumerate(c['turns']):
                    if (c['id'],str(i+1)) not in observed:
                        report['turns'].append(dict(case_id=c['id'],turn_id=str(i+1),question=label['question'],roles=[],queries=[],events=[],answer='',annotations=label,status='not_executed',checks=dict(judge_validity=False,required_call_coverage=False if label['expected_retrieval']=='required' else None,role_coverage=False,returned_evidence=False if label['evidence_quotes'] else None)))
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        report['summary']=summarize(report['turns'])
        p=target/'report.pending.json'; p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); p.replace(target/'report.json')
    return report
def replay_agent_run(run_id):
    r=copy.deepcopy(read_agent_run(run_id))
    identity=r.get('identities',{})
    actual=hashlib.sha256((BASE/'evaluation/agent_scoring.py').read_bytes()).hexdigest()
    quality_hash=hashlib.sha256((BASE/'evaluation/answer_quality.py').read_bytes()).hexdigest()
    if (identity.get('judge') != judge_identity() or identity.get('code_modules',{}).get('evaluation/agent_scoring.py') != actual
            or identity.get('code_modules',{}).get('evaluation/answer_quality.py') != quality_hash):
        raise ValueError('incompatible Judge/scoring identity; historical report remains readable')
    for t in r['turns']:
        if t.get('answer'): t['checks']=score(t['annotations'],t)
    r['summary']=summarize(r['turns']); return r
