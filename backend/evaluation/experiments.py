"""实验报告 v2：评分协议仍为第一批 rag-evidence-v1。"""
import math
from pathlib import Path
from typing import Literal

from evaluation.errors import ExperimentInputError
from evaluation.retrieval import (
    PROTOCOL, StrictModel, Snapshot, Dataset, Execution, CaseReport, MetricSummary,
    canonical_hash, build_snapshot, make_report, score_evidence, summarize,
)
from evaluation.strategies import Strategy, Trace


class EvidenceScores(StrictModel):
    protocol_version: Literal['rag-evidence-v1']
    total: int
    errors: int
    error_rate: float | None
    main: MetricSummary
    successful: MetricSummary
    results: list[CaseReport]


class ExperimentReport(StrictModel):
    protocol_version: Literal['rag-experiment-v2'] = 'rag-experiment-v2'
    scope: Literal['explicit_strategy_pipeline'] = 'explicit_strategy_pipeline'
    completed: bool
    run_id: str
    config_id: str
    run_identity: str
    mode: Literal['synthetic', 'real', 'fixture']
    observation: Literal['executed', 'replayed'] = 'executed'
    source_dataset_hash: str
    evaluated_dataset_hash: str
    knowledge_hash: str
    freeze_hash: str
    split: Literal['dev','validation']
    case_set: list[str]
    config: Strategy
    dependencies: dict
    code: dict
    started_at: str
    finished_at: str
    setup: dict
    scoring: EvidenceScores
    traces: list[Trace]
    statistics: dict
    limitation: str = '无自动选优/基线晋升；synthetic 仅证明程序路径；fixture 耗时可能为占位。'


def load_json(path):
    import json
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_suite(path, split='dev', corpus_path=None):
    path=Path(path)
    manifest=load_json(path)
    if manifest.get('protocol_version')!='rag-suite-v2':
        raise ExperimentInputError('不支持的 suite 版本')
    freeze_hash=manifest.get('freeze_hash')
    if freeze_hash!=canonical_hash({k:v for k,v in manifest.items() if k!='freeze_hash'}):
        raise ExperimentInputError('suite 冻结哈希不一致')
    corpus=load_json(corpus_path or path.parent/manifest['corpus'])
    snapshot=build_snapshot(corpus)
    saved=Snapshot.model_validate(load_json(path.parent/manifest['snapshot']))
    if snapshot!=saved or snapshot.knowledge_hash!=manifest['knowledge_hash']:
        raise ExperimentInputError('语料/导出快照不一致，需要人工复核标注')
    source=Dataset.model_validate(load_json(path.parent/manifest['dataset']))
    source.validate_snapshot(snapshot)
    if source.dataset_hash!=manifest['source_dataset_hash']:
        raise ExperimentInputError('完整数据集不符合冻结身份')
    # 同时检查所有 split，不只校验本次被选中的一半。
    for name, frozen in manifest['splits'].items():
        subset=source.model_copy(update={'cases':[c for c in source.cases if c.split==name]})
        if frozen!={'case_ids':[c.id for c in subset.cases],
                    'groups':sorted({c.group_id for c in subset.cases}),
                    'evaluated_dataset_hash':subset.dataset_hash}:
            raise ExperimentInputError('split 清单或标注冻结身份不匹配')
    if split not in manifest['splits']:
        raise ExperimentInputError('suite 缺少所选 split')
    selected=source.model_copy(update={'cases':[c for c in source.cases if c.split==split]})
    if not selected.cases:
        raise ExperimentInputError('不能运行空用例集合')
    return manifest,snapshot,source,selected


def run_identity(report):
    return canonical_hash({key:getattr(report,key) for key in (
        'source_dataset_hash','evaluated_dataset_hash','knowledge_hash','freeze_hash',
        'split','case_set','config_id','mode','dependencies','run_id','started_at')})


def make_experiment(*, dataset, source, snapshot, manifest, split, config, dependencies,
                    executions, traces, run_id, mode, code, started_at, finished_at, setup):
    # 只取原评分器的证据结果；不把它包装成默认 /search 的 A/B/C/D 报告。
    original=make_report(dataset,snapshot,executions,top_k=config.top_k,run_id=run_id,
                         mode='fixture',started_at=started_at,finished_at=finished_at,
                         code=code,configuration={})
    scoring=EvidenceScores.model_validate(original.model_dump(include=set(EvidenceScores.model_fields)))
    statistics=execution_statistics([row.execution for row in scoring.results],traces,mode)
    result=ExperimentReport(completed=True,run_id=run_id,config_id=canonical_hash(config.model_dump()),
        run_identity='',mode=mode,source_dataset_hash=source.dataset_hash,evaluated_dataset_hash=dataset.dataset_hash,
        knowledge_hash=snapshot.knowledge_hash,freeze_hash=manifest['freeze_hash'],split=split,
        case_set=sorted(c.id for c in dataset.cases),config=config,dependencies=dependencies,code=code,
        started_at=started_at,finished_at=finished_at,setup=setup,scoring=scoring,traces=traces,statistics=statistics)
    result.run_identity=run_identity(result)
    validate_report(result)
    return result


def execution_statistics(executions,traces,mode):
    latencies=sorted(e.elapsed_ms for e in executions)
    n=len(latencies)
    degraded=sum(any(s.status=='degraded' for s in t.effective_steps.values()) for t in traces)
    failed=sum(any(s.status=='failed' for s in t.effective_steps.values()) for t in traces)
    cached=sum(t.caches['pipeline'].hits>0 for t in traces)
    full=sum(e.status=='success' and not t.caches['pipeline'].hits and not any(s.status in ('failed','degraded') for s in t.effective_steps.values())
             for e,t in zip(executions,traces))
    calls=[call for t in traces for call in t.model_calls]
    return {
        'degraded_count':degraded,'degraded_rate':degraded/n if n else None,
        'failed_count':failed,'cache_served_count':cached,
        'complete_count':full,'complete_rate':full/n if n else None,
        'latency':{'n':n,'p50_ms':latencies[math.ceil(.5*n)-1] if n else None,
                   'p95_ms':latencies[math.ceil(.95*n)-1] if n else None,
                   'method':'nearest rank; sequential single pass, request wall clock incl cache; setup excluded; small sample; no repeated-run variance',
                   'scope':mode},
        'model_calls':len(calls), 'embedding_query_calls':sum(t.embedding_calls for t in traces),
        'model_call_boundary':'observed dependency invocations; synthetic has no provider calls; real logical SDK calls with retries disabled',
        'usage':{field:sum(getattr(c,field) for c in calls) if all(getattr(c,field) is not None for c in calls) else None
                 for field in ('input_tokens','output_tokens','cache_read_input_tokens','cache_creation_input_tokens')},
        'usage_reported_calls':sum(c.input_tokens is not None and c.output_tokens is not None for c in calls),
        'cost':None,'cost_reason':'no verified provider price; unknown usage is never zero-filled',
    }


def validate_report(report):
    required={'embedding','model','prompt','distance'}
    if (not required <= set(report.dependencies) or set(report.dependencies)-required-{'lexical'}
            or any(not isinstance(report.dependencies.get(k),str) or not report.dependencies[k] for k in required-{'model'})
            or ('lexical' in report.dependencies and (not isinstance(report.dependencies['lexical'],str) or not report.dependencies['lexical']))):
        raise ExperimentInputError('报告缺少可靠的检索依赖身份')
    if report.config.rewrite and not report.dependencies['model']:
        raise ExperimentInputError('改写策略缺少模型身份')
    if not report.completed or report.run_identity!=run_identity(report):
        raise ExperimentInputError('未完成报告或实验身份无效')
    if report.config_id!=canonical_hash(report.config.model_dump()):
        raise ExperimentInputError('配置身份不一致')
    ids=[r.case.id for r in report.scoring.results]
    if not ids or len(ids)!=len(set(ids)) or sorted(ids)!=report.case_set or report.scoring.total!=len(ids):
        raise ExperimentInputError('报告缺少关键样本')
    if [t.case_id for t in report.traces]!=ids:
        raise ExperimentInputError('轨迹与样本不对应')
    d=Dataset(protocol_version=PROTOCOL,knowledge_hash=report.knowledge_hash,cases=[r.case for r in report.scoring.results])
    if d.dataset_hash!=report.evaluated_dataset_hash or any(c.split!=report.split for c in d.cases):
        raise ExperimentInputError('被评测标注/集合身份不一致')
    if any(t.requested_config!=report.config for t in report.traces):
        raise ExperimentInputError('轨迹配置与报告配置不符')
    if any(t.embedding_identity!=report.dependencies['embedding'] or
           set(t.effective_steps)!={'rewrite','retrieval','rerank'} or
           set(t.caches)!={'pipeline','tool'} or
           any(call.model!=report.dependencies['model'] for call in t.model_calls) for t in report.traces):
        raise ExperimentInputError('轨迹依赖/步骤/缓存身份不完整或不匹配')
    if report.statistics!=execution_statistics([r.execution for r in report.scoring.results],report.traces,report.mode):
        raise ExperimentInputError('报告执行率/用量/耗时汇总不一致')
    # 不信任汇总数字：由原评分器重新核对逐例分数和所有分母。
    for row in report.scoring.results:
        if row.execution.case_id!=row.case.id or row.scores!=score_evidence(row.case,row.execution,report.config.top_k) or row.eligible!=row.case.eligible:
            raise ExperimentInputError('报告逐例评分不一致')
    errors=sum(r.execution.status=='error' for r in report.scoring.results)
    if (report.scoring.main!=summarize(report.scoring.results) or
            report.scoring.successful!=summarize([r for r in report.scoring.results if r.execution.status=='success']) or
            report.scoring.errors!=errors or report.scoring.error_rate!=(errors/len(ids) if ids else None)):
        raise ExperimentInputError('报告评分汇总不一致')


def compare(left: ExperimentReport, right: ExperimentReport):
    for report in (left,right):
        validate_report(report)
    for key in ('knowledge_hash','evaluated_dataset_hash','source_dataset_hash','freeze_hash','split','case_set','mode'):
        if getattr(left,key)!=getattr(right,key):
            raise ExperimentInputError('报告不可比: '+key)
    for key in ('top_k','candidate_k','vector_k','lexical_k','rewrite_count','cache_mode','merge','rrf_constant'):
        if getattr(left.config,key)!=getattr(right.config,key):
            raise ExperimentInputError('非研究变量不同: '+key)
    for key in ('embedding','distance','prompt','lexical'):
        default = 'match-prefix-limit-v1' if key == 'lexical' else None
        if left.dependencies.get(key, default)!=right.dependencies.get(key, default):
            raise ExperimentInputError('检索依赖不同: '+key)
    if left.dependencies.get('model') and right.dependencies.get('model') and left.dependencies['model']!=right.dependencies['model']:
        raise ExperimentInputError('改写/重排模型不同')
    a={r.case.id:r for r in left.scoring.results}
    b={r.case.id:r for r in right.scoring.results}
    common=sorted(i for i in a if a[i].execution.status==b[i].execution.status=='success')
    rows=[]
    counts={'win':0,'tie':0,'loss':0,'not_applicable':0}
    for key in left.case_set:
        x,y=a[key].scores,b[key].scores
        deltas={field:getattr(y,field)-getattr(x,field) for field in ('hit_at_k','evidence_recall_at_k','mrr_at_k')} if x and y else None
        value=deltas['evidence_recall_at_k'] if deltas else None
        outcome='not_applicable' if value is None else 'win' if value>0 else 'loss' if value<0 else 'tie'
        counts[outcome]+=1
        rows.append({'case_id':key,'delta_right_minus_left':deltas,'outcome':outcome,
                     'left_execution':a[key].execution.model_dump(),'right_execution':b[key].execution.model_dump()})
    return {'protocol_version':'rag-comparison-v2','left':left.run_identity,'right':right.run_identity,
            'mode':left.mode,'primary':'evidence_recall_at_k; right relative to left',
            'counts':counts,'eligible_denominator':left.scoring.main.count,
            'left_main':left.scoring.main.model_dump(),'right_main':right.scoring.main.model_dump(),
            'left_execution_summary':{'error_rate':left.scoring.error_rate,**left.statistics},
            'right_execution_summary':{'error_rate':right.scoring.error_rate,**right.statistics},
            'common_success_case_ids':common,'common_success_denominator':len(common),
            'common_success_left':summarize([a[i] for i in common]).model_dump(),
            'common_success_right':summarize([b[i] for i in common]).model_dump(),
            'rows':rows,'regression_verdict':'inconclusive_execution_errors' if left.scoring.errors or right.scoring.errors else 'descriptive_only_no_automatic_promotion',
            'conclusion':'synthetic/fixture 不用于选择真实获胜策略；真实单轮实验也不代表稳定生产收益。'}
