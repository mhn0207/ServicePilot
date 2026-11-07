"""第二批独立检索实验 CLI；默认 dev + synthetic，不加载 .env。"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from uuid import uuid4
from pydantic import ValidationError

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from evaluation.errors import ExperimentInputError
from evaluation.retrieval import Execution, build_snapshot
from evaluation.strategies import StrategyRunner, Trace, preset
from evaluation.experiment_dependencies import SyntheticIndex, SyntheticModel, real_index, ProviderModel
from evaluation.experiments import (
    ExperimentReport, load_json, load_suite, make_experiment, compare, validate_report,
)
from scripts.evaluate_retrieval_evidence import code_identity
from evaluation.provenance import write_source_bundle

OUTPUT_ROOT=ROOT/'data/eval/rag/experiments'
DEFAULT_SUITE=ROOT/'evaluation/corpora/suite.v2.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path,value):
    if hasattr(value,'model_dump'):
        value=value.model_dump()
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


async def execute(args, output, suite):
    manifest,snapshot,source,dataset=suite
    names=list('ABCD') if args.matrix else [args.candidate or args.strategy or 'B']
    if args.with_baselines:
        names=list(dict.fromkeys(['A','B']+names))
    reports=[]
    for name in names:
        config=preset(name,top_k=args.top_k or 3,candidate_k=args.candidate_k or 20,
                      vector_k=args.vector_k or 20,lexical_k=args.lexical_k or 20,
                      cache_mode=args.cache_mode or 'bypass')
        started=now()
        code=code_identity()
        t=time.perf_counter()
        # 每个 run 有自己的 collection/模型客户端/缓存；共享的只是只读快照。
        index=await asyncio.to_thread(real_index,snapshot,args.onnx_cache) if args.mode=='real' else SyntheticIndex(snapshot)
        setup={'index_ms':(time.perf_counter()-t)*1000,'document_embedding_count':len(snapshot.chunks),
               'result_cache':'cold; process-local per strategy', 'embedding_connection_warmup':'not separately performed',
               'storage':'new in-memory collection; no user database'}
        model=None
        executions,traces=[],[]
        try:
            model=ProviderModel() if args.mode=='real' and config.rewrite else SyntheticModel() if config.rewrite else None
            runner=StrategyRunner(snapshot,index,model)
            for case in dataset.cases:
                e,trace=await runner.run(case.id,case.turns[0],config)
                executions.append(e)
                traces.append(trace)
            await asyncio.to_thread(index.verify,snapshot)
            report=make_experiment(dataset=dataset,source=source,snapshot=snapshot,manifest=manifest,
                split=args.split,config=config,dependencies=runner.identity(),executions=executions,traces=traces,
                run_id=output.name+'_'+name,mode=args.mode,code=code,started_at=started,
                finished_at=now(),setup=setup)
            directory=output/name
            directory.mkdir()
            write(directory/'snapshot.json',snapshot)
            write(directory/'source_dataset.json',source)
            write(directory/'dataset.json',dataset)
            write(directory/'suite.json',manifest)
            write(directory/'executions.json',[e.model_dump() for e in executions])
            write(directory/'traces.json',[t.model_dump() for t in traces])
            write(directory/'report.json',report)
            write_source_bundle(ROOT,directory,report.code)
            reports.append(report)
            print(f'{name}: {report.scoring.total} cases, {report.scoring.errors} errors; mode={args.mode}')
        finally:
            if args.mode=='real':
                if model:
                    await model.close()
                await asyncio.to_thread(index.client.delete_collection,index.collection.name)
    if len(reports)>1:
        write(output/'comparisons.json',[compare(reports[0],r) for r in reports[1:]])
    write(output/'completed.json',{'reports':[r.run_id for r in reports],
                                  'mode':args.mode,'conclusion':'实现验收；真实收益需另行实验' if args.mode=='synthetic' else '单次实验；不自动选优/晋升'})
    return 1 if any(r.scoring.errors for r in reports) else 0


def replay(args,output,suite):
    manifest,snapshot,source,dataset=suite
    directory=args.fixture
    original=ExperimentReport.model_validate(load_json(directory/'report.json'))
    validate_report(original)
    if (original.source_dataset_hash!=source.dataset_hash or original.evaluated_dataset_hash!=dataset.dataset_hash or
            original.knowledge_hash!=snapshot.knowledge_hash or original.freeze_hash!=manifest['freeze_hash'] or original.split!=args.split):
        raise ExperimentInputError('重放 suite/split 身份不兼容')
    for key in ('top_k','candidate_k','vector_k','lexical_k','cache_mode'):
        value=getattr(args,key)
        if value is not None and value!=getattr(original.config,key):
            raise ExperimentInputError('重放配置不一致: '+key)
    executions=[Execution.model_validate(row) for row in load_json(directory/'executions.json')]
    traces=[Trace.model_validate(row) for row in load_json(directory/'traces.json')]
    report=make_experiment(dataset=dataset,source=source,snapshot=snapshot,manifest=manifest,split=args.split,
        config=original.config,dependencies=original.dependencies,executions=executions,traces=traces,
        run_id=output.name,mode=original.mode,code=code_identity(),started_at=now(),finished_at=now(),setup=original.setup)
    if report.scoring!=original.scoring or traces!=original.traces:
        raise ExperimentInputError('重放记录与原报告不一致')
    report.observation='replayed'
    report.limitation='重放原始 '+original.mode+' 记录；不执行检索/模型；耗时和用量来自原记录，不是本次实测。'
    write(output/'snapshot.json',snapshot)
    write(output/'source_dataset.json',source)
    write(output/'dataset.json',dataset)
    write(output/'suite.json',manifest)
    write(output/'executions.json',[e.model_dump() for e in executions])
    write(output/'traces.json',[t.model_dump() for t in traces])
    write(output/'report.json',report)
    write_source_bundle(ROOT,output,report.code)
    return 1 if report.scoring.errors else 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--export-snapshot',action='store_true')
    action.add_argument('--validate-suite',action='store_true')
    action.add_argument('--fixture',type=Path,help='重放某次单策略目录，沿用原模式身份，并标记 replayed')
    action.add_argument('--compare',nargs=2,type=Path,metavar=('LEFT_REPORT','RIGHT_REPORT'))
    choice=parser.add_mutually_exclusive_group()
    choice.add_argument('--strategy',choices=list('ABCD'))
    choice.add_argument('--matrix',action='store_true')
    choice.add_argument('--candidate',choices=list('ABCD'),help='明确选择验证候选；模拟候选不代表真实获胜方案')
    parser.add_argument('--with-baselines',action='store_true',help='显式候选同时运行固定 A/B')
    parser.add_argument('--suite',type=Path,default=DEFAULT_SUITE)
    parser.add_argument('--corpus',type=Path,help='显式原始文档 JSON，执行时必须符合 suite 冻结快照')
    parser.add_argument('--split',choices=['dev','validation'],default='dev')
    parser.add_argument('--mode',choices=['synthetic','real'],default='synthetic')
    parser.add_argument('--onnx-cache',type=Path,help='真实模式：已缓存 all-MiniLM-L6-v2 目录，其下必须有 onnx/；不自动下载')
    parser.add_argument('--enable-provider',action='store_true',help='显式启用真实 C/D 模型调用；只读 EVAL 进程环境')
    for name in ('top-k','candidate-k','vector-k','lexical-k'):
        parser.add_argument('--'+name,type=int)
    parser.add_argument('--cache-mode',choices=['bypass','use'])
    parser.add_argument('--run-id')
    args=parser.parse_args(argv)
    phase='arguments'
    try:
        if args.split=='validation' and not (args.candidate or args.fixture or args.compare or args.validate_suite or args.export_snapshot):
            raise ExperimentInputError('验证集必须显式 --candidate，禁止 validation 矩阵')
        if args.with_baselines and not args.candidate:
            raise ExperimentInputError('--with-baselines 需要 --candidate')
        if args.fixture and (args.strategy or args.matrix or args.candidate or args.with_baselines or args.mode!='synthetic' or args.enable_provider):
            raise ExperimentInputError('重放不能同时执行策略或真实模型')
        if args.mode=='real' and not (args.fixture or args.compare or args.validate_suite or args.export_snapshot):
            if not args.onnx_cache:
                raise ExperimentInputError('真实模式需要 --onnx-cache，禁止自动下载')
            needs_model=args.matrix or (args.strategy or args.candidate or 'B') in 'CD'
            if needs_model and not args.enable_provider:
                raise ExperimentInputError('真实 C/D 需要 --enable-provider；可能消耗额度')
        run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid4().hex[:12]
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',run_id):
            raise ExperimentInputError('run-id 格式无效')
        output=OUTPUT_ROOT/run_id
        if args.compare:
            phase='compare.report_identity'
            reports=[ExperimentReport.model_validate(load_json(p)) for p in args.compare]
            comparison=compare(*reports)
            output.mkdir(parents=True,exist_ok=False)
            write(output/'comparison.json',comparison)
            print('compatible comparison; no automatic baseline promotion')
            return 1 if any(r.scoring.errors for r in reports) else 0
        if args.export_snapshot:
            phase='export.corpus'
            manifest=load_json(args.suite)
            snapshot=build_snapshot(load_json(args.corpus or args.suite.parent/manifest['corpus']))
            output.mkdir(parents=True,exist_ok=False)
            write(output/'snapshot.json',snapshot)
            print(f'{len(snapshot.documents)} documents, {len(snapshot.chunks)} chunks; {snapshot.knowledge_hash}')
            return 0
        phase='suite.snapshot_annotations_freeze'
        suite=load_suite(args.suite,args.split,args.corpus)
        if args.validate_suite:
            print(f'frozen suite valid: {suite[0]["freeze_hash"]}; selected {len(suite[3].cases)} cases')
            return 0
        # 在创建目录及依赖之前验证预算；避免无效参数触发模型加载。
        if not args.fixture:
            phase='configuration.budgets'
            for name in list('ABCD') if args.matrix else [args.candidate or args.strategy or 'B']:
                preset(name,**{key:getattr(args,key) for key in ('top_k','candidate_k','vector_k','lexical_k','cache_mode') if getattr(args,key) is not None})
        phase='output.create_unique_directory'
        output.mkdir(parents=True,exist_ok=False)
        phase='fixture.replay' if args.fixture else 'execution.dependencies_and_report'
        return replay(args,output,suite) if args.fixture else asyncio.run(execute(args,output,suite))
    except Exception as error:
        # 不打印可能含输入正文、路径中的秘密或 SDK 响应的异常字符串。
        locations=['.'.join(str(p) for p in row['loc'])+':'+row['type']+':'+row['msg'] for row in error.errors(include_input=False,include_context=False)] if isinstance(error,ValidationError) else []
        safe_reason=str(error) if isinstance(error,ExperimentInputError) else ''
        print(f'实验校验失败 [{phase}] {type(error).__name__}'+(' fields='+','.join(locations) if locations else '')+
              (': '+safe_reason if safe_reason else '')+'；检查参数、冻结 suite、预算、报告完整性及目录是否已存在。',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
