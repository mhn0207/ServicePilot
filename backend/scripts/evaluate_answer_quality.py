"""第三批回答质量 CLI，默认 dev / B 模拟检索 + 固定生成和Judge，不加载 .env。"""
import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.answer_data import DEFAULT_SUITE, DEFAULT_ANSWERS, load_answers, oracle_chunks
from evaluation.answer_dependencies import FixedModel, AnswerProvider, DEFAULT_FIXTURE
from evaluation.answer_quality import AnswerConfig, evaluate_branch
from evaluation.answer_reports import (
    AnswerReport, source_for_answers, make_answer_report, validate_answer_report,
    compare_answers, statistics, paired, review_checklist, report_identity,
    input_identity,
)
from evaluation.experiment_dependencies import SyntheticIndex
from evaluation.experiments import load_json, make_experiment, ExperimentReport
from evaluation.strategies import StrategyRunner, preset
from scripts.evaluate_retrieval_evidence import code_identity
from scripts.evaluate_retrieval_strategies import write

OUTPUT_ROOT = ROOT / 'data/eval/rag/answers'


def now():
    return datetime.now(timezone.utc).isoformat()


async def retrieval_source(suite, runner=None, config=None, run_id='answer-source-B', mode='synthetic'):
    """显式注入第二批执行器；默认一个 B，不扫描策略矩阵。"""
    _, manifest, snapshot, source, dataset = suite
    config = config or preset('B')
    runner = runner or StrategyRunner(snapshot, SyntheticIndex(snapshot))
    started = now()
    records = [await runner.run(c.id, c.turns[0], config) for c in dataset.cases]
    return make_experiment(dataset=dataset, source=source, snapshot=snapshot, manifest=manifest,
        split=dataset.cases[0].split, config=config, dependencies=runner.identity(),
        executions=[r[0] for r in records], traces=[r[1] for r in records],
        run_id=run_id, mode=mode, code=code_identity(), started_at=started, finished_at=now(),
        setup={'storage':'injected isolated runner; default synthetic in-memory', 'selection':'B is an engineering default, not a measured winner'})


async def execute_answers(suite, source, generation, judge, config, case_ids, run_id):
    source_for_answers(source, suite)
    input_identity(suite, case_ids)
    data, _, snapshot, _, _ = suite
    cases = {c.case_id:c for c in data.cases}
    executions = {r.case.id:r.execution for r in source.scoring.results}
    rows = []
    for key in case_ids:
        c, e = cases[key], executions[key]
        for branch in ('retrieved_context', 'oracle_context'):
            chunks = [r for r in e.results if not r.fallback] if branch == 'retrieved_context' else oracle_chunks(c, snapshot)
            rows.append(await evaluate_branch(c, branch, chunks, e, generation, judge, config, run_id))
    return rows


def dependencies(source, gen, judge):
    modes = {source.mode, gen.mode, judge.mode}
    return dict(retrieval=source.mode, generation=dict(mode=gen.mode, identity=gen.identity),
        judge=dict(mode=judge.mode, identity=judge.identity), mode=next(iter(modes)) if len(modes) == 1 else 'mixed')


async def execute(args, suite, output, config, ids):
    source = source_for_answers(ExperimentReport.model_validate(load_json(args.source_report)), suite) if args.source_report else await retrieval_source(suite)
    if args.mode == 'real' and source.mode != 'real' and not args.allow_mixed:
        raise ValueError('real generation requires real source or explicit --allow-mixed')
    started = now()
    gen = judge = None
    try:
        gen = AnswerProvider('generation') if args.mode == 'real' else FixedModel(args.model_fixture)
        if args.mode == 'fixture':
            gen.validate_inputs(suite)
        judge = AnswerProvider('judge') if args.mode == 'real' else gen
        rows = await execute_answers(suite, source, gen, judge, config, ids, output.name)
        report = make_answer_report(suite=suite, source=source, rows=rows, case_ids=ids,
            config=config, dependencies=dependencies(source, gen, judge), run_id=output.name,
            code=code_identity(), started_at=started, finished_at=now())
        save(output, report, suite)
        print(f'{len(ids)} cases; modes={report.dependencies}; generation/Judge/quality denominators in report.json')
        return exit_code(report)
    finally:
        if args.mode == 'real':
            for dep in (gen, judge):
                if dep:
                    await dep.close()


def save(output, report, suite):
    write(output/'report.json', report)
    write(output/'review.json', report.review)
    write(output/'annotations.json', suite[0])
    write(output/'snapshot.json', suite[2])
    write(output/'source_report.json', report.source)


def exit_code(report):
    return int(any(r.errors or r.judge_status == 'partial' for r in report.rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--validate-annotations', action='store_true')
    action.add_argument('--fixture', type=Path, help='纯离线报告重放/重新汇总，不执行生成或Judge')
    action.add_argument('--compare', nargs=2, type=Path)
    parser.add_argument('--source-report', type=Path, help='显式第二批单策略 report.json，完整 split 身份')
    parser.add_argument('--suite', type=Path, default=DEFAULT_SUITE)
    parser.add_argument('--annotations', type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument('--model-fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--split', choices=['dev','validation'], default='dev')
    parser.add_argument('--candidate', choices=['B'], help='validation 必须显式选工程默认 B；其他策略用显式 --source-report')
    parser.add_argument('--case', action='append', help='可重复的明确case清单')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--mode', choices=['fixture','real'], default='fixture')
    parser.add_argument('--enable-provider', action='store_true', help='明确允许真实生成+两个Judge请求，可能收费')
    parser.add_argument('--allow-mixed', action='store_true', help='真实生成消费非真实检索时显式标记混合实验')
    parser.add_argument('--context-chars', type=int, default=16000)
    parser.add_argument('--generation-tokens', type=int, default=2048)
    parser.add_argument('--judge-tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--timeout-s', type=float, default=45.0)
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    try:
        if args.candidate and args.source_report:
            raise ValueError('choose either candidate B or an explicit source report')
        if (args.fixture or args.compare or args.validate_annotations) and (args.enable_provider or args.mode == 'real' or args.source_report or args.allow_mixed):
            raise ValueError('read-only action cannot enable models or change source')
        if args.mode == 'real' and not (args.enable_provider and (args.case or args.limit)):
            raise ValueError('real mode requires explicit cost switch and case/limit')
        if args.enable_provider and args.mode != 'real':
            raise ValueError('provider switch requires real mode')
        suite = load_answers(args.annotations, args.suite, args.split)
        if args.validate_annotations:
            print(f'60 annotations valid; annotation_hash={suite[0].annotation_hash}; freeze_hash={suite[0].freeze_hash}')
            return 0
        if args.split == 'validation' and not (args.candidate or args.source_report or args.fixture or args.compare):
            raise ValueError('validation requires explicit candidate/source')
        config = AnswerConfig(context_chars=args.context_chars, generation_tokens=args.generation_tokens,
            judge_tokens=args.judge_tokens, temperature=args.temperature, timeout_s=args.timeout_s)
        ids = [c.id for c in suite[4].cases]
        if args.case:
            if len(args.case) != len(set(args.case)) or not set(args.case) <= set(ids):
                raise ValueError('unknown or duplicate case')
            ids = [i for i in ids if i in args.case]
        if args.limit is not None:
            if args.limit < 1 or args.limit > len(ids):
                raise ValueError('invalid limit')
            ids = ids[:args.limit]
        run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex[:12]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
            raise ValueError('invalid run id')
        output = OUTPUT_ROOT/run_id
        # Before any model dependency or request: refuse existing output.
        output.mkdir(parents=True, exist_ok=False)
        if args.compare:
            reports = [AnswerReport.model_validate(load_json(p)) for p in args.compare]
            write(output/'comparison.json', compare_answers(*reports, suite))
            print('descriptive comparison saved; no baseline promotion')
            return max(exit_code(r) for r in reports)
        if args.fixture:
            path = args.fixture/'report.json' if args.fixture.is_dir() else args.fixture
            original = AnswerReport.model_validate(load_json(path))
            validate_answer_report(original, suite)
            # CLI subset/budget overrides must never silently change a replay.
            if (args.case or args.limit is not None) and sorted(ids) != original.identity['case_set']:
                raise ValueError('replay case set mismatch')
            supplied_flags = {arg.split('=', 1)[0] for arg in (argv if argv is not None else sys.argv[1:])}
            if any(flag in supplied_flags for flag in (
                    '--context-chars','--generation-tokens','--judge-tokens','--temperature','--timeout-s')) and config != original.config:
                raise ValueError('replay configuration mismatch')
            original.replay_of = original.run_identity
            original.replayed_at = now()
            original.run_id = run_id
            original.code = code_identity()
            original.observation = 'replayed'
            for r in original.rows:
                r.request_id = run_id + ':' + r.case_id + ':' + r.branch
            original.statistics = statistics(original.rows)
            original.pairs = paired(original.rows)
            original.review = review_checklist(original.rows)
            original.run_identity = report_identity(original)
            validate_answer_report(original, suite)
            save(output, original, suite)
            print('replayed without requests; original observations retained')
            return exit_code(original)
        return asyncio.run(execute(args, suite, output, config, ids))
    except Exception as ex:
        # Never echo unknown SDK/error response bodies, private input, credentials or endpoint.
        print('回答评测失败：' + type(ex).__name__ + '；检查冻结身份、case/split、显式模式、配置和唯一输出目录。', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
