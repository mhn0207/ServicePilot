"""回答报告 v1、重算、成对诊断和兼容比较；不复刻检索评分器。"""
from typing import Literal

from evaluation.answer_data import oracle_chunks
from evaluation.answer_quality import (
    AnswerConfig, BranchResult, context_payload, summarize, unresolved, quality_success,
    check_correctness, check_grounding, protocol_identity, request, GEN_SYSTEM,
    CORRECT_SYSTEM, GROUND_SYSTEM, answer_units,
)
from evaluation.experiments import ExperimentReport, validate_report as validate_retrieval
from evaluation.retrieval import StrictModel, canonical_hash, checked_execution


class AnswerReport(StrictModel):
    protocol_version: Literal['rag-answer-report-v1'] = 'rag-answer-report-v1'
    scope: Literal['independent_single_turn_paired_rag'] = 'independent_single_turn_paired_rag'
    completed: bool
    run_id: str
    run_identity: str
    observation: Literal['executed', 'replayed'] = 'executed'
    replay_of: str | None = None
    replayed_at: str | None = None
    started_at: str
    finished_at: str
    code: dict
    identity: dict
    config: AnswerConfig
    prompts: dict
    dependencies: dict
    source: ExperimentReport
    source_trace_hash: str
    rows: list[BranchResult]
    statistics: dict
    pairs: dict
    review: dict
    limitation: str = 'fixture only validates protocol; no real model effectiveness, Chinese quality, human calibration or multi-turn claim. No automatic baseline promotion.'


def source_for_answers(report, suite):
    data, manifest, snapshot, source, selected = suite
    validate_retrieval(report)
    for field, value in dict(knowledge_hash=snapshot.knowledge_hash, source_dataset_hash=source.dataset_hash,
            evaluated_dataset_hash=selected.dataset_hash, freeze_hash=manifest['freeze_hash'],
            split=selected.cases[0].split, case_set=sorted(c.id for c in selected.cases)).items():
        if getattr(report, field) != value:
            raise ValueError('incompatible retrieval source: ' + field)
    for row in report.scoring.results:
        if (checked_execution(row.execution, snapshot) != row.execution or
                len(row.execution.results) > report.config.top_k):
            raise ValueError('invalid source actual chunks or top k')
    return report


def input_identity(suite, case_ids):
    data, manifest, snapshot, source, selected = suite
    subset = selected.model_copy(update={'cases':[c for c in selected.cases if c.id in case_ids]})
    if len(case_ids) != len(set(case_ids)) or sorted(case_ids) != sorted(c.id for c in subset.cases) or not case_ids:
        raise ValueError('invalid selected cases')
    return dict(knowledge_hash=snapshot.knowledge_hash, source_dataset_hash=source.dataset_hash,
        source_evaluated_dataset_hash=selected.dataset_hash, evaluated_dataset_hash=subset.dataset_hash,
        retrieval_freeze_hash=manifest['freeze_hash'], annotation_hash=data.annotation_hash,
        answer_freeze_hash=data.freeze_hash, split=selected.cases[0].split, case_set=sorted(case_ids))


def statistics(rows):
    return {branch:summarize([r for r in rows if r.branch == branch])
            for branch in ('retrieved_context', 'oracle_context')}


def paired(rows):
    by_id = {}
    for row in rows:
        by_id.setdefault(row.case_id, {})[row.branch] = row
    common_generated, common_valid, deltas = [], [], []
    for key, branches in sorted(by_id.items()):
        a, b = branches['retrieved_context'], branches['oracle_context']
        if a.generation_status == b.generation_status == 'success':
            common_generated.append(key)
        if a.judge_status == b.judge_status == 'valid':
            common_valid.append(key)
        def metrics(r):
            return summarize([r])
        sa, sb = metrics(a), metrics(b)
        fields = ('fact_coverage_micro_lower', 'grounding_supported_lower', 'forbidden_violations')
        deltas.append(dict(case_id=key, common_valid=key in common_valid,
            context_counts=[a.context_count,b.context_count], context_chars=[a.context_chars,b.context_chars],
            context_hashes=[a.context_hash,b.context_hash], failures=[a.errors,b.errors],
            oracle_minus_retrieved={f:sb[f]['value']-sa[f]['value'] if key in common_valid and
                sb[f]['value'] is not None and sa[f]['value'] is not None else None for f in fields},
            retrieved_quality_success=quality_success(a), oracle_quality_success=quality_success(b)))
    return dict(common_generated_case_ids=common_generated, common_generated_denominator=len(common_generated),
        common_valid_case_ids=common_valid, common_valid_denominator=len(common_valid),
        common_valid_statistics=statistics([r for r in rows if r.case_id in common_valid]), rows=deltas,
        caution='Oracle is diagnostic with possibly more evidence than Top-K. Difference alone does not identify retrieval as sole cause; preserve main denominators and failures.')


def review_checklist(rows):
    groups = {k:[] for k in ('high', 'low', 'uncertain', 'critical_violation')}
    for r in rows:
        key = r.case_id + ':' + r.branch
        if quality_success(r):
            groups['high'].append(key)
        elif r.applicability == 'applicable':
            groups['low'].append(key)
        if r.judge_status == 'partial':
            groups['uncertain'].append(key)
        if r.correctness and any(f.status == 'violation' for f in r.correctness.forbidden):
            groups['critical_violation'].append(key)
    return dict(protocol_version='rag-answer-review-v1', calibration='not_human_reviewed_fixture_workflow_only',
        categories=groups, rows=[dict(case_id=r.case_id, question=r.question, branch=r.branch, request_id=r.request_id,
        context=r.context, answer=r.answer, correctness=r.correctness.model_dump() if r.correctness else None,
        grounding=r.grounding.model_dump() if r.grounding else None, errors=r.errors,
        human_verdict=None, disagreement=None, reviewer=None) for r in rows])


def report_identity(report):
    return canonical_hash(report.model_dump(exclude={'run_identity', 'observation'}))


def make_answer_report(*, suite, source, rows, case_ids, config, dependencies, run_id, code, started_at, finished_at):
    report = AnswerReport(completed=True, run_id=run_id, run_identity='', started_at=started_at,
        finished_at=finished_at, code=code, identity=input_identity(suite, case_ids), config=config,
        prompts=protocol_identity(), dependencies=dependencies, source=source.model_copy(deep=True),
        source_trace_hash=canonical_hash([t.model_dump() for t in source.traces]), rows=rows,
        statistics=statistics(rows), pairs=paired(rows), review=review_checklist(rows))
    report.run_identity = report_identity(report)
    validate_answer_report(report, suite)
    return report


def validate_answer_report(report, suite):
    data, _, snapshot, _, _ = suite
    legacy_units = report.prompts == protocol_identity(legacy=True)
    source_for_answers(report.source, suite)
    if ((report.observation == 'executed' and (report.replay_of is not None or report.replayed_at is not None))
            or (report.observation == 'replayed' and not (report.replay_of and report.replayed_at))):
        raise ValueError('missing replay provenance')
    if (not report.completed or report.identity != input_identity(suite, report.identity['case_set'])
            or report.run_identity != report_identity(report) or report.prompts not in (protocol_identity(), protocol_identity(legacy=True))
            or report.source_trace_hash != canonical_hash([t.model_dump() for t in report.source.traces])):
        raise ValueError('invalid report identity')
    deps = report.dependencies
    if set(deps) != {'retrieval', 'generation', 'judge', 'mode'} or deps['retrieval'] != report.source.mode:
        raise ValueError('invalid stage dependencies')
    for stage in ('generation', 'judge'):
        if set(deps[stage]) != {'mode', 'identity'} or deps[stage]['mode'] not in ('fixture', 'real', 'synthetic') or not deps[stage]['identity']:
            raise ValueError('invalid model identity')
    modes = {deps['retrieval'], deps['generation']['mode'], deps['judge']['mode']}
    expected_mode = next(iter(modes)) if len(modes) == 1 else 'mixed'
    if deps['mode'] != expected_mode:
        raise ValueError('hidden mixed dependencies')
    cases = {c.case_id:c for c in data.cases}
    executions = {r.case.id:r.execution for r in report.source.scoring.results}
    expected = {(i,b) for i in report.identity['case_set'] for b in ('retrieved_context','oracle_context')}
    keys = [(r.case_id,r.branch) for r in report.rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('missing duplicate branch tasks')
    for r in report.rows:
        c = cases[r.case_id]
        e = executions[r.case_id]
        chunks = ([v for v in e.results if not v.fallback] if r.branch == 'retrieved_context' else oracle_chunks(c, snapshot))
        values, text, status = context_payload(chunks, report.config)
        applicable = r.branch == 'retrieved_context' or bool(c.facts)
        if (r.question != c.question or r.context != values or r.context_text != text or r.context_hash != canonical_hash(text)
                or r.context_chars != len(text) or r.context_count != len(values)
                or r.context_status != (status if applicable else 'not_applicable')
                or r.applicability != ('applicable' if applicable else 'not_applicable')
                or r.retrieval_status != (e.status if r.branch == 'retrieved_context' else 'not_applicable')
                or r.retrieval_error != (e.error_code if r.branch == 'retrieved_context' else None)
                or r.request_id != report.run_id + ':' + r.case_id + ':' + r.branch):
            raise ValueError('invalid branch context or status')
        can_call = applicable and status == 'ready' and r.retrieval_status != 'error'
        if r.context_shown != can_call:
            raise ValueError('context presentation mismatch')
        if not can_call:
            if r.answer is not None or r.calls or r.correctness or r.grounding:
                raise ValueError('unexpected results on unexecuted task')
            if r.generation_status != ('not_called' if applicable else 'not_applicable') or r.judge_status != ('not_called' if applicable else 'not_applicable'):
                raise ValueError('invalid unexecuted status')
        else:
            if not r.calls or r.calls[0].stage != 'generation':
                raise ValueError('missing generation observation')
            generated = r.answer is not None and bool(r.answer.strip())
            if (r.generation_status != ('success' if generated else 'error')
                    or r.calls[0].status != ('success' if generated else 'error')):
                raise ValueError('generation state mismatch')
            payloads = [request(GEN_SYSTEM, dict(question=c.question, actual_context=values))]
            if generated:
                payloads += [request(CORRECT_SYSTEM, dict(question=c.question, answer=r.answer,
                    reference=c.model_dump(exclude={'retrieval_case_hash','notes','split','group_id','optional_facts','scope_rationale'}))),
                    request(GROUND_SYSTEM, dict(question=c.question, answer=r.answer, actual_context=values, units=answer_units(r.answer, legacy=legacy_units)))]
            if [call.stage for call in r.calls] != (['generation','correctness','grounding'] if generated else ['generation']):
                raise ValueError('missing model observations')
            for call, payload in zip(r.calls, payloads):
                identity = deps['generation' if call.stage == 'generation' else 'judge']['identity']
                expected_parameters = dict(temperature=report.config.temperature if call.stage == 'generation' else 0.0,
                    max_tokens=report.config.generation_tokens if call.stage == 'generation' else report.config.judge_tokens)
                if (call.dependency != identity or call.request_hash != canonical_hash(payload)
                        or call.parameters != expected_parameters or (call.status == 'error') != bool(call.error_code)):
                    raise ValueError('invalid call observation')
            if r.correctness:
                check_correctness(r.correctness.model_dump(), c, r.answer)
            if r.grounding:
                check_grounding(r.grounding.model_dump(), r.answer, r.context, legacy_units=legacy_units)
            if generated:
                for call, result in zip(r.calls[1:], [r.correctness,r.grounding]):
                    if (call.status == 'success') != (result is not None):
                        raise ValueError('judge result mismatch')
            expected_judge = 'not_called' if not generated else 'error' if not (r.correctness and r.grounding) else 'partial' if unresolved(r) else 'valid'
            if r.judge_status != expected_judge:
                raise ValueError('invalid Judge completeness')
        errors = ([] if r.retrieval_status != 'error' else ['retrieval:' + e.error_code])
        if applicable and r.retrieval_status != 'error' and status != 'ready':
            errors.append('context:budget_exceeded')
        errors += [call.stage + ':' + call.error_code for call in r.calls if call.status == 'error']
        if r.errors != errors:
            raise ValueError('failure provenance mismatch')
    if report.statistics != statistics(report.rows) or report.pairs != paired(report.rows) or report.review != review_checklist(report.rows):
        raise ValueError('invalid recomputed summary')


def compare_answers(left, right, suite):
    for report in (left, right):
        validate_answer_report(report, suite)
    for field in ('identity', 'config', 'prompts', 'dependencies'):
        if getattr(left, field) != getattr(right, field):
            raise ValueError('incompatible comparison: ' + field)
    # 本协议研究变量为检索方案；生成/Judge/上下文构建固定。检索配置变化不被误拒绝。
    for key in ('embedding', 'distance', 'prompt', 'lexical'):
        default = 'match-prefix-limit-v1' if key == 'lexical' else None
        if left.source.dependencies.get(key, default) != right.source.dependencies.get(key, default):
            raise ValueError('retrieval dependency confound: ' + key)
    a_model, b_model = left.source.dependencies['model'], right.source.dependencies['model']
    if a_model and b_model and a_model != b_model:
        raise ValueError('retrieval model confound')
    a = {(r.case_id,r.branch):r for r in left.rows}
    b = {(r.case_id,r.branch):r for r in right.rows}
    common = sorted(k for k in a if a[k].judge_status == b[k].judge_status == 'valid')
    complete = all(r.judge_status == 'valid' for rep in (left,right) for r in rep.rows if r.applicability == 'applicable')
    return dict(protocol_version='rag-answer-comparison-v1', left=left.run_identity, right=right.run_identity,
        research_variable='retrieval configuration only; all generation/Judge/context policies controlled',
        left_main=left.statistics, right_main=right.statistics,
        common_valid_tasks=[list(k) for k in common], common_valid_denominator=len(common),
        common_valid_left=statistics([a[k] for k in common]), common_valid_right=statistics([b[k] for k in common]),
        rows=[dict(case_id=k[0],branch=k[1],left=summarize([a[k]]),right=summarize([b[k]]),
            quality_regressed=quality_success(a[k]) and not quality_success(b[k]),
            context_changed=a[k].context_hash != b[k].context_hash) for k in sorted(a)],
        regression_verdict='descriptive_only_no_automatic_promotion' if complete else 'inconclusive_incomplete_or_uncertain',
        limitation='No real winner from fixtures; oracle difference alone cannot establish unique causality.')
