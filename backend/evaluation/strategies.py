"""独立检索策略；不导入应用、默认知识库、ToolManager 或在线统计。"""
import asyncio
import copy
import time
from typing import Literal

from pydantic import Field, model_validator

from evaluation.retrieval import (
    StrictModel, Execution, ReturnedChunk, Snapshot, canonical_hash, checked_execution,
)


class Strategy(StrictModel):
    retriever: Literal['vector', 'hybrid'] = 'vector'
    rewrite: bool = False
    rerank: bool = False
    top_k: int = Field(default=3, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=200)
    vector_k: int = Field(default=20, ge=1, le=200)
    lexical_k: int = Field(default=20, ge=1, le=200)
    rewrite_count: int = Field(default=2, ge=1, le=5)
    cache_mode: Literal['bypass', 'use'] = 'bypass'
    merge: Literal['rrf-v1'] = 'rrf-v1'
    rrf_constant: int = Field(default=60, ge=1, le=1000)

    @model_validator(mode='after')
    def budgets(self):
        if self.candidate_k < self.top_k or self.vector_k < self.candidate_k:
            raise ValueError('需要 top_k <= candidate_k <= vector_k')
        if self.retriever == 'hybrid' and self.lexical_k < self.candidate_k:
            raise ValueError('hybrid 需要 candidate_k <= lexical_k')
        if (self.rewrite or self.rerank) and self.retriever != 'hybrid':
            raise ValueError('本批改写/重排策略要求 hybrid')
        if self.rerank and not self.rewrite:
            raise ValueError('本批重排策略要求 rewrite')
        return self


def preset(name: str, **overrides) -> Strategy:
    if name not in 'ABCD' or len(name) != 1:
        raise ValueError('未知策略')
    return Strategy(retriever='vector' if name == 'A' else 'hybrid',
                    rewrite=name in 'CD', rerank=name == 'D', **overrides)


class Step(StrictModel):
    status: Literal['disabled', 'executed', 'skipped', 'degraded', 'failed']
    reason: str


class CacheObservation(StrictModel):
    reads: int = 0
    hits: int = 0
    writes: int = 0


class ModelCall(StrictModel):
    stage: Literal['rewrite', 'rerank']
    model: str
    prompt_version: str = 'rag-strategy-prompts-v2'
    parameters: dict
    status: Literal['success', 'error'] = 'error'
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float = 0.0
    boundary: str = 'logical SDK call; SDK retries disabled; missing provider usage remains null'


class Trace(StrictModel):
    case_id: str
    requested_config: Strategy
    effective_steps: dict[str, Step]
    rewrite_query_list: list[str]
    candidate_counts: list[dict] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    fallback_errors: list[str] = Field(default_factory=list)
    stages_ms: dict[str, float] = Field(default_factory=dict)
    caches: dict[str, CacheObservation] = Field(default_factory=lambda: {
        'pipeline': CacheObservation(), 'tool': CacheObservation()})
    model_calls: list[ModelCall] = Field(default_factory=list)
    embedding_calls: int = 0
    embedding_identity: str
    cache_state_before: str = 'cold'
    rerank_prompt_policy: str = 'all candidates; title[:160], content[:400]; stable index only, chunk IDs remain outside prompt'


def fuse(lists, constant=60):
    """只融合名次，不相加不同查询/分支的原始分数。ID 为稳定次排序。"""
    scores, items = {}, {}
    for rows in lists:
        seen = set()
        for rank, row in enumerate(rows, 1):
            key = row.chunk_id
            if key in seen:
                continue
            seen.add(key)
            items[key] = row
            scores[key] = scores.get(key, 0.0) + 1 / (constant + rank)
    return [items[key] for key in sorted(items, key=lambda key: (-scores[key], key))]


class StrategyRunner:
    """一个实例的缓存/计数仅归该实例所有；模型只接收查询及实际候选。"""
    def __init__(self, snapshot: Snapshot, index, model=None):
        self.snapshot, self.index, self.model = snapshot, index, model
        self.pipeline_cache, self.tool_cache = {}, {}

    def identity(self):
        return {'embedding': self.index.identity, 'model': self.model.identity if self.model else None,
                'prompt': 'rag-strategy-prompts-v2', 'distance': 'cosine',
                'lexical': getattr(self.index, 'lexical_policy', 'unspecified-injected-policy')}

    async def run(self, case_id, query, config: Strategy, context=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError('查询必须是 1–2000 字符的非空文本')
        started = time.perf_counter()
        trace = Trace(case_id=case_id, requested_config=config, embedding_identity=self.index.identity,
                      effective_steps={name: Step(status='disabled', reason='configuration') for name in ('rewrite','retrieval','rerank')},
                      rewrite_query_list=[query])
        base = dict(config=config.model_dump(), knowledge=self.snapshot.knowledge_hash,
                    dependencies=self.identity(), context=context or {})
        key = canonical_hash(dict(base, query=query))
        trace.cache_state_before = 'warm' if self.pipeline_cache or self.tool_cache else 'cold'
        if config.cache_mode == 'use':
            trace.caches['pipeline'].reads += 1
            if key in self.pipeline_cache:
                results = copy.deepcopy(self.pipeline_cache[key])
                trace.caches['pipeline'].hits += 1
                for step in trace.effective_steps:
                    enabled = step == 'retrieval' or getattr(config, step)
                    if enabled:
                        trace.effective_steps[step] = Step(status='skipped', reason='pipeline_cache_hit')
                trace.counts = {'final':len(results)}
                return self._finish(case_id, results, trace, started, cached=True)

        queries = [query]
        if config.rewrite:
            try:
                rewritten = await self._model_call('rewrite', query, [], config.rewrite_count, trace)
                if not isinstance(rewritten, list) or not rewritten or any(
                        not isinstance(q, str) or not q.strip() or len(q) > 2000 for q in rewritten):
                    raise ValueError('invalid rewrite')
                queries = list(dict.fromkeys([query] + rewritten[:config.rewrite_count]))
                trace.effective_steps['rewrite'] = Step(status='executed', reason='validated_model_queries')
            except Exception:
                trace.fallback_errors.append('rewrite_failed')
                trace.effective_steps['rewrite'] = Step(status='degraded', reason='rewrite_failed_original_query')
        trace.rewrite_query_list = queries
        # gather 保持输入顺序；每个任务独有观测，合并时再统一写 trace。
        async def recall(q):
            counts, errors, calls = {}, [], 0
            observation = CacheObservation()
            tool_key = canonical_hash(dict(base, query=q, layer='tool'))
            if config.cache_mode == 'use':
                observation.reads = 1
                if tool_key in self.tool_cache:
                    observation.hits = 1
                    rows, cached_counts = copy.deepcopy(self.tool_cache[tool_key])
                    return rows, dict(cached_counts, cached=True), errors, calls, observation
            branch_lists = []
            for branch, budget in [('vector', config.vector_k)] + ([('lexical',config.lexical_k)] if config.retriever == 'hybrid' else []):
                try:
                    if branch == 'vector':
                        calls += 1
                    rows = await asyncio.to_thread(getattr(self.index, branch), q, budget)
                    if len(rows) > budget or len({r.chunk_id for r in rows}) != len(rows):
                        raise ValueError('invalid index candidates')
                    checked = checked_execution(Execution(case_id=case_id, results=rows), self.snapshot)
                    if checked.status != 'success':
                        raise ValueError('index snapshot mismatch')
                    branch_lists.append(rows)
                    counts[branch] = len(rows)
                except Exception:
                    errors.append(branch + '_failed')
                    counts[branch] = 0
            # 分支排序融合后，显式按单查询 candidate_k 截断。
            merged = fuse(branch_lists, config.rrf_constant)
            counts.update(branch_total=sum(len(r) for r in branch_lists), branch_unique=len(merged),
                          per_query_truncated=max(0,len(merged)-config.candidate_k))
            merged = merged[:config.candidate_k]
            counts['returned'] = len(merged)
            if config.cache_mode == 'use' and not errors:
                self.tool_cache[tool_key] = copy.deepcopy((merged, counts))
                observation.writes = 1
            return merged, counts, errors, calls, observation

        t = time.perf_counter()
        recalled = await asyncio.gather(*(recall(q) for q in queries))
        trace.stages_ms['retrieval'] = (time.perf_counter()-t)*1000
        for i, (_, counts, errors, calls, cache) in enumerate(recalled):
            trace.candidate_counts.append(dict(query_index=i, **counts))
            trace.embedding_calls += calls
            trace.fallback_errors.extend(f'q{i}_{error}' for error in errors)
            for field in ('reads','hits','writes'):
                setattr(trace.caches['tool'], field, getattr(trace.caches['tool'],field)+getattr(cache,field))
        candidates = fuse([r[0] for r in recalled], config.rrf_constant)
        total = sum(len(r[0]) for r in recalled)
        trace.counts = dict(queries=len(queries), across_query_total=total,
                            across_query_unique=len(candidates), deduplicated=total-len(candidates),
                            pre_rerank=len(candidates))
        retrieval_errors = [err for r in recalled for err in r[2]]
        if retrieval_errors and not candidates:
            trace.effective_steps['retrieval'] = Step(status='failed', reason='request_failed_without_evidence')
            if config.rerank:
                trace.effective_steps['rerank'] = Step(status='skipped', reason='retrieval_failed')
            return self._finish(case_id, [], trace, started, error=True)
        trace.effective_steps['retrieval'] = Step(status='degraded' if retrieval_errors else 'executed',
                                                 reason='partial_branch_failure' if retrieval_errors else 'valid_empty' if not candidates else 'retrieved')
        if config.rerank:
            if len(candidates) <= config.top_k:
                trace.effective_steps['rerank'] = Step(status='skipped', reason='candidates_not_more_than_top_k')
            else:
                try:
                    order = await self._model_call('rerank',query,candidates,config.top_k,trace)
                    if not isinstance(order, list) or not order or any(type(i) is not int or not 0 <= i < len(candidates) for i in order):
                        raise ValueError('invalid indices')
                    # 兼容原策略：有效索引去重，遗漏项按原顺序补齐。
                    unique = list(dict.fromkeys(order))
                    trace.counts['rerank_duplicate_indices'] = len(order)-len(unique)
                    trace.counts['rerank_omitted_indices'] = len(candidates)-len(unique)
                    unique.extend(i for i in range(len(candidates)) if i not in unique)
                    candidates = [candidates[i] for i in unique]
                    trace.effective_steps['rerank'] = Step(status='executed', reason='validated_indices_deduplicated_and_completed')
                except Exception:
                    trace.fallback_errors.append('rerank_failed')
                    trace.effective_steps['rerank'] = Step(status='degraded', reason='rerank_failed_original_order')
        trace.counts.update(post_rerank=len(candidates), final=min(config.top_k,len(candidates)),
                            final_truncated=max(0,len(candidates)-config.top_k), shortfall=max(0,config.top_k-len(candidates)))
        if trace.counts['shortfall']:
            trace.fallback_errors.append('candidate_shortfall')
        results = candidates[:config.top_k]
        # 不缓存降级结果，以免后续命中掩盖模型失败。
        if config.cache_mode == 'use' and not any(s.status in ('degraded','failed') for s in trace.effective_steps.values()):
            self.pipeline_cache[key] = copy.deepcopy(results)
            trace.caches['pipeline'].writes = 1
        return self._finish(case_id, results, trace, started)

    async def _model_call(self, stage, query, candidates, n, trace):
        if self.model is None:
            raise ValueError('missing explicit model')
        parameters = {'temperature': .3 if stage == 'rewrite' else 0.0,
                      'max_tokens': 512 if stage == 'rewrite' else 2048}
        observation = ModelCall(stage=stage, model=self.model.identity, parameters=parameters)
        trace.model_calls.append(observation)
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(self.model.call(stage, query, candidates, n, parameters),
                                              timeout=10 if stage == 'rewrite' else 30)
            for field in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
                value = response.get(field)
                setattr(observation, field, value if type(value) is int and value >= 0 else None)
            if response.get('truncated'):
                raise ValueError('truncated output')
            observation.status = 'success'
            return response['value']
        finally:
            observation.elapsed_ms = (time.perf_counter()-started)*1000
            trace.stages_ms[stage] = observation.elapsed_ms

    @staticmethod
    def _finish(case_id, results, trace, started, error=False, cached=False):
        elapsed = (time.perf_counter()-started)*1000
        trace.stages_ms['total'] = elapsed
        return (Execution(case_id=case_id, status='error' if error else 'success', results=results,
                          error_code='request_error' if error else None, elapsed_ms=elapsed, cached=cached,
                          reranked=trace.effective_steps['rerank'].status == 'executed'), trace)
