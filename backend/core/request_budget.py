"""Task-local deadlines for chat. No active context means legacy caller limits."""
import time
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from core.performance import record_budget_event


@dataclass(frozen=True)
class BudgetPolicy:
    """一次 /chat 请求的全部时间参数（秒）。改动任何一项都要通过 validate()。"""
    hard_timeout_s: float = 50        # api 层 asyncio.timeout，兜住预算之外的收尾
    request_s: float = 45             # 整个请求的预算
    primary_s: float = 28             # 主 Agent 阶段上限
    primary_reserve_s: float = 5      # 主 Agent 必须给后续阶段留出的时间
    fallback_min_start_s: float = 11  # 剩余时间低于此值不再启动兜底 Agent
    fallback_s: float = 8             # 兜底 Agent 阶段上限
    fallback_reserve_s: float = 3     # 兜底 Agent 给汇总留出的时间
    composer_min_start_s: float = 2   # 剩余时间低于此值跳过模型汇总
    rag_optional_min_start_s: float = 18  # 剩余时间低于此值跳过查询改写/重排
    tool_handler_s: float = 30        # Agent 内单个异步工具的上限
    agent_model_s: float = 15         # 单次 Agent 模型调用上限
    intent_model_s: float = 5
    rewrite_model_s: float = 2
    rerank_model_s: float = 3
    composer_model_s: float = 5
    # 无请求预算（旧调用方）时各调用点使用的默认超时，可被环境变量覆盖。
    intent_timeout_default_s: float = 8
    agent_timeout_default_s: float = 15
    composer_timeout_default_s: float = 8

    def model_cap(self, stage):
        return {'intent': self.intent_model_s, 'rag.rewrite': self.rewrite_model_s,
                'rag.rerank': self.rerank_model_s, 'composer': self.composer_model_s}.get(stage, self.agent_model_s)

    def validate(self):
        """各阶段之和不得超过请求预算；违反时在导入阶段直接失败。"""
        checks = {
            '外层超时必须大于请求预算': self.hard_timeout_s > self.request_s,
            '意图 + 主 Agent + 兜底启动门槛不得超过请求预算':
                self.intent_model_s + self.primary_s + self.fallback_min_start_s <= self.request_s,
            '主 Agent 预留必须够一次汇总调用': self.primary_reserve_s >= self.composer_model_s,
            '兜底启动门槛必须覆盖兜底阶段及其预留':
                self.fallback_min_start_s >= self.fallback_s + self.fallback_reserve_s,
            '兜底预留必须达到汇总启动门槛': self.fallback_reserve_s >= self.composer_min_start_s,
            '可选检索步骤之后必须还够一次 Agent 模型调用':
                self.rag_optional_min_start_s >= max(self.rewrite_model_s, self.rerank_model_s) + self.agent_model_s,
            '预算内模型上限不得超过无预算默认超时':
                self.intent_model_s <= self.intent_timeout_default_s
                and self.agent_model_s <= self.agent_timeout_default_s
                and self.composer_model_s <= self.composer_timeout_default_s,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError('预算参数不一致：' + '；'.join(failed))
        return self


POLICY = BudgetPolicy().validate()


@dataclass
class Budget:
    deadline: float
    clock: object
    request_deadline: float
    degraded: set = field(default_factory=set)


_current = ContextVar('servicepilot_request_budget', default=None)


@contextmanager
def request_budget(seconds=POLICY.request_s, *, clock=time.monotonic):
    deadline = clock() + seconds
    token = _current.set(Budget(deadline, clock, deadline))
    try:
        yield
    finally:
        _current.reset(token)


@asynccontextmanager
async def chat_budget(seconds=POLICY.request_s):
    with request_budget(seconds):
        yield


def remaining():
    budget = _current.get()
    return max(0.0, budget.deadline-budget.clock()) if budget else None


@contextmanager
def phase_budget(seconds, *, reserve=0):
    parent = _current.get()
    if parent is None:
        yield
        return
    token = _current.set(Budget(min(parent.deadline-reserve, parent.clock()+seconds), parent.clock, parent.request_deadline, parent.degraded))
    try:
        yield
    finally:
        _current.reset(token)


def observe(stage, status, **fields):
    budget = _current.get()
    if budget is not None:
        record_budget_event(dict(stage=stage, status=status, phase_remaining_s=remaining(),
            request_remaining_s=max(0.0, budget.request_deadline-budget.clock()), **fields))


def can_start(stage, minimum):
    value = remaining()
    if value is not None and value < minimum:
        observe(stage, 'skipped', reason='insufficient_remaining_budget', minimum_s=minimum)
        mark_degraded(stage)
        return False
    return True


def mark_degraded(stage):
    budget = _current.get()
    if budget is not None and stage.startswith('rag.'):
        budget.degraded.add(stage)


def budget_degraded():
    budget = _current.get()
    return bool(budget and budget.degraded)


def limit_timeout(limit):
    value = remaining()
    return min(limit, value) if value is not None else limit


def model_timeout(stage):
    if remaining() is None:
        return None
    return limit_timeout(POLICY.model_cap(stage))
