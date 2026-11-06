"""Task-local deadlines for chat. No active context means legacy caller limits."""
import time
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from core.performance import record_budget_event


@dataclass
class Budget:
    deadline: float
    clock: object
    request_deadline: float
    degraded: set = field(default_factory=set)


_current = ContextVar('servicepilot_request_budget', default=None)


@contextmanager
def request_budget(seconds=45, *, clock=time.monotonic):
    deadline = clock() + seconds
    token = _current.set(Budget(deadline, clock, deadline))
    try:
        yield
    finally:
        _current.reset(token)


@asynccontextmanager
async def chat_budget(seconds=45):
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
    cap = {'intent':5, 'rag.rewrite':2, 'rag.rerank':3, 'composer':5}.get(stage,15)
    return limit_timeout(cap)
