"""Optional stage events scoped to a single streamed chat turn."""
from contextlib import contextmanager
from contextvars import ContextVar

_sink = ContextVar("chat_progress", default=None)


@contextmanager
def collect_chat_progress(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def emit_chat_stage(step, **details):
    sink = _sink.get()
    if sink is not None:
        sink({"step": step, **details})
