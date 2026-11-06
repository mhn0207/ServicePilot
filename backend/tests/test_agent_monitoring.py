"""Request lifecycle metrics must describe completed results, not pending SDK calls."""
import asyncio
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry

from agents.agent_orchestrator import AgentOrchestrator, AgentType, BillingAgent, Request
from monitor.performance_monitor import PerformanceMonitor


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'partial', 'error', 'cancelled'])
async def test_pending_request_does_not_report_failure_and_terminal_result_is_counted(outcome):
    entered, release = asyncio.Event(), asyncio.Event()

    async def create(**kwargs):
        entered.set()
        await release.wait()
        if outcome == 'error':
            raise RuntimeError('controlled provider failure')
        return SimpleNamespace(content=[{'type': 'text', 'text': '已提供核验建议'}],
                               stop_reason='max_tokens' if outcome == 'partial' else 'end_turn')

    agent = BillingAgent(SimpleNamespace(messages=SimpleNamespace(create=create)), 'offline')
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {AgentType.BILLING: [agent]}
    monitor = PerformanceMonitor(orchestrator, SimpleNamespace(get_stats=lambda: {}),
                                 registry=CollectorRegistry())
    task = asyncio.create_task(agent.handle(Request(message='测试', user_id='u', conv_id='c')))
    try:
        await entered.wait()
        await monitor._collect()
        pending = orchestrator.get_stats()['billing_0']
        assert pending['success_rate'] == 1.0
        assert not [a for a in monitor._alerts if a.metric == 'agent_success_rate:billing_0']
        assert pending['total'] == 1 and pending['in_flight'] == 1 and pending['completed'] == 0
        if outcome == 'cancelled':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            response = await task
            assert response.status == ('failed' if outcome == 'error' else outcome)
        ended = orchestrator.get_stats()['billing_0']
        assert ended['in_flight'] == 0 and ended['completed'] == 1 and ended['total'] == 1
        assert ended['success_rate'] == (1.0 if outcome == 'success' else 0.0)
        await monitor._collect()
        alerts = [a for a in monitor._alerts if a.metric == 'agent_success_rate:billing_0']
        assert bool(alerts) == (outcome != 'success')
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await monitor.stop()


@pytest.mark.asyncio
async def test_parallel_inflight_requests_do_not_dilute_completed_rate_or_latency():
    release = [asyncio.Event(), asyncio.Event()]
    entered = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        index = calls
        calls += 1
        entered[index].set()
        await release[index].wait()
        return SimpleNamespace(content=[{'type': 'text', 'text': '完成'}], stop_reason='end_turn')

    agent = BillingAgent(SimpleNamespace(messages=SimpleNamespace(create=create)), 'offline')
    # One already-completed successful request with a known 120ms latency.
    agent.stats.total, agent.stats.success, agent.stats.total_ms = 1, 1, 120
    tasks = [asyncio.create_task(agent.handle(Request(message='测试', user_id='u', conv_id=str(i))))
             for i in range(2)]
    try:
        await asyncio.gather(*(event.wait() for event in entered))
        assert agent.stats.success_rate == 1.0
        assert agent.stats.avg_ms == 120
        assert agent.stats.in_flight == 2 and agent.stats.completed == 1
        release[0].set()
        await tasks[0]
        assert agent.stats.in_flight == 1 and agent.stats.completed == 2
        assert agent.stats.success_rate == 1.0
        release[1].set()
        await tasks[1]
        assert agent.stats.in_flight == 0 and agent.stats.completed == 3
        assert agent.stats.success_rate == 1.0
    finally:
        for event in release:
            event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
