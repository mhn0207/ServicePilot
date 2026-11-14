"""Readiness probes must fail when stores fail, without invoking a model."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest

import api.main as api
from memory.conversation_memory import MemoryManager


@pytest.fixture
def healthy_services(monkeypatch):
    monkeypatch.setenv('ZHIYING_ENV', 'development')
    monkeypatch.delenv('ZHIYING_API_TOKENS', raising=False)
    kb = NS(check_health=AsyncMock(return_value=True))
    async def search(params, context):
        raise AssertionError('Health must not search or embed')
    # Production binds knowledge_search to the KnowledgeBase instance.
    kb.search = search
    kb.search_handler = NS(__self__=kb)
    memory = NS(check_health=AsyncMock(return_value=True))
    monkeypatch.setattr(api, '_orchestrator', NS(get_stats=lambda: {}))
    monkeypatch.setattr(api, '_memory', memory)
    monkeypatch.setattr(api, '_tool_manager', NS(_tools={'knowledge_search': NS(handler=kb.search_handler)}))
    return memory, kb


async def health_response():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        return await client.get('/health')


@pytest.mark.asyncio
async def test_readiness_is_healthy_with_available_stores(healthy_services):
    response = await health_response()
    assert response.status_code == 200 and response.json()['status'] == 'ok'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['memory_false', 'memory_exception', 'knowledge_exception', 'uninitialized'])
async def test_readiness_rejects_store_failure(healthy_services, monkeypatch, failure):
    memory, kb = healthy_services
    if failure == 'memory_false': memory.check_health.return_value = False
    if failure == 'memory_exception': memory.check_health.side_effect = ConnectionError('private connection detail')
    if failure == 'knowledge_exception': kb.check_health.side_effect = RuntimeError('private store detail')
    if failure == 'uninitialized': monkeypatch.setattr(api, '_memory', None)
    response = await health_response()
    assert response.status_code == 503
    assert 'private' not in response.text


@pytest.mark.asyncio
async def test_readiness_has_a_bounded_dependency_wait(healthy_services):
    memory, _ = healthy_services
    async def blocked(): await asyncio.Event().wait()
    memory.check_health.side_effect = blocked
    response = await asyncio.wait_for(health_response(), 3)
    assert response.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize('redis_ok,chroma_error,expected', [(True, False, True), (False, False, False), (True, True, False)])
async def test_memory_health_checks_redis_and_chroma(redis_ok, chroma_error, expected):
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = NS(ping=AsyncMock(return_value=redis_ok))
    def count():
        if chroma_error: raise ConnectionError('test store unavailable')
        return 0
    memory._episodic = NS(count=count)
    from core.storage_health import ChromaHealthProbe
    memory._health_probe = ChromaHealthProbe(NS(count=count))
    memory._profile_health_probe = ChromaHealthProbe(NS(count=lambda: 0))
    assert await memory.check_health() is expected


@pytest.mark.asyncio
async def test_cancelled_local_probes_do_not_spawn_more_blocked_threads():
    import threading
    from core.storage_health import ChromaHealthProbe
    release, entered = threading.Event(), threading.Event()
    calls = []
    def count():
        calls.append('count')
        entered.set()
        release.wait(5)
        return 0
    probe = ChromaHealthProbe(NS(count=count))
    try:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(probe.check(), .03)
        assert entered.is_set() and len(calls) == 1
        # A normal shared-pool operation must still be able to run.
        assert await asyncio.wait_for(asyncio.to_thread(lambda: 'free'), .5) == 'free'
    finally:
        release.set()
        await probe.check()


@pytest.mark.asyncio
async def test_remote_probe_has_transport_timeout_and_recovers_without_count_threads(monkeypatch):
    from core.storage_health import ChromaHealthProbe
    original = httpx.AsyncClient
    attempts = []
    def handle(request):
        attempts.append(request.extensions['timeout'])
        if len(attempts) == 1: raise httpx.ReadTimeout('controlled heartbeat timeout', request=request)
        assert request.url.path == '/api/v2/tenants/default_tenant/databases/default_database/collections/collection-one/count'
        return httpx.Response(200, json=0)
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(**kw, transport=httpx.MockTransport(handle)))
    def forbidden_count(): raise AssertionError('Remote health must not use synchronous Chroma count')
    collection = NS(count=forbidden_count, id='collection-one', tenant='default_tenant', database='default_database')
    probe = ChromaHealthProbe(collection, host='review-chroma', port=8000)
    assert await probe.check() is False
    assert await probe.check() is True
    assert attempts and all(0 < timeout <= 1 for call in attempts for timeout in call.values())


@pytest.mark.asyncio
async def test_remote_collection_failure_is_unhealthy_even_when_server_heartbeat_succeeds(monkeypatch):
    from core.storage_health import ChromaHealthProbe
    original = httpx.AsyncClient
    def handle(request):
        return (httpx.Response(200, json={'nanosecond heartbeat': 1}) if request.url.path.endswith('/heartbeat')
                else httpx.Response(404, json={'error': 'collection missing'}))
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(**kw, transport=httpx.MockTransport(handle)))
    collection = NS(count=lambda: 0, id='deleted', tenant='default_tenant', database='default_database')
    assert await ChromaHealthProbe(collection, host='review-chroma').check() is False


@pytest.mark.asyncio
async def test_remote_probes_prepare_verified_tls_before_readiness_checks(monkeypatch):
    import ssl
    from core.storage_health import ChromaHealthProbe

    original = httpx.AsyncClient
    collection = NS(count=lambda: 0, id='one', tenant='default_tenant', database='default_database')
    probes = [ChromaHealthProbe(collection, host='review-chroma') for _ in range(3)]
    contexts = []

    def forbidden_certificate_load(*args, **kwargs):
        raise AssertionError('Readiness must not synchronously reload certificate roots')

    def client(**kwargs):
        context = kwargs['verify']
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        assert kwargs['trust_env'] is False
        contexts.append(context)
        return original(**kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=0)))

    monkeypatch.setattr(ssl, 'create_default_context', forbidden_certificate_load)
    monkeypatch.setattr(httpx, 'AsyncClient', client)
    assert await asyncio.gather(*(probe.check() for probe in probes)) == [True, True, True]
    # Repeated checks retain certificate verification without rebuilding roots.
    assert await probes[0].check() is True
    assert contexts[0] is contexts[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize('api_path,expected', [
    ('/api/v1', '/api/v1/collections/one/count'),
    ('/api/v2', '/api/v2/tenants/default_tenant/databases/default_database/collections/one/count'),
])
async def test_remote_probe_uses_the_configured_chroma_api_version(monkeypatch, api_path, expected):
    from core.storage_health import ChromaHealthProbe
    original = httpx.AsyncClient
    def handle(request):
        return httpx.Response(200, json=0) if request.url.path == expected else httpx.Response(404)
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(**kw, transport=httpx.MockTransport(handle)))
    collection = NS(count=lambda: 0, id='one', tenant='default_tenant', database='default_database')
    assert await ChromaHealthProbe(collection, host='review-chroma', api_path=api_path).check() is True
