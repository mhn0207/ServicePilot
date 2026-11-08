import json
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import api.main as api
from tooling.knowledge_base import KnowledgeBase


class Collection:
    def __init__(self): self.rows = {}
    def upsert(self, ids, documents, metadatas):
        self.rows.update({i: (d, m) for i, d, m in zip(ids, documents, metadatas)})
    def get(self, where=None, include=None, limit=None):
        rows = [(i, d, m) for i, (d, m) in self.rows.items() if not where or all((m or {}).get(k) == v for k, v in where.items())]
        rows = rows[:limit]
        result = {'ids': [r[0] for r in rows], 'metadatas': [r[2] for r in rows]}
        if 'documents' in (include or []): result['documents'] = [r[1] for r in rows]
        return result
    def delete(self, ids):
        for i in ids: self.rows.pop(i, None)
    def count(self): return len(self.rows)


def make_kb():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._write_lock = threading.RLock()
    kb._collection = Collection()
    kb.add_documents([{'document_id': 'A', 'title': '甲', 'content': '甲' * 1100},
                      {'document_id': 'B', 'title': '乙', 'content': '保留的乙文档'}])
    return kb


def test_list_details_order_and_delete_preserve_other_document():
    kb = make_kb()
    listed = kb.list_documents()
    assert len(listed['documents']) == 2
    a = next(d for d in listed['documents'] if d['title'] == '甲')
    assert a['chunk_count'] == 3 and a['deletable']
    detail = kb.get_document(a['document_id'])
    assert [c['chunk_index'] for c in detail['chunks']] == [0, 1, 2]
    assert all(a['version'] in c['chunk_id'] for c in detail['chunks'])
    assert kb.delete_document(a['document_id'], a['version']) == 3
    assert kb.doc_count == 1 and kb.list_documents()['documents'][0]['title'] == '乙'
    with pytest.raises(LookupError): kb.get_document(a['document_id'])


def test_stale_version_never_deletes_updated_document():
    kb = make_kb()
    a = next(d for d in kb.list_documents()['documents'] if d['title'] == '甲')
    kb.add_documents([{'document_id': 'A', 'title': '甲', 'content': '已更新'}])
    with pytest.raises(ValueError): kb.delete_document(a['document_id'], a['version'])
    assert kb.get_document(a['document_id'])['chunks'][0]['content'] == '已更新'


@pytest.mark.parametrize('fault', ['version', 'chunk_index', 'total_chunks', 'missing_chunk', 'mixed_version'])
def test_incomplete_or_legacy_metadata_is_read_only(fault):
    kb = make_kb()
    a = next(d for d in kb.list_documents()['documents'] if d['title'] == '甲')
    ids = [i for i, (_, m) in kb._collection.rows.items() if m['document_id'] == a['document_id']]
    if fault == 'missing_chunk': kb._collection.rows.pop(ids[0])
    elif fault == 'mixed_version': kb._collection.rows[ids[0]][1]['version'] = 'b' * 64
    else: kb._collection.rows[ids[0]][1].pop(fault)
    detail = kb.get_document(a['document_id'])
    assert not detail['deletable']
    count = kb.doc_count
    with pytest.raises(ValueError): kb.delete_document(a['document_id'], a['version'])
    assert kb.doc_count == count


def test_unidentified_chunks_are_reported_without_inventing_document_identity():
    kb = make_kb()
    kb._collection.rows['old'] = ('旧正文', {'title': '旧资料'})
    result = kb.list_documents()
    assert len(result['documents']) == 2
    assert result['unmanaged_chunks'] == [{'title': '旧资料', 'storage_id': 'old'}]


def test_oversized_collection_fails_explicitly():
    kb = make_kb()
    kb._collection.get = Mock(return_value={'ids': ['x'] * 5001, 'metadatas': [{}] * 5001})
    with pytest.raises(ValueError): kb.list_documents()


@pytest.fixture
def api_kb(monkeypatch):
    kb = make_kb()
    monkeypatch.setenv('ZHIYING_ENV', 'production')
    monkeypatch.setenv('ZHIYING_API_TOKENS', json.dumps({'a' * 24: {'user_id': 'admin', 'role': 'admin'}, 'u' * 24: {'user_id': 'user', 'role': 'user'}}))
    monkeypatch.delenv('ZHIYING_METRICS_TOKEN_FILE', raising=False)
    manager = SimpleNamespace(_tools={'knowledge_search': SimpleNamespace(handler=kb.search_handler)}, invalidate_cache=Mock())
    evaluation = SimpleNamespace(invalidate_cache=Mock())
    monkeypatch.setattr(api, '_tool_manager', manager)
    monkeypatch.setattr(api, '_eval_tool_manager', evaluation)
    return kb, manager, evaluation


@pytest.mark.asyncio
async def test_management_routes_enforce_roles_and_versioned_delete(api_kb):
    kb, manager, evaluation = api_kb
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        headers = {'Authorization': 'Bearer ' + 'a' * 24}
        url = '/knowledge/documents'
        assert (await client.get(url)).status_code == 401
        assert (await client.get(url, headers={'Authorization': 'Bearer ' + 'u' * 24})).status_code == 403
        listing = await client.get(url, headers=headers)
        assert listing.status_code == 200
        a = next(d for d in listing.json()['documents'] if d['title'] == '甲')
        path = url + '/' + a['document_id']
        assert (await client.get(path, headers=headers)).json()['chunks'][0]['content']
        assert (await client.delete(path, params={'expected_version': a['version']}, headers={'Authorization': 'Bearer ' + 'u' * 24})).status_code == 403
        assert (await client.delete(path, params={'expected_version': 'b' * 64}, headers=headers)).status_code == 409
        assert (await client.delete(path, params={'expected_version': a['version']}, headers=headers)).json()['deleted_chunks'] == 3
        assert (await client.get(path, headers=headers)).status_code == 404
        assert (await client.get(url + '/invalid', headers=headers)).status_code == 422
        manager.invalidate_cache.assert_called_with('knowledge_search')
        evaluation.invalidate_cache.assert_called_with('knowledge_search')


@pytest.mark.asyncio
async def test_partial_delete_error_invalidates_caches_without_leaking_details(api_kb):
    kb, manager, evaluation = api_kb
    a = next(d for d in kb.list_documents()['documents'] if d['title'] == '甲')
    kb._collection.delete = Mock(side_effect=RuntimeError('private-storage-details'))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        response = await client.delete('/knowledge/documents/' + a['document_id'], params={'expected_version': a['version']}, headers={'Authorization': 'Bearer ' + 'a' * 24})
    assert response.status_code == 503 and 'private-storage-details' not in response.text
    manager.invalidate_cache.assert_called_with('knowledge_search')
    evaluation.invalidate_cache.assert_called_with('knowledge_search')


@pytest.mark.asyncio
async def test_real_tool_caches_are_evicted_by_delete_route(api_kb, monkeypatch):
    from tooling.tool_manager import MCPToolManager, Tool
    kb, _, _ = api_kb
    manager = MCPToolManager(api_key='offline')
    manager.register(Tool('knowledge_search', '', kb.search_handler, {}, cache_ttl=30))
    evaluation = manager.fork()
    for tool_manager in (manager, evaluation):
        tool_manager._set_cache('knowledge_search', {'query': '甲'}, ['cached-A'], 30)
        tool_manager._set_cache('knowledge_search:pipeline', {'query': '甲'}, ['cached-A'], 30)
        assert tool_manager._get_cache('knowledge_search', {'query': '甲'}) is not None
    monkeypatch.setattr(api, '_tool_manager', manager)
    monkeypatch.setattr(api, '_eval_tool_manager', evaluation)
    a = next(d for d in kb.list_documents()['documents'] if d['title'] == '甲')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        result = await client.delete('/knowledge/documents/' + a['document_id'], params={'expected_version': a['version']}, headers={'Authorization': 'Bearer ' + 'a' * 24})
        assert result.status_code == 200
    for tool_manager in (manager, evaluation):
        assert tool_manager._get_cache('knowledge_search', {'query': '甲'}) is None
        assert tool_manager._get_cache('knowledge_search:pipeline', {'query': '甲'}) is None


@pytest.mark.asyncio
async def test_cancelled_http_wait_still_evicts_after_storage_finishes(api_kb):
    kb, manager, evaluation = api_kb
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    original = kb._collection.delete
    def slow_delete(ids):
        entered.set()
        assert release.wait(3)
        original(ids)
        completed.set()
    kb._collection.delete = slow_delete
    a = next(d for d in kb.list_documents()['documents'] if d['title'] == '甲')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        task = asyncio.create_task(client.delete('/knowledge/documents/' + a['document_id'], params={'expected_version': a['version']}, headers={'Authorization': 'Bearer ' + 'a' * 24}))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            before = manager.invalidate_cache.call_count
            release.set()
            assert await asyncio.to_thread(completed.wait, 2)
            for _ in range(20):
                if manager.invalidate_cache.call_count > before: break
                await asyncio.sleep(.01)
            assert manager.invalidate_cache.call_count > before
        finally:
            release.set()
