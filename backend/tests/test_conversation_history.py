import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import api.main as api
from memory.conversation_memory import MemoryManager


@pytest.mark.asyncio
async def test_history_reads_only_authenticated_user_and_excludes_private_metadata(monkeypatch):
    monkeypatch.setenv('ZHIYING_ENV', 'production')
    monkeypatch.setenv('ZHIYING_API_TOKENS', json.dumps({'u' * 24: {'user_id': 'alice', 'role': 'user'}}))
    monkeypatch.delenv('ZHIYING_METRICS_TOKEN_FILE', raising=False)
    read = AsyncMock(return_value=[{'role': 'user', 'content': 'hello', 'timestamp': '2026-11-08T00:00:00'}])
    monkeypatch.setattr(api, '_memory', SimpleNamespace(get_history=read))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        response = await client.get('/conversation/history', params={'conv_id': 'c', 'user_id': 'bob'}, headers={'Authorization': 'Bearer ' + 'u' * 24})
        assert response.status_code == 200
        assert response.json()['messages'][0]['content'] == 'hello'
        read.assert_awaited_once_with('alice', 'c')
        assert (await client.get('/conversation/history', params={'conv_id': 'c'})).status_code == 401


@pytest.mark.asyncio
async def test_history_storage_is_read_only_chronological_and_bounded():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = SimpleNamespace(lrange=AsyncMock(return_value=[json.dumps({'role': role, 'content': content, 'ts': '2026-11-08T00:00:00', 'metadata': {'private': 'hidden'}}) for role, content in [('assistant', 'answer'), ('user', 'question')]]))
    result = await memory.get_history('alice', 'c')
    assert [row['content'] for row in result] == ['question', 'answer']
    assert all(set(row) == {'role', 'content', 'timestamp'} for row in result)
    memory._redis.lrange.assert_awaited_once_with(memory._wm_key('alice', 'c'), 0, memory.WORKING_MAX - 1)


@pytest.mark.asyncio
async def test_history_empty_and_storage_error(monkeypatch):
    monkeypatch.setenv('ZHIYING_ENV', 'development')
    monkeypatch.delenv('ZHIYING_API_TOKENS', raising=False)
    monkeypatch.delenv('ZHIYING_METRICS_TOKEN_FILE', raising=False)
    read = AsyncMock(return_value=[])
    monkeypatch.setattr(api, '_memory', SimpleNamespace(get_history=read))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        assert (await client.get('/conversation/history', params={'conv_id': 'c'})).json()['messages'] == []
        read.side_effect = RuntimeError('private backend details')
        response = await client.get('/conversation/history', params={'conv_id': 'c'})
        assert response.status_code == 503 and 'private backend details' not in response.text
        assert (await client.get('/conversation/history', params={'conv_id': ''})).status_code == 422
