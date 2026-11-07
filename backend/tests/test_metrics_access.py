"""A scrape credential can only read metrics; it cannot become an API identity."""
import json
import httpx
import pytest
import api.main as api


@pytest.fixture
def metrics_auth(monkeypatch, tmp_path):
    monkeypatch.setenv('ZHIYING_ENV', 'production')
    monkeypatch.setenv('ZHIYING_API_TOKENS', json.dumps({
        'a' * 24: {'user_id': 'admin', 'role': 'admin'},
        'u' * 24: {'user_id': 'user', 'role': 'user'},
    }))
    token = tmp_path / 'metrics.token'
    token.write_text('m' * 32 + '\n', encoding='utf-8')
    monkeypatch.setenv('ZHIYING_METRICS_TOKEN_FILE', str(token))
    return token


@pytest.mark.asyncio
async def test_scrape_token_reads_metrics_but_no_business_or_management_routes(metrics_auth):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        headers = {'Authorization': 'Bearer ' + 'm' * 32}
        response = await client.get('/metrics', headers=headers)
        assert response.status_code == 200 and 'text/plain' in response.headers['content-type']
        for route in ['/monitor', '/trace/tools', '/skills', '/search']:
            assert (await client.get(route, headers=headers)).status_code == 401
        assert (await client.post('/chat', headers=headers, json={'message': 'hello'})).status_code == 401


@pytest.mark.asyncio
async def test_metrics_keeps_admin_access_and_rejects_ordinary_or_invalid_tokens(metrics_auth):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        for token, status in [('a' * 24, 200), ('u' * 24, 403), ('wrong', 401), ('', 401)]:
            response = await client.get('/metrics', headers={'Authorization': 'Bearer ' + token})
            assert response.status_code == status


@pytest.mark.asyncio
async def test_empty_scrape_file_does_not_grant_access(metrics_auth):
    metrics_auth.write_text('', encoding='utf-8')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        assert (await client.get('/metrics', headers={'Authorization': 'Bearer '})).status_code == 401


@pytest.mark.asyncio
async def test_missing_scrape_file_fails_closed_without_breaking_other_routes(metrics_auth):
    metrics_auth.unlink()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
        response = await client.get('/metrics', headers={'Authorization': 'Bearer ' + 'm' * 32})
        assert response.status_code == 503 and str(metrics_auth) not in response.text
        assert (await client.get('/skills', headers={'Authorization': 'Bearer invalid'})).status_code == 401
