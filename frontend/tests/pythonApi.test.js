import test from 'node:test'
import assert from 'node:assert/strict'
import {
  createInitialSettings, saveSettings, requestChat, requestHealth, requestHistory,
  requestSearch, addKnowledge, uploadKnowledge, requestToolTrace, runEvaluation,
  requestKnowledgeDocuments, requestKnowledgeDocument, deleteKnowledgeDocument
} from '../src/lib/api.js'

test('knowledge administration sends the current token and previewed version', async t => {
  const calls=[]
  replaceGlobal(t,'fetch',async(url,options)=>{calls.push({url,options});return json({documents:[],deleted_chunks:1})})
  const config={endpoint:'/api/zhiying',userId:'admin',pythonToken:'admin-token'}
  await requestKnowledgeDocuments(config)
  await requestKnowledgeDocument(config,'a'.repeat(64))
  await deleteKnowledgeDocument(config,'a'.repeat(64),'b'.repeat(64))
  assert.equal(calls[0].url,'/api/zhiying/knowledge/documents')
  assert.equal(calls[2].options.method,'DELETE')
  assert.equal(new URL(calls[2].url,'http://local').searchParams.get('expected_version'),'b'.repeat(64))
  assert.ok(calls.every(c=>new Headers(c.options.headers).get('authorization')==='Bearer admin-token'))
})

test('history uses current identity, encoded conversation and access token', t => {
  replaceGlobal(t, 'fetch', async (url, options) => {
    const parsed = new URL(url, 'http://local')
    assert.equal(parsed.searchParams.get('conv_id'), 'a/b & c')
    assert.equal(parsed.searchParams.get('user_id'), 'alice')
    assert.equal(new Headers(options.headers).get('authorization'), 'Bearer test-token')
    return json({ messages: [] })
  })
  return requestHistory({endpoint:'/api/zhiying',userId:'alice',conversationId:'a/b & c',pythonToken:'test-token'})
})

function storage() {
  const data = new Map()
  return {
    getItem: key => data.get(key) ?? null,
    setItem: (key, value) => data.set(key, String(value)),
    removeItem: key => data.delete(key)
  }
}

function replaceGlobal(t, name, value) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name)
  Object.defineProperty(globalThis, name, { configurable: true, value })
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, name, previous)
    else delete globalThis[name]
  })
}

const settings = { endpoint: '/api/zhiying/', userId: 'u-test', conversationId: 'c-test', pythonToken: 'access-token' }
const json = (data, status = 200) => new Response(JSON.stringify(data), { status })

test('migrates the Python endpoint without reusing a legacy non-Python conversation or persisting tokens', t => {
  const local = storage()
  replaceGlobal(t, 'localStorage', local)
  local.setItem('zhiying.frontend.settings', JSON.stringify({
    backend: 'legacy', endpoints: { python: '/python-service' },
    userId: 'alice', conversationId: 'legacy-conversation', pythonToken: 'old-token'
  }))
  const migrated = createInitialSettings()
  assert.deepEqual(migrated, { endpoint: '/python-service', userId: 'alice', conversationId: '', pythonToken: '' })
  saveSettings({ ...migrated, pythonToken: 'secret' })
  assert.deepEqual(JSON.parse(local.getItem('zhiying.frontend.settings')), {
    endpoint: '/python-service', userId: 'alice', conversationId: ''
  })
  local.setItem('zhiying.frontend.settings', 'null')
  assert.equal(createInitialSettings().endpoint, '/api/zhiying')
})

test('Python chat retry reuses its request ID, maps snake_case fields and sends access token', async t => {
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options, body: JSON.parse(options.body) })
    if (calls.length === 1) return json({ detail: 'temporarily unavailable' }, 503)
    return json({ conv_id: 'c-python', request_id: 'r-python', response: '回复', intent: 'refund',
      agent_type: 'billing', primary_agent: 'billing', knowledge_used: true,
      latency_ms: 123, status: 'partial', retryable: false })
  })
  await assert.rejects(requestChat(settings, '退款规则'), /503/)
  const reply = await requestChat(settings, '退款规则')
  assert.equal(calls[0].url, '/api/zhiying/chat')
  assert.equal(calls[0].options.headers.get('Authorization'), 'Bearer access-token')
  assert.equal(calls[0].body.conv_id, 'c-test')
  assert.equal(calls[0].body.user_id, 'u-test')
  assert.ok(calls[0].body.client_request_id)
  assert.equal(calls[0].body.client_request_id, calls[1].body.client_request_id)
  assert.equal(reply.conversationId, 'c-python')
  assert.equal(reply.requestId, 'r-python')
  assert.equal(reply.knowledgeUsed, true)
  assert.equal(reply.status, 'partial')
  await requestChat(settings, '退款规则')
  assert.notEqual(calls[1].body.client_request_id, calls[2].body.client_request_id)
})

test('knowledge, upload and trace requests follow the Python API contract', async t => {
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options })
    return json(url.includes('/trace/') ? { request_id: 'r/1', found: true,
      trace: { tools_used: ['search_knowledge_base'], tool_calls: [{ tool_name: 'search_knowledge_base' }] } } : { status: 'ok', success: true, results: [] })
  })
  await requestHealth(settings)
  await requestSearch(settings, '退款 规则', 3)
  const documents = [{ title: '规则', content: '七天内申请' }]
  await addKnowledge(settings, documents)
  await uploadKnowledge(settings, new File(['测试知识'], 'test.md', { type: 'text/markdown' }))
  const trace = await requestToolTrace(settings, 'r/1')
  assert.equal(calls[0].url, '/api/zhiying/health')
  assert.equal(new URL(calls[1].url, 'http://test').searchParams.get('query'), '退款 规则')
  assert.equal(calls[1].options.method, 'POST')
  assert.deepEqual(JSON.parse(calls[2].options.body), { documents })
  assert.equal(calls[3].options.body.get('file').name, 'test.md')
  assert.equal(calls[3].options.headers.has('Content-Type'), false)
  assert.equal(calls[4].url, '/api/zhiying/trace/tool/r%2F1')
  assert.deepEqual(trace.trace.toolsUsed, ['search_knowledge_base'])
  assert.equal(trace.trace.toolCalls.length, 1)
})

test('resumes a Python evaluation job without submitting again and clears the completed job', async t => {
  const session = storage()
  replaceGlobal(t, 'sessionStorage', session)
  const calls = []
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options })
    return json(url.endsWith('/eval/run') ? { job_id: 'python-job', status: 'queued' }
      : { job_id: 'python-job', status: 'completed', result: { total: 3 } })
  })
  assert.equal(await runEvaluation(settings, null, { isCurrent: () => false }), null)
  assert.equal(session.getItem('zhiying.eval:python:/api/zhiying:u-test'), 'python-job')
  assert.deepEqual(await runEvaluation(settings, null, { interval: 0 }), { total: 3 })
  assert.deepEqual(calls.map(call => call.url), ['/api/zhiying/eval/run', '/api/zhiying/eval/jobs/python-job'])
  assert.equal(session.getItem('zhiying.eval:python:/api/zhiying:u-test'), null)
})

test('search execution failure is rejected even when HTTP succeeds', async t => {
  t.mock.method(globalThis, 'fetch', async () => json({ success: false, results: [], cached: false, reranked: false }))
  await assert.rejects(requestSearch(settings, 'failed search'), /检索.*失败/)
})

test('a successful search with no matching documents stays a valid empty result', async t => {
  t.mock.method(globalThis, 'fetch', async () => json({ success: true, results: [], cached: false, reranked: false }))
  const result = await requestSearch(settings, 'empty search')
  assert.equal(result.success, true)
  assert.deepEqual(result.results, [])
})
