import test from 'node:test'
import assert from 'node:assert/strict'
import {
  createInitialSettings, saveSettings, requestChat, requestHealth,
  requestSearch, addKnowledge, uploadKnowledge, requestToolTrace, runEvaluation
} from '../src/lib/api.js'

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
      trace: { tools_used: ['search_knowledge_base'], tool_calls: [{ tool_name: 'search_knowledge_base' }] } } : { status: 'ok' })
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
