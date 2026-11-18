import { pollEvaluation } from './evaluationJobs.js'
const DEFAULT_API_URL = runtimeConfig().servicepilotApiUrl || runtimeConfig().zhiyingApiUrl || import.meta.env?.VITE_PYTHON_API_URL || '/api/zhiying'

export function createInitialSettings() {
  const saved = readSettings()
  return {
    pythonToken: '',
    endpoint: saved.endpoint || saved.endpoints?.python || DEFAULT_API_URL,
    userId: saved.userId || 'u1001',
    conversationId: saved.backend && saved.backend !== 'python' ? '' : saved.conversationId || ''
  }
}

export function saveSettings(settings) {
  const { endpoint, userId, conversationId } = settings
  localStorage.setItem('zhiying.frontend.settings', JSON.stringify({ endpoint, userId, conversationId }))
}

export function apiBaseUrl(settings) {
  return normalizeBaseUrl(settings.endpoint || DEFAULT_API_URL)
}

export async function requestHealth(settings) {
  return requestFor(settings, '/health')
}

export async function requestHistory(settings) {
  const query = new URLSearchParams({ conv_id: settings.conversationId, user_id: settings.userId })
  return requestFor(settings, `/conversation/history?${query}`)
}

export async function requestMonitor(settings) {
  return requestFor(settings, '/monitor')
}

export async function requestSkills(settings) {
  return requestFor(settings, '/skills')
}

export async function reloadSkills(settings) {
  return requestFor(settings, '/skills/reload', { method: 'POST' })
}

export async function requestKnowledgeStats(settings) {
  return requestFor(settings, '/knowledge/stats')
}

export async function requestKnowledgeDocuments(settings) {
  return requestFor(settings, '/knowledge/documents')
}

export async function requestKnowledgeDocument(settings, documentId) {
  return requestFor(settings, `/knowledge/documents/${encodeURIComponent(documentId)}`)
}

export async function deleteKnowledgeDocument(settings, documentId, version) {
  return requestFor(settings, `/knowledge/documents/${encodeURIComponent(documentId)}?${new URLSearchParams({expected_version:version})}`, {method:'DELETE'})
}

export async function runEvaluation(settings, body = null, options = {}) {
  const taskKey = `zhiying.eval:python:${apiBaseUrl(settings)}:${settings.userId}`
  const savedJob = sessionStorage.getItem(taskKey)
  const initial = savedJob ? {job_id: savedJob, status: 'queued', completed: 0, total: 0} : await requestFor(settings, '/eval/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  })
  if (!initial.job_id) throw new Error('评测接口未返回任务 ID')
  sessionStorage.setItem(taskKey, initial.job_id)
  try {
    const result = await pollEvaluation(initial, jobId => requestFor(settings,
      `/eval/jobs/${encodeURIComponent(jobId)}`), options)
    if (result) sessionStorage.removeItem(taskKey)
    return result
  } catch (error) {
    if (error.terminal || error.status === 404) sessionStorage.removeItem(taskKey)
    throw error
  }
}

export async function requestSearch(settings, query, topK = 5) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  const result = await requestFor(settings, `/search?${params}`, { method: 'POST' })
  if (result?.success !== true) throw new Error('知识检索执行失败，请稍后重试')
  return result
}

export async function ragTaskKey(settings) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(settings.pythonToken || ''))
  const identity = Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('')
  return `zhiying.eval:rag:agent-dev-v1:${apiBaseUrl(settings)}:${settings.userId}:${identity}`
}

export async function requestRagRuns(settings, runId = '') {
  return requestFor(settings, `/eval/rag/runs${runId ? '/' + encodeURIComponent(runId) : ''}`)
}

export async function clearUnknownRagSubmission(settings) {
  const key = await ragTaskKey(settings)
  if (sessionStorage.getItem(key) === 'unknown') sessionStorage.removeItem(key)
}

export async function runRagEvaluation(settings, options = {}, resumeOnly = false) {
  const key = await ragTaskKey(settings)
  let saved = sessionStorage.getItem(key)
  if (saved === 'unknown') throw new Error('首次提交结果未知，请先刷新历史报告核对；不会自动重新提交')
  if (!saved && resumeOnly) return null
  let initial
  if (saved) initial = {job_id: saved, status: 'queued', completed: 0, total: 0}
  else {
    sessionStorage.setItem(key, 'unknown')
    try {
      initial = await requestFor(settings, '/eval/rag/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({suite:'agent-dev-v1'}) })
    } catch (error) {
      if (error.status && error.status < 500) sessionStorage.removeItem(key)
      else error.message = `首次提交结果未知，请核对历史报告：${error.message}`
      throw error
    }
    if (!initial?.job_id) throw new Error('提交结果未知：接口未返回任务 ID，请核对历史报告')
    sessionStorage.setItem(key, initial.job_id)
  }
  try {
    const result = await pollEvaluation(initial, id => requestFor(settings, `/eval/jobs/${encodeURIComponent(id)}`), options)
    if (result) sessionStorage.removeItem(key)
    return result
  } catch (error) {
    if (error.terminal || error.status === 404) sessionStorage.removeItem(key)
    if (error.status === 404) error.message = '任务已过期（可能服务重启）；请查看历史报告，运行中任务不能恢复'
    throw error
  }
}

export async function requestChat(settings, message, { onStage, signal } = {}) {
  const payload = buildChatPayload(settings, message)
  const scope = JSON.stringify([apiBaseUrl(settings), settings.userId, settings.pythonToken || '',
    settings.conversationId || '', message])
  const digest = crypto.subtle && await crypto.subtle.digest('SHA-256', new TextEncoder().encode(scope))
  const storageKey = digest && 'servicepilot.chat:' + Array.from(new Uint8Array(digest),
    byte => byte.toString(16).padStart(2, '0')).join('')
  const attemptKey = storageKey || scope
  let attempt = chatAttempts.get(attemptKey)
  if (!attempt && storageKey) {
    try {
      attempt = JSON.parse(globalThis.sessionStorage?.getItem(storageKey) || 'null')
    } catch { /* Storage may be unavailable in private environments. */ }
  }
  if (typeof attempt?.id !== 'string' || !(attempt.expiresAt > Date.now())) {
    attempt = { id: crypto.randomUUID(), expiresAt: Date.now() + 24 * 60 * 60 * 1000 }
  }
  chatAttempts.set(attemptKey, attempt)
  if (storageKey) {
    try {
      globalThis.sessionStorage?.setItem(storageKey, JSON.stringify(attempt))
    } catch { /* Keep the in-memory attempt when storage is unavailable. */ }
  }
  payload.client_request_id = attempt.id
  const timeout = AbortSignal.timeout(55000)
  const options = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout
  }
  const raw = onStage
    ? await requestStreamingChat(settings, options, onStage)
    : await requestFor(settings, '/chat', options)
  if (raw?.status !== 'failed' && chatAttempts.get(attemptKey) === attempt) {
    chatAttempts.delete(attemptKey)
    if (storageKey) {
      try {
        globalThis.sessionStorage?.removeItem(storageKey)
      } catch { /* The in-memory attempt was already cleared. */ }
    }
  }
  return normalizeChatResponse(raw)
}

async function requestStreamingChat(settings, options, onStage) {
  const response = await fetch(`${apiBaseUrl(settings)}/chat/stream`, {
    ...options,
    headers: authHeaders(settings, { ...options.headers, Accept: 'text/event-stream' })
  })
  // An older backend can reject this route before starting any turn.
  if (response.status === 404 || response.status === 405) {
    await response.body?.cancel()
    return requestFor(settings, '/chat', options)
  }
  if (!response.ok) return readJsonResponse(response)
  if (!response.body || !response.headers.get('content-type')?.startsWith('text/event-stream')) {
    throw new Error('服务没有返回聊天事件流，请重试')
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let pending = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) throw new Error('连接在最终结果返回前中断，请重试')
      pending += value
      const frames = pending.split(/\r?\n\r?\n/)
      pending = frames.pop()
      for (const frame of frames) {
        const lines = frame.split(/\r?\n/)
        const event = lines.find(line => line.startsWith('event:'))?.slice(6).trim()
        if (!['stage', 'final', 'error'].includes(event)) continue
        const data = JSON.parse(lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n'))
        if (event === 'stage') onStage(data)
        else if (event === 'final') {
          if (!data?.conv_id || !data.request_id || typeof data.response !== 'string') {
            throw new Error('聊天最终结果不完整，请重试')
          }
          return data
        } else {
          const error = new Error(data.detail || '请求处理失败')
          error.status = data.status
          error.retryable = data.retryable
          throw error
        }
      }
    }
  } finally {
    await reader.cancel().catch(() => {})
  }
}

export async function requestToolTrace(settings, requestId) {
  if (!requestId) return null
  const raw = await requestFor(settings, `/trace/tool/${encodeURIComponent(requestId)}`)
  return normalizeToolTraceResponse(raw)
}

export async function addKnowledge(settings, documents) {
  return requestFor(settings, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestFor(settings, '/knowledge/upload', {
    method: 'POST',
    body: form
  })
}

function buildChatPayload(settings, message) {
  return {
    message,
    user_id: settings.userId || 'anonymous',
    conv_id: settings.conversationId || undefined
  }
}

function normalizeChatResponse(raw) {
  return {
    conversationId: raw.conv_id || '',
    requestId: raw.request_id || '',
    response: raw.response || '',
    intent: raw.intent || 'other',
    intentGroup: raw.intent_group || 'other',
    agentType: raw.agent_type || '',
    agentTypes: raw.agent_types || [],
    primaryAgent: raw.primary_agent || '',
    supportingAgents: raw.supporting_agents || [],
    routingReason: raw.routing_reason || '',
    routingConfidence: Number(raw.routing_confidence ?? 0),
    entities: raw.entities || {},
    intentConfidence: Number(raw.intent_confidence ?? 0),
    intentSourceScores: raw.intent_source_scores || {},
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used),
    status: raw.status,
    retryable: raw.retryable,
    errorCode: raw.error_code,
    raw
  }
}

function normalizeToolTraceResponse(raw) {
  const trace = raw?.trace || {}
  return {
    requestId: raw?.request_id || '',
    found: Boolean(raw?.found),
    trace: {
      ...trace,
      toolsUsed: trace.tools_used || [],
      toolCalls: trace.tool_calls || []
    },
    raw
  }
}

const chatAttempts = new Map()
function authHeaders(settings, initial) {
  const headers = new Headers(initial)
  if (settings.pythonToken) headers.set('Authorization', `Bearer ${settings.pythonToken}`)
  return headers
}

async function requestFor(settings, path, options = {}) {
  const response = await fetch(`${apiBaseUrl(settings)}${path}`, {
    ...options,
    headers: authHeaders(settings, options.headers),
    signal: options.signal || AbortSignal.timeout(20000)
  })
  return readJsonResponse(response)
}

async function readJsonResponse(response) {
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    const error = new Error(`${response.status} ${response.statusText}: ${detail}`)
    error.status = response.status
    error.retryable = data?.retryable
    throw error
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function readSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem('zhiying.frontend.settings') || '{}')
    return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {}
  } catch {
    return {}
  }
}

function runtimeConfig() {
  if (typeof window === 'undefined') return {}
  return window.__SERVICEPILOT_CONFIG__ || window.__ZHIYING_CONFIG__ || {}
}
