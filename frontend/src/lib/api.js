import { pollEvaluation } from './evaluationJobs.js'
const DEFAULT_API_URL = runtimeConfig().zhiyingApiUrl || import.meta.env?.VITE_PYTHON_API_URL || '/api/zhiying'

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
  return requestFor(settings, `/search?${params}`, { method: 'POST' })
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

export async function requestChat(settings, message) {
  const payload = buildChatPayload(settings, message)
  const attemptKey = JSON.stringify([apiBaseUrl(settings), settings.userId, settings.conversationId, message])
  if (!chatAttempts.has(attemptKey)) {
    if (chatAttempts.size > 100) chatAttempts.clear()
    chatAttempts.set(attemptKey, crypto.randomUUID())
  }
  payload.client_request_id = chatAttempts.get(attemptKey)
  const raw = await requestFor(settings, '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  chatAttempts.delete(attemptKey)
  return normalizeChatResponse(raw)
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
async function requestFor(settings, path, options = {}) {
  const headers = new Headers(options.headers || {})
  if (settings.pythonToken) headers.set('Authorization', `Bearer ${settings.pythonToken}`)
  return requestJson(apiBaseUrl(settings), path, { ...options, headers,
    signal: options.signal || AbortSignal.timeout(path === '/chat' ? 55000 : 20000) })
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, options)
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
  return window.__ZHIYING_CONFIG__ || {}
}
