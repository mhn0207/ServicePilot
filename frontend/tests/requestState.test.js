import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import vm from 'node:vm'
import { createRequestState } from '../src/lib/requestState.js'
import { requestChat, requestSearch } from '../src/lib/api.js'

function state() { return { userId:'a', conversationId:'', endpoint:'/api/zhiying', pythonToken:'' } }

test('changing connection away and back invalidates old responses', () => {
  const settings = state()
  const requests = createRequestState(settings)
  const old = requests.begin('stats')
  settings.endpoint = '/other'; requests.invalidate()
  settings.endpoint = '/api/zhiying'; requests.invalidate()
  assert.equal(old.isCurrent(), false)
  assert.equal(requests.begin('stats').isCurrent(), true)
})

test('latest request wins within each slot and request settings are snapshots', () => {
  const settings = state()
  const requests = createRequestState(settings)
  const old = requests.begin('stats')
  const fresh = requests.begin('stats')
  requests.begin('monitor')
  assert.equal(old.isCurrent(), false)
  assert.equal(fresh.isCurrent(), true)
  settings.endpoint = '/changed'
  assert.equal(fresh.settings.endpoint, '/api/zhiying')
  assert.equal(fresh.isCurrent(), false)
})

const source = fs.readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
for (const outcome of ['success', 'failure', 'stale']) {
  test(`history restoration ${outcome} preserves conversation boundaries`, async () => {
    const settings = { ...state(), conversationId: 'c' }, requests = createRequestState(settings)
    let resolve, reject
    const context = vm.createContext({settings, requestState: requests, busy: {value:false},
      historyError:{value:''}, pendingAction:{value:''}, messages:{value:[]}, requestHistory:() => new Promise((a,b)=>{resolve=a;reject=b}),
      createMessageId:()=> 'restored', nextTick:async()=>{}, messageList:{value:null}})
    loadFunction('restoreConversation',context)
    const task = context.restoreConversation()
    assert.equal(context.busy.value,true)
    if (outcome === 'stale') { requests.invalidate(); context.messages.value=['new']; context.busy.value=true }
    if (outcome === 'failure') reject(new Error('unauthorized'))
    else resolve({messages:[{role:'assistant',content:'old'}]})
    await task
    if (outcome === 'success') {
      assert.equal(context.messages.value[0].content,'old')
      assert.equal(context.busy.value,false)
    } else if (outcome === 'failure') {
      assert.ok(context.historyError.value)
      assert.equal(context.messages.value.length,0)
    } else {
      assert.equal(context.messages.value[0],'new')
      assert.equal(context.busy.value,true)
    }
  })
}

function loadFunction(name, context) {
  const start = source.search(new RegExp(`(?:async )?function ${name}\\(`))
  const next = source.slice(start + 1).search(/\n(?:async )?function /)
  const end = next < 0 ? source.indexOf('</script>',start) : start + 1 + next
  vm.runInContext(source.slice(start,end),context)
}

test('failed restoration blocks sending into unseen context', async () => {
  const context = vm.createContext({draft:{value:'继续处理'},busy:{value:false},historyError:{value:'恢复失败'}})
  loadFunction('sendMessage',context)
  await context.sendMessage()
  assert.equal(context.draft.value,'继续处理')
})

test('a fresh conversation skips history requests', async () => {
  const settings=state(), requests=createRequestState(settings)
  const context=vm.createContext({requestState:requests,historyError:{value:'old'},requestHistory:()=>assert.fail('unexpected history read')})
  loadFunction('restoreConversation',context)
  await context.restoreConversation()
  assert.equal(context.historyError.value,'')
})

test('a knowledge mutation discards an older search result but releases its busy state', async () => {
  let finish
  const settings=state(),requests=createRequestState(settings)
  const context=vm.createContext({requestState:requests,knowledgeSearchRevision:0,searchQuery:{value:'甲'},
    busy:{value:false},searchResults:{value:[]},searched:{value:false},searchError:{value:''},
    requestSearch:()=>new Promise(r=>{finish=r}),loadStats:async()=>{},showToast:()=>{},statusText:{value:''}})
  loadFunction('searchKnowledge',context);loadFunction('knowledgeChanged',context)
  const task=context.searchKnowledge();await context.knowledgeChanged()
  finish({results:[{title:'已删除的甲'}]});await task
  assert.equal(context.searchResults.value.length,0)
  assert.equal(context.searched.value,false)
  assert.equal(context.busy.value,false)
})

for (const [name, requestName, field, result] of [
  ['loadStats','requestKnowledgeStats','knowledgeCount',{total_chunks: 99}],
  ['loadMonitor','requestMonitor','monitorData',{agent_stats:{old:true}}],
  ['loadSkills','requestSkills','skillsData',{skills:['old']}],
]) {
  for (const fail of [false,true]) {
    test(`${name}: stale ${fail ? 'error' : 'success'} cannot overwrite new connection state`, async () => {
      const settings = state()
      const requests = createRequestState(settings)
      let resolve, reject
      const pending = new Promise((a,b)=>{resolve=a;reject=b})
      const context = vm.createContext({settings,requestState:requests,[field]:{value:'new'},[requestName]:()=>pending})
      loadFunction(name,context)
      const task = context[name]()
      settings.endpoint='/other'; requests.invalidate()
      if (fail) reject(new Error('old')); else resolve(result)
      await task
      assert.equal(context[field].value,'new')
    })
  }
}

test('stale eval response does not clear a new operation busy state', async () => {
  const settings=state(), requests=createRequestState(settings)
  let resolve
  const context=vm.createContext({settings, requestState:requests, busy:{value:false},evalData:{value:'new'},
    statusText:{value:''}, showToast:()=>assert.fail('stale toast'), requestEvaluation:()=>new Promise(a=>{resolve=a})})
  loadFunction('runEvaluation',context)
  const task=context.runEvaluation()
  settings.endpoint='/other'; requests.invalidate(); context.busy.value=true
  resolve({avg_scores:{accuracy:0.5}})
  await task
  assert.equal(context.evalData.value,'new')
  assert.equal(context.busy.value,true)
})

for (const outcome of ['pending', 'failed']) {
  test(`a new conversation does not replay a ${outcome} first request with the same text`, async t => {
    const settings = state(), requests = createRequestState(settings), calls = []
    let release
    t.mock.method(globalThis, 'fetch', async (url, options) => {
      calls.push(JSON.parse(options.body))
      if (calls.length === 1) {
        if (outcome === 'failed') return new Response('{}', { status: 503 })
        await new Promise(resolve => { release = resolve })
      }
      return new Response(JSON.stringify({ conv_id: calls.at(-1).conv_id || 'old-server-conversation', response: 'ok' }))
    })
    const old = requestChat({ ...settings }, '相同的首句话')
    if (outcome === 'failed') await assert.rejects(old, /503/)
    const context = vm.createContext({ settings, crypto, requestState: requests, busy: { value: true },
      historyError: { value: 'old' }, messages: { value: ['old'] }, lastResponse: { value: 'old' }, lastTrace: { value: 'old' }, persist() {} })
    loadFunction('clearConversation', context)
    context.clearConversation()
    assert.ok(settings.conversationId)
    await requestChat({ ...settings }, '相同的首句话')
    assert.notEqual(calls[0].client_request_id, calls[1].client_request_id)
    assert.notEqual(calls[0].conv_id, calls[1].conv_id)
    assert.equal(context.messages.value.length, 0)
    if (release) { release(); await old }
  })
}

test('a failed search renders an error rather than a successful empty search', async t => {
  const settings = state(), requests = createRequestState(settings), toasts = []
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ success: false, results: [] })))
  const context = vm.createContext({ settings, requestState: requests, requestSearch, knowledgeSearchRevision: 0,
    searchQuery: { value: '查询' }, busy: { value: false }, searchResults: { value: ['old'] },
    searched: { value: true }, searchError: { value: '' }, statusText: { value: '' }, showToast: value => toasts.push(value) })
  loadFunction('searchKnowledge', context)
  await context.searchKnowledge()
  assert.equal(context.searched.value, false)
  assert.equal(context.searchResults.value.length, 0)
  assert.match(context.searchError.value, /失败/)
  assert.ok(toasts.every(value => !value.includes('检索完成')))
})
