import test from 'node:test'
import assert from 'node:assert/strict'
import { ragTaskKey, runRagEvaluation } from '../src/lib/api.js'
const settings = {endpoint:'/mock', userId:'u', pythonToken:'private-token'}
function storage() { const map=new Map(); return {getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,v),removeItem:k=>map.delete(k)} }
test('RAG binding isolates credentials and never stores raw token', async () => {
 const a=await ragTaskKey(settings)
 assert.ok(!a.includes(settings.pythonToken))
 assert.notEqual(a,await ragTaskKey({...settings,pythonToken:'other'}))
 assert.notEqual(a,await ragTaskKey({...settings,endpoint:'/other'}))
})
test('refresh without task does not POST; uncertain POST never retries', async t => {
 globalThis.sessionStorage=storage(); let calls=0
 t.mock.method(globalThis,'fetch',async()=>{calls++; throw new Error('connection lost')})
 assert.equal(await runRagEvaluation(settings,{},true),null)
 assert.equal(calls,0)
 await assert.rejects(runRagEvaluation(settings),/connection lost/)
 await assert.rejects(runRagEvaluation(settings),/未知/)
 assert.equal(calls,1)
})
test('expired known job clears binding without resubmitting', async t => {
 globalThis.sessionStorage=storage()
 const key=await ragTaskKey(settings); sessionStorage.setItem(key,'known'); const urls=[]
 t.mock.method(globalThis,'fetch',async url=>{urls.push(url); return new Response('{}',{status:404})})
 await assert.rejects(runRagEvaluation(settings,{interval:0},true),/过期/)
 assert.equal(sessionStorage.getItem(key),null)
 assert.deepEqual(urls,['/mock/eval/jobs/known'])
})

test('explicit reconciliation clears only unknown binding and never posts', async t => {
 const {clearUnknownRagSubmission}=await import('../src/lib/api.js')
 globalThis.sessionStorage=storage()
 const key=await ragTaskKey(settings); sessionStorage.setItem(key,'unknown')
 let calls=0;t.mock.method(globalThis,'fetch',async()=>{calls++;return new Response('{}')})
 await clearUnknownRagSubmission(settings)
 assert.equal(sessionStorage.getItem(key),null);assert.equal(calls,0)
 sessionStorage.setItem(key,'known');await clearUnknownRagSubmission(settings)
 assert.equal(sessionStorage.getItem(key),'known')
})
