import test from 'node:test'
import assert from 'node:assert/strict'
import { createDocumentState, createKnowledgeDocuments } from '../src/lib/knowledgeDocuments.js'

const document = {document_id:'a'.repeat(64),version:'b'.repeat(64),title:'甲',deletable:true,chunk_count:1}
function setup(overrides={}) {
  const settings={endpoint:'/api/zhiying',userId:'admin',pythonToken:'test'}
  const state=createDocumentState(), calls=[]
  const service={list:async()=>({documents:[document],unmanaged_chunks:[]}),detail:async()=>({...document,chunks:[{content:'正文'}]}),
    remove:async(...args)=>{calls.push(args);return {deleted_chunks:1}},...overrides}
  let changed=0
  return {settings,state,calls,controller:createKnowledgeDocuments(settings,state,()=>{changed++},service),changed:()=>changed}
}

test('list and detail show actual documents; cancelling deletion never requests', async()=>{
  const x=setup(); await x.controller.load(); assert.equal(x.state.access,'allowed')
  await x.controller.select(document); assert.equal(x.state.detail.chunks[0].content,'正文')
  x.controller.askDelete(x.state.detail); x.controller.cancelDelete(); await x.controller.remove()
  assert.equal(x.calls.length,0)
})
test('valid empty results differ from execution errors and denied access', async()=>{
  const empty=setup({list:async()=>({documents:[],unmanaged_chunks:[]})}); await empty.controller.load()
  assert.equal(empty.state.access,'allowed'); assert.equal(empty.state.listError,'')
  for(const status of [401,403,503]){
    const x=setup({list:async()=>{throw Object.assign(new Error('failed'),{status})}})
    await x.controller.load(); assert.ok(x.state.listError)
    assert.equal(x.state.documents.length,0)
    assert.equal(x.state.access,status===503?'unknown':'denied')
  }
})
test('double confirm sends one delete using the previewed version', async()=>{
  let done; const x=setup({remove:async(...args)=>{x.calls.push(args);await new Promise(r=>{done=r});return {deleted_chunks:1}}})
  await x.controller.load(); x.controller.askDelete(document)
  const task=x.controller.remove(); await x.controller.remove()
  assert.equal(x.calls.length,1); assert.equal(x.calls[0][2],document.version)
  done(); await task; assert.equal(x.changed(),1); assert.equal(x.state.deleting,false)
})
test('version conflict clears stale detail, refreshes list and never retries delete', async()=>{
  let deletes=0,lists=0
  const x=setup({list:async()=>{lists++;return {documents:[document],unmanaged_chunks:[]}},remove:async()=>{deletes++;throw Object.assign(new Error('changed'),{status:409})}})
  await x.controller.load(); await x.controller.select(document); x.controller.askDelete(document)
  await x.controller.remove()
  assert.equal(deletes,1); assert.equal(lists,2); assert.equal(x.state.detail,null); assert.ok(x.state.mutationError.includes('变化'))
})
test('old connection list cannot overwrite a new identity', async()=>{
  let finish; const x=setup({list:()=>new Promise(r=>{finish=r})})
  const old=x.controller.load(); x.settings.userId='other'; x.controller.invalidate()
  x.state.documents=['new']; x.state.loading=true
  finish({documents:[document],unmanaged_chunks:[]}); await old
  assert.equal(x.state.documents[0],'new'); assert.equal(x.state.loading,true)
})
test('legacy and incomplete documents cannot open delete confirmation', async()=>{
  const x=setup();await x.controller.load();x.controller.askDelete({...document,deletable:false})
  assert.equal(x.state.confirmation,null);await x.controller.remove();assert.equal(x.calls.length,0)
})
test('old detail cannot replace a newly selected document', async()=>{
  let finish;const x=setup({detail:async(settings,id)=>id===document.document_id?new Promise(r=>{finish=r}):{title:'乙',chunks:[]}})
  await x.controller.load();const old=x.controller.select(document)
  await x.controller.select({...document,document_id:'c'.repeat(64)})
  finish({...document,chunks:[]});await old;assert.equal(x.state.detail.title,'乙')
})
test('unknown delete outcome refreshes once without automatic retry', async()=>{
  let deletes=0;const x=setup({remove:async()=>{deletes++;throw new Error('network')}})
  await x.controller.load();x.controller.askDelete(document);await x.controller.remove()
  assert.equal(deletes,1);assert.equal(x.state.confirmation,null);assert.ok(x.state.mutationError.includes('尚未确认'))
})
