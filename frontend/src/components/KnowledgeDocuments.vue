<template>
  <section class="workspace-card knowledge-documents" aria-labelledby="knowledge-documents-title">
    <div class="documents-heading"><div><h2 id="knowledge-documents-title">文档管理</h2><p>查看当前文档及片段，删除不再使用的资料。</p></div>
      <button :disabled="state.loading || state.deleting" @click="controller.load">刷新文档</button></div>
    <p v-if="state.loading" role="status">正在读取文档…</p>
    <p v-if="state.listError" role="alert">{{ state.listError }}</p>
    <p v-if="state.notice" role="status">{{ state.notice }}</p>
    <p v-if="state.mutationError" role="alert">{{ state.mutationError }}</p>
    <template v-if="state.access === 'allowed'">
      <p v-if="!state.loading && !state.documents.length && !state.unmanaged.length">知识库暂无文档，请先添加或上传资料。</p>
      <ul class="document-list">
        <li v-for="doc in state.documents" :key="doc.document_id">
          <button class="document-title" :disabled="state.loading || state.deleting" @click="controller.select(doc)">{{ doc.title }}</button>
          <span>{{ doc.chunk_count }} 个片段</span><small>版本 {{ doc.version ? doc.version.slice(0, 12) : '待核验' }}</small>
          <span v-if="!doc.deletable">仅可查看</span>
        </li>
      </ul>
      <details v-if="state.unmanaged.length"><summary>{{ state.unmanaged.length }} 个旧片段待核验（仅展示，不支持删除）</summary>
        <ul><li v-for="item in state.unmanaged" :key="item.storage_id">{{ item.title }} · {{ item.storage_id }}</li></ul>
      </details>
      <p v-if="state.detailLoading" role="status">正在读取片段…</p><p v-if="state.detailError" role="alert">{{ state.detailError }}</p>
      <section v-if="state.detail" class="document-detail" aria-label="文档片段">
        <h3>{{ state.detail.title }}</h3><p>文档 ID：{{ state.detail.document_id }}</p><p>内容版本：{{ state.detail.version || '待核验' }}</p>
        <p v-if="!state.detail.deletable">{{ state.detail.restriction }}</p>
        <article v-for="(chunk, index) in state.detail.chunks" :key="index"><h4>片段 {{ chunk.chunk_index == null ? '待核验' : chunk.chunk_index + 1 }}</h4>
          <pre>{{ chunk.content }}</pre><small>证据 ID：{{ chunk.chunk_id || '待核验' }}</small></article>
        <button v-if="state.detail.deletable" class="delete-document" :disabled="state.deleting || state.loading" @click="controller.askDelete(state.detail)">删除这篇文档</button>
      </section>
      <div v-if="state.confirmation" class="delete-confirmation" role="dialog" aria-label="确认删除文档">
        <p>确认删除“{{ state.confirmation.title }}”及其全部片段？此操作不可恢复。</p>
        <button :disabled="state.deleting" @click="controller.cancelDelete">取消</button>
        <button class="delete-document" :disabled="state.deleting" @click="controller.remove">{{ state.deleting ? '删除中…' : '确认删除' }}</button>
      </div>
    </template>
  </section>
</template>

<script setup>
import { reactive, watch, onMounted, onBeforeUnmount } from 'vue'
import { createDocumentState, createKnowledgeDocuments } from '../lib/knowledgeDocuments.js'
const props=defineProps({settings:{type:Object,required:true},revision:{type:Number,default:0}})
const emit=defineEmits(['changed','access'])
const state=reactive(createDocumentState())
watch(()=>state.access,value=>emit('access',value),{immediate:true})
const controller=createKnowledgeDocuments(props.settings,state,()=>emit('changed'))
watch(()=>[props.settings.endpoint,props.settings.userId,props.settings.pythonToken,props.revision],()=>{controller.invalidate();controller.load()})
onMounted(controller.load)
onBeforeUnmount(controller.dispose)
</script>

<style scoped>
.knowledge-documents{margin-top:24px;padding:24px;min-width:0}.documents-heading{display:flex;justify-content:space-between;gap:16px;align-items:center}.documents-heading p{color:#64748b;margin:8px 0}.knowledge-documents button{border:1px solid #dce2e8;border-radius:8px;padding:8px 12px;background:#fff;color:#253041;cursor:pointer}.knowledge-documents button:disabled{opacity:.5;cursor:wait}.document-list{list-style:none;padding:0}.document-list li{display:flex;align-items:center;flex-wrap:wrap;gap:12px;padding:12px 0;border-bottom:1px solid #e5e7eb}.document-title{text-align:left;font-weight:600}.document-list small{color:#64748b}.document-detail{margin-top:20px;padding:16px;border:1px solid #e5e7eb;border-radius:12px;overflow-wrap:anywhere}.document-detail article{padding:12px 0;border-top:1px solid #eee}.document-detail pre{white-space:pre-wrap;font-family:inherit;line-height:1.7}.document-detail small{color:#64748b;overflow-wrap:anywhere}.knowledge-documents .delete-document{color:#b42318;border-color:#f3b6b1}.delete-confirmation{margin-top:16px;padding:16px;border:1px solid #f3b6b1;border-radius:12px;background:#fff7f6}.delete-confirmation button{margin-right:12px}[role=alert]{color:#b42318}
</style>
