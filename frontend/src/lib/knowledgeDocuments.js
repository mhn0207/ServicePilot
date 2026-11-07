import { createRequestState } from './requestState.js'
import { requestKnowledgeDocuments, requestKnowledgeDocument, deleteKnowledgeDocument } from './api.js'

export function createDocumentState() {
  return {documents:[],unmanaged:[],detail:null,confirmation:null,access:'unknown',
    loading:false,detailLoading:false,deleting:false,listError:'',detailError:'',mutationError:'',notice:''}
}

export function createKnowledgeDocuments(settings, state, onChanged=()=>{}, service={
  list:requestKnowledgeDocuments,detail:requestKnowledgeDocument,remove:deleteKnowledgeDocument
}) {
  const requests=createRequestState(settings)
  const errorText=error=>error.status===403?'权限不足：文档管理需要管理员权限。':
    error.status===401?'访问令牌无效，请在连接配置中填写有效令牌。':'文档读取失败，请检查连接后重试。'
  async function load() {
    if(state.deleting)return
    requests.begin('detail')
    const request=requests.begin('list')
    state.loading=true;state.listError='';state.detail=null;state.confirmation=null;state.detailLoading=false
    state.documents=[];state.unmanaged=[];state.access='unknown'
    try {
      const data=await service.list(request.settings)
      if(!request.isCurrent())return
      state.documents=data.documents;state.unmanaged=data.unmanaged_chunks;state.access='allowed'
    } catch(error) {
      if(!request.isCurrent())return
      state.listError=errorText(error);state.access=[401,403].includes(error.status)?'denied':'unknown'
    } finally {if(request.isCurrent())state.loading=false}
  }
  async function select(doc) {
    if(state.deleting||state.loading||state.access!=='allowed')return
    const request=requests.begin('detail')
    state.detailLoading=true;state.detailError='';state.detail=null;state.confirmation=null
    try {
      const data=await service.detail(request.settings,doc.document_id)
      if(request.isCurrent())state.detail=data
    } catch(error) {if(request.isCurrent())state.detailError=error.status===404?'文档已不存在，请刷新列表。':errorText(error)}
    finally {if(request.isCurrent())state.detailLoading=false}
  }
  function askDelete(doc) {
    if(!doc?.deletable||state.access!=='allowed'||state.loading||state.deleting)return
    state.confirmation={document_id:doc.document_id,version:doc.version,title:doc.title}
    state.mutationError='';state.notice=''
  }
  function cancelDelete(){if(!state.deleting)state.confirmation=null}
  async function remove() {
    if(state.deleting||!state.confirmation)return
    const doc={...state.confirmation},request=requests.begin('delete')
    state.deleting=true;state.mutationError='';state.notice=''
    try {
      const result=await service.remove(request.settings,doc.document_id,doc.version)
      if(!request.isCurrent())return
      state.notice=`已删除“${doc.title}”的 ${result.deleted_chunks} 个片段。`
    } catch(error) {
      if(!request.isCurrent())return
      state.mutationError=error.status===409?'文档已变化或片段信息不完整，已刷新列表，请重新查看。':
        [401,403].includes(error.status)?errorText(error):'删除结果尚未确认，请查看刷新后的列表；不要连续重试。'
    } finally {
      if(request.isCurrent()){
        state.deleting=false;state.confirmation=null;state.detail=null
        onChanged()
        await load()
      }
    }
  }
  function invalidate(){requests.invalidate();Object.assign(state,createDocumentState())}
  return {load,select,askDelete,cancelDelete,remove,invalidate,dispose:()=>requests.invalidate()}
}
