"""显式依赖工厂：默认内存模拟；真实模式只有临时索引及已缓存模型。"""
import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from evaluation.errors import ExperimentInputError
from evaluation.retrieval import ReturnedChunk, canonical_hash
from tooling.knowledge_chunks import evidence_id
from tooling.retrieval_ranking import RetrievalRanking


def public_chunk(chunk):
    return ReturnedChunk(chunk_id=chunk.chunk_id, title=chunk.title, content=chunk.content)


def lexical_rank(query, rows):
    candidates = [dict(title=r.title,content=r.content,document_id=r.chunk_id,
                       chunk=0,chunk_id=r.chunk_id) for r in rows]
    ranked = RetrievalRanking._hybrid_rank(query,candidates,[])
    by_id = {r.chunk_id:r for r in rows}
    return [by_id[r['chunk_id']] for r in ranked]


class SyntheticIndex:
    """特征哈希向量+cosine，验证算法路径；不是语义 embedding。"""
    identity = 'synthetic-feature-hash-64-v1;cosine;no-semantic-model'

    def __init__(self, snapshot):
        self.rows = sorted([public_chunk(c) for c in snapshot.chunks], key=lambda r:r.chunk_id)
        self.vectors = {r.chunk_id:self.embed(r.content) for r in self.rows}
        self.verify(snapshot)

    @staticmethod
    def embed(text):
        vector = [0.0]*64
        for term in RetrievalRanking._search_terms(text):
            digest = hashlib.sha256(term.encode()).digest()
            vector[digest[0]%64] += 1 if digest[1]%2 else -1
        norm = math.sqrt(sum(x*x for x in vector)) or 1
        return [x/norm for x in vector]

    def verify(self, snapshot):
        if self.rows != sorted([public_chunk(c) for c in snapshot.chunks],key=lambda r:r.chunk_id):
            raise ExperimentInputError('整库快照不一致')

    def vector(self, query, k):
        v = self.embed(query)
        return sorted(self.rows,key=lambda r:(-sum(a*b for a,b in zip(v,self.vectors[r.chunk_id])),r.chunk_id))[:k]

    def lexical(self, query, k):
        terms = set(list(dict.fromkeys(RetrievalRanking._search_terms(query)))[:64])
        # 对齐线上 where_document contains（标题仅参与排序）。
        matching = [r for r in self.rows if any(t in r.content for t in terms)][:k]
        return lexical_rank(query,matching)


class SyntheticModel:
    identity = 'synthetic-query-and-rank-v1;no-provider'

    async def call(self, stage, query, candidates, n, parameters):
        if stage == 'rewrite':
            # 固定查询变换，既不读取数据集也不接收标准答案。
            words = query.rstrip('?').split()
            variants = [' '.join(words[:max(1,len(words)//2)]),
                        ' '.join(words[max(0,len(words)//2):])]
            value = variants[:n]
        else:
            ranked = lexical_rank(query,candidates)
            value = [candidates.index(r) for r in ranked]
        return {'value':value,'input_tokens':None,'output_tokens':None}


class ChromaIndex:
    """仅包装显式新建 collection；绝不构造默认 KnowledgeBase。"""
    def __init__(self, collection, snapshot, identity, client=None):
        self.collection, self.identity, self.client = collection, identity, client
        if collection.count():
            raise ExperimentInputError('评测 collection 必须为空')
        self.collection.add(ids=[c.chunk_id for c in snapshot.chunks],
                            documents=[c.content for c in snapshot.chunks],
                            metadatas=[dict(title=c.title,document_id=c.document_id,version=c.version,
                                            chunk_index=c.chunk_index) for c in snapshot.chunks])
        self.verify(snapshot)

    @staticmethod
    def decode(ids, documents, metadatas):
        if not (len(ids)==len(documents)==len(metadatas)):
            raise ExperimentInputError('索引返回长度不一致')
        rows=[]
        for identity, text, meta in zip(ids,documents,metadatas):
            if identity != evidence_id(meta['document_id'],meta['version'],meta['chunk_index']):
                raise ExperimentInputError('索引片段身份不一致')
            rows.append(ReturnedChunk(chunk_id=identity,title=meta['title'],content=text))
        return rows

    def verify(self,snapshot):
        all_rows=self.collection.get(include=['documents','metadatas'])
        rows=self.decode(all_rows['ids'],all_rows['documents'],all_rows['metadatas'])
        if len(rows)!=len(snapshot.chunks) or sorted(rows,key=lambda r:r.chunk_id)!=sorted(
                [public_chunk(c) for c in snapshot.chunks],key=lambda r:r.chunk_id):
            raise ExperimentInputError('整库片段 ID/正文/标题/数量不匹配')
        # 原文哈希与切片身份已由 suite 重建核对，此处再验证整库实际内容。

    def vector(self, query, k):
        if not self.collection.count():
            return []
        data=self.collection.query(query_texts=[query],n_results=min(k,self.collection.count()),
                                   include=['documents','metadatas','distances'])
        rows=self.decode(data['ids'][0],data['documents'][0],data['metadatas'][0])
        return [r for _,r in sorted(zip(data['distances'][0],rows),key=lambda pair:(pair[0],pair[1].chunk_id))]

    def lexical(self, query, k):
        terms=list(dict.fromkeys(RetrievalRanking._search_terms(query)))[:64]
        if not terms:
            return []
        clauses=[{'$contains':t} for t in terms]
        data=self.collection.get(where_document=clauses[0] if len(clauses)==1 else {'$or':clauses},
                                 include=['documents','metadatas'],limit=k)
        rows=self.decode(data['ids'],data['documents'],data['metadatas'])
        return lexical_rank(query,sorted(rows,key=lambda r:r.chunk_id))


def real_index(snapshot, cache_path: Path):
    """同步工厂必须由 to_thread 调用；明确路径，不读用户库/默认模型缓存。"""
    import chromadb
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    class LocalOnlyEmbedding(ONNXMiniLM_L6_V2):
        DOWNLOAD_PATH = cache_path.resolve()

        def _download_model_if_not_exists(self):
            # 父实现的下载入口被彻底禁止，不因缺文件静默联网。
            required = ['model.onnx','tokenizer.json','config.json','special_tokens_map.json',
                        'tokenizer_config.json','vocab.txt']
            if not all((self.DOWNLOAD_PATH/'onnx'/name).is_file() for name in required):
                raise ExperimentInputError('指定 ONNX 模型缓存不完整；不会自动下载')

    folder=cache_path.resolve()/'onnx'
    if not (folder/'model.onnx').is_file():
        raise ExperimentInputError('需要显式已缓存 ONNX 模型目录；不会自动下载')
    ef=LocalOnlyEmbedding()
    ef._download_model_if_not_exists()
    # 不信任可改名的模型标识，记录实际模型及 tokenizer 内容哈希。
    digest=hashlib.sha256()
    for path in sorted(folder.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):
                    digest.update(block)
    client=chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    collection=client.create_collection(name='eval_'+uuid4().hex,embedding_function=ef,
                                        metadata={'hnsw:space':'cosine'})
    return ChromaIndex(collection,snapshot,'onnx-MiniLM-L6-v2:'+digest.hexdigest(),client)


class ProviderModel:
    """直接 SDK；不经过会更新模块全局统计的 core.llm_utils.create_message。"""
    def __init__(self):
        from anthropic import AsyncAnthropic
        model=os.environ.get('SERVICEPILOT_EVAL_MODEL')
        key=os.environ.get('SERVICEPILOT_EVAL_API_KEY')
        if not model or not key:
            raise ExperimentInputError('真实 C/D 需要显式进程环境 SERVICEPILOT_EVAL_MODEL/API_KEY')
        self.model=model
        # base_url 不落盘，只保存不可逆身份哈希以拒绝混比不同服务。
        url=os.environ.get('SERVICEPILOT_EVAL_BASE_URL','https://api.anthropic.com')
        self.identity=model+':endpoint-'+canonical_hash(url)
        self.client=AsyncAnthropic(api_key=key,base_url=url,max_retries=0,timeout=30)

    async def call(self,stage,query,candidates,n,parameters):
        if stage=='rewrite':
            prompt=f'Rewrite this search query into at most {n} distinct search queries. Return only a JSON string array. Query: {query}'
        else:
            items=[{'index':i,'title':c.title[:160],'content':c.content[:400]} for i,c in enumerate(candidates)]
            prompt=f'Rank all these candidates for the query. Return only a JSON array of integer indices, most relevant first. Query: {query}\nCandidates: '+json.dumps(items,ensure_ascii=False)
        response=await self.client.messages.create(model=self.model,**parameters,
                                                   messages=[{'role':'user','content':prompt}])
        # 仅协议值和供应商用量离开适配器；异常/未知响应正文不写入报告。
        raw=''.join(b.text for b in response.content if getattr(b,'type',None)=='text')
        usage=getattr(response,'usage',None)
        try:
            value=json.loads(raw)
        except (ValueError,TypeError):
            value=None
        return {'value':value,'truncated':response.stop_reason=='max_tokens',
                **{field:getattr(usage,field,None) for field in (
                    'input_tokens','output_tokens','cache_read_input_tokens','cache_creation_input_tokens')}}

    async def close(self):
        await self.client.close()
