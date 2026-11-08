import sys
import types
import threading

try:
    import chromadb
except ModuleNotFoundError:
    sys.modules['chromadb'] = types.ModuleType('chromadb')

from tooling.knowledge_base import KnowledgeBase


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)
        for i, doc, meta in zip(kwargs['ids'], kwargs['documents'], kwargs['metadatas']):
            self.rows[i] = (doc, meta)

    def get(self, where, include):
        rows = [(i, d, m) for i, (d, m) in self.rows.items() if all(m.get(k) == v for k, v in where.items())]
        return {'ids': [r[0] for r in rows], 'metadatas': [r[2] for r in rows]}

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)


def make_kb():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._write_lock = threading.RLock()
    kb._collection = FakeCollection()
    return kb


def test_repeated_document_import_is_idempotent():
    kb = make_kb()
    doc = {'title': '退款政策', 'content': '7 天内可以申请退款。'}
    assert kb.add_documents([doc, doc]) == 1
    assert kb.add_documents([doc]) == 1
    assert len(kb._collection.rows) == 1


def test_shorter_document_replaces_chunks_and_legacy_data():
    kb = make_kb()
    kb._collection.rows['legacy'] = ('旧内容', {'title': '政策'})
    kb.add_documents([{'title': '政策', 'content': ('旧规则' * 100 + '。') * 5}])
    assert len(kb._collection.rows) > 1
    assert 'legacy' not in kb._collection.rows
    kb.add_documents([{'title': '政策', 'content': '新的政策'}])
    assert len(kb._collection.rows) == 1
    assert next(iter(kb._collection.rows.values()))[0] == '新的政策'


def test_explicit_ids_allow_documents_with_same_title():
    kb = make_kb()
    kb.add_documents([{'document_id': 'a', 'title': '政策', 'content': '甲'},
                      {'document_id': 'b', 'title': '政策', 'content': '乙'}])
    assert len(kb._collection.rows) == 2
    kb.add_documents([{'document_id': 'a', 'title': '改名', 'content': '新甲'}])
    assert len(kb._collection.rows) == 2
    assert {d for d, m in kb._collection.rows.values()} == {'新甲', '乙'}


def test_long_unpunctuated_text_respects_chunk_size():
    kb = make_kb()
    chunks = kb._chunk_text('长' * 1201, chunk_size=500)
    assert [len(c) for c in chunks] == [500, 500, 201]
    assert ''.join(chunks) == '长' * 1201


def test_search_exposes_snapshot_identity_on_vector_and_lexical_paths():
    from evaluation.retrieval import build_snapshot
    docs = [{'title': '认证', 'content': '登录401检查认证凭据。'},
            {'title': '账单', 'content': '重复扣款需要核验。'}]
    kb = make_kb()
    kb.add_documents(docs)
    stored = list(kb._collection.rows.values())
    kb._collection.count = lambda: 2
    kb._collection.query = lambda **kw: {
        'documents': [[stored[0][0]]], 'metadatas': [[stored[0][1]]], 'distances': [[.1]]}
    kb._collection.get = lambda **kw: {'documents': [stored[1][0]], 'metadatas': [stored[1][1]]}
    snap = build_snapshot(docs)
    expected = {c.title: c.chunk_id for c in snap.chunks}
    rows = kb.search('重复扣款401', top_k=2)
    assert {r['title']: r['chunk_id'] for r in rows} == expected
    # 旧数据缺少身份/版本时不伪造新证据 ID。
    stored[0][1].pop('version')
    assert next(r for r in kb.search('重复扣款401', 2) if r['title'] == '认证')['chunk_id'] == ''


def test_snapshot_preserves_exact_imported_text_and_explicit_identity():
    from evaluation.retrieval import build_snapshot
    from tooling.knowledge_chunks import evidence_id
    docs = [{'document_id': ' external id ', 'title': ' 标题 ', 'content': ' 正文。\n'}]
    kb = make_kb()
    kb.add_documents(docs)
    content, meta = next(iter(kb._collection.rows.values()))
    c = build_snapshot(docs).chunks[0]
    assert (c.content, c.title) == (content, meta['title'])
    assert c.chunk_id == evidence_id(meta['document_id'], meta['version'], meta['chunk_index'])
