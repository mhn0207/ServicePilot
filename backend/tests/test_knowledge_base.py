import sys
import types
import threading

import pytest

try:
    import chromadb
except ModuleNotFoundError:
    sys.modules['chromadb'] = types.ModuleType('chromadb')

from tooling.knowledge_base import KnowledgeBase
from tooling.knowledge_chunks import chunk_text, document_identity, evidence_id


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


@pytest.mark.parametrize("fence", ["~~~", chr(96) * 3])
def test_markdown_sections_preserve_nested_lists_and_fences(fence):
    checklist = "- 核验订单\n\n  - 比对入账时间\n  - 保留流水编号\n"
    code = f"{fence}sh\n    {fence}\n# 这行是代码注释\n  printf '原始缩进'\n\n{fence}\n"
    text = f"# 账单\n\n## 重复扣款\n\n{checklist}\n{code}\n## 退款\n\n退款需确认。\n"
    chunks = chunk_text(text, 110, strategy="markdown-v1")

    assert all(len(chunk) <= 110 for chunk in chunks)
    assert any(chunk.startswith("[章节: 账单 > 重复扣款]\n") and checklist in chunk for chunk in chunks)
    assert any(chunk.startswith("[章节: 账单 > 重复扣款]\n") and code in chunk for chunk in chunks)
    assert any(chunk.startswith("[章节: 账单 > 退款]\n") and "退款需确认。" in chunk for chunk in chunks)
    assert all(" > 这行是代码注释]" not in chunk for chunk in chunks)


@pytest.mark.parametrize(("checklist", "next_indent"), [
    ("- 恢复步骤\n  # 网关诊断\n  使用备用地址。\n", ""),
    ("1. 恢复步骤\n\n   # 网关诊断\n   使用备用地址。\n", "  "),
    ("- 恢复步骤\n  - # 网关诊断\n    使用备用地址。\n", ""),
])
def test_markdown_list_headings_stay_inside_the_list(checklist, next_indent):
    text = f"# 手册\n\n{checklist}\n{next_indent}## 核验结果\n查询状态。\n"
    chunks = chunk_text(text, strategy="markdown-v1")

    assert [chunk.splitlines()[0] for chunk in chunks] == ["[章节: 手册]", "[章节: 手册 > 核验结果]"]
    assert checklist in chunks[0]
    assert "查询状态。" in chunks[1]


@pytest.mark.parametrize(("list_prefix", "fence"), [
    ("- ", "```"),
    ("  10. ", "~~~"),
    ("- - ", "```"),
])
def test_markdown_fence_opening_on_list_item_keeps_next_heading(list_prefix, fence):
    indent = " " * len(list_prefix)
    code = (f"{list_prefix}{fence}python\n"
            f"{indent}    {fence}\n{indent}# 代码注释\n{indent}print(1)\n"
            f"{indent}{fence}\n")
    chunks = chunk_text(f"# 手册\n\n{code}\n## 下一节\n\n后续说明。\n", strategy="markdown-v1")

    assert [chunk.splitlines()[0] for chunk in chunks] == ["[章节: 手册]", "[章节: 手册 > 下一节]"]
    assert code in chunks[0]


def test_markdown_unclosed_list_fence_ends_at_list_dedent():
    text = "# 手册\n\n- ~~~python\n  # 代码注释\n  print(1)\n\n## 下一节\n\n后续说明。\n"
    chunks = chunk_text(text, strategy="markdown-v1")

    assert [chunk.splitlines()[0] for chunk in chunks] == ["[章节: 手册]", "[章节: 手册 > 下一节]"]
    assert "  # 代码注释\n  print(1)\n" in chunks[0]


@pytest.mark.parametrize("code", [
    "- 项目\n\n  ~~~python\n  # 代码注释\n",
    "- 上级\n  - 下级\n\n    ~~~python\n    # 代码注释\n",
    "    ~~~python\n    # 缩进代码，不是围栏\n",
])
def test_markdown_indented_marker_keeps_heading_after_list_or_code(code):
    chunks = chunk_text(f"# 手册\n\n{code}\n## 下一节\n\n后续说明。\n", strategy="markdown-v1")

    assert [chunk.splitlines()[0] for chunk in chunks] == ["[章节: 手册]", "[章节: 手册 > 下一节]"]
    assert code in chunks[0]


def test_markdown_oversized_list_fence_keeps_indented_closers_and_code():
    long_line = "a" * 1000
    text = f"# 手册\n\n- ~~~python\n  # 代码注释\n  {long_line}\n\n## 下一节\n"
    chunks = chunk_text(text, strategy="markdown-v1")

    code_chunks = [chunk.splitlines() for chunk in chunks if "- ~~~python" in chunk]
    assert len(code_chunks) > 1
    assert all(lines[0] == "[章节: 手册]" and lines[-1] == "  ~~~" for lines in code_chunks)
    assert all(line.startswith("  ") for lines in code_chunks for line in lines[2:-1] if line)
    assert sum(line.count("a") for lines in code_chunks for line in lines[2:-1]) == len(long_line)
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert chunks[-1].startswith("[章节: 手册 > 下一节]\n")


def test_markdown_oversized_blocks_keep_context_and_code_fences():
    lines = [f"  setting_{i} = '{'值' * 48}'\n" for i in range(14)]
    prose = "".join(chr(0x4E00 + i) for i in range(900))
    text = "# 配置\n\n~~~python\n" + "".join(lines) + "~~~\n\n" + prose
    chunks = chunk_text(text, 200, strategy="markdown-v1")

    assert all(len(chunk) <= 200 and chunk.startswith("[章节: 配置]\n") for chunk in chunks)
    code_chunks = [chunk for chunk in chunks if "~~~python" in chunk]
    assert len(code_chunks) > 1 and all(chunk.rstrip().endswith("~~~") for chunk in code_chunks)
    assert all(any(line in chunk for chunk in code_chunks) for line in lines)
    bodies = [chunk.split("\n", 1)[1] for chunk in chunks]
    prose_chunks = [body for body in bodies if body and body in prose]
    assert len(prose_chunks) > 1
    assert prose.startswith(prose_chunks[0]) and prose.endswith(prose_chunks[-1])
    assert all(left[-50:] == right[:50] for left, right in zip(prose_chunks, prose_chunks[1:]))


@pytest.mark.parametrize("identity", [{}, {"document_id": "billing"}])
def test_markdown_and_legacy_coexist_with_matching_online_and_offline_ids(identity):
    from evaluation.retrieval import build_snapshot

    original = {"title": "账单", "content": "# 查账\n\n请保留流水编号。", **identity}
    markdown = dict(original, chunk_strategy="markdown-v1")
    snapshot = build_snapshot([original, markdown])
    assert build_snapshot([d.model_dump() for d in snapshot.documents]) == snapshot
    assert build_snapshot([dict(original, chunk_strategy="legacy")]) == build_snapshot([original])
    assert len({c.document_id for c in snapshot.chunks}) == 2
    assert len({c.chunk_id for c in snapshot.chunks}) == 2
    assert len({c.version for c in snapshot.chunks}) == 1  # 原文哈希相同

    kb = make_kb()
    kb._collection.rows['pre-version'] = ('需人工核验', {'title': '账单'})
    kb.add_documents([markdown])
    assert 'pre-version' in kb._collection.rows
    kb.add_documents([original])
    kb.add_documents([markdown])
    assert {evidence_id(m['document_id'], m['version'], m['chunk_index']): doc
            for doc, m in kb._collection.rows.values() if m.get('document_id')} == {
                c.chunk_id: c.content for c in snapshot.chunks}


def test_legacy_id_migration_cannot_remove_markdown_chunks():
    kb = make_kb()
    kb.add_documents([{"title": "Runbook", "content": "# Section\n\nMarkdown rule.",
                       "chunk_strategy": "markdown-v1"}])
    markdown_ids = set(kb._collection.rows)
    # The historical scheme hashed arbitrary explicit IDs without a namespace.
    legacy_id = document_identity("Runbook", strategy="markdown-v1")
    kb.add_documents([{"document_id": legacy_id, "title": "Runbook", "content": "Legacy rule."}])
    assert markdown_ids < set(kb._collection.rows)
