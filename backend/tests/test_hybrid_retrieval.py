from types import SimpleNamespace

import pytest

from tooling.knowledge_base import KnowledgeBase


def make_kb(vector, lexical):
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = SimpleNamespace(
        count=lambda: 50,
        query=lambda **kwargs: {"documents": [[v[1] for v in vector]],
                                "metadatas": [[{"title": v[0]} for v in vector]],
                                "distances": [[.1 + i * .01 for i in range(len(vector))]]},
        get=lambda **kwargs: {"documents": [v[1] for v in lexical],
                              "metadatas": [{"title": v[0]} for v in lexical]},
    )
    return kb


@pytest.mark.parametrize("query,title,content", [
    ("企业订单重复扣款，退款需要审核吗？", "退款审核", "重复扣款需要人工审核退款资格。"),
    ("登录401错误怎么排查", "认证故障", "登录返回401时检查认证令牌与时间。"),
    ("合同续签需要哪些审批材料", "合同管理", "合同续签需要准备材料并走审批流程。"),
])
def test_lexical_branch_recovers_relevant_document_outside_vector_candidates(query, title, content):
    kb = make_kb([("无关配送", "配送物流进度"), ("无关会员", "会员积分优惠")], [(title, content)])
    hits = kb.search(query, top_k=1)
    assert hits[0]["title"] == title
    assert hits[0]["retrieval_method"] == "hybrid"


def test_no_lexical_overlap_preserves_semantic_results():
    kb = make_kb([("Returns", "Purchases can be returned."), ("Shipping", "Delivery tracking.")], [])
    assert [hit["title"] for hit in kb.search("退款政策", top_k=2)] == ["Returns", "Shipping"]


def test_lexical_failure_falls_back_without_losing_semantic_results():
    kb = make_kb([("退款政策", "需要审核")], [])
    def fail(**kwargs):
        raise RuntimeError("lexical unavailable")
    kb._collection.get = fail
    assert kb.search("退款", top_k=1)[0]["title"] == "退款政策"


def test_import_changes_are_visible_without_stale_lexical_cache():
    kb = make_kb([("配送", "快递信息")], [("账单", "退款需要审核")])
    assert kb.search("退款", top_k=1)[0]["title"] == "账单"
    kb._collection.get = lambda **kwargs: {"documents": ["退款新规则"], "metadatas": [{"title": "新政策"}]}
    assert kb.search("退款", top_k=1)[0]["title"] == "新政策"
