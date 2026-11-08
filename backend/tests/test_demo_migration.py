from scripts.migrate_demo_knowledge import migration_candidates


def test_migration_preserves_custom_documents_even_with_legacy_title():
    legacy = [{"title": "退款政策", "content": "旧演示原文"}]
    basic = {"title": "退款政策", "chunk_index": 0, "total_chunks": 1}
    rows = {
        "ids": ["seed", "modified", "imported", "unknown", "split"],
        "documents": ["旧演示原文", "用户自定义合同", "旧演示原文", "旧演示原文", "旧演示原文"],
        "metadatas": [basic, basic, {**basic, "document_id": "user-import"},
                      None, {**basic, "total_chunks": 2}],
    }
    assert migration_candidates(rows, legacy) == ["seed"]


def test_enterprise_documents_are_never_selected_as_legacy():
    rows = {"ids": ["new"], "documents": ["人工审核"],
            "metadatas": [{"title": "重复扣款与退款审核", "chunk_index": 0, "total_chunks": 1}]}
    assert migration_candidates(rows, [{"title": "退款政策", "content": "旧演示原文"}]) == []
