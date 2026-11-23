import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from core.concurrency import KeyedLocks
from core.llm_utils import create_message
from core.knowledge_revision import KnowledgeRevision
from core.performance import collect_performance, performance_snapshot, measure_stage
from memory.conversation_memory import MemoryManager, Message, MsgRole
from tooling.tool_manager import MCPToolManager, Tool


def search_manager(ttl=30):
    manager = MCPToolManager("test")
    async def retrieve(params, context):
        return [{"content": "退款规则", "document_id": "refund"}]
    manager.register(Tool("search", "", retrieve, {"type": "object"}, cache_ttl=ttl))
    manager.rewrite_query = AsyncMock(return_value=["退款"])
    return manager


@pytest.mark.asyncio
async def test_shared_knowledge_revision_invalidates_raw_and_pipeline_caches_across_workers():
    async with fakeredis.aioredis.FakeRedis(decode_responses=True) as redis:
        writer = KnowledgeRevision(redis)
        reader = KnowledgeRevision(redis)
        manager = search_manager()
        body = "old"

        async def retrieve(params, context):
            return [{"content": body, "document_id": "refund"}]

        manager.register(Tool("search", "", retrieve, {"type": "object"},
                              cache_ttl=30, cache_epoch=reader.current))
        assert (await manager.search_with_rewrite("search", "退款")).data[0]["content"] == "old"
        assert (await manager.search_with_rewrite("search", "退款")).cached
        async with writer.mutation():
            body = "new"
            result = await manager.search_with_rewrite("search", "退款")
            assert result.data[0]["content"] == "new" and not result.cached
        result = await manager.search_with_rewrite("search", "退款")
        assert result.data[0]["content"] == "new" and not result.cached
        assert (await manager.search_with_rewrite("search", "退款")).cached
        await manager._client.close()


@pytest.mark.asyncio
async def test_full_search_cache_skips_models_and_returns_defensive_copy():
    manager = search_manager()
    try:
        first = await manager.search_with_rewrite("search", "退款")
        first.data[0]["content"] = "changed"
        with collect_performance():
            second = await manager.search_with_rewrite("search", "退款")
            assert performance_snapshot()["events"]["rag_pipeline_cache_hit"] == 1
        assert second.cached and second.data[0]["content"] == "退款规则"
        assert manager.rewrite_query.await_count == 1
    finally:
        await manager._client.close()


@pytest.mark.asyncio
async def test_concurrent_identical_searches_share_pipeline():
    manager = search_manager()
    started, release = asyncio.Event(), asyncio.Event()
    async def rewrite(*args, **kwargs):
        started.set()
        await release.wait()
        return ["退款"]
    manager.rewrite_query = AsyncMock(side_effect=rewrite)
    try:
        first = asyncio.create_task(manager.search_with_rewrite("search", "退款"))
        await started.wait()
        rest = [asyncio.create_task(manager.search_with_rewrite("search", "退款")) for _ in range(4)]
        release.set()
        results = await asyncio.gather(first, *rest)
        assert manager.rewrite_query.await_count == 1
        assert sum(row.cached for row in results) == 4
        assert not manager._search_locks._locks
    finally:
        await manager._client.close()


@pytest.mark.asyncio
async def test_pipeline_cache_respects_user_top_k_and_invalidation():
    manager = search_manager()
    try:
        await manager.search_with_rewrite("search", "退款", context={"user_id": "a"})
        await manager.search_with_rewrite("search", "退款", context={"user_id": "b"})
        await manager.search_with_rewrite("search", "退款", top_k=2, context={"user_id": "a"})
        assert manager.rewrite_query.await_count == 3
        manager.invalidate_cache("search")
        assert not (await manager.search_with_rewrite("search", "退款", context={"user_id": "a"})).cached
        assert manager.rewrite_query.await_count == 4
        clone = manager.fork()
        clone.rewrite_query = AsyncMock(return_value=["退款"])
        assert not (await clone.search_with_rewrite("search", "退款", context={"user_id": "a"})).cached
    finally:
        await manager._client.close()


@pytest.mark.asyncio
async def test_invalidated_inflight_search_cannot_restore_old_pipeline_result():
    manager = search_manager()
    started, release = asyncio.Event(), asyncio.Event()
    async def rewrite(*args, **kwargs):
        started.set()
        await release.wait()
        return ["退款"]
    manager.rewrite_query = AsyncMock(side_effect=rewrite)
    try:
        pending = asyncio.create_task(manager.search_with_rewrite("search", "退款"))
        await started.wait()
        manager.invalidate_cache("search")
        release.set()
        await pending
        assert not (await manager.search_with_rewrite("search", "退款")).cached
        assert manager.rewrite_query.await_count == 2
    finally:
        await manager._client.close()


@pytest.mark.asyncio
async def test_zero_ttl_and_failed_searches_are_not_pipeline_cached():
    manager = search_manager(ttl=0)
    try:
        await manager.search_with_rewrite("search", "退款")
        assert not (await manager.search_with_rewrite("search", "退款")).cached
        assert manager.rewrite_query.await_count == 2
        manager.register(Tool("search", "", AsyncMock(side_effect=RuntimeError("down")), {}, cache_ttl=30))
        assert not (await manager.search_with_rewrite("search", "退款")).success
        assert not (await manager.search_with_rewrite("search", "退款")).success
        assert manager.rewrite_query.await_count == 4
    finally:
        await manager._client.close()


@pytest.mark.asyncio
async def test_cancelled_search_releases_single_flight_lock():
    manager = search_manager()
    started = asyncio.Event()
    async def stall(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()
    manager.rewrite_query = AsyncMock(side_effect=stall)
    task = asyncio.create_task(manager.search_with_rewrite("search", "退款"))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        manager.rewrite_query = AsyncMock(return_value=["退款"])
        assert (await manager.search_with_rewrite("search", "退款")).success
        assert not manager._search_locks._locks
    finally:
        await manager._client.close()


@pytest.mark.asyncio
async def test_pipeline_hit_skips_rerank_and_expiration_recomputes():
    manager = search_manager()
    manager._tools["search"].handler = AsyncMock(return_value=["a", "b", "c"])
    manager._rerank_with_status = AsyncMock(return_value=(["b"], True))
    try:
        assert (await manager.search_with_rewrite("search", "退款", top_k=1)).reranked
        second = await manager.search_with_rewrite("search", "退款", top_k=1)
        assert second.cached and second.reranked and second.data == ["b"]
        assert manager._rerank_with_status.await_count == 1
        for key, (data, expiry, reranked) in list(manager._cache.items()):
            if key.startswith("search:pipeline:"):
                manager._cache[key] = (data, 0, reranked)
        assert not (await manager.search_with_rewrite("search", "退款", top_k=1)).cached
        assert manager._rerank_with_status.await_count == 2
    finally:
        await manager._client.close()


@pytest.mark.asyncio
async def test_usage_is_request_scoped_and_contains_no_prompts():
    async def request(model, tokens):
        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            usage={"input_tokens": tokens, "output_tokens": 3, "cache_read_input_tokens": 2}))))
        with collect_performance():
            with measure_stage("execution"):
                await asyncio.sleep(0)
                await create_message(client, stage="agent.general", model=model, messages=[{"role":"user", "content":"private prompt"}])
            return performance_snapshot()
    a, b = await asyncio.gather(request("a", 10), request("b", 20))
    assert a["llm"]["input_tokens"] == 10 and b["llm"]["input_tokens"] == 20
    assert a["llm"]["calls"] == b["llm"]["calls"] == 1
    assert "execution" in a["stages_ms"] and "private prompt" not in json.dumps(a)
    assert performance_snapshot() == {}


@pytest.mark.asyncio
async def test_unreported_usage_and_failed_calls_are_visible():
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[SimpleNamespace(), RuntimeError("private error") ])))
    with collect_performance():
        await create_message(client, model="test")
        with pytest.raises(RuntimeError):
            await create_message(client, model="test")
        snapshot = performance_snapshot()
    assert snapshot["llm"]["calls"] == 2 and snapshot["llm"]["errors"] == 1
    assert snapshot["llm"]["usage_reported_calls"] == 0
    assert "private error" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_background_profile_is_throttled_but_preferences_and_force_are_immediate(monkeypatch):
    monkeypatch.setenv("PROFILE_UPDATE_MIN_INTERVAL_S", "60")
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    memory._update_profile = AsyncMock(return_value=True)
    memory._get_working_memory = AsyncMock(return_value=[Message(MsgRole.USER, "退款多久到？")])
    try:
        await memory.update_profile("u", "c", only_if_due=True)
        for _ in range(4):
            await memory.update_profile("u", "c", only_if_due=True)
        assert memory._update_profile.await_count == 1
        memory._get_working_memory.return_value = [Message(MsgRole.USER, "请记住，以后请用简短回答")]
        await memory.update_profile("u", "c", only_if_due=True)
        assert memory._update_profile.await_count == 2
        await memory.update_profile("u", "c")
        assert memory._update_profile.await_count == 3
        monkeypatch.setenv("PROFILE_UPDATE_MIN_INTERVAL_S", "0")
        await memory.update_profile("u", "c", only_if_due=True)
        assert memory._update_profile.await_count == 4
    finally:
        await memory._redis.aclose()


@pytest.mark.asyncio
async def test_profile_failure_does_not_suppress_retry():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    memory._update_profile = AsyncMock(side_effect=[False, True])
    try:
        await memory.update_profile("u", "c", only_if_due=True)
        await memory.update_profile("u", "c", only_if_due=True)
        assert memory._update_profile.await_count == 2
    finally:
        await memory._redis.aclose()


@pytest.mark.asyncio
async def test_get_context_runs_storage_reads_concurrently_and_preserves_data():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        await memory._redis.lpush(
            memory._wm_key("u1", "c1"),
            json.dumps({"role": "user", "content": "发票抬头怎么改", "ts": "2026-11-24T00:00:00", "metadata": {}}),
        )
        await memory._redis.set(memory._summary_key("u1", "c1"), "历史摘要")
        await memory._redis.set(memory._state_key("u1", "c1"), json.dumps({"entities": {"order_id": ["ORD-1"]}}))

        active = 0
        all_entered = asyncio.Event()

        async def barrier():
            nonlocal active
            active += 1
            if active == 3:
                all_entered.set()
            await asyncio.wait_for(all_entered.wait(), timeout=1.0)

        original_read_session = memory._read_session_state

        async def read_session(user_id, conv_id):
            await barrier()
            return await original_read_session(user_id, conv_id)

        async def search_episodic(user_id, conv_id, query):
            await barrier()
            return ["相关历史片段"]

        async def get_profile(user_id):
            await barrier()
            return {"preferences": ["简短回答"]}

        memory._read_session_state = read_session
        memory._search_episodic = search_episodic
        memory._get_profile = get_profile

        ctx = await memory.get_context("u1", "c1", query="发票")
        assert [m.content for m in ctx.recent_messages] == ["发票抬头怎么改"]
        assert ctx.summary == "历史摘要"
        assert ctx.state == {"entities": {"order_id": ["ORD-1"]}}
        assert ctx.relevant_history == ["相关历史片段"]
        assert ctx.user_profile == {"preferences": ["简短回答"]}
    finally:
        await memory._redis.aclose()


@pytest.mark.asyncio
async def test_conversation_migrates_legacy_keys_once_per_lease():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    memory._request_locks = KeyedLocks()
    memory._search_episodic = AsyncMock(return_value=[])
    memory._get_profile = AsyncMock(return_value={})
    try:
        legacy_msg = json.dumps({"role": "user", "content": "旧消息", "ts": "2026-11-24T00:00:00", "metadata": {}})
        await memory._redis.rpush("wm:u1:c1", legacy_msg)
        await memory._redis.set("summary:u1:c1", "旧摘要")

        eval_calls = 0
        original_eval = memory._redis.eval

        async def tracked_eval(*args, **kwargs):
            nonlocal eval_calls
            eval_calls += 1
            return await original_eval(*args, **kwargs)

        memory._redis.eval = tracked_eval

        async with memory.conversation("u1", "c1"):
            ctx = await memory.get_context("u1", "c1", query="继续")
            await memory.commit_turn(
                "u1",
                "c1",
                "req-1",
                "继续",
                {"request_id": "r-1", "response": "好的"},
            )

        assert eval_calls == 1
        assert [m.content for m in ctx.recent_messages] == ["旧消息"]
        assert ctx.summary == "旧摘要"
        assert not await memory._redis.exists("wm:u1:c1", "summary:u1:c1")

        # 脱离 conversation() 上下文直接调用，仍正常执行一次原子迁移。
        await memory._redis.rpush("wm:u1:c2", legacy_msg)
        await memory._redis.set("summary:u1:c2", "另一段旧摘要")
        await memory._migrate_legacy_keys("u1", "c2")
        assert eval_calls == 2
        assert await memory._redis.get(memory._summary_key("u1", "c2")) == "另一段旧摘要"
    finally:
        await memory._redis.aclose()

