import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from core.intent_recognizer import IntentCategory, IntentRecognizer, extract_error_codes


def make_recognizer(threshold=0.5):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled = True
    recognizer.threshold = threshold
    return recognizer


def vote(recognizer, llm, emb, pat):
    return recognizer._vote(
        {"intent": llm[0], "confidence": llm[1]},
        {"intent": emb[0], "confidence": emb[1]},
        {"intent": pat[0], "confidence": pat[1]},
    )


def test_pattern_cannot_refine_a_different_intent_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.BILLING, 0.9),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.BILLING
    assert confidence == pytest.approx(0.63)
    assert "refined_by_pattern" not in sources


def test_coarse_and_fine_scores_choose_the_same_group_together():
    intent, confidence, sources = vote(
        make_recognizer(threshold=0.2),
        (IntentCategory.BILLING, 0.3),       # weighted score 0.21
        (IntentCategory.TECHNICAL_LOGIN, 0.9),  # weighted score 0.18
        (IntentCategory.TECHNICAL, 0.4),     # weighted score 0.04
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.22)
    assert "refined_by_pattern" not in sources


def test_pattern_can_still_refine_within_the_winning_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.TECHNICAL, 0.8),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.61)
    assert sources["refined_by_pattern"] == pytest.approx(0.5)


def test_unavailable_embedding_is_removed_from_weight_denominator():
    intent, confidence, sources = make_recognizer()._vote(
        {"intent": IntentCategory.BILLING, "confidence": .6},
        {"intent": IntentCategory.OTHER, "confidence": 0, "source": "unavailable"},
        {"intent": IntentCategory.OTHER, "confidence": 0},
    )
    assert intent is IntentCategory.BILLING
    assert confidence == pytest.approx((.7 * .6) / .8)
    assert sources == {"llm": .6, "embedding": 0, "pattern": 0}


@pytest.mark.parametrize("failed", [
    {"confidence": .9, "failed": True},
    {"confidence": float("nan")},
    {"confidence": True},
    {"confidence": 1.01},
])
def test_llm_failure_cannot_promote_a_weak_vector(failed):
    llm = {"intent": IntentCategory.REFUND, **failed}
    intent, confidence, sources = make_recognizer()._vote(
        llm, {"intent": IntentCategory.REFUND, "confidence": .35},
        {"intent": IntentCategory.OTHER, "confidence": 0},
    )
    assert intent is IntentCategory.OTHER
    assert confidence == pytest.approx((.2 * .35) / .3)
    assert sources["llm"] == 0


def test_failed_llm_and_vector_keep_strong_rule_fallback():
    intent, confidence, _ = make_recognizer()._vote(
        {"intent": IntentCategory.OTHER, "confidence": 0, "failed": True},
        {"intent": IntentCategory.OTHER, "confidence": 0, "source": "unavailable"},
        {"intent": IntentCategory.REFUND, "confidence": .5},
    )
    assert (intent, confidence) == (IntentCategory.REFUND, .5)


def test_pattern_refinement_still_requires_the_group_threshold():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.TECHNICAL, .3),
        (IntentCategory.OTHER, 0),
        (IntentCategory.TECHNICAL_LOGIN, .5),
    )
    assert intent is IntentCategory.OTHER
    assert confidence == pytest.approx(.26)
    assert "refined_by_pattern" not in sources


@pytest.mark.parametrize(("message", "expected"), [
    ("报 500 元，实际只扣了 401 元。", []),
    ("401 和 500", []),
    ("接口返回 HTTP 502. 登录报401", ["502", "401"]),
    ("页面报错 401.5 元", []),
])
def test_http_code_requires_a_code_context_and_numeric_boundary(message, expected):
    assert extract_error_codes(message) == expected


@pytest.fixture
def fast_recognizer(monkeypatch):
    monkeypatch.setenv("SERVICEPILOT_INTENT_FAST_PATH", "true")
    monkeypatch.setenv("INTENT_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("INTENT_EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr("core.intent_recognizer.AsyncAnthropic", Mock())
    return IntentRecognizer("offline")


@pytest.mark.asyncio
async def test_slow_embedding_head_start_does_not_cancel_or_duplicate_lookup(fast_recognizer, monkeypatch):
    monkeypatch.setattr("core.intent_recognizer._FAST_PATH_WAIT_S", .001)
    llm_started = asyncio.Event()

    async def embedding(_):
        await llm_started.wait()
        return {"intent": IntentCategory.REFUND, "confidence": .95, "source": "local-ngram"}

    async def llm(*_):
        llm_started.set()
        return {"intent": IntentCategory.REFUND, "confidence": .95}

    fast_recognizer._embedding_recognize = AsyncMock(side_effect=embedding)
    fast_recognizer._llm_recognize = AsyncMock(side_effect=llm)
    result = await asyncio.wait_for(fast_recognizer.recognize("我要申请退款"), 1)
    assert result.intent is IntentCategory.REFUND
    fast_recognizer._embedding_recognize.assert_awaited_once()
    fast_recognizer._llm_recognize.assert_awaited_once()


@pytest.mark.parametrize(("category", "confidence", "source"), [
    (IntentCategory.REFUND, .81, "local-ngram"),
    (IntentCategory.INVOICE, .99, "local-ngram"),
    (IntentCategory.REFUND, .99, "unavailable"),
])
@pytest.mark.asyncio
async def test_vector_must_be_available_strong_and_agree_to_shortcut(fast_recognizer, category, confidence, source):
    fast_recognizer._embedding_recognize = AsyncMock(return_value={
        "intent": category, "confidence": confidence, "source": source,
    })
    fast_recognizer._llm_recognize = AsyncMock(return_value={"intent": IntentCategory.REFUND, "confidence": .95})
    assert (await fast_recognizer.recognize("我要申请退款")).intent is IntentCategory.REFUND
    fast_recognizer._llm_recognize.assert_awaited_once()


@pytest.mark.asyncio
async def test_shortcut_is_cached_by_history(fast_recognizer):
    fast_recognizer._llm_recognize = AsyncMock(return_value={"intent": IntentCategory.REFUND, "confidence": .95})
    fresh = await fast_recognizer.recognize("我要申请退款")
    assert await fast_recognizer.recognize("我要申请退款") is fresh
    assert fresh.source_scores["llm"] == 0
    fast_recognizer._llm_recognize.assert_not_awaited()
    with_history = await fast_recognizer.recognize("我要申请退款", [{"role": "user", "content": "修改金额"}])
    assert with_history.source_scores["llm"] == .95
    fast_recognizer._llm_recognize.assert_awaited_once()


def test_learn_isolates_templates_per_recognizer_instance():
    from core.intent_recognizer import _TEMPLATES
    first = IntentRecognizer(api_key="test")
    second = IntentRecognizer(api_key="test")
    sample = "测试专用样本_企业对公账户开票"
    first.learn(sample, IntentCategory.INVOICE)
    assert sample in first._templates_map()[IntentCategory.INVOICE]
    assert sample not in second._templates_map()[IntentCategory.INVOICE]
    assert sample not in _TEMPLATES[IntentCategory.INVOICE]


def test_cors_origins_restricts_production_by_default(monkeypatch):
    from api.main import _cors_origins
    monkeypatch.delenv("SERVICEPILOT_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("ZHIYING_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("SERVICEPILOT_ENV", "production")
    assert _cors_origins() == []
    monkeypatch.setenv("SERVICEPILOT_CORS_ORIGINS", "https://console.example.com, https://admin.example.com")
    assert _cors_origins() == ["https://console.example.com", "https://admin.example.com"]
    monkeypatch.delenv("SERVICEPILOT_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("SERVICEPILOT_ENV", "development")
    assert _cors_origins() == ["*"]
