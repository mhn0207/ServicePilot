"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
启用短路时，明确的首轮请求会给向量分支短暂的确认窗口，其余请求并行执行。
"""
import asyncio
import math
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import create_message, extract_text_content
from core.amount_entities import AMOUNT, amount_mentions
from core.env import env_bool, env_value
from core.request_budget import POLICY

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 要求升级/转人工
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单/退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    ORDER_STATUS = "order_status"        # 订单状态
    LOGISTICS = "logistics"              # 物流配送
    REFUND = "refund"                    # 退款/退货
    INVOICE = "invoice"                  # 发票
    PAYMENT_ISSUE = "payment_issue"      # 支付/扣款异常
    ACCOUNT_SECURITY = "account_security" # 账户安全
    TECHNICAL_LOGIN = "technical_login"  # 登录认证故障
    TECHNICAL_CRASH = "technical_crash"  # 崩溃/错误码
    HUMAN_HANDOFF = "human_handoff"      # 转人工
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    embedding_source: str = "local-ngram"


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"],
    IntentCategory.COMPLAINT:  ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST:    ["帮我取消企业订单", "我需要更新企业资料", "请协助退款"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "转人工客服", "找你们经理"],
    IntentCategory.TECHNICAL:  ["应用一直崩溃", "无法登录", "出现500错误"],
    IntentCategory.BILLING:    ["为什么扣了两次款？", "申请退款", "发票问题"],
    IntentCategory.ACCOUNT:    ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
    IntentCategory.ORDER_STATUS: ["我的订单现在是什么状态？", "订单对应的服务开通了吗？", "订单处理到哪一步了？"],
    IntentCategory.LOGISTICS: ["快递什么时候到？", "物流一直不更新", "配送要多久？"],
    IntentCategory.REFUND: ["我要申请退款", "企业合同退款怎么处理？", "退款多久到账？"],
    IntentCategory.INVOICE: ["帮我开发票", "发票抬头怎么改？", "电子发票在哪里？"],
    IntentCategory.PAYMENT_ISSUE: ["为什么重复扣款？", "支付失败怎么办？", "这个月多扣了钱"],
    IntentCategory.ACCOUNT_SECURITY: ["账户被盗了", "发现异常登录", "我要重置密码"],
    IntentCategory.TECHNICAL_LOGIN: ["登录一直报401", "验证码收不到", "无法登录账号"],
    IntentCategory.TECHNICAL_CRASH: ["应用一直崩溃", "页面报500错误", "系统闪退"],
    IntentCategory.HUMAN_HANDOFF: ["转人工客服", "我要找人工", "请升级处理"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.ORDER_STATUS,
    IntentCategory.LOGISTICS,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
    IntentCategory.HUMAN_HANDOFF,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.BILLING,
    IntentCategory.TECHNICAL,
    IntentCategory.ACCOUNT,
    IntentCategory.ESCALATION,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.ORDER_STATUS: IntentCategory.QUERY,
    IntentCategory.LOGISTICS: IntentCategory.QUERY,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.ACCOUNT_SECURITY: IntentCategory.ACCOUNT,
    IntentCategory.TECHNICAL_LOGIN: IntentCategory.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: IntentCategory.TECHNICAL,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.ESCALATION,
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}

_FAST_PATH_PATTERNS = {
    IntentCategory.TECHNICAL_LOGIN: r"登录(?:一直|时)?报(?:错)?\s*(?:http\s*)?401|无法登录账号|验证码收不到",
    IntentCategory.TECHNICAL_CRASH: r"应用一直崩溃|系统闪退|页面(?:报|返回)\s*(?:http\s*)?(?:500|502)(?:错误)?",
    IntentCategory.REFUND: r"(?:我(?:想|要)|请帮我)申请退款|退款多久到账",
    IntentCategory.INVOICE: r"(?:请)?帮我开发票|电子发票在哪里",
    IntentCategory.PAYMENT_ISSUE: r"为什么重复扣款|支付失败怎么办",
    IntentCategory.HUMAN_HANDOFF: r"转人工客服|我要找人工|请帮我转人工",
}
_FAST_PATH_EMBEDDING_MIN = 0.82
_FAST_PATH_WAIT_S = 0.05
_SKIPPED_LLM = {"intent": IntentCategory.OTHER, "confidence": 0.0, "skipped": True,
                "reasoning": "首轮明确请求与向量意图一致"}

_CODE_END = r"(?![A-Za-z0-9]|\.\d|\s*(?:元|块|rmb|cny|usd|美元|客服|热线|电话|分|天|小时|人|次|个|件|条|页|mb|gb))"
_EXPLICIT_ERROR_CODE = re.compile(
    rf"(?:error(?:_code)?|错误码|状态码|http|报错|返回|异常|报)\s*[:：#]?\s*([45]\d{{2}}){_CODE_END}", re.I,
)
_CONTEXTUAL_ERROR_CODE = re.compile(rf"(?<![A-Za-z0-9¥￥.])([45]\d{{2}}){_CODE_END}", re.I)
_ERROR_CONTEXT = ("报错", "错误", "状态码", "异常", "失败", "登录", "接口", "请求", "返回", "页面", "崩溃",
                  "error", "http", "status", "code")


def extract_error_codes(message: str) -> List[str]:
    """只提取技术上下文里的三位错误码，供意图规则和路由共同使用。"""
    amount_spans = [match.span() for match in AMOUNT.finditer(message)]

    def outside_amount(match):
        start, end = match.span(1)
        return not any(left <= start and end <= right for left, right in amount_spans)

    codes = [match[1] for match in _EXPLICIT_ERROR_CODE.finditer(message) if outside_amount(match)]
    for match in _CONTEXTUAL_ERROR_CODE.finditer(message):
        start, end = match.span(1)
        window = message[max(0, start - 12):end + 12].lower()
        if outside_amount(match) and any(keyword in window for keyword in _ERROR_CONTEXT):
            codes.append(match[1])
    return list(dict.fromkeys(codes))


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    LLM 使用 Anthropic Messages 客户端，向量使用独立的硅基流动客户端。
    模板向量在首次请求时生成并缓存；在线不可用时整次匹配回退本地 n-gram。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        from core.intent_embedding import IntentEmbedding
        self._embedding = IntentEmbedding()
        embedding_enabled = os.getenv('INTENT_EMBEDDING_ENABLED', 'true').strip().lower()
        if embedding_enabled not in {'true', 'false', '1', '0', 'yes', 'no', 'on', 'off'}:
            raise ValueError('INTENT_EMBEDDING_ENABLED 必须为 true/false')
        self._embedding_enabled = embedding_enabled in {'true', '1', 'yes', 'on'}
        self._fast_path_enabled = env_bool("SERVICEPILOT_INTENT_FAST_PATH")
        self._templates: Dict[IntentCategory, List[str]] = {cat: list(tpls) for cat, tpls in _TEMPLATES.items()}
        self._cache: Dict[str, IntentResult] = {}
        self._cache_expiries: Dict[str, float] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache and time.monotonic() < self._cache_expiries.get(key, 0):
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        entities = self._extract_entities(message)
        pat = self._pattern_recognize(message, entities["error_code"])
        llm, emb = await self._recognize_sources(message, history, pat)
        intent, confidence, source_scores = self._vote(llm, emb, pat)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
            embedding_source=emb.get("source", "unavailable"),
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
                self._cache_expiries.pop(k, None)
        self._cache[key] = result
        self._cache_expiries[key] = time.monotonic() + (60 if result.embedding_source == "local-ngram" and self._embedding.enabled else 300)
        return result

    async def _recognize_sources(self, message, history, pattern):
        if not self._embedding_enabled:
            return await self._llm_recognize(message, history), {
                "intent": IntentCategory.OTHER, "confidence": 0.0, "source": "disabled",
            }
        async with asyncio.TaskGroup() as tasks:
            embedding = tasks.create_task(self._embedding_recognize(message))
            shortcut_pattern = _FAST_PATH_PATTERNS.get(pattern["intent"]) if self._fast_path_enabled and not history else None
            shortcut_message = message.strip().lower().rstrip("。！？!?").rstrip()
            if shortcut_pattern and re.fullmatch(shortcut_pattern, shortcut_message):
                # 向量慢时立即启动原 LLM 分支，复用仍在途的向量请求。
                done, _ = await asyncio.wait({embedding}, timeout=_FAST_PATH_WAIT_S)
                if done:
                    result = embedding.result()
                    confidence = self._voting_confidence(result)
                    if (result.get("intent") == pattern["intent"] and confidence is not None
                            and confidence >= _FAST_PATH_EMBEDDING_MIN
                            and self._vote(_SKIPPED_LLM, result, pattern)[0] == pattern["intent"]):
                        return _SKIPPED_LLM, result
            llm = await self._llm_recognize(message, history)
            return llm, await embedding

    async def check_embedding(self, message: str) -> Dict[str, Any]:
        """独立验证向量链路；不调用 LLM，也不写入意图识别结果缓存。"""
        if not self._embedding_enabled:
            return {'success': False, 'source': 'disabled', 'error': 'embedding_branch_disabled'}
        started = time.monotonic()
        result = await self._embedding_recognize(message)
        source = result.get('source', 'unavailable')
        return {
            'success': source == 'siliconflow/' + self._embedding.MODEL,
            'source': source,
            'intent': result['intent'].value,
            'confidence': result['confidence'],
            'elapsed_ms': round((time.monotonic() - started) * 1000, 1),
            'status': self._embedding.status(),
        }

    def _templates_map(self) -> Dict[IntentCategory, List[str]]:
        if not hasattr(self, "_templates"):
            self._templates = {cat: list(tpls) for cat, tpls in _TEMPLATES.items()}
        return self._templates

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入当前实例模板，清除对应 Embedding 缓存。"""
        tpls = self._templates_map().setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._embedding.remote_templates.clear()
            self._embedding.local_templates.clear()
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            self._cache_expiries.clear()
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in self._templates_map().items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是客服意图分析专家。根据示例判断用户意图，返回 JSON。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如退款优先返回 refund，发票优先返回 invoice，登录故障优先返回 technical_login。

示例:
{examples}

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            timeout_s = env_value("SERVICEPILOT_INTENT_TIMEOUT_S", str(POLICY.intent_timeout_default_s))
            resp = await asyncio.wait_for(create_message(self.client,
                stage="intent",
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            ), timeout=max(0.1, float(timeout_s)))
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            confidence = data.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("意图 confidence 必须是 0 到 1 的有限数字")
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            templates, msg_vec, source = await self._embedding.vectors(
                self._clean_text(message), self._templates_map(), self._local_embedding)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in templates.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": min(1.0, max(0.0, best_score)), "source": source}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "source": "unavailable"}

    def _pattern_recognize(self, message: str, error_codes: List[str] | None = None) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        if error_codes is None:
            error_codes = extract_error_codes(msg)
        specific_patterns = {
            IntentCategory.HUMAN_HANDOFF: ["转人工", "人工客服", "找人工"],
            IntentCategory.ORDER_STATUS: ["订单状态", "发货了吗", "处理到哪", "order status"],
            IntentCategory.LOGISTICS: ["物流", "快递", "配送", "运单", "delivery", "shipping"],
            IntentCategory.REFUND: ["退款", "退货", "refund", "return"],
            IntentCategory.INVOICE: ["发票", "抬头", "税号", "invoice"],
            IntentCategory.PAYMENT_ISSUE: ["重复扣款", "多扣", "支付失败", "扣费", "payment failed"],
            IntentCategory.ACCOUNT_SECURITY: ["被盗", "异常登录", "重置密码", "两步验证", "安全"],
            IntentCategory.TECHNICAL_LOGIN: ["无法登录", "登录失败", "验证码"],
            IntentCategory.TECHNICAL_CRASH: ["崩溃", "闪退", "报错", "crash"],
        }
        if "401" in error_codes:
            specific_patterns[IntentCategory.TECHNICAL_LOGIN].append("401")
        specific_patterns[IntentCategory.TECHNICAL_CRASH].extend(code for code in error_codes if code.startswith("5"))
        generic_patterns = {
            IntentCategory.ESCALATION: ["投诉", "经理", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    ["退款", "扣款", "发票", "refund"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
        }

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _voting_confidence(source: Dict) -> float | None:
        value = source.get("confidence")
        if (source.get("failed") or source.get("skipped") or source.get("source") in {"disabled", "unavailable"}
                or type(value) not in (int, float) or not 0 <= value <= 1):
            return None
        return float(value)

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """只用实际参与的分支归一化权重，返回组得分和原始分支得分。"""
        if self._embedding_enabled:
            branches = [("llm", llm, 0.7), ("embedding", emb, 0.2), ("pattern", pat, 0.1)]
        else:
            branches = [("llm", llm, 0.85), ("pattern", pat, 0.15)]
        source_scores = {"llm": 0.0, "embedding": 0.0, "pattern": 0.0}
        active_weight = 0.0
        scores: Dict[IntentCategory, float] = {}
        group_scores: Dict[IntentCategory, float] = {}
        for name, result, weight in branches:
            conf = self._voting_confidence(result)
            if conf is None:
                continue
            source_scores[name] = conf
            active_weight += weight
            cat = result.get("intent", IntentCategory.OTHER)
            weighted_score = weight * conf
            scores[cat] = scores.get(cat, 0.0) + weighted_score
            group = _INTENT_GROUPS.get(cat, cat)
            group_scores[group] = group_scores.get(group, 0.0) + weighted_score

        if not group_scores:
            return IntentCategory.OTHER, 0.0, source_scores
        best_group = max(group_scores, key=group_scores.get)
        best_score = min(1.0, group_scores[best_group] / active_weight)
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        best = max(
            (cat for cat in scores if _INTENT_GROUPS.get(cat, cat) == best_group),
            key=scores.get,
        )
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = source_scores["pattern"]
        if (best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS
                and _INTENT_GROUPS.get(pat_intent, pat_intent) == best_group
                and pat_conf >= 0.5 and best_score < 0.8):
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        return {
            "order_id": self._unique(re.findall(r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)),
            "product": [],
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": amount_mentions(message)[0],
            "error_code": extract_error_codes(message),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        msg = re.sub(r"(?:不(?:是|用|需要)?|无需|无须|非|并不|没那么|暂不)\s*(?:紧急|立刻|马上|尽快|快点)", "", msg)
        msg = re.sub(r"\b(?:not|isn't|is not|no)\s+(?:an?\s+)?(?:emergency|urgent|asap|now)\b", "", msg)
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", msg) if kw.isascii() else kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", "")),
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
