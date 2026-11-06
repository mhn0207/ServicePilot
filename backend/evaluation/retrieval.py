"""证据级检索协议 v1。纯数据与评分，不导入应用或数据库客户端。"""
import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from tooling.knowledge_chunks import chunk_text, content_version, document_identity, evidence_id

PROTOCOL = "rag-evidence-v1"
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RawText = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Document(StrictModel):
    # 与真实入库一致，保留原文和显式 ID 的空白，不能在哈希前偷偷修剪。
    title: RawText
    content: RawText
    document_id: RawText | None = None


class SnapshotChunk(StrictModel):
    chunk_id: Text
    document_id: Text
    version: Text
    chunk_index: int = Field(ge=0)
    title: RawText
    content: RawText


class Snapshot(StrictModel):
    protocol_version: Literal["knowledge-snapshot-v1"] = "knowledge-snapshot-v1"
    chunk_size: Literal[500] = 500
    documents: list[Document]
    chunks: list[SnapshotChunk]
    knowledge_hash: Text


def build_snapshot(documents: list[dict]) -> Snapshot:
    """固定原文构建快照；顺序无关，重复身份拒绝，绝不读取本机已有库。"""
    docs = [Document.model_validate(doc) for doc in documents]
    docs.sort(key=lambda d: document_identity(d.title, d.document_id))
    chunks, identities = [], set()
    for doc in docs:
        identity = document_identity(doc.title, doc.document_id)
        if identity in identities:
            raise ValueError("重复文档身份")
        identities.add(identity)
        document_id = content_version(identity)
        version = content_version(doc.content)
        for index, content in enumerate(chunk_text(doc.content)):
            chunks.append(SnapshotChunk(chunk_id=evidence_id(document_id, version, index),
                                        document_id=document_id, version=version, chunk_index=index,
                                        title=doc.title, content=content))
    payload = {"protocol_version": "knowledge-snapshot-v1", "chunk_size": 500,
               "documents": [d.model_dump() for d in docs],
               "chunks": [c.model_dump() for c in sorted(chunks, key=lambda c: c.chunk_id)]}
    return Snapshot(**payload, knowledge_hash=canonical_hash(payload))


class EvidenceGroup(StrictModel):
    id: Text
    acceptable_chunk_ids: list[Text] = Field(min_length=1)
    # 人工核对原文的摘录；每个替代片段都必须包含该摘录。
    source_quote: Text


class RetrievalCase(StrictModel):
    id: Text
    group_id: Text
    category: Literal["direct", "paraphrase", "near_concept", "multi_evidence", "partial", "unanswerable"]
    split: Literal["dev", "validation", "test"]
    turns: list[Text] = Field(min_length=1)
    answerability: Literal["answerable", "partial", "unanswerable"]
    expected_retrieval: Literal["required", "optional", "not_applicable"]
    evidence_groups: list[EvidenceGroup]
    required_facts: list[Text] = Field(default_factory=list)
    forbidden_claims: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_labels(self):
        if self.answerability == "unanswerable":
            if self.expected_retrieval != "not_applicable" or self.evidence_groups:
                raise ValueError("无答案用例必须 not_applicable 且无证据组")
        elif self.expected_retrieval == "required" and not self.evidence_groups:
            raise ValueError("required 有答案用例必须标注证据")
        if self.expected_retrieval == "not_applicable" and self.evidence_groups:
            raise ValueError("not_applicable 不能标注正向证据")
        group_ids = [g.id for g in self.evidence_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("证据组 ID 重复")
        for group in self.evidence_groups:
            if len(group.acceptable_chunk_ids) != len(set(group.acceptable_chunk_ids)):
                raise ValueError("同组替代片段 ID 重复")
        return self

    @property
    def eligible(self) -> bool:
        return self.answerability in {"answerable", "partial"} and bool(self.evidence_groups)


class Dataset(StrictModel):
    protocol_version: Literal["rag-evidence-v1"]
    knowledge_hash: Text
    cases: list[RetrievalCase]

    def validate_snapshot(self, snapshot: Snapshot) -> None:
        if self.knowledge_hash != snapshot.knowledge_hash:
            raise ValueError("知识哈希不匹配，需要复核标注")
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("用例 ID 重复")
        chunks = {c.chunk_id: c for c in snapshot.chunks}
        splits = {}
        for case in self.cases:
            if case.group_id in splits and splits[case.group_id] != case.split:
                raise ValueError("同源 group_id 不能跨 split")
            splits[case.group_id] = case.split
            # 结构保留多轮，但本批执行器拒绝，避免只跑首轮冒充多轮。
            if len(case.turns) != 1:
                raise ValueError("第一批仅支持单轮执行")
            for group in case.evidence_groups:
                for chunk_id in group.acceptable_chunk_ids:
                    if chunk_id not in chunks:
                        raise ValueError("未知或过期的标注证据 ID")
                    if group.source_quote not in chunks[chunk_id].content:
                        raise ValueError("证据摘录不在标注片段原文中")

    @property
    def dataset_hash(self) -> str:
        data = self.model_dump()
        data["cases"] = sorted(data["cases"], key=lambda c: c["id"])
        return canonical_hash(data)


class ReturnedChunk(StrictModel):
    chunk_id: str | None = None
    title: str | None = None
    content: str | None = None
    fallback: bool = False


ErrorCode = Literal["request_error", "invalid_response", "snapshot_mismatch", "fixture_error"]


class Execution(StrictModel):
    case_id: Text
    status: Literal["success", "error"] = "success"
    results: list[ReturnedChunk] = Field(default_factory=list)
    elapsed_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    error_code: ErrorCode | None = None
    cached: bool | None = None
    reranked: bool | None = None

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "error") != (self.error_code is not None):
            raise ValueError("error 状态必须有错误码，success 不能有错误码")
        if self.status == "error" and self.results:
            raise ValueError("错误执行不能携带可评分结果")
        return self


class Fixture(StrictModel):
    protocol_version: Literal["rag-fixture-v1"]
    knowledge_hash: Text
    dataset_hash: Text
    top_k: int = Field(ge=1, le=20)
    description: Text
    executions: list[Execution]

    def validate_inputs(self, dataset: Dataset, top_k: int):
        if (self.knowledge_hash != dataset.knowledge_hash or self.dataset_hash != dataset.dataset_hash
                or self.top_k != top_k):
            raise ValueError("fixture 的知识/数据集哈希或 Top-K 不匹配")
        ids = [e.case_id for e in self.executions]
        if len(ids) != len(set(ids)) or set(ids) != {c.id for c in dataset.cases}:
            raise ValueError("fixture 必须为每条用例提供且只提供一次执行记录")


class Scores(StrictModel):
    hit_at_k: float
    evidence_recall_at_k: float
    mrr_at_k: float
    covered_groups: list[str]


class CaseReport(StrictModel):
    case: RetrievalCase
    execution: Execution
    eligible: bool
    scores: Scores | None


class MetricSummary(StrictModel):
    count: int
    hit_at_k: float | None
    evidence_recall_at_k: float | None
    mrr_at_k: float | None


class RetrievalReport(StrictModel):
    protocol_version: Literal["rag-evidence-v1"] = PROTOCOL
    run_id: Text
    mode: Literal["fixture", "online"]
    scope: Literal["default_search_pipeline"] = "default_search_pipeline"
    knowledge_hash: Text
    dataset_hash: Text
    top_k: int
    started_at: Text
    finished_at: Text
    code: dict[str, str | bool | None]
    configuration: dict[str, str | bool | None]
    total: int
    errors: int
    error_rate: float | None
    main: MetricSummary
    successful: MetricSummary
    results: list[CaseReport]


def checked_execution(execution: Execution, snapshot: Snapshot) -> Execution:
    """丢弃私有/未知正文和 fallback 文案，仅保留已核对公开快照的结果。"""
    if execution.status == "error":
        return execution
    chunks = {c.chunk_id: c for c in snapshot.chunks}
    safe = []
    for result in execution.results:
        if result.fallback:
            safe.append(ReturnedChunk(fallback=True))
            continue
        chunk = chunks.get(result.chunk_id)
        if chunk is None or result.content != chunk.content or result.title != chunk.title:
            return Execution(case_id=execution.case_id, status="error", error_code="snapshot_mismatch",
                             elapsed_ms=execution.elapsed_ms, cached=execution.cached, reranked=execution.reranked)
        safe.append(ReturnedChunk(chunk_id=chunk.chunk_id, title=chunk.title, content=chunk.content))
    return execution.model_copy(update={"results": safe})


def score_evidence(case: RetrievalCase, execution: Execution, top_k: int) -> Scores | None:
    if not 1 <= top_k <= 20:
        raise ValueError("Top-K 必须在 1–20 之间")
    if not case.eligible:
        return None
    covered, first_rank, seen = set(), None, set()
    if execution.status == "success":
        for rank, result in enumerate(execution.results[:top_k], 1):
            if result.fallback or not result.chunk_id or result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            for group in case.evidence_groups:
                if result.chunk_id in group.acceptable_chunk_ids:
                    covered.add(group.id)
                    if first_rank is None:
                        first_rank = rank
    return Scores(hit_at_k=float(bool(covered)), evidence_recall_at_k=len(covered) / len(case.evidence_groups),
                  mrr_at_k=1 / first_rank if first_rank else 0.0, covered_groups=sorted(covered))


def summarize(rows: list[CaseReport]) -> MetricSummary:
    scores = [row.scores for row in rows if row.scores is not None]
    return MetricSummary(count=len(scores), **{
        key: sum(getattr(s, key) for s in scores) / len(scores) if scores else None
        for key in ("hit_at_k", "evidence_recall_at_k", "mrr_at_k")})


def make_report(dataset: Dataset, snapshot: Snapshot, executions: list[Execution], *, top_k: int,
                run_id: str, mode: str, started_at: str, finished_at: str,
                code: dict, configuration: dict) -> RetrievalReport:
    dataset.validate_snapshot(snapshot)
    by_id = {e.case_id: e for e in executions}
    if len(by_id) != len(executions) or set(by_id) != {c.id for c in dataset.cases}:
        raise ValueError("执行记录和用例不对应")
    rows = []
    for case in dataset.cases:
        execution = checked_execution(by_id[case.id], snapshot)
        rows.append(CaseReport(case=case, execution=execution, eligible=case.eligible,
                               scores=score_evidence(case, execution, top_k)))
    errors = sum(row.execution.status == "error" for row in rows)
    return RetrievalReport(run_id=run_id, mode=mode, knowledge_hash=snapshot.knowledge_hash,
                           dataset_hash=dataset.dataset_hash, top_k=top_k, started_at=started_at,
                           finished_at=finished_at, code=code, configuration=configuration,
                           total=len(rows), errors=errors, error_rate=errors / len(rows) if rows else None,
                           main=summarize(rows),
                           successful=summarize([r for r in rows if r.execution.status == "success"]), results=rows)
