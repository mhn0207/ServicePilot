"""独立回答标注 sidecar；不改变已冻结的检索协议。"""
from pathlib import Path
from typing import Literal

from pydantic import Field

from evaluation.experiments import load_json, load_suite
from evaluation.retrieval import StrictModel, Text, canonical_hash

DEFAULT_SUITE = Path(__file__).parent / 'corpora/suite.v2.json'
DEFAULT_ANSWERS = Path(__file__).parent / 'corpora/answers.v4.json'


class Citation(StrictModel):
    chunk_id: Text
    quote: Text


class Fact(StrictModel):
    fact_id: Text
    text: Text
    evidence: list[Citation] = Field(min_length=1)


class Forbidden(StrictModel):
    forbidden_claim_id: Text
    conclusion: Text
    condition: Text
    critical: bool = True


class AnswerCase(StrictModel):
    case_id: Text
    question: Text
    group_id: Text
    split: Literal['dev', 'validation']
    retrieval_case_hash: Text
    answerability: Literal['answerable', 'partial', 'unanswerable']
    language: Literal['en'] = 'en'
    facts: list[Fact]
    forbidden: list[Forbidden]
    unknown: list[Text]
    handling: Text
    reference: Text
    notes: Text


class ScopedAnswerCase(AnswerCase):
    """v4显式保存可选补充；旧版case不增加默认字段，保持历史哈希和请求身份。"""
    optional_facts: list[Fact]
    scope_rationale: Text


class AnswerDataset(StrictModel):
    protocol_version: Literal['rag-answer-dataset-v1']
    annotation_version: Literal['answer-annotations-v1', 'answer-annotations-v2', 'answer-annotations-v3', 'answer-annotations-v4']
    knowledge_hash: Text
    source_dataset_hash: Text
    retrieval_freeze_hash: Text
    splits: dict
    cases: list[ScopedAnswerCase | AnswerCase]
    annotation_hash: Text
    freeze_hash: Text


def load_answers(path=DEFAULT_ANSWERS, suite=DEFAULT_SUITE, split='dev'):
    manifest, snapshot, source, selected = load_suite(suite, split)
    data = AnswerDataset.model_validate(load_json(path))
    payload = data.model_dump(exclude={'annotation_hash', 'freeze_hash'})
    if data.annotation_hash != canonical_hash(payload) or data.freeze_hash != canonical_hash(
            dict(annotation_hash=data.annotation_hash, version=data.annotation_version)):
        raise ValueError('answer freeze mismatch')
    if (data.knowledge_hash != snapshot.knowledge_hash or data.source_dataset_hash != source.dataset_hash
            or data.retrieval_freeze_hash != manifest['freeze_hash'] or data.splits != manifest['splits']):
        raise ValueError('answer source identity mismatch')
    ids = [c.case_id for c in data.cases]
    if len(ids) != len(set(ids)) or set(ids) != {c.id for c in source.cases}:
        raise ValueError('answer annotation case set mismatch')
    originals = {c.id: c for c in source.cases}
    chunks = {c.chunk_id: c for c in snapshot.chunks}
    for case in data.cases:
        if (data.annotation_version == 'answer-annotations-v4') != isinstance(case, ScopedAnswerCase):
            raise ValueError('answer scope schema/version mismatch')
        original = originals[case.case_id]
        if (case.question != original.turns[0] or case.group_id != original.group_id
                or case.split != original.split or case.answerability != original.answerability
                or case.retrieval_case_hash != canonical_hash(original.model_dump())):
            raise ValueError('answer case identity mismatch')
        optional = case.optional_facts if isinstance(case, ScopedAnswerCase) else []
        for rows, field in ((case.facts + optional, 'fact_id'), (case.forbidden, 'forbidden_claim_id')):
            item_ids = [getattr(r, field) for r in rows]
            if len(item_ids) != len(set(item_ids)):
                raise ValueError('duplicate annotation id')
        if bool(case.facts) != original.eligible or (case.answerability != 'answerable' and not case.unknown):
            raise ValueError('missing answerability annotations')
        allowed = {i for g in original.evidence_groups for i in g.acceptable_chunk_ids}
        for fact in case.facts + optional:
            for ref in fact.evidence:
                if ref.chunk_id not in allowed or ref.chunk_id not in chunks or ref.quote not in chunks[ref.chunk_id].content:
                    raise ValueError('invalid answer evidence')
    return data, manifest, snapshot, source, selected


def oracle_chunks(case, snapshot):
    """按 fact_id、chunk_id 排序，替代证据取最小 ID，跨事实稳定去重。"""
    by_id = {c.chunk_id: c for c in snapshot.chunks}
    ids = []
    for fact in sorted(case.facts, key=lambda f: f.fact_id):
        key = min(ref.chunk_id for ref in fact.evidence)
        if key not in ids:
            ids.append(key)
    return [by_id[key] for key in ids]
