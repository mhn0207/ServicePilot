"""进程内轻量倒排 BM25 预筛索引；分词与权重与 RetrievalRanking 同源。"""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tooling.retrieval_ranking import RetrievalRanking


@dataclass(frozen=True)
class LexicalEntry:
    storage_id: str
    document_id: str
    version: str
    title: str
    chunk: int
    term_freqs: Dict[str, int]
    doc_len: int


def build_lexical_entry(
    storage_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> LexicalEntry:
    """使用 RetrievalRanking._search_terms 构建单条片段的词频索引项（标题 3 倍权重）。"""
    meta = metadata or {}
    title = str(meta.get("title") or "")
    document_id = str(meta.get("document_id") or "")
    version = str(meta.get("version") or "")
    raw_chunk = meta.get("chunk_index", 0)
    chunk = raw_chunk if type(raw_chunk) is int else 0
    terms = RetrievalRanking._search_terms(content or "") + RetrievalRanking._search_terms(title) * 3
    return LexicalEntry(
        storage_id=storage_id,
        document_id=document_id,
        version=version,
        title=title,
        chunk=chunk,
        term_freqs=dict(Counter(terms)),
        doc_len=len(terms),
    )


class LexicalIndex:
    """按 KnowledgeRevision 同步的内存倒排 BM25 索引，仅用于预筛候选 storage_id。"""

    def __init__(
        self,
        revision: str,
        postings: Dict[str, List[Tuple[str, int]]],
        docs: Dict[str, LexicalEntry],
        avg_dl: float,
    ):
        self.revision = revision
        self.postings = postings
        self.docs = docs
        self.avg_dl = avg_dl

    @classmethod
    def build(
        cls,
        revision: str,
        rows: Iterable[Tuple[str, str, Optional[Dict[str, Any]]]],
        *,
        previous: Optional["LexicalIndex"] = None,
    ) -> "LexicalIndex":
        """从全量分页扫描结果构建倒排表；相同 (document_id, version, title, chunk) 复用旧词频。"""
        docs: Dict[str, LexicalEntry] = {}
        postings: Dict[str, List[Tuple[str, int]]] = {}
        total_len = 0
        prev_docs = previous.docs if previous is not None else {}

        for storage_id, content, meta in rows:
            meta_dict = meta or {}
            doc_id = str(meta_dict.get("document_id") or "")
            version = str(meta_dict.get("version") or "")
            title = str(meta_dict.get("title") or "")
            raw_chunk = meta_dict.get("chunk_index", 0)
            chunk = raw_chunk if type(raw_chunk) is int else 0

            prev = prev_docs.get(storage_id)
            if (
                prev is not None
                and doc_id
                and version
                and prev.document_id == doc_id
                and prev.version == version
                and prev.title == title
                and prev.chunk == chunk
            ):
                entry = prev
            else:
                entry = build_lexical_entry(storage_id, content, meta_dict)

            docs[storage_id] = entry
            total_len += entry.doc_len
            for term, tf in entry.term_freqs.items():
                postings.setdefault(term, []).append((storage_id, tf))

        avg_dl = (total_len / len(docs)) if docs else 1.0
        return cls(revision=revision, postings=postings, docs=docs, avg_dl=avg_dl)

    def query_candidates(self, query: str, limit: int = 200) -> List[str]:
        """按全库 BM25 预筛 Top-N storage_id；无命中词项时返回空列表。"""
        if limit <= 0 or not self.docs:
            return []
        terms = set(RetrievalRanking._search_terms(query))
        if not terms:
            return []

        n_docs = len(self.docs)
        avg_dl = self.avg_dl or 1.0
        scores: Dict[str, float] = {}
        for term in terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for storage_id, tf in posting:
                dl = self.docs[storage_id].doc_len
                scores[storage_id] = scores.get(storage_id, 0.0) + (
                    idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avg_dl))
                )

        if not scores:
            return []
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.docs[item[0]].title, self.docs[item[0]].chunk, item[0]),
        )
        return [storage_id for storage_id, _ in ranked[:limit]]
