"""Rebuild or verify the frozen public_zh_expanded.v1 retrieval suite."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.experiments import load_json, load_suite
from evaluation.retrieval import Dataset, Snapshot, build_snapshot, canonical_hash
from tooling.knowledge_chunks import content_version, document_identity

CORPORA = ROOT / "evaluation/corpora"
PUBLIC = CORPORA / "public_zh.v1"
BACKGROUND = CORPORA / "service_zh.v1"
EXPANDED = CORPORA / "public_zh_expanded.v1"


def build_files():
    parent, public_snapshot, public_cases, _ = load_suite(PUBLIC / "suite.json")
    public_corpus = load_json(PUBLIC / "corpus.json")
    public_sources = load_json(PUBLIC / "sources.json")
    if canonical_hash(public_sources) != parent["sources_hash"]:
        raise ValueError("Public source manifest differs from the parent suite")

    background_corpus = load_json(BACKGROUND / "corpus.json")
    background_sources = load_json(BACKGROUND / "sources.json")
    background_snapshot = build_snapshot(background_corpus)
    if background_snapshot != Snapshot.model_validate(load_json(BACKGROUND / "snapshot.json")):
        raise ValueError("Synthetic background corpus differs from its frozen snapshot")

    distractors = load_json(EXPANDED / "distractors.json")
    authored_docs = [
        {
            "document_id": f"public-zh-expanded:{row['id']}",
            "title": f"{row['scope']} · {row['title']}",
            "content": f"【合成评测规则】适用范围：{row['scope']}。{row['rule']}",
        }
        for row in distractors
    ]
    corpus = public_corpus + background_corpus + authored_docs
    snapshot = build_snapshot(corpus)
    dataset = Dataset.model_validate(
        public_cases.model_dump() | {"knowledge_hash": snapshot.knowledge_hash}
    )
    dataset.validate_snapshot(snapshot)

    public_ids = {chunk.chunk_id for chunk in public_snapshot.chunks}
    background_ids = {chunk.chunk_id for chunk in background_snapshot.chunks}
    inherited_ids = public_ids | background_ids
    chunks = {chunk.chunk_id: chunk for chunk in snapshot.chunks}
    if public_ids & background_ids or not inherited_ids <= chunks.keys():
        raise ValueError("Parent and background chunks must be preserved and disjoint")

    authored_chunks = {
        chunk.document_id: chunk for chunk in snapshot.chunks
        if chunk.chunk_id not in inherited_ids
    }
    if len(authored_chunks) != len(authored_docs) or len(chunks) != len(inherited_ids) + len(authored_docs):
        raise ValueError("Each authored distractor must produce exactly one chunk")
    positive_ids = {
        case.id: {chunk_id for group in case.evidence_groups for chunk_id in group.acceptable_chunk_ids}
        for case in dataset.cases if case.eligible
    }
    negatives = []
    for doc, row in zip(authored_docs, distractors, strict=True):
        chunk = authored_chunks[content_version(document_identity(doc["title"], doc["document_id"]))]
        case_ids = row["cases"]
        if not case_ids or len(case_ids) != len(set(case_ids)) or not row["reason"].strip():
            raise ValueError(f"Missing or duplicate negative annotations: {row['id']}")
        if any(case_id not in positive_ids or chunk.chunk_id in positive_ids[case_id] for case_id in case_ids):
            raise ValueError(f"Invalid hard-negative target: {row['id']}")
        negatives.append({
            "document_id": doc["document_id"],
            "chunk_id": chunk.chunk_id,
            "case_ids": case_ids,
            "scope": row["scope"],
            "reason": row["reason"],
        })
    if {case_id for row in negatives for case_id in row["case_ids"]} != positive_ids.keys():
        raise ValueError("Every positive case must have a scoped distractor")

    review_status = "AI_AUTHORED_SCOPE_REVIEW_PENDING_INDEPENDENT_HUMAN_REVIEW"
    negative_manifest = {
        "review_status": review_status,
        "hard_negatives": negatives,
    }
    sources = {
        "version": "public-zh-expanded-sources-v1",
        "public_parent": {
            "path": "../public_zh.v1/sources.json",
            "sources_hash": canonical_hash(public_sources),
            "freeze_hash": parent["freeze_hash"],
            "documents": [doc["document_id"] for doc in public_corpus],
        },
        "synthetic_background": {
            "path": "../service_zh.v1/sources.json",
            "sources_hash": canonical_hash(background_sources),
            "knowledge_hash": background_snapshot.knowledge_hash,
            "documents": [doc["document_id"] for doc in background_corpus],
        },
        "synthetic_distractors": {
            "path": "distractors.json",
            "sources_hash": canonical_hash(distractors),
            "documents": [doc["document_id"] for doc in authored_docs],
            "declaration": "墨桥与澄屿规则为本套件自编的虚构资料，适用范围随片段保留。",
        },
    }
    splits = {}
    for name in parent["splits"]:
        cases = [case for case in dataset.cases if case.split == name]
        splits[name] = {
            "case_ids": [case.id for case in cases],
            "groups": sorted({case.group_id for case in cases}),
            "evaluated_dataset_hash": dataset.model_copy(update={"cases": cases}).dataset_hash,
        }
    suite = {
        "protocol_version": "rag-suite-v2",
        "corpus": "corpus.json",
        "snapshot": "snapshot.json",
        "dataset": "cases.json",
        "sources": "sources.json",
        "hard_negatives": "hard-negatives.json",
        "knowledge_hash": snapshot.knowledge_hash,
        "source_dataset_hash": dataset.dataset_hash,
        "sources_hash": canonical_hash(sources),
        "hard_negatives_hash": canonical_hash(negative_manifest),
        "review_status": review_status,
        "grouping_rationale": parent["grouping_rationale"],
        "limitations": [
            "Same 24 public_zh.v1 questions and evidence labels. Both splits use the same expanded pool.",
            "Background rules and authored distractors are synthetic. Targeted negatives are authored against the visible question set.",
            "The author reviewed applicability against the frozen evidence. Independent human annotation review and current public source audit are pending.",
        ],
        "splits": splits,
    }
    suite["freeze_hash"] = canonical_hash(suite)
    return {
        "corpus.json": corpus,
        "snapshot.json": snapshot.model_dump(),
        "cases.json": dataset.model_dump(),
        "sources.json": sources,
        "hard-negatives.json": negative_manifest,
        "suite.json": suite,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate frozen JSON from the source inputs")
    args = parser.parse_args()
    files = build_files()
    for name, value in files.items():
        path = EXPANDED / name
        if args.write:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif not path.is_file() or load_json(path) != value:
            parser.exit(1, f"Frozen artifact differs from source inputs: {name}\n")
    load_suite(EXPANDED / "suite.json")
    print(f"{len(files['corpus.json'])} documents, {len(files['snapshot.json']['chunks'])} chunks, "
          f"{len(files['cases.json']['cases'])} cases; {files['suite.json']['freeze_hash']}")


if __name__ == "__main__":
    main()
