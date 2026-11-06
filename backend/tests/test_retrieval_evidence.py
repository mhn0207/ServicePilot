import asyncio
import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from evaluation.retrieval import (
    Dataset, EvidenceGroup, Execution, Fixture, RetrievalCase, RetrievalReport,
    ReturnedChunk, build_snapshot, checked_execution, make_report, score_evidence,
)
from scripts import evaluate_retrieval_evidence as cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("证据级测试不得使用真实 HTTP 网络")
    async def reject_async(*args, **kwargs):
        reject()
    # MockTransport 不经过真实传输层。不要封禁 socket.connect：Windows 的
    # asyncio 自身也用本地 socketpair 唤醒事件循环，会导致假失败。
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reject_async)


@pytest.fixture
def snapshot():
    return build_snapshot([{"title": "alpha", "content": "共同证据。甲的细节。"},
                           {"title": "beta", "content": "共同证据。乙的细节。"},
                           {"title": "other", "content": "无关资料。"}])


def chunk(snapshot, title):
    return next(c for c in snapshot.chunks if c.title == title)


def result(snapshot, title):
    c = chunk(snapshot, title)
    return ReturnedChunk(chunk_id=c.chunk_id, title=c.title, content=c.content)


def case(snapshot, *, alternatives=False, same_chunk=False):
    a, b = chunk(snapshot, "alpha"), chunk(snapshot, "beta")
    return RetrievalCase(id="c", group_id="g", category="multi_evidence", split="dev", turns=["问题"],
                         answerability="answerable", expected_retrieval="required", evidence_groups=[
                             EvidenceGroup(id="a", acceptable_chunk_ids=[a.chunk_id, b.chunk_id] if alternatives else [a.chunk_id], source_quote="共同证据"),
                             EvidenceGroup(id="b", acceptable_chunk_ids=[a.chunk_id if same_chunk else b.chunk_id], source_quote="共同证据")])


def execution(snapshot, *titles, **kwargs):
    return Execution(case_id="c", results=[result(snapshot, t) for t in titles], **kwargs)


def report(snapshot, cases, executions, top_k=3):
    return make_report(Dataset(protocol_version="rag-evidence-v1", knowledge_hash=snapshot.knowledge_hash, cases=cases),
                       snapshot, executions, top_k=top_k, run_id="test", mode="fixture", started_at="start",
                       finished_at="end", code={}, configuration={})


def test_one_of_two_groups_and_alternatives(snapshot):
    scores = score_evidence(case(snapshot), execution(snapshot, "alpha"), 3)
    assert (scores.hit_at_k, scores.evidence_recall_at_k, scores.mrr_at_k) == (1, .5, 1)
    alternate = case(snapshot, alternatives=True)
    alternate.evidence_groups = alternate.evidence_groups[:1]
    scores = score_evidence(alternate, execution(snapshot, "alpha", "beta"), 3)
    assert scores.covered_groups == ["a"] and scores.evidence_recall_at_k == 1


def test_one_chunk_can_cover_multiple_groups(snapshot):
    assert score_evidence(case(snapshot, same_chunk=True), execution(snapshot, "alpha"), 3).evidence_recall_at_k == 1


def test_third_rank_and_duplicate_positions_are_preserved(snapshot):
    scores = score_evidence(case(snapshot), execution(snapshot, "other", "other", "alpha"), 3)
    assert scores.mrr_at_k == 1 / 3 and scores.evidence_recall_at_k == .5
    assert score_evidence(case(snapshot), execution(snapshot, "other", "other", "alpha"), 2).hit_at_k == 0
    scores = score_evidence(case(snapshot), execution(snapshot, "alpha", "alpha"), 3)
    assert scores.evidence_recall_at_k == .5


@pytest.mark.parametrize("titles", [[], ["other"]])
def test_empty_or_irrelevant_results_are_misses(snapshot, titles):
    assert score_evidence(case(snapshot), execution(snapshot, *titles), 3).hit_at_k == 0


def test_fallback_does_not_count_or_expose_text(snapshot):
    row = Execution(case_id="c", results=[ReturnedChunk(chunk_id=chunk(snapshot, "alpha").chunk_id,
                                                     content="private fallback", fallback=True)])
    checked = checked_execution(row, snapshot)
    assert score_evidence(case(snapshot), checked, 3).hit_at_k == 0
    assert checked.results == [ReturnedChunk(fallback=True)]


@pytest.mark.parametrize("change", ["unknown_id", "old_id", "wrong_text", "wrong_title", "missing_id"])
def test_unknown_or_forged_results_rejected_and_private_text_removed(snapshot, change):
    item = result(snapshot, "alpha")
    if change == "unknown_id":
        item.chunk_id = "unknown"
    elif change == "old_id":
        item.chunk_id = chunk(snapshot, "alpha").document_id + ":0"
    elif change == "wrong_text":
        item.content = "private unrelated content"
    elif change == "wrong_title":
        item.title = "wrong"
    else:
        item.chunk_id = None
    r = report(snapshot, [case(snapshot)], [Execution(case_id="c", results=[item])])
    assert r.errors == 1 and r.main.hit_at_k == 0
    assert r.results[0].execution.error_code == "snapshot_mismatch"
    assert "private unrelated" not in r.model_dump_json()


def test_correct_title_wrong_chunk_cannot_pass():
    snap = build_snapshot([{"document_id": "a", "title": "same", "content": "证据"},
                           {"document_id": "b", "title": "same", "content": "无关"}])
    good = next(c for c in snap.chunks if c.content == "证据")
    bad = next(c for c in snap.chunks if c.content == "无关")
    c = RetrievalCase(id="c", group_id="g", category="direct", split="dev", turns=["问题"],
                      answerability="answerable", expected_retrieval="required",
                      evidence_groups=[EvidenceGroup(id="e", acceptable_chunk_ids=[good.chunk_id], source_quote="证据")])
    r = report(snap, [c], [Execution(case_id="c", results=[ReturnedChunk(chunk_id=bad.chunk_id, title=bad.title, content=bad.content)])])
    assert r.errors == 0 and r.main.hit_at_k == 0


def test_errors_stay_in_main_denominator_and_success_subset(snapshot):
    a = case(snapshot)
    b = a.model_copy(update={"id": "b"})
    r = report(snapshot, [a, b], [execution(snapshot, "alpha"), Execution(case_id="b", status="error", error_code="request_error")])
    assert r.error_rate == .5 and r.main.count == 2
    assert r.main.hit_at_k == .5 and r.main.evidence_recall_at_k == .25
    assert r.successful.count == 1 and r.successful.hit_at_k == 1


def test_unanswerable_and_empty_denominators_are_null(snapshot):
    c = case(snapshot).model_dump()
    c.update(answerability="unanswerable", expected_retrieval="not_applicable", evidence_groups=[])
    r = report(snapshot, [RetrievalCase.model_validate(c)], [execution(snapshot, "alpha")])
    assert r.main.count == 0 and r.main.hit_at_k is None and r.results[0].scores is None
    empty = report(snapshot, [], [])
    assert empty.error_rate is None and empty.main.mrr_at_k is None and empty.successful.count == 0


def test_optional_and_partial_with_annotations_are_eligible(snapshot):
    c = case(snapshot).model_dump()
    c.update(expected_retrieval="optional", answerability="partial")
    assert RetrievalCase.model_validate(c).eligible


@pytest.mark.parametrize("mutation", ["required_empty", "unanswerable_evidence", "duplicate_groups", "string_turns", "extra_field"])
def test_case_schema_rejects_invalid_labels(snapshot, mutation):
    raw = case(snapshot).model_dump()
    if mutation == "required_empty":
        raw['evidence_groups'] = []
    elif mutation == "unanswerable_evidence":
        raw['answerability'] = 'unanswerable'
    elif mutation == "duplicate_groups":
        raw['evidence_groups'][1]['id'] = 'a'
    elif mutation == "string_turns":
        raw['turns'] = 'question'
    else:
        raw['typo'] = 'value'
    with pytest.raises(ValueError):
        RetrievalCase.model_validate(raw)


@pytest.mark.parametrize("mutation", ["hash", "id", "quote", "multiturn", "duplicate_case", "split"])
def test_dataset_validation(snapshot, mutation):
    d = Dataset(protocol_version="rag-evidence-v1", knowledge_hash=snapshot.knowledge_hash, cases=[case(snapshot)])
    if mutation == 'hash':
        d.knowledge_hash = 'wrong'
    elif mutation == 'id':
        d.cases[0].evidence_groups[0].acceptable_chunk_ids = ['old']
    elif mutation == 'quote':
        d.cases[0].evidence_groups[0].source_quote = 'invented'
    elif mutation == 'multiturn':
        d.cases[0].turns.append('second')
    elif mutation == 'duplicate_case':
        d.cases.append(d.cases[0])
    else:
        d.cases.append(d.cases[0].model_copy(update={'id': 'b', 'split': 'validation'}))
    with pytest.raises(ValueError):
        d.validate_snapshot(snapshot)


def test_snapshot_and_ids_repeat_across_rebuilds_and_change_with_content():
    docs = [{"title": "a", "content": "old"}, {"title": "b", "content": "same"}]
    a = build_snapshot(docs)
    assert a == build_snapshot(list(reversed(docs)))
    b = build_snapshot([dict(docs[0], content="new"), docs[1]])
    assert a.knowledge_hash != b.knowledge_hash
    assert chunk(a, 'a').document_id == chunk(b, 'a').document_id
    assert chunk(a, 'a').chunk_id != chunk(b, 'a').chunk_id
    with pytest.raises(ValueError, match="重复"):
        build_snapshot([docs[0], docs[0]])


def test_shipped_annotations_and_fixture_are_valid_and_legacy_preserved():
    snap = build_snapshot(cli.load_json(ROOT / 'data/demo_docs/sample_knowledge.json'))
    d = Dataset.model_validate(cli.load_json(ROOT / 'evaluation/rag_evidence_cases.v1.json'))
    d.validate_snapshot(snap)
    f = Fixture.model_validate(cli.load_json(ROOT / 'evaluation/fixtures/retrieval_results.v1.json'))
    f.validate_inputs(d, 3)
    assert len(d.cases) == 24 and len(snap.chunks) == 6
    assert len(cli.load_json(ROOT / 'evaluation/rag_cases.json')) == 12
    assert {c.category for c in d.cases} == {'direct', 'paraphrase', 'near_concept', 'multi_evidence', 'partial', 'unanswerable'}


def test_cli_reports_replay_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(cli, 'code_identity', lambda: {'head': 'test', 'worktree_dirty': True})
    assert cli.main(['--run-id', 'first']) == 1  # fixture 有一次有意模拟的错误
    saved = tmp_path / 'first/report.json'
    original = saved.read_bytes()
    r = RetrievalReport.model_validate_json(original)
    assert r.total == 24 and r.main.count == 22 and r.errors == 1
    assert r.main.hit_at_k == 6 / 22 and r.main.evidence_recall_at_k == 5.5 / 22
    assert r.main.mrr_at_k == pytest.approx((5 + 1 / 3) / 22)
    assert r.results[16].scores.evidence_recall_at_k == .5
    assert cli.main(['--run-id', 'first']) == 2
    assert saved.read_bytes() == original
    assert cli.main(['--run-id', 'replay', '--fixture', str(tmp_path / 'first/executions.json')]) == 1
    replay = RetrievalReport.model_validate_json((tmp_path / 'replay/report.json').read_bytes())
    assert replay.main == r.main and replay.successful == r.successful


def test_snapshot_export_does_not_require_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'OUTPUT_ROOT', tmp_path)
    assert cli.main(['--export-snapshot', '--cases', 'does-not-exist.json', '--run-id', 'snapshot']) == 0
    exported = cli.load_json(tmp_path / 'snapshot/snapshot.json')
    assert len(exported['chunks']) == 6
    assert not (tmp_path / 'snapshot/report.json').exists()


@pytest.mark.parametrize('mutation', ['unknown_evidence', 'empty_required', 'bad_fixture', 'unknown_version'])
def test_cli_validation_errors_nonzero(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(cli, 'OUTPUT_ROOT', tmp_path / 'runs')
    d = cli.load_json(ROOT / 'evaluation/rag_evidence_cases.v1.json')
    if mutation == 'unknown_evidence':
        d['cases'][0]['evidence_groups'][0]['acceptable_chunk_ids'] = ['unknown']
    elif mutation == 'empty_required':
        d['cases'][0]['evidence_groups'] = []
    elif mutation == 'unknown_version':
        d['protocol_version'] = 'v999'
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(d), encoding='utf-8')
    args = ['--cases', str(path)]
    if mutation == 'bad_fixture':
        f = cli.load_json(ROOT / 'evaluation/fixtures/retrieval_results.v1.json')
        f['executions'].pop()
        fixture_path = tmp_path / 'fixture.json'
        fixture_path.write_text(json.dumps(f), encoding='utf-8')
        args += ['--fixture', str(fixture_path)]
    assert cli.main(args) == 2


def test_online_auth_and_error_reporting_with_mock_transport(snapshot, monkeypatch):
    a = case(snapshot)
    b = a.model_copy(update={'id': 'b', 'turns': ['fails']})
    d = Dataset(protocol_version='rag-evidence-v1', knowledge_hash=snapshot.knowledge_hash, cases=[a, b])
    def handler(request):
        assert request.headers['authorization'] == 'Bearer secret-test-token'
        assert request.url.path == '/search' and request.url.params['top_k'] == '3'
        if request.url.params['query'] == 'fails':
            return httpx.Response(503, json={'detail': 'private backend failure'})
        return httpx.Response(200, json={'results': [result(snapshot, 'alpha').model_dump()], 'cached': False, 'reranked': True})
    client = httpx.AsyncClient
    monkeypatch.setenv('ZHIYING_EVAL_TOKEN', 'secret-test-token')
    monkeypatch.setattr(cli.httpx, 'AsyncClient', lambda **kw: client(**kw, transport=httpx.MockTransport(handler)))
    rows = asyncio.run(cli.request_executions('http://test', d, 3))
    r = report(snapshot, [a, b], rows)
    assert r.main.hit_at_k == .5 and rows[1].error_code == 'request_error'
    assert rows[0].reranked and not rows[0].cached
    assert 'secret-test-token' not in r.model_dump_json() and 'private backend' not in r.model_dump_json()


@pytest.mark.parametrize('payload', [None, {'results': 'bad'}, {'results': [42]}, {'results': [], 'cached': 'yes'}])
def test_invalid_online_response_is_execution_error(snapshot, monkeypatch, payload):
    d = Dataset(protocol_version='rag-evidence-v1', knowledge_hash=snapshot.knowledge_hash, cases=[case(snapshot)])
    client = httpx.AsyncClient
    monkeypatch.setattr(cli.httpx, 'AsyncClient', lambda **kw: client(**kw, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))))
    rows = asyncio.run(cli.request_executions('http://test', d, 3))
    assert rows[0].status == 'error' and rows[0].error_code == 'invalid_response'


def test_direct_script_runs_without_application_or_database_import(tmp_path):
    # 使用绝对脚本路径从临时目录运行，也验证没有依赖 backend 当前目录补 sys.path。
    proc = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/evaluate_retrieval_evidence.py'),
                           '--help'], cwd=tmp_path, capture_output=True, text=True, encoding='utf-8')
    assert proc.returncode == 0 and '--online' in proc.stdout


def test_offline_run_never_opens_env_database_or_online_statistics(tmp_path):
    # 独立进程审计整个默认离线执行，避免父测试已导入的 API/监控掩盖副作用。
    program = '''
import pathlib, sys
sys.path.insert(0, sys.argv[1])
def guard(event, args):
    if event == "socket.connect":
        raise AssertionError("offline socket connection")
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = str(args[0]).replace(chr(92), "/").lower()
        if name.endswith("/.env") or "/data/chroma/" in name or ".sqlite" in name:
            raise AssertionError("offline private data access")
sys.addaudithook(guard)
from scripts import evaluate_retrieval_evidence as cli
cli.OUTPUT_ROOT = pathlib.Path(sys.argv[2])
status = cli.main(["--run-id", "isolated"])
for name in ("api.main", "tooling.knowledge_base", "monitor.performance_monitor", "memory.conversation_memory"):
    assert name not in sys.modules, name
assert status == 1
'''
    proc = subprocess.run([sys.executable, '-X', 'utf8', '-c', program, str(ROOT), str(tmp_path / 'runs')],
                          cwd=tmp_path, capture_output=True, text=True, encoding='utf-8')
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / 'runs/isolated/report.json').exists()
