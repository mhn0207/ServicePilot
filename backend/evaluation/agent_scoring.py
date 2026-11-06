"""Fixed, authored Agent Judge verdicts through batch-3 validators; never a semantic model."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from evaluation.answer_quality import check_correctness, check_grounding, answer_units
from evaluation.retrieval import canonical_hash

JUDGE_PATH = Path(__file__).parent/'fixtures/agent.judge.v1.json'

def judge_identity():
    return canonical_hash(json.loads(JUDGE_PATH.read_text(encoding='utf-8')))

def score(label, turn):
    events = turn['events']
    queries = [event['query'] for event in events]
    chunks = [chunk for event in events for chunk in event.get('shown_chunks', [])]
    shown = '\n'.join(chunk['content'] for chunk in chunks)
    answer = turn['answer']
    key = turn['case_id'] + ':' + turn['turn_id']
    fixture = json.loads(JUDGE_PATH.read_text(encoding='utf-8'))['rows'].get(key)
    checks = dict(
        required_call_coverage=bool(events) if label['expected_retrieval']=='required' else None,
        execution_success=all(event['status']=='success' for event in events) if events else None,
        query_entities=all(all(entity in query for entity in label['required_entities']) for query in queries) if events and label['required_entities'] else None,
        old_entity_clean=all('11111' not in query for query in queries) and '11111' not in answer if '22222' in label['required_entities'] else None,
        returned_evidence=all(quote in shown for quote in label['evidence_quotes']) if label['evidence_quotes'] else None,
        role_coverage=set(label['expected_roles']).issubset(turn['roles']),
        required_facts=None, forbidden_clear=None, judge_validity=False,
        judge_status='error', final_direct_grounding=None,
        final_grounding_scope='indirect role answers only' if len(turn['roles'])>1 else 'actual shown tool evidence',
    )
    turn['judge_identity'] = judge_identity()
    turn['judge_scope'] = 'fixture verdict; no real semantic Judge or human calibration'
    if not fixture or fixture['answer'] != answer or fixture.get('label_hash') != canonical_hash(label):
        turn['judge_error'] = 'fixture_request_mismatch'
        return checks
    case = NS(facts=[NS(fact_id=str(i)) for i in range(len(label['required_facts']))],
              forbidden=[NS(forbidden_claim_id=str(i)) for i in range(len(label['forbidden_claims']))])
    span = dict(start=0,end=len(answer),quote=answer) if answer else None
    value = dict(facts=[dict(fact_id=str(i),status=status,answer_span=span if status in ('covered','contradicted') else None) for i,status in enumerate(fixture['fact_statuses'])],
                 forbidden=[dict(forbidden_claim_id=str(i),status=status,answer_span=span if status=='violation' else None) for i,status in enumerate(fixture['forbidden_statuses'])],
                 handling=dict(status=fixture['handling'],rationale='Authored fixed response verdict; no calibrated model claim'))
    try:
        correctness = check_correctness(value,case,answer)
    except ValueError:
        turn['judge_error'] = 'fixture_validation_error'
        return checks
    turn['correctness'] = correctness.model_dump()
    checks.update(required_facts=all(f.status=='covered' for f in correctness.facts) if correctness.facts else None,
                  forbidden_clear=all(f.status=='no_violation' for f in correctness.forbidden),
                  handling=correctness.handling.status=='pass',judge_validity=True,judge_status='valid_fixture')
    # Final Composer's inputs only contain role answers; assess retention above, direct grounding N/A.
    def grounded(text, context):
        evidence=[]
        for quote in fixture['evidence_quotes']:
            match=next((chunk for chunk in context if quote in chunk['content']),None)
            if match: evidence.append(dict(chunk_id=match['chunk_id'],quote=quote))
        factual=not fixture['non_factual']
        status='supported' if evidence and len(evidence)==len(fixture['evidence_quotes']) else 'unsupported'
        claims=[dict(claim_id=str(unit['unit_id']),unit_ids=[unit['unit_id']],kind='factual' if factual else 'non_factual',
                     status=status if factual else 'not_applicable',answer_span=dict(start=unit['start'],end=unit['end'],quote=unit['text']),
                     evidence=evidence,rationale='Fixed authored claim checked against actual role-visible context only') for unit in answer_units(text)]
        return check_grounding(dict(extraction='complete',claims=claims),text,context).model_dump()
    if len(turn['roles']) == 1:
        turn['grounding']=grounded(answer,chunks)
        applicable=[claim for claim in turn['grounding']['claims'] if claim['kind']=='factual']
        checks['final_direct_grounding']=all(c['status']=='supported' for c in applicable) if applicable else None
    else:
        turn['grounding']=None
        turn['role_grounding']=[]
        for role_answer in turn.get('role_answers',[]):
            role=role_answer['role']; text=role_answer['answer']
            context=[chunk for event in events if event.get('role')==role for chunk in event.get('shown_chunks',[])]
            if text==fixture['answer']:
                turn['role_grounding'].append(dict(role=role,result=grounded(text,context)))
            else:
                turn['role_grounding'].append(dict(role=role,status='fixture_request_mismatch'))
    return checks
