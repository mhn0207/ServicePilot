import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from evaluation.agent_scoring import score
from evaluation.answer_quality import check_grounding

def turn(cid, answer, chunks=None):
 return dict(case_id=cid,turn_id='1',answer=answer,roles=['billing'],events=[dict(query='q',status='success',shown_chunks=chunks or [])])
def label():
 data=json.loads((Path(__file__).parents[1]/'evaluation/corpora/agent.cases.v1.json').read_text(encoding='utf-8-sig'))
 return next(c['turns'][0] for c in data['cases'] if c['id']=='missed-required')
def test_fixed_judge_validity_does_not_replace_required_call():
 t=turn('missed-required','取消自动续费不等于退款。');t['events']=[]
 checks=score(label(),t)
 assert checks['required_call_coverage'] is False
 assert checks['required_facts'] is True
 assert checks['final_direct_grounding'] is False
 assert checks['judge_validity'] is True

def test_altered_answer_is_unresolved_not_fixed_fixture_success():
 t=turn('subscription','已退款成功\n保证明日到账')
 checks=score(label(),t)
 assert checks['judge_validity'] is False
 assert checks['required_facts'] is None
 assert t['judge_error']=='fixture_request_mismatch'

def test_unpunctuated_line_cannot_be_omitted_by_judge():
 text='取消不等于退款\n保证明日到账'
 value={'extraction':'complete','claims':[{'claim_id':'one','unit_ids':[0],'kind':'factual','status':'unsupported','answer_span':{'start':0,'end':7,'quote':text[:7]},'evidence':[],'rationale':'fixed'}]}
 with pytest.raises(ValueError):check_grounding(value,text,[])
