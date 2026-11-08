"""Run or replay the fixed offline Agent dev suite; never imports api.main."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.agent_evaluation import run_agents, replay_agent_run, load_suite

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    parser.add_argument('--replay', metavar='RUN_ID')
    parser.add_argument('--validate-annotations',action='store_true')
    args=parser.parse_args()
    if args.validate_annotations:
        data,snapshot,manifest=load_suite()
        print(json.dumps(dict(suite=data['suite'],cases=len(data['cases']),turns=sum(len(c['turns']) for c in data['cases']),knowledge_hash=snapshot.knowledge_hash),ensure_ascii=False)); return 0
    if args.replay and args.run_id: parser.error('--replay cannot combine with --run-id; replay is read-only')
    report=replay_agent_run(args.replay) if args.replay else asyncio.run(run_agents(run_id=args.run_id))
    print(json.dumps({key:report[key] for key in ('run_id','status','scope','mode','summary')},ensure_ascii=False,indent=2))
    # Completion is execution, not regression acceptance; unresolved judgements remain explicit.
    return 0 if report['status']=='completed' else 1
if __name__=='__main__':
    raise SystemExit(main())
