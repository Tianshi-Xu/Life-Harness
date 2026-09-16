from pathlib import Path
import json
R=Path(__file__).resolve().parent
for d in ('airline','retail','telecom','alfworld','dbbench','webshop'):
 p=R/d/'state.json';s=json.loads(p.read_text()) if p.exists() else {'status':'not_started','rounds':[]}
 print(f"{d:9} {s['status']:11} rounds={len(s['rounds'])} batch={s.get('batch',0)} incumbent={s.get('incumbent','baseline_zero.py')}")
 if s['rounds']:
  x=s['rounds'][-1];print(f"  r{x['iteration']}: train {x['baseline_score']:.3f}->{x['candidate_score']:.3f} accepted={x['accepted']} test={x['test_score']:.3f}")
 phases=sorted((R/d/'evals').glob('*'),key=lambda x:x.stat().st_mtime) if (R/d/'evals').exists() else []
 for q in phases[-2:]:
  summary=next((x for x in (q/'summary.json',q/'val.json',q/'test.json') if x.exists()),None)
  if summary:
   x=json.loads(summary.read_text());print(f"  {q.name}: n={len(x.get('per_task',{}))} score={x.get('accuracy')} infra={x.get('infra_errors','n/a')}")
