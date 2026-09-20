"""Artificial plot fixture only. No model, study data or prediction arrays."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('plotter', ROOT/'scripts/plot_dialogue_objective.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
source = HERE/'synthetic-report'; source.mkdir()


def write(name, value):
    with (source/name).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


def cell(n, correct, offset, loss):
    error=n-correct; branch=error*3//4; value=error-branch
    vals={'accuracy':correct/n,'error':error/n,'wrong_selected_branch':branch/n,'wrong_value':value/n,
          'nll':loss,'brier':min(2.,loss*.4)}
    return {'rows':n,'counts':{'correct':correct,'error':error,'wrong_selected_branch':branch,'wrong_value':value},
            'metrics':{m:{'row':v,'equal_service':max(0.,min(1.,v+offset)) if m in ('accuracy','error','wrong_selected_branch','wrong_value') else v*(1+offset),
                          'equal_dialogue':v} for m,v in vals.items()},'rare':{},'exact_tie_rows':0}


def averaged(tree):
    if isinstance(tree[0],dict):return {k:averaged([v[k] for v in tree]) for k in tree[0]}
    return math.fsum(tree)/len(tree)


fresh,historic={},{}
for i,cat in enumerate(p.CATEGORIES):
    for j,seed in enumerate(p.SEEDS):
        changed=round(578*(.65+.022*i+.009*(j-1)))
        ucorrect=4032-round(4032*(.015+.002*i+.001*(j-1)))
        acorrect=3209-round(3209*(.026+.003*i+.002*(j-1)))
        corrects={'changed':changed,'unmentioned_retention':ucorrect,'assigned_retention':acorrect,
                  'retained':ucorrect+acorrect,'all':changed+ucorrect+acorrect}
        cells={}
        for panel in ('all','seen_service','heldout_service'):
            for s,n in p.SUPPORT.items():
                cells[panel+'/'+s]=cell(n,corrects[s],.003*(j-1),(.32-.012*i+.008*(j-1))*(2 if s=='changed' else 1))
        f={'seed':seed,'synthetic':True,'cells':cells,'services':{}}
        (fresh if cat in p.FRESH else historic)[f'{cat}-{seed}']=f
checks={name:{'passed':i not in (2,6,10)} for i,name in enumerate(sorted(p.CHECKS))}
objective=all(v['passed'] for k,v in checks.items() if k.startswith('objective_'))
practical=all(v['passed'] for k,v in checks.items() if k.startswith('practical_'))
rule={'passed':objective and practical,'objective_passed':objective,'practical_passed':practical,
      'checks_passed':sum(v['passed'] for v in checks.values()),'total_checks':13,'checks':checks}
summary={'version':p.REPORT_VERSION,'status':'completed','technical_validity_passed':True,'synthetic':True,
         'scope':'SYNTHETIC PLOTTING FIXTURE ONLY. Metrics, costs and decisions are invented.',
         'execution_completed_sha256':'1'*64,'plan_sha256':'2'*64,'fits':fresh,
         'means':{cat:{'cells':averaged([fresh[f'{cat}-{s}']['cells'] for s in p.SEEDS]),'services':{}} for cat in p.FRESH},
         'historical':{'fits':historic,'means':{cat:{'cells':averaged([historic[f'{cat}-{s}']['cells'] for s in p.SEEDS]),'services':{}} for cat in p.HISTORICAL}},
         'continuation':rule,'costs':{'whole_wall_seconds':4700.,'process_lifetime_peak_rss_bytes':2*1024**3,
         'per_fit':{f'{m}-{s}':{'training_wall_seconds':710.,'evaluation_wall_seconds':30.,'wall_seconds':750.}
                    for m in ('stratum','uniform') for s in p.SEEDS}}}
write('summary.json',summary);write('started.json',{'synthetic':True,'scope':'Artificial plotting fixture'})
(source/'report.md').write_text('# SYNTHETIC ONLY\nNo real experiment, data, predictions or performance.\n')
receipt={'version':p.REPORT_VERSION,'status':'completed','technical_validity_passed':True,'synthetic':True,
         'execution_completed_sha256':'1'*64,'plan_sha256':'2'*64,'continuation_passed':rule['passed'],
         'files':{f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in source.iterdir()}}
write('receipt.json',receipt)
print(json.dumps({'summary_sha256':p.digest(source/'summary.json'),'receipt_sha256':p.digest(source/'receipt.json'),'run_sha256':'1'*64}))
