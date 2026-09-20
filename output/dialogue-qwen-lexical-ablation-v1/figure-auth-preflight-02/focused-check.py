import copy,importlib.util,math
from pathlib import Path
from types import SimpleNamespace
p=Path('scripts/plot_dialogue_qwen_lexical_ablation.py');s=importlib.util.spec_from_file_location('p',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
base=Path('output/dialogue-qwen-lexical-ablation-v1/figure-preflight-01')
r=base/'synthetic-report';a=base/'synthetic-audit'
args=SimpleNamespace(summary=r/'summary.json',summary_sha256=m.digest(r/'summary.json'),receipt=r/'receipt.json',receipt_sha256=m.digest(r/'receipt.json'),audit=a/'receipt.json',audit_sha256=m.digest(a/'receipt.json'),synthetic=True)
summary,audited,_=m.authenticate(args)
reported=summary['continuation']; changed=copy.deepcopy(audited['continuation']);key='current/all/nll/row'
changed['checks'][key]['difference']=math.nextafter(changed['checks'][key]['difference'],math.inf)
m.same_decisions(reported,changed)
checks={'original16component_envelope':True,'roundoff_delta_accepted':True}
for damage in ('large_delta','pass_flag','threshold','key','count'):
 x=copy.deepcopy(changed)
 if damage=='large_delta':x['checks'][key]['difference']+=1e-5
 elif damage=='pass_flag':x['checks'][key]['passed']=not x['checks'][key]['passed']
 elif damage=='threshold':x['checks'][key]['threshold']=1e-15
 elif damage=='key':x['checks']['wrong']=x['checks'].pop(key)
 else:x['total_checks']=15
 try:m.same_decisions(reported,x)
 except ValueError:checks[damage+'_rejected']=True
 else:raise AssertionError(damage+' accepted')
print(checks)
