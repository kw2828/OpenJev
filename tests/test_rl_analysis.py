import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3", reason="Install the rl extra for RL analysis tests")

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
spec=importlib.util.spec_from_file_location('rl_analysis',Path(__file__).resolve().parents[1]/'research/analyze_rl_doom.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('line_kills,passed',[(11,True),(8,False)])
def test_kill_gain_cannot_rescue_shift_failure(tmp_path,line_kills,passed):
    rows=[]
    scenarios=['defend_the_center','defend_the_line']
    for scenario in scenarios:
        for seed in [1,2,3]:
            for arm in ['ppo_current','rules','rule_event']:
                for rep in range(3) if arm=='ppo_current' else [None]:
                    kills=(12 if scenario=='defend_the_center' else line_kills) if rep is not None else 10
                    rows.append({'scenario': scenario,'seed': seed,'arm': arm,'replicate': rep,'kills': kills,
                        'game_seconds': 20.,'utility': 5.,'net_utility': 4.99,'engine_reward': kills-1,
                        'firing_windows': 10,'capped': False,'decision_compute_seconds': .01,'steps': 100,'p95_decision_ms': .1})
    ep=tmp_path/'episodes.jsonl'
    ep.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    protocol={'confirmation_seeds': [1,2,3],'replicates': 3,'scenarios': scenarios,
                  'bootstrap_seed': 1,'bootstrap_draws': 100,'interval_quantiles': [.00625,.99375]}
    manifest={'status': 'completed_confirmation','stage': 'confirmation','protocol': protocol,
                  'selection': {'selected':'ppo_current'},'arms': ['ppo_current'],
                  'artifact_sha256': {'episodes.jsonl':hashlib.sha256(ep.read_bytes()).hexdigest()}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert module.analyze(tmp_path)['confirmation_passed'] is passed
    ep.write_text(ep.read_text()+'\n')
    with pytest.raises(ValueError,match='Changed artifact'):
        module.analyze(tmp_path)
