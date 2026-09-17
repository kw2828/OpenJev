import importlib
import json
from pathlib import Path

import pytest


def analysis_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'research'))
    return importlib.import_module('analyze_cadence_doom')


@pytest.mark.parametrize(('candidate_kills','expected'), [(5,True),(3,False)])
def test_confirmation_requires_utility_gain_and_kill_noninferiority(tmp_path, monkeypatch, candidate_kills, expected):
    module = analysis_module(monkeypatch)
    protocol = json.loads((Path(__file__).resolve().parents[1]/'research/protocols/cadence-doom-v1.json').read_text())
    protocol.update(confirmation_seeds=[81000,81001,81002], bootstrap_draws=100)
    rows = []
    arms = ['selected','rules','matched_no_rest','command_rest1','history_map']
    for scenario in protocol['confirmation_scenarios']:
        for seed in protocol['confirmation_seeds']:
            for arm in arms:
                for rep in (range(5) if arm == 'history_map' else [None]):
                    rows.append({'scenario': scenario,'seed': seed,'arm': arm,'replicate': rep,
                        'utility': 2 if arm=='selected' else 0, 'net_utility': 2 if arm=='selected' else 0,
                        'kills': candidate_kills if arm=='selected' else 5, 'engine_reward': 0,'ammo_used': 1,
                        'firing_windows': 1,'success_windows': 1,'game_seconds': 1,'decision_compute_seconds': .001,
                        'steps': 1,'truncated': False,'p95_decision_ms': .1})
    episodes = tmp_path/'episodes.jsonl'
    episodes.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest = {'status': 'completed_confirmation','protocol': protocol,'selection': {'selected':'ammo_rest1'},
                    'arms': {a:{} for a in arms if a!='history_map'},
                    'artifact_sha256': {'episodes.jsonl':module.digest(episodes)}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    result = module.analyze(tmp_path)
    assert result['confirmation_passed'] is expected
    for scenario in protocol['confirmation_scenarios']:
        assert result['primary'][scenario]['rules']['net_utility']['interval'] == [2,2]
    # A changed outcome cannot be silently analyzed under the old receipt.
    episodes.write_text(episodes.read_text()+'\n')
    with pytest.raises(ValueError, match='Changed artifact'):
        module.analyze(tmp_path)
