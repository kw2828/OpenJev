"""Frozen, bounded real-ViZDoom development pilot. No paid/model API calls."""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import vizdoom as vzd

from openjev.domain import hard_decision, teacher_action
from openjev.game import Doom
from openjev.policies import LocalPolicy
from openjev.research.bayesian import BayesianLogistic

ROOT = Path(__file__).resolve().parents[1]


def features(obs):
    return [1., float(obs.visible), min(abs(obs.aim_error), 1.), obs.half_width,
            min(obs.distance/1000, 2.), obs.health/100, min(obs.ammo/30, 2.)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_episode(scenario, seed, mode, protocol, model=None, rng=None, trace=None, replicate=None):
    unit = f'{scenario}:{seed}'
    tiny = LocalPolicy() if mode == 'tiny_imitation' else None
    decisions, records, latencies = [], [], []
    hits, shots = 0., 0.
    with Doom(scenario, seed) as doom:
        for step in range(protocol['max_steps_per_episode']):
            obs = doom.observe()
            if obs is None:
                break
            start = time.perf_counter()
            x = features(obs)
            steer, rule_fire = teacher_action(obs)
            belief = None
            if mode == 'collect':
                if rng.random() < protocol['training_random_steer_probability']:
                    steer = str(rng.choice(['left', 'hold', 'right']))
                fire = rng.random() < protocol['training_fire_probability']
            elif mode == 'rules':
                fire = rule_fire
            elif mode == 'tiny_imitation':
                # Isolate the firing decision: every evaluated controller shares rule steering.
                fire = tiny.decide(obs).fire
            else:
                belief = model.predict(x, unit_id=unit, z=protocol['lower_credible_z'])
                value = {'map_hit': belief.map_probability,
                         'bayes_mean_hit': belief.mean_probability,
                         'bayes_lower_hit': belief.lower_probability}[mode]
                fire = value > protocol['action_cost']
            latencies.append((time.perf_counter()-start)*1000)
            before_hits = doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)
            before_ammo = obs.ammo
            doom.step(hard_decision(steer, bool(fire), mode), obs, protocol['tics_per_step'])
            hit_change = max(0., doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)-before_hits)
            ammo_change = max(0., before_ammo-doom.game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO))
            hits += hit_change
            shots += ammo_change
            row = {'unit_id': unit, 'replicate': replicate, 'step': step, 'x': x, 'fire_requested': bool(fire),
                   'steer': steer, 'ammo_spent': ammo_change, 'hit_count_change': hit_change,
                   'outcome': int(hit_change > 0) if ammo_change > 0 else None,
                   'belief': asdict(belief) if belief else None}
            decisions.append(row)
            if ammo_change > 0:
                records.append(row)
            if trace:
                trace.write(json.dumps({'policy': mode, **row})+'\n')
        stats = doom.stats()
    success_windows = sum(r['outcome'] for r in records)
    forecasts = [r for r in records if r['belief'] is not None]
    p = np.asarray([r['belief']['mean_probability'] if mode != 'map_hit'
                    else r['belief']['map_probability'] for r in forecasts])
    y = np.asarray([r['outcome'] for r in forecasts])
    return {
        'scenario': scenario, 'seed': seed, 'policy': mode, 'replicate': replicate, 'steps': len(decisions),
        'hits': hits, 'shots': shots, 'successful_firing_windows': success_windows,
        'utility': success_windows-protocol['action_cost']*shots,
        'kills': stats['kills'], 'health': stats['health'], 'game_seconds': stats['game_seconds'],
        'finished': stats['finished'], 'truncated': not stats['finished'],
        'firing_windows': len(records),
        'selected_brier': float(np.mean((p-y)**2)) if len(p) else None,
        'selected_log_loss': float(-np.mean(y*np.log(np.clip(p, 1e-12, 1)) +
                                          (1-y)*np.log(np.clip(1-p, 1e-12, 1)))) if len(p) else None,
        'median_decision_ms': float(np.median(latencies)),
    }, records


def common_audit(model, rows):
    y = np.asarray([r['outcome'] for r in rows])
    beliefs = [model.predict(r['x'], unit_id=r['unit_id']) for r in rows]
    result = {}
    for label, attr in [('map', 'map_probability'), ('bayes_mean', 'mean_probability')]:
        p = np.array([getattr(b, attr) for b in beliefs])
        result[label] = {'brier': float(np.mean((p-y)**2)),
                         'log_loss': float(-np.mean(y*np.log(np.clip(p, 1e-12, 1))+
                                                   (1-y)*np.log(np.clip(1-p, 1e-12, 1))))}
    return {'n': len(rows), 'empirical_hit_rate': float(y.mean()), **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT/'research/protocols/bayesian-doom-v1.json'
    protocol = json.loads(protocol_path.read_text())
    sources = [Path(__file__), ROOT/'src/openjev/research/bayesian.py', ROOT/'src/openjev/game.py',
               ROOT/'src/openjev/domain.py', ROOT/'src/openjev/policies.py',
               ROOT/'src/openjev/weights/doom.npz']
    receipt = {'status': 'running', 'protocol': protocol, 'protocol_sha256': digest(protocol_path),
               'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
               'python': platform.python_version(), 'platform': platform.platform(),
               'numpy': np.__version__, 'vizdoom': vzd.__version__, 'episodes_completed': 0}
    (output/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    started = time.monotonic()
    episodes, models, audit_rows = [], [], []
    try:
        with (output/'trace.jsonl').open('x') as trace, (output/'episodes.jsonl').open('x') as episode_file:
            def episode(scenario, seed, mode, **kwargs):
                if time.monotonic()-started > 1800 or receipt['episodes_completed'] >= 600:
                    raise RuntimeError('Frozen wall-time or episode budget exhausted')
                summary, rows = run_episode(scenario, seed, mode, protocol, trace=trace, **kwargs)
                receipt['episodes_completed'] += 1
                episode_file.write(json.dumps(summary)+'\n')
                episode_file.flush()
                return summary, rows
            for replicate, training_seed in enumerate(protocol['training_seeds']):
                rng = np.random.default_rng(training_seed)
                rows = []
                for index in range(protocol['training_episodes_per_seed']):
                    _, collected = episode(protocol['train_scenario'], 10000+100*replicate+index,
                                           'collect', rng=rng, replicate=replicate)
                    rows.extend(collected)
                model = BayesianLogistic(7, protocol['prior_precision']).fit(
                    [r['x'] for r in rows], [r['outcome'] for r in rows], [r['unit_id'] for r in rows],
                    protocol['newton_iterations_max'])
                np.savez(output/f'posterior-{replicate}.npz', mean=model.mean, covariance=model.covariance)
                models.append(model)
                print(json.dumps({'phase': 'fit', 'replicate': replicate, 'firing_windows': len(rows),
                                  'hit_rate': float(np.mean([r['outcome'] for r in rows]))}), flush=True)
            # Common held-out behavior audit: same data for MAP and posterior mean.
            audit_rng = np.random.default_rng(76543)
            for scenario in protocol['evaluation_scenarios']:
                for seed in protocol['evaluation_episode_seeds']:
                    _, rows = episode(scenario, seed, 'collect', rng=audit_rng)
                    audit_rows.extend(rows)
            for scenario in protocol['evaluation_scenarios']:
                for mode in protocol['policies']:
                    replicates = range(5) if mode not in ('rules', 'tiny_imitation') else [None]
                    for replicate in replicates:
                        for seed in protocol['evaluation_episode_seeds']:
                            result, _ = episode(scenario, seed, mode,
                                                model=models[replicate] if replicate is not None else None,
                                                replicate=replicate)
                            result['replicate'] = replicate
                            episodes.append(result)
                        print(json.dumps({'phase': 'evaluate', 'scenario': scenario, 'policy': mode,
                                          'replicate': replicate, 'episodes': receipt['episodes_completed']}), flush=True)
        audits = {scenario: [common_audit(model, [r for r in audit_rows if r['unit_id'].startswith(scenario+':')])
                             for model in models] for scenario in protocol['evaluation_scenarios']}
        (output/'results.json').write_text(json.dumps({'episodes': episodes, 'common_audit': audits}, indent=2)+'\n')
        receipt.update(status='completed_development_pilot', wall_seconds=time.monotonic()-started)
    except Exception as exc:
        receipt.update(status='failed_no_efficacy_claim', error=f'{type(exc).__name__}: {exc}',
                       wall_seconds=time.monotonic()-started)
        raise
    finally:
        receipt['artifact_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.name != 'manifest.json'}
        (output/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
