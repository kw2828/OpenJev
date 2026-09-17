"""Prospective portable-head and matched probability-ensemble comparison."""
import argparse
import gzip
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import vizdoom as vzd
from memory_doom import base_features, training_data

from openjev.domain import hard_decision, teacher_action
from openjev.game import Doom
from openjev.research.bayesian import BayesianLogistic, sigmoid
from openjev.research.event_cadence import EventCadenceControl
from openjev.research.memory import DecisionHistory
from openjev.research.portable import portable_features

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'research/protocols/portable-head-doom-v1.json'
SOURCES = ['research/portable_head_doom.py', 'research/analyze_portable_head_doom.py',
           'src/openjev/research/portable.py', 'research/protocols/memory-doom-v1.json',
           'src/openjev/research/event_cadence.py',
           'research/memory_doom.py', 'src/openjev/research/cadence.py',
           'src/openjev/research/memory.py', 'src/openjev/research/bayesian.py',
           'src/openjev/domain.py', 'src/openjev/game.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_run(root, expected_status):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != expected_status:
        raise ValueError('Required completed run missing')
    for name, expected in manifest['artifact_sha256'].items():
        if digest(root/name) != expected:
            raise ValueError(f'Changed artifact: {name}')
    return manifest


def episode(scenario, seed, arm, protocol, settings=None, weights=None, replicate=None, trace=None):
    head = settings['head'] if settings is not None else 'history_bank'
    cadence = EventCadenceControl(margin=.1 if head=='rules' else 2., event='ammo_or_hit',rest_windows=1) if settings is not None and settings['gate'] else None
    uses_history = head in ('original','portable_history','history_bank')
    history = DecisionHistory()
    times = []
    fired = success = 0
    with Doom(scenario, seed) as doom:
        initial_ammo = doom.observe().ammo
        for step in range(protocol['max_steps_per_episode']):
            obs = doom.observe()
            if obs is None:
                break
            start = time.perf_counter()
            x = base_features(obs)
            probability = None
            steer, fire = teacher_action(obs)
            if head != 'rules':
                hx = history.features(x) if uses_history else None
                features = hx if head in ('original','history_bank') else portable_features(x,hx)
                probability = float(np.mean(sigmoid(weights @ features)))
                fire = probability > protocol['action_cost']
                if head != 'history_bank':
                    fire = fire and obs.visible
            if cadence is not None:
                _, allowed = cadence.decide(obs)
                fire = fire and allowed
            decision = hard_decision(steer, bool(fire), arm)
            issued = bool(fire and obs.ammo > 0 and obs.directive != 'pacifist')
            elapsed = time.perf_counter()-start
            before = doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)
            doom.step(decision, obs, protocol['tics_per_step'])
            hit = max(0., doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)-before)
            start = time.perf_counter()
            if uses_history:
                history.update(x, issued, hit)
            if cadence is not None:
                cadence.observe_outcome(hit)
            elapsed += time.perf_counter()-start
            times.append(elapsed)
            fired += issued
            success += int(issued and hit > 0)
            if trace is not None:
                trace.write(json.dumps({'scenario':scenario, 'seed':seed, 'arm':arm,
                    'replicate':replicate, 'step':step, 'observation':asdict(obs),
                    'issued_fire':issued, 'steer':steer, 'hit_count_change':hit,
                    'probability':probability, 'decision_seconds':elapsed}, allow_nan=False)+'\n')
        stats = doom.stats()
    compute = float(sum(times))
    utility = success-protocol['action_cost']*fired
    return {'scenario':scenario, 'seed':seed, 'arm':arm, 'replicate':replicate,
            'steps':len(times), 'firing_windows':fired, 'success_windows':success,
            'utility':utility, 'net_utility':utility-protocol['net_utility_compute_cost_per_second']*compute,
            'decision_compute_seconds':compute, 'p95_decision_ms':float(np.quantile(times,.95)*1000),
            'kills':stats['kills'], 'engine_reward':stats['reward'], 'ammo_used':initial_ammo-stats['ammo'],
            'game_seconds':stats['game_seconds'], 'truncated':not stats['finished']}


def select_development(rows, protocol):
    means = {}
    for scenario in protocol['development_scenarios']:
        means[scenario] = {}
        for arm in protocol['arms']:
            cells = [r for r in rows if r['arm'] == arm and r['scenario'] == scenario]
            if len(cells) != len(protocol['development_seeds']):
                raise ValueError('Incomplete development arm')
            means[scenario][arm] = {m:float(np.mean([r[m] for r in cells])) for m in
                ('net_utility','utility','kills','firing_windows','success_windows')}
    eligible, scores = [], {}
    for arm in protocol['arms']:
        if arm == 'rules':
            continue
        gains, checks = [], []
        for scenario in protocol['development_scenarios']:
            c,r = means[scenario][arm],means[scenario]['rules']
            gains.append(c['net_utility']-r['net_utility'])
            checks.append(gains[-1] >= protocol['selection']['minimum_development_net_gain_vs_rules'])
            checks.append(c['kills']-r['kills'] >= protocol['selection']['minimum_development_kill_difference_vs_rules'])
        scores[arm] = min(gains)
        if all(checks):
            eligible.append(arm)
    selected = min(eligible,key=lambda a:(-scores[a],a)) if eligible else None
    return {'selected':selected, 'eligible':eligible, 'minimum_scenario_gain':scores, 'means':means,
            'status':'selected_for_one_confirmation' if selected else 'no_candidate_qualified_for_confirmation'}


def fit_portable(output):
    original = json.loads((ROOT/'research/protocols/memory-doom-v1.json').read_text())
    xs,ys,units = training_data(original)
    fitted, checks = {}, []
    for head in ('portable_current','portable_history'):
        bank = []
        for rep in range(5):
            x = np.array([portable_features(row[:7], row if head=='portable_history' else None)
                          for row in xs['history'][rep]])
            y = np.array(ys[rep])
            model = BayesianLogistic(x.shape[1],1.).fit(x,y,units[rep],40)
            gradient = x.T @ (sigmoid(x @ model.mean)-y)+model.mean
            norm = float(np.linalg.norm(gradient))
            if norm > 1e-5:
                raise ValueError('Portable fit failed convergence check')
            checks.append({'head':head,'replicate':rep,'n':len(y),'gradient_norm':norm})
            bank.append(model.mean)
        fitted[head] = np.array(bank)
        np.savez(output/f'{head}.npz',mean=fitted[head])
    return fitted, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['development','confirmation'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--development', type=Path)
    args = ap.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    hashes = {s:digest(ROOT/s) for s in SOURCES}
    selection = None
    history_root = ROOT/protocol['history_run']
    history_manifest = verify_run(history_root, 'completed_frozen_memory_ablation')
    original_weights = np.array([np.load(history_root/f'history-{r}.npz')['mean'] for r in range(5)])
    history_receipt = {'manifest_sha256':digest(history_root/'manifest.json'),
                      'model_hashes':{f'history-{r}.npz':history_manifest['artifact_sha256'][f'history-{r}.npz'] for r in range(5)}}
    models = {'original':original_weights}
    if args.stage == 'confirmation':
        if args.development is None:
            raise ValueError('Development receipt required')
        parent = verify_run(args.development, 'completed_development')
        if parent['source_sha256'] != hashes or parent['protocol_sha256'] != digest(PROTOCOL):
            raise ValueError('Protocol or execution sources changed after development')
        selection = json.loads((args.development/'selection.json').read_text())
        if selection['selected'] is None:
            raise ValueError('No candidate qualified; confirmation is not authorized by this protocol')
        selected = protocol['arms'][selection['selected']]
        arms = {'selected':selected,'rules':protocol['arms']['rules'],
                'original_ensemble':protocol['arms']['original_ensemble'],
                'rule_event':protocol['arms']['rule_event'],
                'selected_no_gate':{**selected,'gate':False}}
        for head in ('portable_current','portable_history'):
            models[head] = np.load(args.development/f'{head}.npz')['mean']
    else:
        arms = protocol['arms']
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'status':'running', 'stage':args.stage, 'protocol':protocol,
                'protocol_sha256':digest(PROTOCOL), 'source_sha256':hashes,
                'platform':platform.platform(), 'python':platform.python_version(),
                'numpy':np.__version__, 'vizdoom':vzd.__version__, 'episodes_completed':0,
                'selection':selection, 'arms':arms, 'history_receipt':history_receipt,
                'training_source_sha256':digest(ROOT/protocol['training_source'])}
    if args.development:
        manifest['development_manifest_sha256'] = digest(args.development/'manifest.json')
    mp = args.output/'manifest.json'
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    started = time.monotonic()
    try:
        if args.stage == 'development':
            portable, checks = fit_portable(args.output)
            models.update(portable)
            (args.output/'fit-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        jobs = []
        for scenario in protocol[f'{args.stage}_scenarios']:
            for seed in protocol[f'{args.stage}_seeds']:
                jobs.extend((scenario,seed,arm,None) for arm in arms)
                if args.stage == 'confirmation':
                    jobs.extend((scenario,seed,'history_map',r) for r in range(5))
        assert len(jobs) == protocol[f'max_{args.stage}_episodes']
        np.random.default_rng(protocol[f'{args.stage}_schedule_seed']).shuffle(jobs)
        rows = []
        with gzip.open(args.output/'trace.jsonl.gz','wt') as trace, (args.output/'episodes.jsonl').open('x') as stream:
            for scenario, seed, arm, rep in jobs:
                if time.monotonic()-started > protocol['max_stage_wall_seconds']:
                    raise RuntimeError('Frozen wall budget exhausted')
                weights = original_weights[rep] if rep is not None else models.get(arms[arm]['head'])
                row = episode(scenario,seed,arm,protocol,arms.get(arm),weights,rep,trace)
                rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                stream.flush()
                manifest['episodes_completed'] += 1
                if manifest['episodes_completed'] % 100 == 0:
                    mp.write_text(json.dumps(manifest,indent=2)+'\n')
                    print(json.dumps({'stage':args.stage,'episodes':manifest['episodes_completed']}),flush=True)
        if args.stage == 'development':
            selection = select_development(rows,protocol)
            (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
            print(json.dumps(selection),flush=True)
        manifest['status'] = f'completed_{args.stage}'
    except BaseException as exc:
        manifest.update(status='failed_no_efficacy_claim',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['wall_seconds'] = time.monotonic()-started
        manifest['artifact_sha256'] = {p.name:digest(p) for p in args.output.iterdir() if p.name != 'manifest.json'}
        mp.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
