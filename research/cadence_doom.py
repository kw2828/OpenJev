"""Prospective cadence development and one selected-policy confirmation."""
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
from memory_doom import base_features

from openjev.domain import hard_decision, teacher_action
from openjev.game import Doom
from openjev.research.bayesian import sigmoid
from openjev.research.cadence import CadenceControl
from openjev.research.memory import DecisionHistory

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'research/protocols/cadence-doom-v1.json'
SOURCES = ['research/cadence_doom.py', 'research/analyze_cadence_doom.py',
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
    cadence = CadenceControl(**settings) if settings is not None else None
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
            if cadence is not None:
                steer, fire = cadence.decide(obs)
            else:
                steer, _ = teacher_action(obs)
                probability = float(sigmoid(history.features(x) @ weights))
                fire = probability > protocol['action_cost']
            decision = hard_decision(steer, bool(fire), arm)
            issued = bool(fire and obs.ammo > 0 and obs.directive != 'pacifist')
            elapsed = time.perf_counter()-start
            before = doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)
            doom.step(decision, obs, protocol['tics_per_step'])
            hit = max(0., doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)-before)
            start = time.perf_counter()
            if cadence is None:
                history.update(x, issued, hit)
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
    for arm in protocol['arms']:
        cells = [r for r in rows if r['arm'] == arm]
        if len(cells) != len(protocol['development_seeds']):
            raise ValueError('Incomplete development arm')
        means[arm] = {m:float(np.mean([r[m] for r in cells])) for m in
                      ('net_utility', 'utility', 'kills', 'firing_windows', 'success_windows')}
    rule = means['rules']
    eligible = [a for a in means if a != 'rules'
                and means[a]['net_utility']-rule['net_utility'] >= protocol['selection']['minimum_development_net_gain_vs_rules']
                and means[a]['kills']-rule['kills'] >= protocol['selection']['minimum_development_kill_difference_vs_rules']]
    selected = min(eligible, key=lambda a:(-means[a]['net_utility'],a)) if eligible else None
    return {'selected':selected, 'eligible':eligible, 'means':means,
            'status':'selected_for_one_confirmation' if selected else 'no_candidate_qualified_for_confirmation'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['development','confirmation'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--development', type=Path)
    args = ap.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    hashes = {s:digest(ROOT/s) for s in SOURCES}
    selection = None
    models = {}
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
        arms = {'selected':selected, 'rules':protocol['arms']['rules'],
                'matched_no_rest':{**selected,'rest_windows':0},
                'command_rest1':protocol['arms']['command_rest1']}
        history_root = ROOT/protocol['history_run']
        history_manifest = verify_run(history_root, 'completed_frozen_memory_ablation')
        for rep in range(protocol['history_replicates']):
            models[rep] = np.load(history_root/f'history-{rep}.npz')['mean']
        history_receipt = {'manifest_sha256':digest(history_root/'manifest.json'),
                           'model_hashes':{f'history-{r}.npz':history_manifest['artifact_sha256'][f'history-{r}.npz'] for r in models}}
    else:
        arms = protocol['arms']
        history_receipt = None
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'status':'running', 'stage':args.stage, 'protocol':protocol,
                'protocol_sha256':digest(PROTOCOL), 'source_sha256':hashes,
                'platform':platform.platform(), 'python':platform.python_version(),
                'numpy':np.__version__, 'vizdoom':vzd.__version__, 'episodes_completed':0,
                'selection':selection, 'arms':arms, 'history_receipt':history_receipt}
    if args.development:
        manifest['development_manifest_sha256'] = digest(args.development/'manifest.json')
    mp = args.output/'manifest.json'
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    started = time.monotonic()
    try:
        jobs = []
        for scenario in protocol[f'{args.stage}_scenarios']:
            for seed in protocol[f'{args.stage}_seeds']:
                jobs.extend((scenario,seed,arm,None) for arm in arms)
                if args.stage == 'confirmation':
                    jobs.extend((scenario,seed,'history_map',r) for r in models)
        assert len(jobs) == protocol[f'max_{args.stage}_episodes']
        np.random.default_rng(protocol[f'{args.stage}_schedule_seed']).shuffle(jobs)
        rows = []
        with gzip.open(args.output/'trace.jsonl.gz','wt') as trace, (args.output/'episodes.jsonl').open('x') as stream:
            for scenario, seed, arm, rep in jobs:
                if time.monotonic()-started > protocol['max_stage_wall_seconds']:
                    raise RuntimeError('Frozen wall budget exhausted')
                row = episode(scenario,seed,arm,protocol,arms.get(arm),models.get(rep),rep,trace)
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
