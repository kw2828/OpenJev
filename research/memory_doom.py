"""Frozen matched history ablation; no new training labels and no paid APIs."""
import argparse
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import vizdoom as vzd

from openjev.domain import hard_decision, teacher_action
from openjev.game import Doom
from openjev.policies import LocalPolicy
from openjev.research.bayesian import BayesianLogistic, sigmoid
from openjev.research.memory import DecisionHistory, select_features

ROOT = Path(__file__).resolve().parents[1]


def base_features(obs):
    return np.array([1., float(obs.visible), min(abs(obs.aim_error),1.), obs.half_width,
                     min(obs.distance/1000,2.), obs.health/100, min(obs.ammo/30,2.)])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_data(protocol):
    source = ROOT/protocol['training_source']
    parent = json.loads((source.parent/'manifest.json').read_text())
    raw = gzip.decompress(source.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == parent['artifact_sha256']['trace.jsonl']
    datasets = {k: [[] for _ in range(5)] for k in protocol['features']}
    labels, units = [[] for _ in range(5)], [[] for _ in range(5)]
    previous_unit, history = None, None
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if row['policy'] != 'collect' or row['replicate'] is None:
            continue
        unit = row['unit_id']
        if unit != previous_unit:
            assert row['step'] == 0
            history = DecisionHistory()
            previous_unit = unit
        rep = row['replicate']
        if row['outcome'] is not None:
            for kind in datasets:
                datasets[kind][rep].append(select_features(kind,row['x'],history))
            labels[rep].append(row['outcome'])
            units[rep].append(unit)
        history.update(row['x'],row['issued_fire'],row['hit_count_change'])
    assert all(len(set(u)) == 16 for u in units)
    return datasets, labels, units


def probability(model, x, mode, unit):
    if unit in model.training_units:
        raise ValueError('Training/evaluation overlap')
    if mode == 'map':
        return float(sigmoid(x @ model.mean))
    return model.predict(x, unit_id=unit).mean_probability


def episode(scenario, seed, policy, protocol, model=None, replicate=None, rng=None, trace=None):
    history = DecisionHistory()
    tiny = LocalPolicy() if policy == 'tiny_imitation' else None
    unit = f'{scenario}:{seed}'
    kind, mode = policy.rsplit('_',1) if policy.startswith(('current_','expanded_','history_')) else (None,None)
    times, records = [], []
    success = fired = 0
    with Doom(scenario,seed) as doom:
        for step in range(protocol['max_steps_per_episode']):
            obs = doom.observe()
            if obs is None:
                break
            start = time.perf_counter()
            x = base_features(obs)
            steer, fire = teacher_action(obs)
            hx = history.features(x) if kind == 'history' or policy == 'collect' else None
            p = None
            if policy == 'collect':
                if rng.random() < protocol['training_random_steer_probability']:
                    steer = str(rng.choice(['left','hold','right']))
                fire = rng.random() < protocol['training_fire_probability']
            elif policy == 'tiny_imitation':
                fire = tiny.decide(obs).fire
            elif kind:
                fx = hx if kind == 'history' else select_features(kind,x,history)
                p = probability(model,fx,mode,unit)
                fire = p > protocol['action_cost']
            issued = bool(fire and obs.ammo > 0 and obs.directive != 'pacifist')
            decision = hard_decision(steer,bool(fire),policy)
            elapsed = time.perf_counter()-start
            before = doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)
            doom.step(decision,obs,protocol['tics_per_step'])
            hit = max(0.,doom.game.get_game_variable(vzd.GameVariable.HITCOUNT)-before)
            y = int(hit > 0) if issued else None
            start = time.perf_counter()
            if kind == 'history' or policy == 'collect':
                history.update(x,issued,hit)
            elapsed += time.perf_counter()-start
            times.append(elapsed)
            fired += issued
            success += int(issued and hit > 0)
            row = {'scenario':scenario,'seed':seed,'unit_id':unit,'policy':policy,'replicate':replicate,
                   'step':step,'x':x.tolist(),'history_x':hx.tolist() if hx is not None else None,
                   'issued_fire':issued,'steer':steer,'outcome':y,'hit_count_change':hit,
                   'p':p,'decision_seconds':elapsed}
            if policy == 'collect' and issued:
                records.append(row)
            if trace:
                trace.write(json.dumps(row,allow_nan=False)+'\n')
        stats = doom.stats()
    compute = float(sum(times))
    utility = success-protocol['action_cost']*fired
    return {'scenario':scenario,'seed':seed,'policy':policy,'replicate':replicate,
            'steps':len(times),'utility':utility,'net_utility':utility-compute,'decision_compute_seconds':compute,
            'p95_decision_ms':float(np.quantile(times,.95)*1000),'kills':stats['kills'],
            'game_seconds':stats['game_seconds'],'firing_windows':fired,'success_windows':success,
            'truncated':not stats['finished']}, records


def audit(models, rows):
    if not rows:
        raise ValueError('Empty audit')
    output = {}
    y = np.array([r['outcome'] for r in rows])
    for kind, fits in models.items():
        output[kind] = []
        for model in fits:
            result = {'n':len(y)}
            for mode in ('map','bayes'):
                ps = [probability(model, np.array(r['history_x']) if kind == 'history'
                                  else select_features(kind,r['x'],DecisionHistory()),mode,r['unit_id']) for r in rows]
                p = np.clip(ps,1e-12,1-1e-12)
                result[mode] = {'brier':float(np.mean((p-y)**2)),
                                'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}
            output[kind].append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True,exist_ok=False)
    pp = ROOT/'research/protocols/memory-doom-v1.json'
    protocol = json.loads(pp.read_text())
    sources = [Path(__file__),ROOT/'research/analyze_memory_doom.py',ROOT/'src/openjev/research/memory.py',
               ROOT/'src/openjev/research/bayesian.py',ROOT/'src/openjev/game.py',ROOT/'src/openjev/domain.py',
               ROOT/'src/openjev/policies.py',ROOT/'src/openjev/weights/doom.npz']
    manifest = {'status':'running','protocol':protocol,'protocol_sha256':digest(pp),
                'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
                'training_source_sha256':digest(ROOT/protocol['training_source']),
                'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
                'vizdoom':vzd.__version__,'episodes_completed':0}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started = time.monotonic()
    try:
        xs, ys, units = training_data(protocol)
        models, checks = {}, []
        for kind, dim in protocol['features'].items():
            models[kind] = []
            for rep in range(5):
                x,y = np.array(xs[kind][rep]),np.array(ys[rep])
                model = BayesianLogistic(dim,protocol['prior_precision']).fit(x,y,units[rep],protocol['newton_iterations_max'])
                gradient = x.T @ (sigmoid(x @ model.mean)-y)+model.mean
                checks.append({'kind':kind,'replicate':rep,'n':len(y),'gradient_norm':float(np.linalg.norm(gradient)),
                               'min_covariance_eigenvalue':float(np.linalg.eigvalsh(model.covariance).min())})
                if checks[-1]['gradient_norm'] > 1e-5:
                    raise ValueError('Fit failed convergence check')
                np.savez(output/f'{kind}-{rep}.npz',mean=model.mean,covariance=model.covariance)
                models[kind].append(model)
        print(json.dumps({'phase':'fitted','models':15,'training_windows':[len(y) for y in ys]}),flush=True)
        results, audits = [], {}
        with gzip.open(output/'trace.jsonl.gz','wt') as trace, (output/'episodes.jsonl').open('x') as ef:
            def run(scenario,seed,policy,**kwargs):
                if time.monotonic()-started > protocol['max_wall_seconds'] or manifest['episodes_completed'] >= protocol['max_episodes']:
                    raise RuntimeError('Frozen run budget exhausted')
                r, rows = episode(scenario,seed,policy,protocol,trace=trace,**kwargs)
                manifest['episodes_completed'] += 1
                ef.write(json.dumps(r,allow_nan=False)+'\n'); ef.flush()
                if manifest['episodes_completed'] % 100 == 0:
                    print(json.dumps({'phase':'running','episodes':manifest['episodes_completed']}),flush=True)
                return r,rows
            jobs = []
            for scenario in protocol['evaluation_scenarios']:
                for policy in protocol['policies']:
                    for rep in ([None] if policy in ('rules','tiny_imitation') else range(5)):
                        jobs.extend((scenario,seed,policy,rep) for seed in protocol['evaluation_episode_seeds'])
            np.random.default_rng(protocol['schedule_rng_seed']).shuffle(jobs)
            for scenario,seed,policy,rep in jobs:
                model = models[policy.rsplit('_',1)[0]][rep] if rep is not None else None
                r,_ = run(scenario,seed,policy,model=model,replicate=rep)
                results.append(r)
            rng = np.random.default_rng(protocol['audit_rng_seed'])
            for scenario in protocol['evaluation_scenarios']:
                rows = []
                for seed in protocol['audit_episode_seeds']:
                    _, collected = run(scenario,seed,'collect',rng=rng)
                    rows.extend(collected)
                audits[scenario] = audit(models,rows)
        assert manifest['episodes_completed'] == protocol['planned_episodes']
        (output/'results.json').write_text(json.dumps({'episodes':results,'common_audit':audits,'fit_checks':checks},indent=2,allow_nan=False)+'\n')
        manifest['status'] = 'completed_frozen_memory_ablation'
    except Exception as exc:
        manifest.update(status='failed_no_efficacy_claim',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['wall_seconds'] = time.monotonic()-started
        manifest['artifact_sha256'] = {p.name:digest(p) for p in output.iterdir() if p.name != 'manifest.json'}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
