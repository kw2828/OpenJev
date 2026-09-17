"""Prospective equal-interaction PPO / DQN experiment using Stable Baselines3."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from openjev.domain import teacher_action
from openjev.research.event_cadence import EventCadenceControl
from openjev.research.rl_env import FiringEnv

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'research/protocols/rl-doom-v1.json'
SOURCES = ['research/rl_doom.py','research/analyze_rl_doom.py','src/openjev/research/rl_env.py',
           'src/openjev/research/memory.py','src/openjev/research/event_cadence.py',
           'src/openjev/research/cadence.py','src/openjev/game.py','src/openjev/domain.py','uv.lock']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, status):
    m = json.loads((root/'manifest.json').read_text())
    if m['status'] != status:
        raise ValueError('Required completed stage missing')
    for path, expected in m['artifact_sha256'].items():
        if digest(root/path) != expected:
            raise ValueError(f'Changed artifact: {path}')
    return m


class Budget(BaseCallback):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def _on_step(self):
        if time.monotonic() > self.deadline:
            raise RuntimeError('Frozen stage wall budget exhausted')
        if self.num_timesteps % 8192 == 0:
            print(json.dumps({'training_steps':self.num_timesteps}),flush=True)
        return True


def episode(scenario, seed, arm, protocol, model=None, kind='current', rep=None, trace=None):
    cadence = EventCadenceControl(margin=.1,event='ammo_or_hit',rest_windows=1) if arm=='rule_event' else None
    env = FiringEnv(kind,scenario,horizon=protocol['horizon'],tics=protocol['tics'],trace=trace,
                    label={'arm':arm,'replicate':rep})
    times = []
    try:
        state,_ = env.reset(options={'game_seed':seed})
        while True:
            start = time.perf_counter()
            if model is not None:
                action = int(model.predict(state,deterministic=True)[0])
            elif cadence:
                _,action = cadence.decide(env.obs)
            else:
                _,action = teacher_action(env.obs)
            elapsed = time.perf_counter()-start
            state,_,done,_,info = env.step(action)
            start = time.perf_counter()
            if cadence:
                cadence.observe_outcome(info['hit_count_change'])
            elapsed += time.perf_counter()-start + info['decision_seconds']
            times.append(elapsed)
            if done:
                break
        return {**info['result'],'arm':arm,'replicate':rep,'decision_compute_seconds':sum(times),
                'p95_decision_ms':float(np.quantile(times,.95)*1000),
                'net_utility':info['result']['utility']-sum(times)}
    finally:
        env.close()


def select(rows, protocol):
    summary, eligible, gains = {},[],{}
    for arm in protocol['arms']:
        summary[arm] = {}
        ok = True
        for scenario in protocol['scenarios']:
            a = [r for r in rows if r['arm']==arm and r['scenario']==scenario]
            b = [r for r in rows if r['arm']=='rule_event' and r['scenario']==scenario]
            summary[arm][scenario] = {k:float(np.mean([r[k] for r in a])-np.mean([r[k] for r in b]))
                                     for k in ('kills','game_seconds','utility')}
            v = summary[arm][scenario]
            minimum = .5 if scenario == 'defend_the_center' else -.5
            ok = ok and v['kills'] >= minimum and v['game_seconds'] >= -.5
        gains[arm] = summary[arm]['defend_the_center']['kills']
        if ok:
            eligible.append(arm)
    chosen = min(eligible,key=lambda arm:(-gains[arm],arm)) if eligible else None
    return {'selected':chosen,'eligible':eligible,'mean_differences_vs_event_rule':summary,
            'status':'selected_for_confirmation' if chosen else 'no_candidate_qualified'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['development','confirmation'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--development',type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads(PROTOCOL.read_text())
    hashes = {s:digest(ROOT/s) for s in SOURCES}
    versions = {p:importlib.metadata.version(p) for p in ('stable-baselines3','torch','gymnasium','numpy','vizdoom')}
    parent = None
    selection = None
    models = {}
    if args.stage == 'confirmation':
        if args.development is None:
            raise ValueError('Development required')
        parent = verify(args.development,'completed_development')
        if parent['source_sha256'] != hashes or parent['protocol_sha256'] != digest(PROTOCOL) or parent['versions'] != versions:
            raise ValueError('Sources, protocol or runtime changed after development')
        selection = json.loads((args.development/'selection.json').read_text())
        if selection['selected'] is None:
            raise ValueError('No candidate qualified')
        # Preserve all three fits; never select a lucky training seed.
        arms = sorted({selection['selected'],'ppo_current'})
        for arm in arms:
            cls = PPO if protocol['arms'][arm]['algorithm']=='PPO' else DQN
            for rep in range(protocol['replicates']):
                models[arm,rep] = cls.load(args.development/f'{arm}-{rep}.zip',device='cpu')
    else:
        arms = list(protocol['arms'])
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    deadline = started + protocol['max_stage_wall_seconds']
    manifest = {'status':'running','stage':args.stage,'protocol':protocol,'protocol_sha256':digest(PROTOCOL),
        'source_sha256':hashes,'versions':versions,'platform':platform.platform(),'python':platform.python_version(),
        'selection':selection,'arms':arms,'training_steps_completed':0,'evaluation_episodes_completed':0}
    if parent:
        manifest['development_manifest_sha256'] = digest(args.development/'manifest.json')
    mp = args.output/'manifest.json'
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        if args.stage == 'development':
            jobs = [(arm,rep) for arm in arms for rep in range(protocol['replicates'])]
            np.random.default_rng(protocol['training_order_seed']).shuffle(jobs)
            for arm,rep in jobs:
                settings = protocol['arms'][arm]
                print(json.dumps({'training':arm,'replicate':rep}),flush=True)
                with gzip.open(args.output/f'{arm}-{rep}-training.jsonl.gz','wt') as trace:
                    env = FiringEnv(settings['features'],seed_start=protocol['training_seed_starts'][rep],
                        horizon=protocol['horizon'],tics=protocol['tics'],trace=trace,
                        label={'arm':arm,'replicate':rep})
                    cls = PPO if settings['algorithm']=='PPO' else DQN
                    model = cls('MlpPolicy',env,seed=protocol['model_seeds'][rep],device='cpu',
                        policy_kwargs={'net_arch':[64,64]},**protocol[settings['algorithm'].lower()])
                    model.set_logger(configure(str(args.output/f'{arm}-{rep}-logs'),['csv']))
                    train_start = time.monotonic()
                    try:
                        model.learn(total_timesteps=protocol['steps_per_fit'],callback=Budget(deadline))
                        if model.num_timesteps != protocol['steps_per_fit']:
                            raise ValueError('Interaction budget mismatch')
                        model.save(args.output/f'{arm}-{rep}.zip')
                        (args.output/f'{arm}-{rep}-training-summary.json').write_text(json.dumps({
                            'steps':model.num_timesteps,'wall_seconds':time.monotonic()-train_start,
                            'completed_episodes':env.training_episodes,
                            'unfinished_episode':None if env.done else env.result(),
                            'parameter_count':sum(p.numel() for p in model.policy.parameters())},indent=2)+'\n')
                    finally:
                        env.close()
                models[arm,rep] = model
                manifest['training_steps_completed'] += model.num_timesteps
                mp.write_text(json.dumps(manifest,indent=2)+'\n')
        jobs = []
        for scenario in protocol['scenarios']:
            for seed in protocol[f'{args.stage}_seeds']:
                jobs.extend((scenario,seed,arm,rep) for arm in arms for rep in range(protocol['replicates']))
                jobs.extend((scenario,seed,arm,None) for arm in ('rules','rule_event'))
        np.random.default_rng(protocol[f'{args.stage}_order_seed']).shuffle(jobs)
        rows = []
        with gzip.open(args.output/'evaluation-trace.jsonl.gz','wt') as trace, (args.output/'episodes.jsonl').open('x') as out:
            for scenario,seed,arm,rep in jobs:
                if time.monotonic()>deadline:
                    raise RuntimeError('Frozen stage wall budget exhausted')
                row = episode(scenario,seed,arm,protocol,models.get((arm,rep)),
                              protocol['arms'].get(arm,{}).get('features','current'),rep,trace)
                out.write(json.dumps(row,allow_nan=False)+'\n')
                out.flush()
                rows.append(row)
                manifest['evaluation_episodes_completed'] += 1
                if len(rows)%100==0:
                    print(json.dumps({'evaluation_episodes':len(rows)}),flush=True)
            if args.stage == 'development':
                selection = select(rows,protocol)
                (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
                print(json.dumps(selection,indent=2),flush=True)
        manifest['status'] = f'completed_{args.stage}'
        manifest['selection'] = selection
    except BaseException as e:
        manifest.update(status='failed_no_efficacy_claim',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        manifest['wall_seconds'] = time.monotonic()-started
        manifest['artifact_sha256'] = {str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*')
                                      if p.is_file() and p != mp}
        mp.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
