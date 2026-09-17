"""Replay a preserved RL evaluation episode and verify gameplay before saving GIF."""
import argparse
import json
from pathlib import Path

import rl_doom as runner
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import DQN, PPO

import openjev.research.rl_env as env_module


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--arm',choices=['ppo_current','ppo_history','dqn_history'],required=True)
    ap.add_argument('--replicate',type=int,default=0)
    ap.add_argument('--scenario',default='defend_the_center')
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    info=json.loads((args.run/'manifest.json').read_text())
    runner.verify(args.run,'completed_'+info['stage'])
    runner.verify(args.models,'completed_development')
    if info['stage']=='confirmation' and runner.digest(args.models/'manifest.json')!=info['development_manifest_sha256']:
        raise ValueError('Incorrect model provenance')
    if info['stage']=='development' and args.models.resolve()!=args.run.resolve():
        raise ValueError('Development models must come from the recorded run')
    rows=[json.loads(s) for s in (args.run/'episodes.jsonl').read_text().splitlines()]
    matches=[r for r in rows if r['arm']==args.arm and r['replicate']==args.replicate
             and r['seed']==args.seed and r['scenario']==args.scenario]
    if len(matches)!=1 or args.output.exists():
        raise ValueError('One preserved reference and new output required')
    settings=info['protocol']['arms'][args.arm]
    cls=PPO if settings['algorithm']=='PPO' else DQN
    model=cls.load(args.models/f'{args.arm}-{args.replicate}.zip',device='cpu')
    frames=[]
    original=env_module.Doom

    class RecordedDoom(original):
        def step(self,decision,observation,tics=7):
            frame=self.frame()
            if frame is not None:
                pic=Image.fromarray(frame).resize((480,360))
                canvas=Image.new('RGB',(480,398),'#101820')
                canvas.paste(pic,(0,38))
                draw=ImageDraw.Draw(canvas)
                draw.text((10,6),f'OpenJev | {args.arm} | training fit {args.replicate}',fill='white')
                draw.text((10,21),f'Seed {args.seed} | decision {len(frames)+1} | fire: {decision.fire}',fill='#b3e886')
                frames.append(canvas)
            return super().step(decision,observation,tics)

    env_module.Doom=RecordedDoom
    try:
        actual=runner.episode(args.scenario,args.seed,args.arm,info['protocol'],model,settings['features'],args.replicate)
    finally:
        env_module.Doom=original
    fields=['steps','kills','utility','game_seconds','engine_reward','firing_windows','success_windows','ammo_used','capped','dead']
    for f in fields:
        if actual[f]!=matches[0][f]:
            raise ValueError(f'Replay differs: {f}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    duration=round(info['protocol']['tics']/35*1000)
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=duration,loop=0,optimize=True)
    receipt={'status':'recorded_gameplay_matches_preserved_episode','arm':args.arm,'replicate':args.replicate,
             'scenario':args.scenario,'seed':args.seed,'frames':len(frames),'frame_duration_ms':duration,
             'matched_fields':{f:actual[f] for f in fields},'gif_sha256':runner.digest(args.output),
             'evaluation_manifest_sha256':runner.digest(args.run/'manifest.json'),
             'model_sha256':runner.digest(args.models/f'{args.arm}-{args.replicate}.zip'),
             'claim_boundary':'One illustrative recorded episode, not efficacy evidence or an inference-speed benchmark.'}
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
