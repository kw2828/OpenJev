"""Record the first JEPA fit and first evaluation seed, verifying preserved outcomes."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from rl_doom import digest, episode, verify
from stable_baselines3 import PPO

import openjev.research.rl_env as env_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    m = verify(args.run, 'completed_pilot')
    p = m['protocol']
    arm, rep, seed, scenario = 'srpo_vjepa', 0, p['evaluation_seeds'][0], p['scenarios'][0]
    rows = [json.loads(line) for line in (args.run/'episodes.jsonl').read_text().splitlines()]
    matches = [r for r in rows if (r['arm'], r['replicate'], r['seed'], r['scenario']) == (arm, rep, seed, scenario)]
    if len(matches) != 1 or args.output.exists():
        raise ValueError('Expected one preserved episode and a new output')
    model = PPO.load(args.run/f'{arm}-{rep}.zip', device='cpu')
    frames = []
    original = env_module.Doom

    class RecordedDoom(original):
        def step(self, decision, observation, tics=7):
            pic = Image.fromarray(self.frame()).resize((480, 360))
            canvas = Image.new('RGB', (480, 412), '#101820')
            canvas.paste(pic, (0, 52))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 5), 'OpenJev | policy trained with V-JEPA 2 rewards', fill='white')
            draw.text((10, 21), 'Small policy acts; JEPA used during training only', fill='#b3e886')
            draw.text((10, 36), f'First fit, seed {seed} | fire: {decision.fire}', fill='#b3e886')
            frames.append(canvas)
            return super().step(decision, observation, tics)

    env_module.Doom = RecordedDoom
    try:
        actual = episode(scenario, seed, arm, p, model, 'history', rep)
    finally:
        env_module.Doom = original
    fields = ['steps', 'kills', 'game_seconds', 'engine_reward', 'utility', 'firing_windows',
              'success_windows', 'ammo_used', 'capped', 'dead']
    for field in fields:
        if actual[field] != matches[0][field]:
            raise ValueError(f'Replay differs: {field}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = round(p['tics']/35*1000)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    receipt = {'status': 'recorded_gameplay_matches_preserved_episode', 'arm': arm, 'replicate': rep,
        'scenario': scenario, 'seed': seed, 'selection': 'First fit and first protocol evaluation seed, not best episode',
        'frames': len(frames), 'frame_duration_ms': duration, 'matched_fields': {f: actual[f] for f in fields},
        'gif_sha256': digest(args.output), 'evaluation_manifest_sha256': digest(args.run/'manifest.json'),
        'model_sha256': digest(args.run/f'{arm}-{rep}.zip'),
        'claim_boundary': 'Recorded demonstration at game speed. Not efficacy evidence or an inference-speed benchmark.'}
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
