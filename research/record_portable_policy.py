"""Record the selected policy, checking gameplay against its preserved episode."""
import argparse
import json
from pathlib import Path

import numpy as np
import portable_head_doom as runner
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--scenario', default='defend_the_center')
    parser.add_argument('--seed', type=int, default=101000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = runner.verify_run(args.run,'completed_confirmation')
    runner.verify_run(args.development,'completed_development')
    if runner.digest(args.development/'manifest.json') != manifest['development_manifest_sha256']:
        raise ValueError('Development receipt does not match confirmation')
    protocol = manifest['protocol']
    settings = manifest['arms']['selected']
    head = settings['head']
    if head == 'rules':
        weights = None
    elif head == 'original':
        root = runner.ROOT/protocol['history_run']
        weights = np.array([np.load(root/f'history-{r}.npz')['mean'] for r in range(5)])
    else:
        weights = np.load(args.development/f'{head}.npz')['mean']
    episodes = [json.loads(line) for line in (args.run/'episodes.jsonl').read_text().splitlines()]
    expected = [r for r in episodes if r['arm']=='selected' and r['scenario']==args.scenario and r['seed']==args.seed]
    if len(expected) != 1 or args.output.exists():
        raise ValueError('One preserved reference and a new output path are required')
    frames = []
    original_doom = runner.Doom

    class RecordedDoom(original_doom):
        def step(self, decision, observation, tics=2):
            frame = self.frame()
            if frame is not None:
                picture = Image.fromarray(frame).resize((480,360))
                canvas = Image.new('RGB',(480,398),'#101820')
                canvas.paste(picture,(0,38))
                draw = ImageDraw.Draw(canvas)
                draw.text((10,6),'OpenJev | portable current-state ensemble + event memory',fill='white')
                draw.text((10,21),f'Seed {args.seed} | decision {len(frames)+1} | fire: {decision.fire}',fill='#b3e886')
                frames.append(canvas)
            return super().step(decision,observation,tics)

    runner.Doom = RecordedDoom
    try:
        result = runner.episode(args.scenario,args.seed,'selected',protocol,settings,weights)
    finally:
        runner.Doom = original_doom
    fields = ['steps','utility','kills','firing_windows','success_windows','game_seconds','engine_reward','ammo_used','truncated']
    for field in fields:
        if result[field] != expected[0][field]:
            raise ValueError(f'Recorded gameplay differs from preserved episode: {field}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(args.output,save_all=True,append_images=frames[1:],
                   duration=round(protocol['tics_per_step']/35*1000),loop=0,optimize=True)
    receipt = {'status':'recorded_gameplay_matches_preserved_confirmation_episode',
               'scenario':args.scenario,'seed':args.seed,'selected':manifest['selection']['selected'],
               'settings':settings,'reference_manifest_sha256':runner.digest(args.run/'manifest.json'),
               'frames':len(frames),'frame_duration_ms':round(protocol['tics_per_step']/35*1000),
               'matched_gameplay':{f:result[f] for f in fields},'gif_sha256':runner.digest(args.output),
               'claim_boundary':'Illustration of the first confirmation seed, not a new efficacy run. Frame capture changes wall timing; this recording is not a speed benchmark.'}
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
