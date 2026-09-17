"""Generate readable measured tables from preserved RL analysis only."""
import argparse
import json
from pathlib import Path

LABELS={'ppo_current':'PPO current','ppo_history':'PPO history','dqn_history':'DQN history',
        'rules':'Rules','rule_event':'Rules + event memory'}


def report(analysis_path,training_path):
    a=json.loads(analysis_path.read_text())
    manifest=json.loads((training_path/'manifest.json').read_text())
    protocol=manifest['protocol']
    selected=a['selection']['selected']
    is_confirmation='confirmation_passed' in a
    lines=['## '+('Fresh confirmation' if is_confirmation else 'Completed development results'),'']
    if is_confirmation:
        lines += [f"The selected family was **{LABELS[selected]}**. The full confirmation gate **{'passed' if a['confirmation_passed'] else 'failed'}**.",'']
    else:
        selection=f'**{LABELS[selected]}** qualified for one fresh confirmation.' if selected else '**No family qualified for confirmation.** The frozen continuation rule stopped the study here.'
        lines += [f"Completed {manifest['training_steps_completed']:,} training interactions and {manifest['evaluation_episodes_completed']:,} development evaluation episodes. {selection}",'']
    lines += ['| Controller | Center kills | Center game seconds | Line kills | Line game seconds |','|---|---:|---:|---:|---:|']
    center=a['summary']['defend_the_center'];line=a['summary']['defend_the_line']
    for arm in center:
        c,l=center[arm],line[arm]
        lines.append(f"| {LABELS[arm]} | {c['kills']:.3f} | {c['game_seconds']:.3f} | {l['kills']:.3f} | {l['game_seconds']:.3f} |")
    lines += ['', 'Learned rows average all three training fits and matched game seeds. Fixed rules are run once per seed, not repeated as independent data. These finite-horizon durations can include capped episodes.','',
              '| Family | Center kills by fit | Line kills by fit |','|---|---|---|']
    for arm in manifest['arms']:
        if arm in center:
            lines.append(f"| {LABELS[arm]} | {', '.join(f'{x:.2f}' for x in center[arm]['per_fit_kills'])} | {', '.join(f'{x:.2f}' for x in line[arm]['per_fit_kills'])} |")
    lines += ['', 'PPO scores are action-selection probabilities. DQN scores are discounted action-value estimates, not probabilities of success. Neither is a calibrated success probability.','']
    if is_confirmation:
        lines += ['| Primary contrast versus event-memory rules | Difference | 98.75% interval |','|---|---:|---|']
        for scenario in protocol['scenarios']:
            for metric in ('kills','game_seconds'):
                item=a['contrasts'][scenario][selected]['rule_event'][metric]
                label=('Center' if scenario.endswith('center') else 'Line')+' '+metric.replace('_',' ')
                lo,hi=item['interval']
                lines.append(f"| {label} | {item['difference']:+.3f} | [{lo:.3f}, {hi:.3f}] |")
        lines += ['', 'Intervals resample paired game seeds and the three training fits. Only these four primary contrasts use the within-study multiplicity adjustment. Three fits give limited information about training variability. Secondary comparisons cannot rescue a failed gate.','']
    else:
        lines += ['Development uses the disclosed selection thresholds, not a statistical superiority claim. The analysis includes descriptive crossed-bootstrap intervals; development selection and small training-seed counts limit their interpretation.','']
    return '\n'.join(lines)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('analysis',type=Path)
    ap.add_argument('--training',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with args.output.open('x') as out:
        out.write(report(args.analysis,args.training))
