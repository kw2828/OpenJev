"""Write paper tables only from an audited completed chess-spatial release."""

import argparse
import json
from pathlib import Path

from publish_chess_spatial import audit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project
    audit(root)
    summary = json.loads((root/'evidence/chess-spatial-v1/results/summary.json').read_text())
    means = summary['means']
    passed = summary['continuation_gate']['passed']
    result = ('met its descriptive development threshold' if passed else 'did not meet its descriptive development threshold')
    dev_gain = 100*(means['dev']['predict']['top1_teacher_agreement']-
                   means['dev']['reconstruct']['top1_teacher_agreement'])
    shift_gain = 100*(means['shift']['predict']['top1_teacher_agreement']-
                     means['shift']['reconstruct']['top1_teacher_agreement'])
    interpretation = ('The frozen criterion supports a fresh confirmation experiment; it does not establish novelty.'
                      if passed else 'The results do not establish that future prediction improves decisions over the controls.')
    discussion = (f'Prediction minus reconstruction teacher agreement is {dev_gain:+.2f} percentage points '
                  f'on ordinary development and {shift_gain:+.2f} points under the specified shift. '
                  'All twelve final fits are retained. No best seed or checkpoint is selected. '
                  'Auxiliary accuracy and policy quality answer different questions.')
    macros = {
        'PredictDev': f"{100*means['dev']['predict']['top1_teacher_agreement']:.2f}",
        'PredictShift': f"{100*means['shift']['predict']['top1_teacher_agreement']:.2f}",
        'ReconstructDev': f"{100*means['dev']['reconstruct']['top1_teacher_agreement']:.2f}",
        'ReconstructShift': f"{100*means['shift']['reconstruct']['top1_teacher_agreement']:.2f}",
        'GateOutcome': result, 'ResultInterpretation': interpretation, 'ResultDiscussion': discussion,
        'TrainingMinutes': f"{summary['training_wall_seconds']/60:.2f}",
    }
    with (root/'paper/chess-spatial-results.tex').open('x') as stream:
        stream.write('% Generated from the audited completed release.\n')
        for name, value in macros.items():
            stream.write('\\newcommand{\\'+name+'}{'+value+'}\n')
    lines = [r'\begin{table}[ht]', r'\centering\small', r'\begin{tabular}{lrrrrr}',
             r'\toprule', r'& & \multicolumn{2}{c}{Teacher agreement (\%)} & \multicolumn{2}{c}{Bounded engine difference} \\',
             r'Model & Parameters & Dev & Shift & Dev & Shift \\', r'\midrule']
    for mode in ('cnn', 'recurrent', 'reconstruct', 'predict'):
        params = next(row['parameter_count'] for row in summary['fits'] if row['mode'] == mode)
        regrets = [sum(summary['regret']['means'][split][f'{mode}-{seed}']['mean_bounded_regret']
                       for seed in summary['protocol']['seeds'])/len(summary['protocol']['seeds'])
                   for split in ('dev', 'shift')]
        lines.append(f"{('CNN' if mode == 'cnn' else mode.capitalize())} & {params:,} & {100*means['dev'][mode]['top1_teacher_agreement']:.2f} & "
                     f"{100*means['shift'][mode]['top1_teacher_agreement']:.2f} & {regrets[0]:.3f} & {regrets[1]:.3f} \\\\")
    greedy = summary['baselines']
    lines.extend([r'\midrule',
                  (f"Material heuristic & 0 & {100*greedy['dev']['greedy_material']['metrics']['top1_teacher_agreement']:.2f} & "
                  f"{100*greedy['shift']['greedy_material']['metrics']['top1_teacher_agreement']:.2f} & "
                  f"{summary['regret']['means']['dev']['greedy_material']['mean_bounded_regret']:.3f} & "
                  f"{summary['regret']['means']['shift']['greedy_material']['mean_bounded_regret']:.3f} \\\\"),
                  r'\bottomrule', r'\end{tabular}',
                  r'\caption{Mean of three fits for learned models. Higher agreement is better; lower bounded engine difference is better. Engine differences use 128 fixed positions per split and may be negative because searches are finite.}',
                  r'\end{table}', r'\begin{table}[ht]', r'\centering\small', r'\begin{tabular}{llrrr}',
                  r'\toprule', r'Model & Split & Changed squares (\%) & All squares (\%) & Exact boards (\%) \\', r'\midrule'])
    for mode in ('reconstruct', 'predict'):
        for split in ('dev', 'shift'):
            row = means[split][mode]
            lines.append(f"{mode.capitalize()} & {split.capitalize()} & {100*row['future_changed_accuracy']:.2f} & "
                         f"{100*row['future_square_accuracy']:.2f} & {100*row['future_exact_board_accuracy']:.2f} \\\\")
    lines.extend([r'\bottomrule', r'\end{tabular}',
                  r'\caption{Decoder predictions assessed against actual successors. Reconstruction is trained to output current pieces, so this table is not an equal-objective reconstruction-quality comparison.}',
                  r'\end{table}'])
    with (root/'paper/chess-spatial-tables.tex').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'status': 'written', 'plan_sha256': summary['plan_sha256'], 'gate_passed': passed}))


if __name__ == '__main__':
    main()
