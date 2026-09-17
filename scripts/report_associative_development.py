"""Export the unsuccessful development screen without opening confirmation."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openjev.research.text_distillation import file_hash, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or (args.run/'confirmation').exists():
        raise ValueError('Fresh report required; confirmation must not have been opened')
    selection = json.loads((args.run/'selection.json').read_text())
    manifest = json.loads((args.run/'packet/manifest.json').read_text())
    configs = {c['name']: c for c in selection['configs']}
    selected = configs[selection['selected']]
    write_new(args.out/'development.json', {
        'status': 'stopped_after_development', 'confirmation_executed': False,
        'frozen_code_commit': '02f60f5', 'selection': selection,
        'audit': manifest['audit'], 'parts': {k: {n: v for n, v in info.items() if n != 'ids'}
                                           for k, info in manifest['parts'].items()},
        'source_sha256': manifest['source_sha256'], 'selection_sha256': file_hash(args.run/'selection.json'),
        'reason': 'Prototype dominates tested recurrent variants on development utility and memory-search cost. '
                  'Selected recurrent gate exceeds development cost target; do not consume confirmation.',
        'next_test': 'Learned metric and recurrent head versus parameter-matched feedforward/zero-step controls. '
                     'Choose using development data; preserve unused confirmation and calibration.',
        'limitations': ['Development results used for architecture decisions, not held-out efficacy.',
                       'Only 50 OOS development examples.', 'Encoder remains a pretrained transformer.',
                       'Earlier Astra supervision still pending credentials.']})
    fig, ax = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    for name, row in configs.items():
        x, y = row['memory_passes_per_query'], row['metrics']['balanced_utility']*100
        color = '#169875' if name == 'prototype' else '#6b7280'
        if name == selected['name']:
            color = '#b44242'
        ax.scatter(x, y, color=color, s=75 if name in ('prototype', selected['name']) else 35)
        if name in ('prototype', 'nearest', 'recurrent', selected['name']):
            label = name.replace('gated_recurrent-0.75', 'Selected recurrent gate')
            ax.annotate(label, (x, y), xytext=(5, 7), textcoords='offset points', fontsize=9)
    ax.axvline(.5, color='#b5b9c0', linestyle=':', label='Confirmation cost ceiling')
    ax.set(xlim=(-.12, 2.3), ylim=(80, 95), xlabel='Full memory-bank searches per request',
           ylabel='Development balanced utility (%)',
           title='Associative routing: simpler prototype wins the development screen')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right')
    fig.savefig(args.out/'development.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
