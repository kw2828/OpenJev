"""Visualize sealed expert-coordination results, without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {
    'fast': 'Recency + robust expert', 'slow': 'GRU expert',
    'position_fast': 'Fast position + GRU rotation', 'position_slow': 'GRU position + fast rotation',
    'half': 'Fixed half mixing', 'constant': 'Learned constants',
    'summary': 'Context summary MLP', 'recurrent': 'Recurrent selector',
}
PANELS = ('test_sin', 'test_zigzag')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(completed, summary, out):
    done, audited = (json.loads(p.read_text()) for p in (completed, summary))
    if done['status'] != 'completed' or len(done['rows']) != 48:
        raise ValueError('completed48-row study required')
    if audited['execution_completed_sha256'] != sha(completed):
        raise ValueError('audit completion binding')
    for name, binding in done['files'].items():
        if sha(completed.parent / name) != binding['sha256']:
            raise ValueError('altered payload: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#07836e', '#3b6391', '#809889', '#9b9eb1', '#a2937c', '#d2964d', '#9875b4', '#cf514b']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for col, panel in enumerate(PANELS):
        for row, metric in enumerate(('position_rmse_m', 'rotation_rmse_rad')):
            ax = axes[row, col]
            for y, variant in enumerate(NAMES):
                values = np.array([r['metrics'][metric] for r in done['rows']
                                   if r['panel'] == panel and r['variant'] == variant])
                if values.shape != (3,):
                    raise ValueError('three fits per row required')
                ax.scatter(values, np.full(3, y), color=colors[y], s=26)
                ax.scatter(np.sqrt(np.mean(values**2)), y, marker='|', s=170, color=colors[y])
            ax.set_yticks(range(len(NAMES)), NAMES.values())
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            ax.set_xlabel(('Position RMSE (meters)' if row == 0 else 'Rotation RMSE (radians)') + '; lower is better')
            if row == 0:
                ax.set_title('Plain archive' if col == 0 else 'Zigzag archive')
    axes[0, 0].invert_yaxis()
    fig.suptitle('OpenJev: does recurrent expert coordination help?', fontsize=19, weight='bold')
    gate = audited['continuation_gate']
    fig.tight_layout(rect=(0, .1, 1, .96))
    fig.text(.02, .02, f"Continuation: {'PASS' if gate['passed'] else 'FAIL'}, "
             f"{gate['requirements_passed']}/17 requirements. Dots: all three seeds; bars: pooled RMSE.\n"
             'Two frozen experts; nine new selector fits. All 160 windows per archive and all 25 forecast steps.\n'
             'Exposed simulated-robot development data. The full gate also retains all 20 previous baselines.', fontsize=10)
    fig.savefig(out / 'physical-errors.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    for y, variant in enumerate(NAMES):
        values = np.asarray([v for r in done['rows'] if r['variant'] == variant for v in r['latency_ms']])
        median, p95 = np.median(values), np.percentile(values, 95)
        ax.plot([median, p95], [y, y], color=colors[y], linewidth=3)
        ax.scatter(median, y, s=45, color=colors[y])
        ax.text(p95 + .06, y, f'{median:.2f} / {p95:.2f} ms', va='center', fontsize=9)
    ax.set_yticks(range(len(NAMES)), NAMES.values())
    ax.invert_yaxis()
    ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.17)
    ax.set_xlabel('Full context plus 25-step forecast, median / p95 milliseconds')
    ax.grid(axis='x', alpha=.2)
    ax.set_title('Combinations pay for both experts and the selector', fontsize=16, weight='bold')
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.text(.015, .025, 'Apple M5 Max, one CPU thread. 120 timed windows per configuration, after warmups.\n'
             'Includes expert computation, context tokens, gate and mixing; excludes loading and reporting.', fontsize=10)
    fig.savefig(out / 'prediction-cost.png', dpi=160)
    plt.close(fig)
    receipt = {'status': 'completed', 'completed_sha256': sha(completed), 'summary_sha256': sha(summary),
               'source_sha256': sha(__file__), 'new_model_calls': 0,
               'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in out.iterdir()}}
    with (out / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.completed, args.summary, args.out)
