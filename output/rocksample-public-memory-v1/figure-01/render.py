"""Render every saved map and fixed predictor; no new predictions or episodes."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[3]
source = root / 'output/rocksample-public-memory-v1/run-01/summary.json'
s = json.loads(source.read_text())
out = Path(__file__).parent / 'render-01'
out.mkdir(exist_ok=False)
arms = ['prior', 'recent32', 'recent128', 'latest_check', 'full']
labels = ['Prior', 'Recent 32', 'Recent 128', 'Latest check', 'Full history']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.1), sharey=True)
fig.patch.set_facecolor('#f8fafc')
values = {}
for ax, scope, title in zip(axes, ['overall', 'delayed'], ['All 1,651 checks', '42 checks with earlier reading >128 steps ago']):
    ax.set_facecolor('#f8fafc')
    rows = [m['scores'] if scope == 'overall' else m['ages']['gt128']['scores'] for m in s['by_map']]
    matrix = np.array([[row[a]['nll'] for a in arms] for row in rows])
    mean = s['scores_equal_map_means'] if scope == 'overall' else s['delayed_equal_map_means']
    values[scope] = {'per_map': matrix.tolist(), 'map_seeds': [m['map_seed'] for m in s['by_map']], 'means': [mean[a]['nll'] for a in arms]}
    for row in matrix:
        ax.plot(range(5), row, color='#94a3b8', alpha=.55, linewidth=.8, marker='o', markersize=3, zorder=1)
    means = np.array(values[scope]['means'])
    ax.scatter(range(5), means, c=['#334155']*4+['#2563eb'], s=86, marker='D', edgecolor='white', linewidth=.7, zorder=3)
    for i, value in enumerate(means):
        ax.annotate(f'{value:.3f}', (i, value), xytext=(0, -19), textcoords='offset points', ha='center', weight='bold', color='#0f172a')
    ax.set_xticks(range(5), labels, rotation=18)
    ax.set_title(title, loc='left', pad=15, weight='bold', fontsize=11)
    ax.set_ylim(0, max(1.05, float(matrix.max())+.1))
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Check prediction log loss (lower is better)')
fig.suptitle('RockSample: older history helps, but misses the overall continuation rule', x=.055, ha='left', fontsize=15, weight='bold', color='#0f172a')
fig.text(.055,.87, '8 constructor maps · 32 fixed exploration fragments · raw public inputs · no trained models', color='#475569', fontsize=10)
fig.text(.055,.135, 'Full history: 3.1% better than recent 128 and 8.5% better than latest check. Required: at least 10% against both.', color='#9a3412', fontsize=10, weight='bold')
fig.text(.055,.088, '4/6 conditions pass. No learned-memory pilot admitted. Delayed queries are only 2.54% of checks.', color='#9a3412', fontsize=10)
fig.text(.055,.047, 'Gray lines: every map. Diamonds: equal-map means. Fixed exploration avoids exiting; this is not a task-return benchmark.', color='#475569', fontsize=9)
fig.subplots_adjust(left=.075, right=.975, top=.78, bottom=.245, wspace=.18)
fig.savefig(out/'public-memory.png', dpi=180, facecolor=fig.get_facecolor())
fig.savefig(out/'public-memory.pdf', facecolor=fig.get_facecolor())
(out/'plotted-values.json').write_text(json.dumps(values,indent=2)+'\n')
receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},'scope':'saved-only figure; no prediction or environment calls'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
