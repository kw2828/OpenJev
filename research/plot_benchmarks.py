"""Generate benchmark figures and paper tables from preserved evidence, without rerunning experiments."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/benchmarks'
PAPER = ROOT / 'paper/generated'
SOURCES = ['evidence/defend-center.json', 'evidence/rust-python-score-kernel.json',
           'evidence/bayesian-doom-v2/analysis-001.json']
LABELS = {'rules': 'Rule baseline', 'tiny_imitation': 'Tiny imitation', 'map_hit': 'Logistic MAP',
          'bayes_mean_hit': 'Bayesian mean', 'bayes_lower_hit': 'Bayesian lower bound'}


def save(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    gameplay, speed, study = [json.loads((ROOT / p).read_text()) for p in SOURCES]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'openjev-benchmarks-v1', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8, 3.7))
    keys = ['random', 'local', 'rules']
    for i, (key, label) in enumerate(zip(keys, ['Random', 'Tiny imitation', 'Rule teacher'])):
        rows = sorted((r for r in gameplay['episodes'] if r['policy'] == key), key=lambda r: r['seed'])
        assert len(rows) == 20 and [r['seed'] for r in rows] == list(range(1000, 1020))
        y = np.array([r['kills'] for r in rows])
        ax.scatter(i + np.linspace(-.12, .12, len(y)), y, alpha=.55, s=22, color='#3674a5')
        ax.hlines(y.mean(), i-.22, i+.22, color='#ad5348', linewidth=3)
        ax.text(i, max(y)+1, f'mean {y.mean():.2f}', ha='center')
    ax.set(xticks=range(3), xticklabels=['Random', 'Tiny imitation', 'Rule teacher'],
           ylabel='Kills per episode', ylim=(-1, 27), title='Original gameplay benchmark: imitation tracks its teacher')
    ax.grid(axis='y', alpha=.15)
    fig.text(.5, .01, '20 matched seeds per policy; 2-tic actions; dots = episodes, red line = mean. Separate from the Bayesian pilot.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1])
    save(fig, 'gameplay')

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, key in enumerate(['python_numpy', 'rust']):
        row = speed[key]
        values = np.array(row['raw_batch_ms']) / speed['rows']
        assert np.isclose(np.median(values), row['median_ms_per_row'])
        ax.scatter(values, i+np.linspace(-.1,.1,len(values)), s=23, color='#3674a5', alpha=.7)
        ax.errorbar(row['median_ms_per_row'], i, xerr=[[row['median_ms_per_row']-row['p25_ms_per_row']],
                    [row['p75_ms_per_row']-row['median_ms_per_row']]], fmt='|', markersize=20,
                    color='#ad5348', capsize=5, linewidth=2)
    ax.set(yticks=[0,1], yticklabels=['Python / NumPy', 'Rust release'], xlim=(0,.65), ylim=(-.5,1.5),
           xlabel='Milliseconds per row (batch time / 32); lower is better',
           title=f"Score kernel only: {speed['python_time_divided_by_rust_time']:.2f}x ratio of medians")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15)
    fig.text(.5,.01,'11 timed batches each; dots = batches, red mark = median and IQR. Excludes model inference, Doom and I/O.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1])
    save(fig, 'score-kernel')

    fig, axes = plt.subplots(1,2,figsize=(9,3.8), sharey=True)
    for ax, scenario, name in zip(axes, ['defend_the_center','defend_the_line'], ['Center','Shifted line']):
        audit = study['common_audit'][scenario]
        for r in audit:
            ax.plot([0,1], [r['map']['brier'],r['bayes_mean']['brier']], color='#3674a5', marker='o',alpha=.6)
        means = [np.mean([r[k]['brier'] for r in audit]) for k in ['map','bayes_mean']]
        ax.scatter([0,1], means, marker='D', color='#ad5348', s=65, zorder=5)
        for x,y in enumerate(means):
            ax.annotate(f'{y:.6f}', (x,y), xytext=(0,12), textcoords='offset points',ha='center',fontsize=9)
        ax.set(xticks=[0,1],xticklabels=['MAP','Bayesian mean'],xlim=(-.4,1.4), title=f"{name}: {audit[0]['n']} audit events")
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Brier score; lower is better')
    fig.suptitle('Shared audit: probability accuracy across five fitted models',fontsize=12)
    fig.text(.5,.01,'Each line = one training replicate; diamond = mean. Same audit events for all fits; 10 episodes per scenario.\nBrier score measures overall probability accuracy, not calibration alone. Events within episodes are dependent.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.12,1,.94])
    save(fig, 'audit-brier')

    table = [r'\begin{tabular}{lrrr}', r'\toprule', r'Controller & Center utility & Line utility & Line kills \\', r'\midrule']
    for key,label in LABELS.items():
        c = study['summary']['defend_the_center'][key]
        l = study['summary']['defend_the_line'][key]
        table.append(f"{label} & {c['utility']:.3f} & {l['utility']:.3f} & {l['kills']:.2f} " + r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}'])
    (PAPER/'results-table.tex').write_text('\n'.join(table)+'\n')
    manifest = {'purpose': 'Visualization of existing development evidence; no new experimental runs',
                'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
                'figures': ['gameplay','score-kernel','audit-brier']}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
