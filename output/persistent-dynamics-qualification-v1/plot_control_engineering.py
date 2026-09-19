"""Plot all six completed engineering rows; no new simulation or inference."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
RUN = BASE / 'control-engineering-01'
ARMS = ('nominal', 'adaptive', 'frozen', 'public_gain', 'true_state', 'zero')
LABELS = ('Nominal gain', 'Rolling estimate', 'Frozen estimate', 'Known current gain', 'Known state + gain', 'Zero command')
COLORS = ('#64748b', '#2563eb', '#d97706', '#059669', '#7c3aed', '#b91c1c')


def read(path):
    return json.loads(path.read_text())


def main():
    done = read(RUN / 'completed.json')
    review = read(BASE / 'control-results-review-01' / 'control-results-review.json')
    assert done['status'] == 'completed' and not (RUN / 'failed.json').exists()
    assert review['status'] == 'completed'
    for item in done['audits']:
        p = RUN / ('audit-' + item['arm'] + '.json')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
    episodes = {arm: read(RUN / 'rows' / arm / 'episode.json') for arm in ARMS}
    rewards = {arm: np.array([t['reward'] for t in ep['audit']['transitions']]) for arm, ep in episodes.items()}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1, 1.1]})
    for arm, label, color in zip(ARMS, LABELS, COLORS, strict=True):
        axes[0].plot(np.arange(1, 201), -np.cumsum(rewards[arm]), label=label, color=color, lw=1.9)
    axes[0].set(title='Native cumulative cost', xlabel='Executed action', ylabel='Cost (lower is better)')
    axes[0].legend(fontsize=8, loc='upper left', frameon=False)
    axes[0].axvline(80, c='#111827', ls='--', lw=1)
    post = [-rewards[arm][80:].mean() for arm in ARMS]
    axes[1].barh(LABELS, post, color=COLORS)
    axes[1].invert_yaxis()
    axes[1].set(title='After the hidden gain change', xlabel='Native cost per action, actions 80-199')
    for i, value in enumerate(post):
        axes[1].text(value + max(post)*.02, i, f'{value:.5f}', va='center', fontsize=8)
    axes[1].set_xlim(0, max(post)*1.2)
    with np.load(RUN / 'case.npz', allow_pickle=False) as case:
        gains = case['gains'].copy()
    axes[2].step(np.arange(200), gains, where='post', c='#111827', lw=2, label='Actual gain (audit only)')
    for arm, label, color in [('adaptive', 'Rolling estimate at action', COLORS[1]), ('frozen', 'Frozen estimate at action', COLORS[2])]:
        estimates=[]
        for t in range(200):
            with np.load(RUN/'rows'/arm/'decisions'/f'{t:03d}.npz', allow_pickle=False) as values:
                estimates.append(float(values['planning_gain']))
        axes[2].step(np.arange(200), estimates, where='post', color=color, lw=1.5, label=label)
    axes[2].set(title='Parameter used for planning', xlabel='Executed action', ylabel='Motor gain', ylim=(.45,1.55))
    axes[2].legend(fontsize=8, frameon=False, loc='lower right')
    for ax in (axes[0], axes[2]):
        for t in (50, 100, 150):
            ax.axvline(t, c='#94a3b8', lw=.7, ls=':')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle('Changing motor strength: one engineering case, all six controllers\nSeed 410 reused; no training, scientific qualification or architecture claim', fontsize=13)
    fig.savefig(BASE/'control-engineering-results.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
