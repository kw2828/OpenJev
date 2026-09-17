"""Plot preserved pilot results without rerunning or changing the experiment."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('analysis', type=Path)
p.add_argument('--prefix', type=Path, required=True)
a = p.parse_args()
r = json.loads(a.analysis.read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.patch.set_facecolor('white')
labels = {'group_binary': 'Group binary', 'srpo_raw': 'SRPO raw adaptation',
          'srpo_latent': 'SRPO latent adaptation', 'ppo_frozen': 'Frozen PPO history',
          'always_fire': 'Always fire'}
colors = {'group_binary': '#3975a7', 'srpo_raw': '#9b70ac', 'srpo_latent': '#268e63',
          'ppo_frozen': '#787f89', 'always_fire': '#b78254'}
arms = list(labels)
for ax, scenario, title in zip(axes[0], ['defend_the_center', 'defend_the_line'], ['Center', 'Line'], strict=True):
    for i, arm in enumerate(arms):
        entry = r['summary'][scenario][arm]
        ax.scatter(entry['kills'], i, color=colors[arm], s=50, zorder=3)
        if arm != 'always_fire':
            ax.scatter(entry['per_fit_kills'], [i]*3, color=colors[arm], marker='x', s=24, alpha=.5)
    ax.set_yticks(range(len(arms)), [labels[x] for x in arms])
    ax.invert_yaxis()
    ax.set_title(f'{title}: final mean kills')
    ax.set_xlabel('Kills per episode')
for ax, metric, title in zip(axes[1], ['interactions', 'wall'],
                            ['New training interactions', 'Additional training wall time'], strict=True):
    for i, arm in enumerate(arms[:3]):
        entry = r['training'][arm]
        val = entry['interactions'] if metric == 'interactions' else (
            entry['actor_training_wall_seconds']+entry['encoder_training_wall_seconds'])
        ax.barh(i, val, color=colors[arm], alpha=.8)
        ax.text(val*.98, i, f'{val:,.0f}', ha='right', va='center', color='white', fontsize=9)
    ax.set_yticks(range(3), [labels[x] for x in arms[:3]])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Transitions across three fits' if metric == 'interactions' else 'Workstation seconds across three fits')
for ax in axes.ravel():
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#e9edf1', linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
status = 'eligible for fresh confirmation' if r['eligible_for_fresh_confirmation'] else 'continuation criteria not met'
fig.suptitle(f'OpenJev | SRPO adaptation pilot: {status}', fontsize=13, y=.99)
fig.text(.5, .018, 'Dots: means; crosses: three individual fit means, not confidence intervals. 24 paired seeds per scenario.\n'
         'Equal trajectory-group budgets. Time includes rollouts, updates, encoding and I/O; prior PPO training excluded.\n'
         'Structured dynamics encoder, not V-JEPA/OpenVLA replication. Development results only.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, .10, 1, .95), h_pad=2.0, w_pad=2.0)
a.prefix.parent.mkdir(parents=True, exist_ok=True)
for extension in ['png', 'svg']:
    fig.savefig(a.prefix.with_suffix('.'+extension), dpi=180, facecolor='white')
plt.close(fig)
