"""Plot all frozen pilot methods, paired contrasts and measured local costs."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'grpo': 'GRPO', 'dr_grpo': 'Dr. GRPO adaptation', 'dapo_clip_loss': 'DAPO loss adaptation',
          'rloo': 'RLOO adaptation', 'srpo_pixels': 'SRPO pixels', 'srpo_random_video': 'SRPO random video',
          'srpo_vjepa': 'SRPO pretrained V-JEPA 2', 'ppo_sparse': 'Sparse PPO', 'a2c_sparse': 'Sparse A2C',
          'ppo_frozen': 'Unchanged PPO', 'always_fire': 'Always fire'}

ap = argparse.ArgumentParser()
ap.add_argument('analysis', type=Path)
ap.add_argument('--prefix', type=Path, required=True)
a = ap.parse_args()
r = json.loads(a.analysis.read_text())
arms = list(LABELS)
colors = {arm: '#198366' if arm == 'srpo_vjepa' else '#707986' if arm in ('ppo_frozen', 'always_fire')
          else '#497eaa' for arm in arms}
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, scenario, name in zip(axes[0], ['defend_the_center', 'defend_the_line'], ['Center', 'Line'], strict=True):
    for i, arm in enumerate(arms):
        row = r['summary'][scenario][arm]
        ax.scatter(row['kills'], i, color=colors[arm], s=38, zorder=3)
        if arm != 'always_fire':
            ax.scatter(row['per_fit_kills'], [i]*3, color=colors[arm], marker='x', alpha=.6, s=24)
    ax.set_yticks(range(len(arms)), [LABELS[arm] for arm in arms])
    ax.invert_yaxis()
    ax.set_xlabel('Kills per episode')
    ax.set_title(f'{name}: final policy evaluation')
ax = axes[1, 0]
for i, arm in enumerate(arms[:9]):
    val = r['training'][arm]['total_fit_wall_seconds']
    ax.barh(i, val, color=colors[arm], alpha=.85)
    ax.text(val+2, i, f'{val:.0f}s', va='center', fontsize=8)
ax.set_yticks(range(9), [LABELS[arm] for arm in arms[:9]])
ax.invert_yaxis()
ax.set_title('Measured additional training time')
ax.set_xlabel('Workstation seconds, all three fits')
ax.margins(x=.17)
ax = axes[1, 1]
controls = ['grpo', 'srpo_pixels', 'srpo_random_video', 'ppo_sparse', 'ppo_frozen', 'always_fire']
for i, arm in enumerate(controls):
    entry = r['contrasts']['defend_the_center']['srpo_vjepa'][arm]['kills']
    low, high = entry['exploratory_95_interval']
    ax.plot([low, high], [i, i], color='#198366', linewidth=2)
    ax.scatter(entry['difference'], i, color='#198366', zorder=3)
ax.axvline(0, color='#777', linewidth=.8)
ax.axvline(.5, color='#b87940', linestyle=':', linewidth=1.)
ax.set_yticks(range(len(controls)), [f'vs {LABELS[arm]}' for arm in controls])
ax.invert_yaxis()
ax.set_title('Pretrained JEPA: paired Center differences')
ax.text(.99, .98, 'Dotted line: +0.5 continuation threshold', transform=ax.transAxes,
        ha='right', va='top', fontsize=7, color='#8a592e')
ax.set_xlabel('Kill difference; exploratory 95% bootstrap interval')
for ax in axes.ravel():
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#e8edf2', linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
status = 'continuation screen passed' if r['eligible_for_fresh_confirmation'] else 'continuation criteria not met'
fig.suptitle(f'OpenJev | JEPA rewards and RL: {status}', fontsize=15, y=.98)
fig.text(.5, .018, '27 fits, 8,192 new interactions each; 16 paired game seeds per scenario. Dots: means; crosses: fit means.\n'
    'Intervals resample both game and training seeds; three fits, no multiple-comparison correction. Development pilot only.\n'
    'Local time includes video processing and updates; historical PPO training and external JEPA pretraining excluded.\n'
    'JEPA supplies training rewards only. The deployed policy is the same small structured-observation actor.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, .105, 1, .95), h_pad=2.5, w_pad=2.2)
a.prefix.parent.mkdir(parents=True, exist_ok=True)
for ext in ['png', 'svg']:
    fig.savefig(a.prefix.with_suffix('.'+ext), dpi=180, facecolor='white')
plt.close(fig)
