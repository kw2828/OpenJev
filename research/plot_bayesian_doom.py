"""Render the preserved development result; matplotlib is an optional plotting dependency."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
data = json.loads((root/'evidence/bayesian-doom-v2/analysis-001.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
comparisons = ['bayes_mean_hit', 'bayes_lower_hit', 'rules', 'tiny_imitation']
labels = ['Bayesian mean', 'Bayesian lower bound', 'Rule baseline', 'Tiny imitation baseline']
colors = ['#3674a5', '#ad5348', '#478064', '#77808e']
for ax, scenario, title in zip(axes, data['paired_differences'], ['Same scenario', 'Shifted scenario']):
    for index, (method, color) in enumerate(zip(comparisons, colors)):
        row = data['paired_differences'][scenario][method+'_minus_map']
        center = row['mean_utility_difference']
        low, high = row['exploratory_95_interval']
        ax.errorbar(center, index, xerr=[[center-low], [high-center]], fmt='o', color=color,
                    capsize=4, markersize=7, linewidth=1.6)
    ax.axvline(0, color='#808080', linestyle='--', linewidth=1)
    ax.set_title(title+'\n'+scenario.replace('_', ' '), fontsize=12)
    ax.set_xlabel('Utility difference versus MAP\nHigher is better', fontsize=10)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(axis='x', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].invert_yaxis()
fig.suptitle('Doom pilot: posterior averaging did not establish a control gain', fontsize=14, y=.98)
fig.text(.5, .015, '440 episodes; five fitted models; ten paired evaluation seeds per scenario. Exploratory 95% crossed-bootstrap intervals.\nUtility = successful fire-command windows minus 0.25 per fire-command window. No novelty or safety claim.',
         ha='center', fontsize=8)
fig.tight_layout(rect=[0,.12,1,.9])
for extension in ['png', 'svg']:
    fig.savefig(root/f'evidence/bayesian-doom-v2/utility-comparison.{extension}', dpi=180)
