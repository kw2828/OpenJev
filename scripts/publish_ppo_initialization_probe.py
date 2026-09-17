"""Publish saved first-rollout gradient measurements without model execution."""

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['gru', 'feedforward', 'fast_global', 'fast_selective']
SEEDS = [101, 113, 127]
GROUPS = ['all', 'shared_trunk', 'actor_head', 'value_head']
COMPONENTS = ['actor_normalized', 'actor_raw', 'actor_centered_raw', 'entropy_weighted', 'critic_weighted']
CONDITIONS = list(itertools.product(['original', 'zero'], ['actual', 'synthetic_all_zero']))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=2e-5, abs_tol=1e-10)


def data_digest(data):
    digest = hashlib.sha256()
    types = {'float32': 'torch.float32', 'int64': 'torch.int64', 'bool': 'torch.bool'}
    for name in sorted(data):
        value = np.ascontiguousarray(data[name])
        digest.update(json.dumps([name, types[str(value.dtype)], list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate(run, plan):
    """Check receipts, saved arrays and numeric identities; never replay a model."""
    frozen, summary, completed = read(plan), read(run/'summary.json'), read(run/'completed.json')
    require(summary['protocol']['version'] == 'ppo-initialization-probe-v1', 'Wrong protocol')
    require(all(summary[key] == value for key, value in frozen.items()), 'Summary differs from plan')
    require(summary['plan_sha256'] == sha(plan) == completed['plan_sha256'], 'Plan hash mismatch')
    require(completed['summary_sha256'] == sha(run/'summary.json'), 'Summary hash mismatch')
    require(summary['status'] == completed['status'] == 'completed', 'Incomplete panel')
    require(summary['smoke'] is False and completed['smoke'] is False, 'Smoke is not the panel')
    require(summary['optimizer_steps'] == completed['optimizer_steps'] == 0, 'Unexpected optimization')
    require(summary['rollouts'] == completed['rollouts'] == len(summary['results']) == 12, 'Wrong rollout count')
    require(summary['environment_interactions'] == 12288, 'Wrong interaction count')
    require({(r['arm'], r['seed']) for r in summary['results']} == set(itertools.product(ARMS, SEEDS)),
            'Wrong arm/seed identities')
    start = read(run/'started.json')
    require(all(start[key] == value for key, value in frozen.items()), 'Start receipt differs from plan')
    require(start['plan_sha256'] == sha(plan) and start['smoke'] is False, 'Wrong start identity')
    for source, digest in frozen['source_sha256'].items():
        require(sha(ROOT/source) == digest, f'Frozen source changed: {source}')
    for row in summary['results']:
        directory = run/f"{row['arm']}-{row['seed']}"
        require(read(directory/'result.json') == row, 'Per-rollout result differs from summary')
        require(sha(directory/'rollout.npz') == row['rollout_npz_sha256'], 'Saved rollout changed')
        require(row['parameters_unchanged'] is True and row['optimizer_steps'] == 0 and row['smoke'] is False,
                'Wrong per-rollout scope')
        require(row['initial_parameter_sha256'] == row['final_parameter_sha256'], 'Parameters changed')
        rollout = row['rollout']
        require(rollout['paired_logits_states_and_actions_identical'] is True, 'Paired policies differ')
        with np.load(directory/'rollout.npz', allow_pickle=False) as data:
            require(data_digest(data) == rollout['data_sha256'], 'Array content digest mismatch')
            require(data['rewards'].shape == (64, 16), 'Wrong saved rollout shape')
            require(rollout['interactions'] == data['rewards'].size == 1024, 'Wrong rollout interactions')
            require(rollout['reward_sum'] == float(data['rewards'].sum()), 'Reward total differs')
            require(rollout['positive_reward_events'] == int((data['rewards'] > 0).sum()), 'Reward count differs')
            require(rollout['terminated_events'] == int(data['terminated'].sum()), 'Terminal count differs')
            require(rollout['truncation_only_events'] == int((data['ended'] & ~data['terminated']).sum()),
                    'Truncation count differs')
            require(len(rollout['completed_episodes']) == int(data['ended'].sum()), 'Episode count differs')
            require(rollout['action_counts'] == np.bincount(data['actions'].ravel(), minlength=7).tolist(),
                    'Action counts differ')
            require(not np.count_nonzero(data['values_zero']) and not np.count_nonzero(data['next_values_zero']),
                    'Zero head emitted nonzero values')
        conditions = {(a['value_head'], a['reward_condition']): a for a in row['analyses']}
        require(len(row['analyses']) == len(conditions) == 4 and set(conditions) == set(CONDITIONS),
                'Missing or duplicated analysis condition')
        for (head, reward), analysis in conditions.items():
            require(analysis['synthetic_rewards'] == (reward == 'synthetic_all_zero'), 'Wrong reward label')
            require(0 <= analysis['policy_entropy_nats'] <= math.log(7)+1e-6, 'Invalid entropy')
            require(set(analysis['gradient_components']) == set(GROUPS), 'Wrong gradient groups')
            for group, components in analysis['gradient_components'].items():
                norms, cosines = components['l2_norms'], components['cosines']
                require(set(norms) == set(COMPONENTS), 'Wrong gradient components')
                require(all(math.isfinite(x) and x >= 0 for x in norms.values()), 'Invalid gradient norm')
                require(set(cosines) == {f'{a}__{b}' for a, b in itertools.combinations(COMPONENTS, 2)},
                        'Missing gradient cosines')
                for pair, value in cosines.items():
                    a, b = pair.split('__')
                    require(value is None if norms[a]*norms[b] == 0 else
                            value is not None and math.isfinite(value) and abs(value) <= 1+1e-6,
                            'Invalid cosine or undefined-cosine convention')
                ratio = components['normalized_actor_to_weighted_entropy_norm_ratio']
                require(ratio is None if norms['entropy_weighted'] == 0 else
                        close(ratio, norms['actor_normalized']/norms['entropy_weighted']), 'Incorrect norm ratio')
                require(close(norms['actor_normalized'],
                              norms['actor_centered_raw']/analysis['normalization_divisor']),
                        'Normalization gradient identity failed')
            if head == 'zero' and (reward == 'synthetic_all_zero' or rollout['positive_reward_events'] == 0):
                require(analysis['raw_advantage']['nonzero_count'] == 0, 'Zero critic/reward GAE not zero')
                for components in analysis['gradient_components'].values():
                    require(all(components['l2_norms'][key] == 0 for key in COMPONENTS if key != 'entropy_weighted'),
                            'Zero critic/reward has actor or critic gradient')
        for reward in ['actual', 'synthetic_all_zero']:
            original, zero = conditions['original', reward], conditions['zero', reward]
            require(original['policy_entropy_nats'] == zero['policy_entropy_nats'], 'Paired entropy differs')
            for group in GROUPS:
                require(original['gradient_components'][group]['l2_norms']['entropy_weighted'] ==
                        zero['gradient_components'][group]['l2_norms']['entropy_weighted'], 'Entropy gradient differs')
        if rollout['positive_reward_events'] == 0 and rollout['reward_sum'] == 0:
            for head in ['original', 'zero']:
                actual, synthetic = conditions[head, 'actual'], conditions[head, 'synthetic_all_zero']
                ignored = {'reward_condition', 'synthetic_rewards'}
                require({k: v for k, v in actual.items() if k not in ignored} ==
                        {k: v for k, v in synthetic.items() if k not in ignored}, 'Zero-reward conditions differ')
    return summary


def compact(summary):
    rows = []
    for result in summary['results']:
        actual = next(a for a in result['analyses'] if a['value_head'] == 'original' and a['reward_condition'] == 'actual')
        trunk = actual['gradient_components']['shared_trunk']
        norms = trunk['l2_norms']
        rows.append({'arm': result['arm'], 'seed': result['seed'],
                     'raw_advantage_std': actual['raw_advantage']['std_population'],
                     'normalization_multiplier': 1/actual['normalization_divisor'],
                     'initial_policy_entropy_nats': actual['policy_entropy_nats'],
                     'shared_trunk_gradient_norms': norms,
                     'shared_trunk_actor_to_entropy_ratio': trunk['normalized_actor_to_weighted_entropy_norm_ratio'],
                     'shared_trunk_critic_to_actor_ratio': norms['critic_weighted']/norms['actor_normalized'],
                     'shared_trunk_actor_critic_cosine': trunk['cosines']['actor_normalized__critic_weighted'],
                     'all_gradient_norms': actual['gradient_components']['all']['l2_norms'],
                     'all_actor_to_entropy_ratio': actual['gradient_components']['all'][
                         'normalized_actor_to_weighted_entropy_norm_ratio']})
    return {'scope': 'Initial gradient audit; no optimizer updates or effectiveness evaluation',
            'rollouts': len(rows), 'environment_interactions': summary['environment_interactions'],
            'positive_reward_events': sum(r['rollout']['positive_reward_events'] for r in summary['results']),
            'completed_episodes': sum(len(r['rollout']['completed_episodes']) for r in summary['results']),
            'uniform_policy_entropy_nats': math.log(7), 'elapsed_seconds': summary['elapsed_seconds'],
            'per_fit_original_actual': rows, 'limitations': summary['limitations']}


def gradients_csv(summary):
    rows = []
    for result in summary['results']:
        for analysis in result['analyses']:
            for group, components in analysis['gradient_components'].items():
                rows.append({'arm': result['arm'], 'seed': result['seed'],
                             'value_head': analysis['value_head'], 'reward_condition': analysis['reward_condition'],
                             'group': group, 'entropy_nats': analysis['policy_entropy_nats'],
                             'raw_advantage_std': analysis['raw_advantage']['std_population'],
                             'normalization_divisor': analysis['normalization_divisor'],
                             **{f'norm_{k}': v for k, v in components['l2_norms'].items()},
                             **{f'cosine_{k}': v for k, v in components['cosines'].items()},
                             'actor_to_entropy_norm_ratio': components['normalized_actor_to_weighted_entropy_norm_ratio']})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def figure(audit, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), gridspec_kw={'width_ratios': [1.5, 1]})
    rows = audit['per_fit_original_actual']
    labels = {'gru': 'GRU', 'feedforward': 'Adapter', 'fast_global': 'Global', 'fast_selective': 'Selective'}
    y = np.arange(len(rows))
    for ax in axes:
        ax.set_yticks(y, [f"{labels[r['arm']]} / {r['seed']}" for r in rows])
        ax.invert_yaxis()
        for start in (0, 6):
            ax.axhspan(start-.5, start+2.5, color='#f1f5f9', zorder=0)
        ax.grid(axis='x', alpha=.2)
    styles = [('actor_centered_raw', 'Actor, centered raw GAE', '#90a4b8', 'x'),
              ('actor_normalized', 'Actor, normalized GAE', '#3166a3', 'o'),
              ('entropy_weighted', 'Entropy, coefficient 0.01', '#179082', 's'),
              ('critic_weighted', 'Critic, coefficient 0.5', '#be773a', '^')]
    for key, label, color, marker in styles:
        axes[0].scatter([r['shared_trunk_gradient_norms'][key] for r in rows], y,
                        label=label, c=color, marker=marker, s=34, zorder=3)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Shared-trunk gradient L2 norm (log scale)')
    axes[0].set_title('Original critic: component gradients', loc='left', fontweight='bold')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.43, .085),
               ncol=2, frameon=False, fontsize=8.5)
    axes[1].scatter([r['normalization_multiplier'] for r in rows], y, c='#3166a3', s=38)
    axes[1].axvline(1, color='#68788b', linestyle='--', linewidth=1)
    axes[1].set_xlim(0, 26)
    axes[1].set_xlabel('Normalized / centered-raw actor gradient norm')
    axes[1].set_title('Unit-variance rescaling', loc='left', fontweight='bold')
    fig.suptitle('First-rollout gradients before any learning', x=.04, ha='left', fontsize=17, fontweight='bold')
    fig.text(.04, .925, '12 fresh policies | 12,288 actual interactions | 0 reward events | 0 optimizer steps',
             color='#40516a', fontsize=10.5)
    fig.text(.04, .025, 'Zero critic: actor and critic gradients are exactly zero; entropy gradients are unchanged.\n'
             'Policies begin nearly uniform. Gradient ratios do not show a harmful update or a training benefit.',
             fontsize=9, color='#40516a')
    fig.subplots_adjust(left=.13, right=.985, top=.84, bottom=.27, wspace=.42)
    for suffix in ['png', 'svg']:
        fig.savefig(out/f'initialization-gradients.{suffix}', dpi=190, facecolor='white')
    plt.close(fig)


def publish(run, plan, out):
    summary = validate(run, plan)
    audit = compact(summary)
    # This fixed completed panel has no rewards. Do not reuse the zero-reward caption for another result.
    require(audit['positive_reward_events'] == audit['completed_episodes'] == 0, 'Figure assumes this unrewarded panel')
    filenames = ['summary.json', 'completed.json', 'audit.json', 'gradients.csv',
                 'initialization-gradients.png', 'initialization-gradients.svg', 'publication.json']
    require(not any((out/name).exists() for name in filenames), 'Publication never overwrites existing evidence')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        require(sha(out/'plan.json') == sha(plan), 'Existing published plan differs')
    # The frozen plan is managed separately and is never written by this publisher.
    for name in ['summary.json', 'completed.json']:
        (out/name).write_bytes((run/name).read_bytes())
    (out/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    (out/'gradients.csv').write_text(gradients_csv(summary))
    figure(audit, out)
    manifest = {'publisher_sha256': sha(__file__), 'plan_sha256': sha(plan),
                'source_summary_sha256': sha(run/'summary.json'),
                'source_completed_sha256': sha(run/'completed.json'),
                'validation': 'Receipt, source, raw-array hashes and gradient identities; no model execution',
                'raw_rollouts': {f"{r['arm']}-{r['seed']}": r['rollout_npz_sha256'] for r in summary['results']},
                'raw_rollouts_published': False,
                'files': {name: sha(out/name) for name in filenames if name != 'publication.json'}}
    (out/'publication.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit = publish(args.run, args.plan, args.out)
    print(json.dumps({key: audit[key] for key in ['rollouts', 'environment_interactions', 'positive_reward_events']}))


if __name__ == '__main__':
    main()
