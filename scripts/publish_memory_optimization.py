"""Verify and plot the completed four-context memory optimization diagnostic.

Checks frozen sources, receipts, logs, data, all saved predictions and model
replays. Loads existing weights for inference and exact baseline comparison only;
never trains or changes a checkpoint. Figures are descriptive, not novelty claims.
"""

import argparse
import hashlib
import importlib.util
import io
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CELLS = [('recursive', 'ce7'), ('cache_h0', 'ce7'), ('recursive', 'ce2'), ('cache_h0', 'ce2')]
LABELS = ['Recursive + 7-action loss', 'Cached h0 + 7-action loss',
          'Recursive + 2-choice loss', 'Cached h0 + 2-choice loss']
SHORT_LABELS = ['Recursive\n7-action loss', 'Cached h0\n7-action loss',
                'Recursive\n2-choice loss', 'Cached h0\n2-choice loss']
COLORS = ['#53708f', '#cf8642', '#178b87', '#7856b6']
METRICS = ['accuracy7', 'accuracy2', 'conditional_binary_cross_entropy',
           'target_probability7', 'target_probability2']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, label, atol=3e-6):
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual-expected) > atol:
        raise ValueError(f'{label} does not replay: {actual!r} versus {expected!r}')


def validate_source(plan_path, version):
    plan = read(plan_path)
    if plan['protocol']['version'] != version:
        raise ValueError('Wrong frozen plan version')
    for name, expected in plan['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    if sha(ROOT/'uv.lock') != plan['lock_sha256']:
        raise ValueError('Frozen lockfile changed')
    return plan


def load_completed(run_dir, plan_path, expected_version):
    plan = validate_source(plan_path, expected_version)
    completed, summary = read(run_dir/'completed.json'), read(run_dir/'summary.json')
    if (completed['status'] != 'completed' or completed['fits'] != 12
            or completed['optimizer_updates'] != 6000 or completed['plan_sha256'] != sha(plan_path)
            or completed['summary_sha256'] != sha(run_dir/'summary.json')
            or summary['plan_sha256'] != sha(plan_path) or summary['protocol'] != plan['protocol']
            or summary['optimizer_updates'] != 6000):
        raise ValueError('Completion, summary or frozen-plan identity mismatch')
    training = read(run_dir/'training-completed.json')
    if (training['status'] != 'completed' or training['plan_sha256'] != sha(plan_path)
            or training['fits'] != summary['fits'] or len(summary['fits']) != 12):
        raise ValueError('Training completion receipts disagree')
    for key in ('rl_training', 'rl_warmstart', 'novelty_established', 'independent_confirmation'):
        if summary[key] is not False:
            raise ValueError('Incorrect diagnostic claim boundary')
    return plan, summary


def validate_fit(directory, receipt, plan_hash, optimization):
    if (read(directory/'completed.json') != receipt or receipt['status'] != 'completed'
            or receipt['plan_sha256'] != plan_hash or receipt['updates'] != 500
            or receipt['sequences_seen'] != 2000 or receipt['smoke'] is not False
            or receipt['supervised_only'] is not True or receipt['rl_warmstart'] is not False):
        raise ValueError('Wrong fit receipt, budget or scope')
    start = read(directory/'started.json')
    identity_keys = ('path', 'loss', 'seed') if optimization else ('mode', 'seed')
    if (any(start[key] != receipt[key] for key in identity_keys) or start['updates'] != 500
            or start['plan_sha256'] != plan_hash or start['smoke'] is not False):
        raise ValueError('Started/completed fit identity mismatch')
    artifacts = [('model.pt', 'checkpoint_sha256'), ('learning.jsonl', 'learning_sha256')]
    if optimization:
        artifacts += [('before.json', 'before_sha256'), ('after.json', 'after_sha256')]
    for filename, key in artifacts:
        if sha(directory/filename) != receipt[key]:
            raise ValueError(f'Fit artifact changed: {filename}')
    logs = [json.loads(line) for line in (directory/'learning.jsonl').read_text().splitlines()]
    if len(logs) != 500 or [row['update'] for row in logs] != list(range(1, 501)):
        raise ValueError('Expected every one of the 500 optimizer update receipts')
    field = 'optimized_loss' if optimization else 'final_turn_cross_entropy'
    if (any(row['sequences_seen'] != 4*row['update'] or not np.isfinite(row[field])
            or row[field] < 0 or not np.isfinite(row['elapsed_seconds']) for row in logs)
            or any(a['elapsed_seconds'] > b['elapsed_seconds'] for a, b in pairwise(logs))):
        raise ValueError('Invalid learning log sequence count, loss or timing')
    return logs


def decode_observation(encoded):
    cells = encoded[:-4].reshape(7, 7, 20)
    groups = [cells[..., :11], cells[..., 11:17], cells[..., 17:20], encoded[-4:]]
    for group in groups:
        if not np.isin(group, [0., 1.]).all() or not np.all(group.sum(-1) == 1):
            raise ValueError('Native categorical inputs are not one-hot')
    return {'image': np.stack([group.argmax(-1) for group in groups[:3]], axis=-1),
            'direction': int(groups[-1].argmax())}


def validate_packet(run_dir, stem, metadata, study, size):
    with np.load(run_dir/'data'/f'{stem}.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'observations', 'previous', 'resets', 'labels'}:
            raise ValueError('Unexpected packet arrays')
        data = {name: archive[name] for name in archive.files}
    if (data['observations'].shape != (size-2, 4, 984) or data['observations'].dtype != np.float32
            or data['previous'].shape != (size-2, 4) or data['resets'].shape != (size-2, 4)
            or data['labels'].shape != (4,) or data['resets'].dtype != bool
            or not np.issubdtype(data['previous'].dtype, np.integer)
            or not np.issubdtype(data['labels'].dtype, np.integer)
            or not data['resets'][0].all() or data['resets'][1:].any()
            or not np.all(data['previous'][0] == 0) or not np.all(data['previous'][1:] == 2)):
        raise ValueError('Packet input shapes, forced actions or reset masks changed')
    hashes = {key: study.teacher.array_sha(value) for key, value in data.items()}
    if hashes != metadata['array_sha256']:
        raise ValueError('Packet arrays do not match their recorded hashes')
    if len(metadata['contexts']) != 4 or metadata['size'] != size:
        raise ValueError('Expected exactly four contexts at the declared size')
    combinations, sequence_hashes = set(), set()
    for i, context in enumerate(metadata['contexts']):
        sequence = data['observations'][:, i]
        for observation in sequence:
            decode_observation(observation)
        first, last = decode_observation(sequence[0]), decode_observation(sequence[-1])
        label, combination = study.teacher.teacher_choice(first, last)
        if (label != data['labels'][i] or label != context['target_turn']
                or combination != (context['cue'], context['upper_branch'])
                or label != int(context['cue'] != context['upper_branch'])
                or study.teacher.array_sha(sequence) != context['observations_sha256']):
            raise ValueError('Observable teacher labels or context identities do not replay')
        combinations.add(combination)
        sequence_hashes.add(context['observations_sha256'])
    if (combinations != set(study.teacher.CONTEXTS) or len(sequence_hashes) != 4
            or metadata['unique_observation_sequences'] != 4
            or metadata['target_counts'] != {'0': 2, '1': 2}
            or np.bincount(data['labels'], minlength=2).tolist() != [2, 2]):
        raise ValueError('Contexts are not the four balanced unique native sequences')
    return data


def validate_result(result, data, metadata, replay_probabilities):
    if result['data_array_sha256'] != metadata['array_sha256'] or len(result['examples']) != 4:
        raise ValueError('Evaluation packet or example count mismatch')
    p7, p2 = [], []
    for i, (row, context) in enumerate(zip(result['examples'], metadata['contexts'], strict=True)):
        if any(row[key] != value for key, value in context.items()):
            raise ValueError('Evaluation example context, label or seed changed')
        a, b = np.asarray(row['probabilities7']), np.asarray(row['probabilities2'])
        if (a.shape != (7,) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all()
                or (a < 0).any() or (b < 0).any() or (a > 1).any() or (b > 1).any()):
            raise ValueError('Invalid recorded action probabilities')
        close(a.sum(), 1., 'seven-action normalization', 2e-6)
        close(b.sum(), 1., 'two-candidate normalization', 2e-6)
        if (row['predicted_action7'] != int(a.argmax()) or row['predicted_candidate2'] != int(b.argmax())
                or not np.allclose(b, a[:2]/a[:2].sum(), rtol=0, atol=2e-6)
                or not np.array_equal(a, replay_probabilities[0][i])
                or not np.array_equal(b, replay_probabilities[1][i])):
            raise ValueError('Recorded predictions do not match probabilities or final-checkpoint replay')
        p7.append(a)
        p2.append(b)
    p7, p2 = np.asarray(p7), np.asarray(p2)
    labels = data['labels']
    target7, target2 = p7[np.arange(4), labels], p2[np.arange(4), labels]
    if (target7 <= 0).any() or (target2 <= 0).any():
        raise ValueError('Zero target probability prevents finite loss replay')
    measured = {'accuracy7': np.mean(p7.argmax(-1) == labels), 'accuracy2': np.mean(p2.argmax(-1) == labels),
                'conditional_binary_cross_entropy': -np.log(target2).mean(),
                'cross_entropy7': -np.log(target7).mean(),
                'target_probability7': target7.mean(), 'target_probability2': target2.mean()}
    for key, value in measured.items():
        close(result[key], value, key)
    return measured


def same_diagnostic(saved, replayed):
    if saved.keys() != replayed.keys():
        raise ValueError('Activation diagnostic fields changed')
    for key, value in replayed.items():
        if isinstance(value, str):
            if saved[key] != value:
                raise ValueError('Activation interpretation changed')
        elif isinstance(value, list):
            if len(saved[key]) != len(value):
                raise ValueError('Activation sequence length changed')
            for a, b in zip(saved[key], value, strict=True):
                close(a, b, f'activation {key}', 1e-7)
        else:
            close(saved[key], value, f'activation {key}', 1e-7)


def verify(run_dir, baseline_run, plan_path, baseline_plan_path):
    plan, summary = load_completed(run_dir, plan_path, 'memory-optimization-v1')
    baseline_plan, baseline = load_completed(baseline_run, baseline_plan_path, 'associative-learnability-v1')
    spec = importlib.util.spec_from_file_location('_optimization_figure_frozen',
                                                  ROOT/'scripts/memory_optimization_diagnostic.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    if summary['protocol'] != study.PROTOCOL:
        raise ValueError('Publisher requires the exact frozen factorial protocol')
    torch.set_num_threads(1)
    training_metadata = read(run_dir/'data/training.json')
    if (sha(run_dir/'data/training.npz') != training_metadata['npz_sha256']
            or {k: v for k, v in training_metadata.items() if k != 'npz_sha256'} != plan['training_data']
            or summary['training_data'] != plan['training_data']
            or summary['training_data'] != baseline['training_data']
            or summary['training_data'] != baseline_plan['training_data']):
        raise ValueError('Training packets differ from frozen or earlier contexts')
    train_data = validate_packet(run_dir, 'training', summary['training_data'], study, 11)
    baseline_metadata = read(baseline_run/'data/training.json')
    if sha(baseline_run/'data/training.npz') != baseline_metadata['npz_sha256']:
        raise ValueError('Baseline training packet archive changed')
    earlier_data = validate_packet(baseline_run, 'training', baseline['training_data'], study, 11)
    if any(not np.array_equal(train_data[key], earlier_data[key]) for key in train_data):
        raise ValueError('Earlier GRU and factorial training inputs differ')
    panels = {}
    for size in plan['protocol']['evaluation_sizes']:
        metadata = read(run_dir/'data'/f'evaluation-{size}.json')
        if metadata != summary['evaluation_data'][str(size)]:
            raise ValueError('Evaluation metadata differs from the summary')
        if [r['seed'] for r in metadata['contexts']] != [r['seed'] for r in training_metadata['contexts']]:
            raise ValueError('Size probes must reuse the exact selected context seed IDs')
        panels[size] = validate_packet(run_dir, f'evaluation-{size}', metadata, study, size), metadata
    expected = {(path, loss, seed) for path in study.PATHS for loss in study.LOSSES for seed in [101, 113, 127]}
    fits = {(r['path'], r['loss'], r['seed']): r for r in summary['fits']}
    if len(fits) != 12 or set(fits) != expected:
        raise ValueError('Factorial fit identities are incomplete or duplicated')
    activations = {(r['path'], r['loss'], r['seed']): r for r in summary['activation_changes']}
    evaluations = {(r['path'], r['loss'], r['seed']): r for r in summary['evaluations']}
    if (len(summary['activation_changes']) != 12 or set(activations) != expected
            or len(summary['evaluations']) != 12 or set(evaluations) != expected):
        raise ValueError('Missing activation or evaluation conditions')
    earlier_fits = {(r['mode'], r['seed']): r for r in baseline['fits']}
    comparisons, records, log_count = [], {}, 0
    for identity in sorted(expected):
        path, loss, seed = identity
        directory = run_dir/'fits'/f'{path}-{loss}-{seed}'
        receipt = fits[identity]
        log_count += len(validate_fit(directory, receipt, sha(plan_path), True))
        before, after = read(directory/'before.json'), read(directory/'after.json')
        if activations[identity] != {'path': path, 'loss': loss, 'seed': seed, 'before': before, 'after': after}:
            raise ValueError('Summary activation data differ from their hashed receipts')
        model = study.teacher.new_model('gru', seed)
        same_diagnostic(before, study.activation_diagnostic(model, train_data, summary['training_data'], path))
        weights = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.eval()
        same_diagnostic(after, study.activation_diagnostic(model, train_data, summary['training_data'], path))
        if path == 'recursive' and loss == 'ce7':
            previous = earlier_fits[('gru', seed)]
            previous_dir = baseline_run/'fits'/f'gru-{seed}'
            validate_fit(previous_dir, previous, sha(baseline_plan_path), False)
            previous_weights = torch.load(previous_dir/'model.pt', map_location='cpu', weights_only=True)
            if weights.keys() != previous_weights.keys() or any(
                    not torch.equal(weights[key], previous_weights[key]) for key in weights):
                raise ValueError('Recursive ce7 did not reproduce the earlier negative GRU checkpoint exactly')
            comparisons.append({'seed': seed, 'tensor_equal': True, 'tensor_count': len(weights),
                                'parameter_elements': sum(value.numel() for value in weights.values()),
                                'factorial_checkpoint_sha256': receipt['checkpoint_sha256'],
                                'earlier_checkpoint_sha256': previous['checkpoint_sha256']})
        record = read(run_dir/'evaluation'/f'{path}-{loss}-{seed}.json')
        if record != evaluations[identity] or record['checkpoint_sha256'] != receipt['checkpoint_sha256']:
            raise ValueError('Evaluation file/checkpoint identity mismatch')
        wanted_results = {(size, condition) for size in [11, 17, 23]
                          for condition in ('intact', 'fork_state_erased')}
        seen = set()
        if len(record['results']) != 6:
            raise ValueError('Each fit needs all six size/ablation evaluations')
        for result in record['results']:
            pair = result['size'], result['condition']
            if pair not in wanted_results or pair in seen:
                raise ValueError('Unexpected or duplicated evaluation condition')
            seen.add(pair)
            data, metadata = panels[result['size']]
            with torch.inference_mode():
                logits = study.outputs(model, data, path, ablate_fork=result['condition'] == 'fork_state_erased')[0][-1]
                replay = logits.softmax(-1).numpy(), logits[:, :2].softmax(-1).numpy()
            measured = validate_result(result, data, metadata, replay)
            records[(*identity, *pair)] = measured
        if seen != wanted_results:
            raise ValueError('Missing size or ablation result')
    if len(records) != 72 or log_count != 6000 or len(comparisons) != 3:
        raise ValueError('Final verified coverage mismatch')
    expected_means = {(path, loss, size, condition) for path, loss in CELLS for size in [11, 17, 23]
                      for condition in ('intact', 'fork_state_erased')}
    seen = set()
    for average in summary['averages']:
        identity = average['path'], average['loss'], average['size'], average['condition']
        if identity not in expected_means or identity in seen:
            raise ValueError('Mean conditions are unexpected or duplicated')
        seen.add(identity)
        path, loss, size, condition = identity
        for key in METRICS:
            mean = np.mean([records[(path, loss, seed, size, condition)][key] for seed in [101, 113, 127]])
            close(average[key], mean, f'mean {identity} {key}')
    if seen != expected_means or summary['optimizer_updates'] != 6000:
        raise ValueError('Means or optimizer budget incomplete')
    close(summary['training_seconds'], sum(r['training_seconds'] for r in summary['fits']), 'training time', 1e-9)
    return summary, records, activations, comparisons


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#cdd4df')
    ax.grid(axis='y', alpha=.2, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#445266', labelsize=9)


def figure(records, activations):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.2, hspace=.44, wspace=.23)
    fig.suptitle('Four-context memory diagnostic: loss and state path matter',
                 x=.075, y=.975, ha='left', fontsize=19, fontweight='bold', color='#233044')
    fig.text(.075, .935, 'Supervised final-turn choice  |  12 fits  |  500 updates per fit  |  Forced paths, not navigation',
             fontsize=11, color='#5c6878')
    fig.legend([Line2D([0], [0], color=c, linewidth=2.2) for c in COLORS], LABELS,
               loc='upper left', bbox_to_anchor=(.068, .903), ncol=2, frameon=False,
               fontsize=10, handlelength=2.2, columnspacing=2.4)
    seed_offsets = np.array([-.024, 0., .024])
    for column, metric in enumerate(('accuracy7', 'accuracy2')):
        ax = axes[0, column]
        for index, ((path, loss), color) in enumerate(zip(CELLS, COLORS, strict=True)):
            x = np.arange(3)+(index-1.5)*.075
            values = np.array([[records[(path, loss, seed, size, 'intact')][metric]*100
                                for size in [11, 17, 23]] for seed in [101, 113, 127]])
            for row, offset in zip(values, seed_offsets, strict=True):
                ax.scatter(x+offset, row, color=color, s=25, alpha=.8, zorder=4)
            ax.plot(x, values.mean(0), color=color, linewidth=2.2, zorder=3,
                    linestyle='--' if loss == 'ce7' else '-')
        ax.axhline(50, color='#8a96a5', linestyle=':', linewidth=1, zorder=1)
        ax.set(xticks=range(3), xticklabels=['11 (trained)', '17', '23'], ylim=(0, 107),
               yticks=[0, 25, 50, 75, 100], xlabel='Native grid size; forced observations to fork',
               ylabel='Correct final turns (%)')
        ax.set_title('Seven-action argmax' if metric == 'accuracy7' else 'Two-candidate argmax: left / right',
                     loc='left', fontweight='bold', pad=12)
        style_axis(ax)
    for column, (metric, title) in enumerate([
            ('encoder_saturation_fraction', 'Encoder activations with |value| > 0.95'),
            ('fork_state_saturation_fraction', 'Fork hidden state with |value| > 0.95')]):
        ax = axes[1, column]
        for i, ((path, loss), color) in enumerate(zip(CELLS, COLORS, strict=True)):
            rows = [activations[(path, loss, seed)] for seed in [101, 113, 127]]
            values = [100*row['after'][metric] for row in rows]
            before = [100*row['before'][metric] for row in rows]
            ax.bar(i, np.mean(values), width=.62, color=color, alpha=.72, zorder=2)
            ax.scatter(i+np.array([-.12, 0, .12]), values, s=28, color=color,
                       edgecolors='#26364a', linewidths=.45, zorder=4)
            ax.scatter(i+np.array([-.12, 0, .12]), before, s=25, marker='D',
                       facecolors='white', edgecolors='#586779', linewidths=.8, zorder=4)
        ax.set(xticks=range(4), xticklabels=SHORT_LABELS, ylim=(-3, 106), yticks=[0, 25, 50, 75, 100],
               ylabel='Activation components (%)')
        ax.set_title(title, loc='left', fontweight='bold', pad=12)
        style_axis(ax)
    same_definitions = all(row['accuracy7'] == row['accuracy2'] for row in records.values())
    erased = {row['accuracy2'] for key, row in records.items() if key[-1] == 'fork_state_erased'}
    fig.text(.075, .135, 'Filled points: every final fit. Lines / bars: means. Hollow diamonds: initialization. '
             'Horizontal offsets separate overlaps.', fontsize=9, color='#5c6878')
    fig.text(.075, .109, f'Accuracy definitions agree in these recorded results: {same_definitions}. '
             f'Fork-memory erasure: {", ".join(f"{x*100:g}%" for x in sorted(erased))}. '
             '50% is the balanced constant-turn baseline.', fontsize=9, color='#5c6878')
    fig.text(.075, .079, 'Cached h0 reuses the first-observation state unchanged. Intermediate frames cannot '
             'overwrite it: this is a fixed memory shortcut.', fontsize=9.3, color='#26364a', fontweight='bold')
    fig.text(.075, .050, 'Four supervised contexts only; size probes reuse their seed IDs. '
             'No learned navigation, RL gain, unseen-map result or novelty established.', fontsize=9.3, color='#26364a')
    return fig


def publish(args):
    if args.out.exists():
        raise FileExistsError('Figure output directory must be fresh')
    plan = args.plan or args.run.parent/'plan.json'
    baseline_plan = args.baseline_plan or args.baseline_run.parent/'plan.json'
    summary, records, activations, comparisons = verify(args.run, args.baseline_run, plan, baseline_plan)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.hashsalt': 'memory-optimization-v1'})
    fig = figure(records, activations)
    content = {}
    for suffix in ('png', 'svg'):
        buffer = io.BytesIO()
        metadata = {'Software': 'OpenJev verified memory diagnostic'} if suffix == 'png' else {'Date': None}
        fig.savefig(buffer, format=suffix, dpi=170, facecolor='white', metadata=metadata)
        content[f'memory-optimization.{suffix}'] = buffer.getvalue()
    plt.close(fig)
    args.out.mkdir(parents=True, exist_ok=False)
    for filename, value in content.items():
        with (args.out/filename).open('xb') as stream:
            stream.write(value)
    receipt = {'kind': 'verified_supervised_memory_optimization_figures',
               'summary_sha256': sha(args.run/'summary.json'), 'plan_sha256': sha(plan),
               'completed_sha256': sha(args.run/'completed.json'),
               'baseline_summary_sha256': sha(args.baseline_run/'summary.json'),
               'baseline_plan_sha256': sha(baseline_plan), 'publisher_sha256': sha(Path(__file__)),
               'verified_fits': 12, 'verified_optimizer_log_rows': 6000,
               'verified_evaluation_results': 72, 'verified_context_predictions': 288,
               'four_unique_balanced_contexts_each_result': True, 'checkpoint_prediction_replay': True,
               'activation_before_after_replay': True, 'baseline_exact_tensor_reproduction': comparisons,
               'training_packet_identical_to_baseline': True,
               'accuracy7_equals_accuracy2_in_recorded_results': all(r['accuracy7'] == r['accuracy2']
                                                                    for r in records.values()),
               'artifacts': {filename: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
                             for filename, value in content.items()},
               'new_training': False, 'rl_training': False, 'unseen_unique_map_test': False,
               'novelty_established': False, 'independent_confirmation': False,
               'scope': summary['protocol']['scope']}
    with (args.out/'memory-optimization-figures.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--baseline-plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    publish(parser.parse_args())


if __name__ == '__main__':
    main()
