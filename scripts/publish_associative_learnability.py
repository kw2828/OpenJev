"""Publish a completed four-context supervised diagnostic without loading models."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

VERSION = 'associative-learnability-v1'
MODES = ('gru', 'feedforward', 'fast_global', 'fast_selective')
LABELS = ['GRU', 'GRU + feedforward', 'GRU + global store', 'GRU + selective store']
SHORT = ['GRU', '+ feedforward', '+ global', '+ selective']
COLORS = ['#416d9c', '#8392a7', '#8d71aa', '#248a76']
CONDITIONS = ('intact', 'reset_all', 'reset_store')
STEM = 'associative-learnability'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def close(actual, expected, description, tolerance=1e-10):
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > tolerance:
        raise ValueError(f'Invalid or inconsistent {description}')


def packet(run, name, expected_metadata, selected_seeds):
    path = run / 'data' / f'{name}.npz'
    metadata_path = path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    extra = {'npz_sha256'} if name == 'training' else {
        'npz_sha256', 'same_selected_seed_ids_as_training', 'unseen_unique_maps'}
    if ({key: value for key, value in metadata.items() if key not in extra} != expected_metadata
            or metadata['npz_sha256'] != sha(path)):
        raise ValueError('Saved observation packet differs from its metadata or frozen summary')
    if name != 'training' and (metadata['same_selected_seed_ids_as_training'] is not True
                               or metadata['unseen_unique_maps'] is not False):
        raise ValueError('Length-transfer packets must retain their training-context scope')
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {'observations', 'previous', 'resets', 'labels'}:
            raise ValueError('Unexpected saved packet fields')
        data = {key: saved[key] for key in saved.files}
    if {key: array_sha(value) for key, value in data.items()} != metadata['array_sha256']:
        raise ValueError('Saved packet array hashes do not match')
    length = metadata['size'] - 2
    obs, previous, resets, labels = (data[key] for key in ('observations', 'previous', 'resets', 'labels'))
    if (obs.shape != (length, 4, 984) or obs.dtype != np.float32
            or previous.shape != (length, 4) or previous.dtype != np.int64
            or resets.shape != (length, 4) or resets.dtype != bool
            or labels.shape != (4,) or labels.dtype != np.int64):
        raise ValueError('Unexpected sequence shape or dtype')
    if (not np.all(previous[0] == 0) or not np.all(previous[1:] == 2)
            or not np.all(resets[0]) or np.any(resets[1:])):
        raise ValueError('Previous forced actions or initial-only reset flags changed')
    cells = obs[..., :-4].reshape(length, 4, 7, 7, 20)
    if not np.isin(obs, [0., 1.]).all() or not np.all(obs[..., -4:] == [1., 0., 0., 0.]):
        raise ValueError('Expected east-facing categorical partial observations')
    if any(not np.all(cells[..., start:end].sum(-1) == 1) for start, end in [(0, 11), (11, 17), (17, 20)]):
        raise ValueError('Observation channels are not valid one-hot categories')
    objects = cells[..., :11].argmax(-1)
    contexts = metadata['contexts']
    if (len(contexts) != 4 or [row['seed'] for row in contexts] != selected_seeds
            or [(row['cue'], row['upper_branch']) for row in contexts] != [(5, 5), (5, 6), (6, 5), (6, 6)]):
        raise ValueError('Expected four canonical cue and upper-branch contexts with fixed seed IDs')
    for index, context in enumerate(contexts):
        initial, fork = objects[0, index], objects[-1, index]
        first = np.argwhere(np.isin(initial, [5, 6]))
        last = np.argwhere(np.isin(fork, [5, 6]))
        if len(first) != 1 or last.tolist() != [[1, 6], [5, 6]]:
            raise ValueError('Teacher requires exactly one visible initial cue and both final branch objects')
        cue, upper, lower = int(initial[tuple(first[0])]), int(fork[1, 6]), int(fork[5, 6])
        target = 0 if cue == upper else 1
        if ({upper, lower} != {5, 6} or (cue, upper) != (context['cue'], context['upper_branch'])
                or target != context['target_turn'] or target != labels[index]
                or array_sha(obs[:, index]) != context['observations_sha256']):
            raise ValueError('Teacher label does not match the saved visible observations')
    if (len({array_sha(obs[:, index]) for index in range(4)}) != 4
            or metadata['unique_observation_sequences'] != 4 or metadata['target_counts'] != {'0': 2, '1': 2}
            or labels.tolist() != [0, 1, 1, 0]):
        raise ValueError('Four-context panel must be distinct and balanced')
    return {'metadata_sha256': sha(metadata_path), 'npz_sha256': sha(path)}


def load_completed(run, plan_path):
    plan, summary = json.loads(plan_path.read_text()), json.loads((run / 'summary.json').read_text())
    completed = json.loads((run / 'completed.json').read_text())
    plan_hash = sha(plan_path)
    p = summary['protocol']
    if (p != plan['protocol'] or p['version'] != VERSION or summary['plan_sha256'] != plan_hash
            or completed['status'] != 'completed' or completed['plan_sha256'] != plan_hash
            or completed['summary_sha256'] != sha(run / 'summary.json')
            or completed['fits'] != 12 or completed['optimizer_updates'] != 6000):
        raise ValueError('Expected a complete summary with the matching frozen plan and completion receipt')
    if (tuple(p['modes']) != MODES or p['seeds'] != [101, 113, 127] or p['updates'] != 500
            or p['batch_size'] != 4 or p['evaluation_sizes'] != [11, 17, 23]
            or tuple(p['conditions']) != CONDITIONS or p['train_size'] != 11):
        raise ValueError('Unexpected frozen fit, data or evaluation coverage')
    for key in ('gameplay_evaluated', 'rl_training', 'rl_warmstart', 'unseen_unique_map_test',
                'novelty_established', 'independent_confirmation'):
        if summary[key] is not False:
            raise ValueError('This publisher only accepts the bounded supervised diagnostic')
    if summary['training_data'] != plan['training_data']:
        raise ValueError('Training contexts changed from the frozen plan')
    selected = [row['seed'] for row in summary['training_data']['contexts']]
    packet_hashes = {'training': packet(run, 'training', summary['training_data'], selected)}
    if set(summary['evaluation_data']) != {'11', '17', '23'}:
        raise ValueError('Unexpected evaluation packets')
    for size in p['evaluation_sizes']:
        packet_hashes[f'evaluation-{size}'] = packet(
            run, f'evaluation-{size}', summary['evaluation_data'][str(size)], selected)
    expected = {(mode, seed) for mode in MODES for seed in p['seeds']}
    fits = {(row['mode'], row['seed']): row for row in summary['fits']}
    if len(summary['fits']) != 12 or set(fits) != expected:
        raise ValueError('Missing, duplicated or unexpected fit identities')
    training_completed = json.loads((run / 'training-completed.json').read_text())
    if (training_completed['status'] != 'completed' or training_completed['plan_sha256'] != plan_hash
            or training_completed['fits'] != summary['fits']):
        raise ValueError('Training completion receipt differs from the final summary')
    sources = {}
    for (mode, seed), fit in fits.items():
        folder = run / 'fits' / f'{mode}-{seed}'
        if (json.loads((folder / 'completed.json').read_text()) != fit or fit['status'] != 'completed'
                or fit['plan_sha256'] != plan_hash or fit['updates'] != 500 or fit['sequences_seen'] != 2000
                or fit['unique_training_sequences'] != 4 or fit['smoke'] is not False
                or fit['supervised_only'] is not True or fit['rl_warmstart'] is not False
                or sha(folder / 'model.pt') != fit['checkpoint_sha256']
                or sha(folder / 'learning.jsonl') != fit['learning_sha256']
                or not np.isfinite(fit['training_seconds']) or fit['training_seconds'] <= 0):
            raise ValueError('Fit receipt, fixed budget, checkpoint or learning hash mismatch')
        rows = [json.loads(line) for line in (folder / 'learning.jsonl').read_text().splitlines()]
        if len(rows) != 500:
            raise ValueError('Expected exactly 500 logged optimizer updates per fit')
        previous_elapsed = 0.
        for update, row in enumerate(rows, 1):
            if (row['update'] != update or row['sequences_seen'] != update * 4
                    or not np.isfinite(row['final_turn_cross_entropy']) or row['final_turn_cross_entropy'] < 0
                    or not np.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] < previous_elapsed):
                raise ValueError('Learning log sequence, loss or timing is invalid')
            previous_elapsed = row['elapsed_seconds']
        sources[f'{mode}-{seed}'] = {'checkpoint_sha256': fit['checkpoint_sha256'],
                                   'learning_sha256': fit['learning_sha256'],
                                   'completed_sha256': sha(folder / 'completed.json')}
    close(summary['optimizer_updates'], 6000, 'total optimizer updates')
    close(summary['training_seconds'], sum(fit['training_seconds'] for fit in fits.values()), 'training seconds')
    observed, scores = set(), {}
    if len(summary['evaluations']) != 12:
        raise ValueError('Expected one final evaluation file for each of 12 fits')
    for record in summary['evaluations']:
        mode, seed = record['mode'], record['seed']
        if (mode, seed) not in expected or (mode, seed) in observed:
            raise ValueError('Unexpected or duplicated evaluation fit identity')
        observed.add((mode, seed))
        path = run / 'evaluation' / f'{mode}-{seed}.json'
        if json.loads(path.read_text()) != record or record['checkpoint_sha256'] != fits[(mode, seed)]['checkpoint_sha256']:
            raise ValueError('Saved final evaluation or checkpoint identity differs from the summary')
        sources[f'{mode}-{seed}']['evaluation_sha256'] = sha(path)
        conditions = {(size, condition) for size in p['evaluation_sizes'] for condition in CONDITIONS}
        seen = set()
        if len(record['results']) != len(conditions):
            raise ValueError('Expected nine size and state-ablation results per fit')
        for result in record['results']:
            size, condition = result['size'], result['condition']
            if (size, condition) not in conditions or (size, condition) in seen:
                raise ValueError('Unexpected or duplicate evaluation size and condition')
            seen.add((size, condition))
            metadata = summary['evaluation_data'][str(size)]
            if (result['data_array_sha256'] != metadata['array_sha256'] or result['unique_contexts'] != 4
                    or len(result['examples']) != 4):
                raise ValueError('Evaluation must use the preserved four-context packet')
            probabilities, targets, correct = [], [], []
            for row, context in zip(result['examples'], metadata['contexts'], strict=True):
                if {key: row[key] for key in context} != context:
                    raise ValueError('Evaluation example identity, teacher target or observation hash changed')
                prob = np.asarray(row['probabilities'], dtype=np.float64)
                if prob.shape != (7,) or not np.isfinite(prob).all() or np.any(prob < 0) or np.any(prob > 1):
                    raise ValueError('Invalid seven-action probabilities')
                close(float(prob.sum()), 1., 'probability normalization', tolerance=2e-6)
                target, prediction = row['target_turn'], int(prob.argmax())
                if (type(row['predicted_action']) is not int or row['predicted_action'] != prediction
                        or type(row['correct']) is not bool or row['correct'] != (prediction == target)
                        or prob[target] <= 0 or prob[:2].sum() <= 0):
                    raise ValueError('Predicted action or correctness does not follow saved probabilities')
                probabilities.append(prob)
                targets.append(target)
                correct.append(prediction == target)
            prob, targets = np.array(probabilities), np.array(targets)
            target_prob = prob[np.arange(4), targets]
            close(result['accuracy'], np.mean(correct), 'final-turn accuracy')
            # Saved probabilities and CE came from float32 softmax/log-softmax, so allow its rounding error.
            close(result['cross_entropy'], -np.log(target_prob).mean(), 'seven-action cross entropy', 2e-6)
            close(result['target_probability'], target_prob.mean(), 'target probability', 1e-7)
            conditional_ce = float(-np.log(target_prob / prob[:, :2].sum(-1)).mean())
            scores[(mode, seed, size, condition)] = {
                **{key: result[key] for key in ('accuracy', 'cross_entropy', 'target_probability')},
                'conditional_turn_cross_entropy': conditional_ce,
                'conditional_ce_minus_log2': conditional_ce - float(np.log(2)),
                'min_absolute_turn_probability_margin': float(np.abs(prob[:, 0] - prob[:, 1]).min()),
                'max_absolute_turn_probability_margin': float(np.abs(prob[:, 0] - prob[:, 1]).max()),
                'exact_turn_probability_ties': int(np.count_nonzero(prob[:, 0] == prob[:, 1])),
            }
    if observed != expected or len(scores) != 108:
        raise ValueError('Final evaluation coverage is incomplete')
    for mode in MODES:
        for size in p['evaluation_sizes']:
            for condition in CONDITIONS:
                for key in ('accuracy', 'cross_entropy', 'target_probability'):
                    close(summary['averages'][mode][str(size)][condition][key],
                          np.mean([scores[(mode, seed, size, condition)][key] for seed in p['seeds']]),
                          'across-fit evaluation aggregate')
    return summary, plan, scores, sources, packet_hashes


def figure(summary, scores):
    p = summary['protocol']
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8), sharey='row')
    fig.subplots_adjust(left=.085, right=.985, top=.78, bottom=.2, hspace=.35, wspace=.15)
    fig.suptitle('Four-context supervised learnability diagnostic', x=.085, y=.974,
                 ha='left', fontsize=19, fontweight='bold', color='#233044')
    fig.text(.085, .928, 'Final turn after a forced path  |  Three fits per architecture  |  500 fixed updates per fit',
             fontsize=11, color='#5c6878')
    fig.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS, strict=True)],
               loc='upper left', bbox_to_anchor=(.079, .897), ncol=4, frameon=False, fontsize=10, columnspacing=1.8)
    offsets = np.linspace(-.12, .12, 3)
    delta_values = []
    for column, size in enumerate(p['evaluation_sizes']):
        for index, mode in enumerate(MODES):
            rows = [scores[(mode, seed, size, 'intact')] for seed in p['seeds']]
            values = [[100 * row['accuracy'] for row in rows],
                      [1e6 * row['conditional_ce_minus_log2'] for row in rows]]
            delta_values.extend(values[1])
            for row_index, value in enumerate(values):
                ax = axes[row_index, column]
                ax.scatter(index + offsets, value, color=COLORS[index], s=45,
                           marker='o', alpha=.85, edgecolors='white', linewidths=.8, zorder=4)
                ax.plot([index - .23, index + .23], [np.mean(value)] * 2,
                        color=COLORS[index], linewidth=2.5, zorder=3)
        for row_index in range(2):
            ax = axes[row_index, column]
            ax.set(xticks=range(4), xticklabels=SHORT, xlim=(-.5, 3.5))
            ax.tick_params(axis='x', labelsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.2)
        axes[0, column].set(ylim=(-3, 103), yticks=[0, 25, 50, 75, 100])
        axes[0, column].axhline(50, color='#9aa3ad', linestyle='--', linewidth=.8)
        scope = 'training path length' if size == p['train_size'] else 'longer path, same four seeds'
        axes[0, column].set_title(f'{size} × {size}: {scope}', loc='left', fontsize=10.5, fontweight='bold', pad=12)
        axes[1, column].axhline(0, color='#9aa3ad', linestyle='--', linewidth=.8)
    lo, hi = min(0., min(delta_values)), max(0., max(delta_values))
    padding = max(.03, (hi - lo) * .25)
    axes[1, 0].set_ylim(lo - padding, hi + padding)
    axes[0, 0].set_ylabel('Correct final turns, four contexts (%)')
    axes[1, 0].set_ylabel('Conditional turn CE minus ln(2)\n(millionths of a nat)')
    fig.text(.085, .135, 'Dots: all three final fits. Horizontal marks: fit means. '
             'Only intact state is plotted; both reset conditions are also validated.', fontsize=9, color='#5c6878')
    fig.text(.085, .104, 'Conditional CE renormalizes saved probabilities to left/right. '
             'Zero marks the equal-probability baseline. The lower axis shows millionths of a nat.', fontsize=9, color='#5c6878')
    fig.text(.085, .073, 'A 75% point is three of four contexts, with ties or tiny turn-probability margins. '
             'It does not establish a robust solution.', fontsize=9, color='#364354', fontweight='bold')
    fig.text(.085, .041, 'Same four selected seeds at all path lengths. Supervised diagnostic only: '
             'no gameplay, RL, unseen-map evaluation, or novelty claim.', fontsize=9, color='#364354')
    return fig


def publish(args):
    summary, plan, scores, sources, packets = load_completed(args.run, args.plan)
    names = [f'{STEM}.png', f'{STEM}.svg', f'{STEM}-figures.json']
    if any((args.out / name).exists() for name in names):
        raise FileExistsError('Publication output already exists; use a fresh output directory')
    rendered = {}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': VERSION, 'savefig.facecolor': 'white'}):
        fig = figure(summary, scores)
        try:
            for suffix in ('png', 'svg'):
                buffer = io.BytesIO()
                metadata = {'Date': None, 'Creator': 'OpenJev four-context supervised diagnostic'} if suffix == 'svg' else {}
                fig.savefig(buffer, format=suffix, dpi=180, bbox_inches='tight', pad_inches=.18, metadata=metadata)
                content = buffer.getvalue()
                if suffix == 'svg':
                    content = ('\n'.join(line.rstrip() for line in content.decode().splitlines()) + '\n').encode()
                rendered[f'{STEM}.{suffix}'] = content
        finally:
            plt.close(fig)
    receipt = {
        'protocol_version': VERSION, 'plan_sha256': sha(args.plan),
        'summary_sha256': sha(args.run / 'summary.json'), 'completed_sha256': sha(args.run / 'completed.json'),
        'publisher_sha256': sha(Path(__file__)), 'frozen_study_sources_sha256': plan['source_sha256'],
        'packet_sources': packets, 'fit_sources': sources,
        'fits': 12, 'optimizer_updates': 6000, 'validated_evaluation_results': 108, 'contexts_per_result': 4,
        'figures': {name: hashlib.sha256(content).hexdigest() for name, content in rendered.items()},
        'derived_metrics': [{'mode': mode, 'seed': seed, 'size': size, 'condition': condition, **values}
                            for (mode, seed, size, condition), values in sorted(scores.items())],
        'probability_replay_tolerances': {'normalization': 2e-6, 'seven_action_cross_entropy': 2e-6,
                                         'mean_target_probability': 1e-7},
        'scope': 'Four-context supervised forced-path diagnostic. Same four selected seeds; not gameplay, RL or unseen maps',
        'new_training': False, 'novelty_established': False, 'independent_confirmation': False,
        'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        with (args.out / name).open('xb') as stream:
            stream.write(content)
    with (args.out / names[-1]).open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.out), 'figures': receipt['figures'],
                      'validated_evaluation_results': len(scores), 'fits': 12}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed supervised execution directory')
    parser.add_argument('--plan', type=Path, required=True, help='Frozen learnability plan')
    parser.add_argument('--out', type=Path, required=True, help='Fresh figure output directory')
    publish(parser.parse_args())
