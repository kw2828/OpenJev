"""Verify and plot completed four-context candidate-conditioned memory results.

This publisher only replays final checkpoints and existing packets. It verifies
all saved predictions, state/write telemetry and the predeclared continuation
rule before writing figures. It never trains or changes frozen study artifacts.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import random
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import publish_memory_optimization as audit
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MODES = ['gru', 'feedforward', 'fast_global', 'fast_selective']
SEEDS = [101, 113, 127]
SIZES = [11, 17, 23]
CONDITIONS = ['intact', 'reset_all', 'reset_store']
LABELS = ['GRU', 'GRU + feedforward adapter', 'Global-write associative memory',
          'Selective-write associative memory']
COLORS = ['#53708f', '#cf8642', '#178b87', '#7856b6']
METRICS = ['accuracy7', 'accuracy2', 'cross_entropy7', 'conditional_binary_cross_entropy',
           'target_probability7', 'target_probability2']
sha, read, close = audit.sha, audit.read, audit.close


def same_tree(saved, replay, label):
    """Compare complete nested telemetry/means, including nulls and finite floats."""
    if isinstance(replay, dict):
        if not isinstance(saved, dict) or saved.keys() != replay.keys():
            raise ValueError(f'{label}: mapping fields changed')
        for key in replay:
            same_tree(saved[key], replay[key], f'{label}.{key}')
    elif isinstance(replay, list):
        if not isinstance(saved, list) or len(saved) != len(replay):
            raise ValueError(f'{label}: list length changed')
        for i, (a, b) in enumerate(zip(saved, replay, strict=True)):
            same_tree(a, b, f'{label}[{i}]')
    elif isinstance(replay, float):
        if not isinstance(saved, (float, int)) or isinstance(saved, bool):
            raise TypeError(f'{label}: expected numeric value')
        close(saved, replay, label, 1e-7)
    elif type(saved) is not type(replay) or saved != replay:
        raise ValueError(f'{label}: value changed')


def validate_fit(directory, receipt, plan_hash, model):
    if (read(directory/'completed.json') != receipt or receipt['status'] != 'completed'
            or receipt['plan_sha256'] != plan_hash or receipt['updates'] != 500
            or receipt['sequences_seen'] != 2000 or receipt['unique_training_contexts'] != 4
            or receipt['smoke'] is not False or receipt['supervised_only'] is not True
            or receipt['rl_warmstart'] is not False):
        raise ValueError('Wrong candidate fit receipt, budget or scope')
    start = read(directory/'started.json')
    if (any(start[key] != receipt[key] for key in ('mode', 'seed', 'updates', 'plan_sha256', 'smoke'))
            or not np.isfinite(start['started_unix'])):
        raise ValueError('Fit started/completed identities disagree')
    for filename, key in [('model.pt', 'checkpoint_sha256'), ('learning.jsonl', 'learning_sha256')]:
        if sha(directory/filename) != receipt[key]:
            raise ValueError(f'Candidate fit artifact changed: {filename}')
    logs = [json.loads(line) for line in (directory/'learning.jsonl').read_text().splitlines()]
    if len(logs) != 500 or [r['update'] for r in logs] != list(range(1, 501)):
        raise ValueError('Missing or duplicated optimizer updates')
    if (any(r['sequences_seen'] != 4*r['update']
            or not np.isfinite(r['conditional_binary_cross_entropy'])
            or r['conditional_binary_cross_entropy'] < 0
            or not np.isfinite(r['elapsed_seconds']) or r['elapsed_seconds'] < 0 for r in logs)
            or any(a['elapsed_seconds'] > b['elapsed_seconds'] for a, b in pairwise(logs))
            or not np.isfinite(receipt['training_seconds'])
            or receipt['training_seconds'] < logs[-1]['elapsed_seconds']):
        raise ValueError('Invalid learning log loss, sequence count or timing')
    parameters = dict(model.named_parameters())
    active = sorted(name for name, p in parameters.items() if p.requires_grad and not name.startswith('value.'))
    expected = {'registered_parameters': sum(p.numel() for p in parameters.values()),
                'trainable_parameters': sum(p.numel() for p in parameters.values() if p.requires_grad),
                'gradient_parameters': sum(parameters[name].numel() for name in active),
                'gradient_parameter_names': active}
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError('Candidate parameter or gradient participation counts disagree')
    return logs


def validate_telemetry(saved, replay, mode, size, metadata):
    same_tree(saved, replay, f'telemetry {mode} size{size}')
    if (saved['size'] != size or saved['condition'] != 'intact'
            or saved['context_seed_ids'] != [r['seed'] for r in metadata['contexts']]
            or saved['data_array_sha256'] != metadata['array_sha256']
            or len(saved['per_step']) != size-2
            or [r['step'] for r in saved['per_step']] != list(range(size-2))):
        raise ValueError('Telemetry trajectory coverage changed')
    store_fields = ['store_opposite_cue_frobenius', 'write_strengths',
                    'mean_write_strength', 'mean_store_update_frobenius']
    global_strength = None
    for row in saved['per_step']:
        if row['gru_opposite_cue_l2'] < 0:
            raise ValueError('Negative cue distance')
        if mode in ('gru', 'feedforward'):
            if any(row[key] is not None for key in store_fields):
                raise ValueError('No-store control has matrix telemetry')
            continue
        strengths = np.asarray(row['write_strengths'])
        if (strengths.shape != (4,) or not np.isfinite(strengths).all()
                or np.any(strengths < 0) or np.any(strengths > 1)
                or row['store_opposite_cue_frobenius'] < 0 or row['mean_store_update_frobenius'] < 0):
            raise ValueError('Invalid store distances or write strengths')
        close(row['mean_write_strength'], strengths.mean(), 'Mean write strength', 1e-7)
        if mode == 'fast_global':
            if global_strength is None:
                global_strength = float(strengths[0])
            if not np.all(strengths == global_strength):
                raise ValueError('Global write strength varies with context or time')


def verify_reference(run_dir, reference, reference_plan, reproduction, summary):
    _, earlier = audit.load_completed(reference, reference_plan, 'memory-optimization-v1')
    common = ('seeds', 'updates', 'batch_size', 'learning_rate', 'betas', 'epsilon', 'weight_decay')
    if summary['training_data'] != earlier['training_data'] or any(
            summary['protocol'][key] != earlier['protocol'][key] for key in common):
        raise ValueError('GRU reference data or optimizer differ')
    receipt = read(reproduction)
    if (receipt['status'] != 'completed' or receipt['candidate_summary_sha256'] != sha(run_dir/'summary.json')
            or receipt['reference_summary_sha256'] != sha(reference/'summary.json')
            or receipt['comparison_source_sha256'] != sha(ROOT/'scripts/associative_candidate_diagnostic.py')
            or receipt['training_checkpoints_reused'] is not False or receipt['all_tensors_equal'] is not True):
        raise ValueError('Invalid GRU reproduction receipt')
    records = []
    for seed in SEEDS:
        prior = [r for r in earlier['fits'] if (r['path'], r['loss'], r['seed']) == ('recursive', 'ce2', seed)]
        current = [r for r in summary['fits'] if (r['mode'], r['seed']) == ('gru', seed)]
        if len(prior) != 1 or len(current) != 1:
            raise ValueError('GRU reproduction fit identity missing or duplicated')
        old_dir = reference/'fits'/f'recursive-ce2-{seed}'
        audit.validate_fit(old_dir, prior[0], sha(reference_plan), True)
        paths = [run_dir/'fits'/f'gru-{seed}'/'model.pt', old_dir/'model.pt']
        for path, row in zip(paths, [current[0], prior[0]], strict=True):
            if sha(path) != row['checkpoint_sha256']:
                raise ValueError('Reproduction checkpoint hash changed')
        a, b = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
        if a.keys() != b.keys() or any(not torch.equal(a[key], b[key]) for key in a):
            raise ValueError('Candidate GRU did not exactly reproduce previous recursive CE2 tensors')
        records.append({'seed': seed, 'candidate_checkpoint_sha256': sha(paths[0]),
                        'reference_checkpoint_sha256': sha(paths[1]),
                        'tensor_equality': {key: True for key in a}, 'all_tensors_equal': True})
    if receipt['fits'] != records:
        raise ValueError('GRU reproduction records do not replay')
    return records


def verify(run_dir, plan_path, reference, reference_plan, reproduction):
    plan, summary = audit.load_completed(run_dir, plan_path, 'associative-candidate-v1')
    spec = importlib.util.spec_from_file_location('_candidate_publisher_frozen',
                                                  ROOT/'scripts/associative_candidate_diagnostic.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    if plan != study.signature():
        raise ValueError('Frozen candidate signature differs from current sources, dependencies or native teacher data')
    torch.set_num_threads(1)
    if any(summary[key] is not False for key in ('gameplay_evaluated', 'unseen_unique_map_test')):
        raise ValueError('Incorrect gameplay or unseen-map claim')
    if read(run_dir/'started.json')['plan_sha256'] != sha(plan_path):
        raise ValueError('Run started with a different plan')
    training_metadata = read(run_dir/'data/training.json')
    if (sha(run_dir/'data/training.npz') != training_metadata['npz_sha256']
            or {k: v for k, v in training_metadata.items() if k != 'npz_sha256'} != plan['training_data']
            or summary['training_data'] != plan['training_data']):
        raise ValueError('Training packet differs from frozen context set')
    training = audit.validate_packet(run_dir, 'training', summary['training_data'], study, 11)
    panels = {}
    for size in SIZES:
        metadata = summary['evaluation_data'][str(size)]
        file_metadata = read(run_dir/'data'/f'evaluation-{size}.json')
        expected = {**metadata, 'npz_sha256': sha(run_dir/'data'/f'evaluation-{size}.npz'),
                    'same_selected_seed_ids_as_training': True, 'unseen_unique_maps': False}
        if file_metadata != expected:
            raise ValueError('Evaluation packet metadata/hash differs from summary')
        seeds = [r['seed'] for r in metadata['contexts']]
        if seeds != [r['seed'] for r in training_metadata['contexts']]:
            raise ValueError('Length probes do not reuse the four selected context seeds')
        data = audit.validate_packet(run_dir, f'evaluation-{size}', metadata, study, size)
        native, native_metadata = study.teacher.packet([study.teacher.collect_sequence(size, seed) for seed in seeds])
        if native_metadata != metadata or any(not np.array_equal(data[key], native[key]) for key in data):
            raise ValueError('Saved evaluation sequence differs from the actual native partial-observation path')
        panels[size] = data, metadata
    if any(not np.array_equal(training[key], panels[11][0][key]) for key in training):
        raise ValueError('Size11 evaluation is not the original four-context training packet')
    expected = {(mode, seed) for mode in MODES for seed in SEEDS}
    collections = {}
    for key in ('fits', 'evaluations', 'telemetry'):
        rows = {(r['mode'], r['seed']): r for r in summary[key]}
        if len(summary[key]) != 12 or set(rows) != expected:
            raise ValueError(f'Incomplete or duplicated {key} identities')
        collections[key] = rows
    order = [(mode, seed) for mode in study.PROTOCOL['modes'] for seed in study.PROTOCOL['seeds']]
    random.Random(study.PROTOCOL['fit_order_seed']).shuffle(order)
    if (read(run_dir/'fit-order.json') != [list(x) for x in order]
            or [(r['mode'], r['seed']) for r in summary['fits']] != order):
        raise ValueError('Frozen randomized fit order differs from receipts')
    wanted_results = {(size, condition) for size in SIZES for condition in CONDITIONS}
    records, log_count, step_count = {}, 0, 0
    for mode, seed in sorted(expected):
        directory = run_dir/'fits'/f'{mode}-{seed}'
        fit = collections['fits'][(mode, seed)]
        model = study.teacher.new_model(mode, seed)
        log_count += len(validate_fit(directory, fit, sha(plan_path), model))
        weights = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
        if any(not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError('Nonfinite final checkpoint tensor')
        model.load_state_dict(weights, strict=True)
        model.eval()
        evaluation = read(run_dir/'evaluation'/f'{mode}-{seed}.json')
        if (evaluation != collections['evaluations'][(mode, seed)]
                or evaluation['checkpoint_sha256'] != fit['checkpoint_sha256']):
            raise ValueError('Evaluation file/summary/checkpoint identity mismatch')
        results = evaluation['results']
        if len(results) != 9 or {(r['size'], r['condition']) for r in results} != wanted_results:
            raise ValueError('Missing or duplicated size/state conditions')
        for result in results:
            size, condition = result['size'], result['condition']
            data, metadata = panels[size]
            if result['unique_contexts'] != 4:
                raise ValueError('Incorrect unique-context evaluation count')
            with torch.inference_mode():
                logits = study.teacher.final_logits(model, data, condition)
                replay = logits.softmax(-1).numpy(), logits[:, :2].softmax(-1).numpy()
            measured = audit.validate_result(result, data, metadata, replay)
            for row in result['examples']:
                for count, prediction in [(7, 'predicted_action7'), (2, 'predicted_candidate2')]:
                    if row[f'correct{count}'] is not (row[prediction] == row['target_turn']):
                        raise ValueError('Correctness flags disagree with saved predictions')
            records[(mode, seed, size, condition)] = measured
        telemetry = read(run_dir/'telemetry'/f'{mode}-{seed}.json')
        if (telemetry != collections['telemetry'][(mode, seed)]
                or telemetry['checkpoint_sha256'] != fit['checkpoint_sha256']
                or len(telemetry['results']) != 3
                or {r['size'] for r in telemetry['results']} != set(SIZES)):
            raise ValueError('Telemetry file/summary/checkpoint/size coverage mismatch')
        for row in telemetry['results']:
            size = row['size']
            data, metadata = panels[size]
            validate_telemetry(row, study.telemetry(model, data, metadata), mode, size, metadata)
            step_count += len(row['per_step'])
    if len(records) != 108 or log_count != 6000 or step_count != 540:
        raise ValueError('Unexpected final verification coverage')
    means = {mode: {str(size): {condition: {key: float(np.mean([
        records[(mode, seed, size, condition)][key] for seed in SEEDS])) for key in METRICS}
        for condition in CONDITIONS} for size in SIZES} for mode in MODES}
    # CE from stored float32 probabilities and original logits can differ by a few ulps.
    for mode in MODES:
        for size in SIZES:
            for condition in CONDITIONS:
                for key in METRICS:
                    close(summary['averages'][mode][str(size)][condition][key],
                          means[mode][str(size)][condition][key], f'mean {mode}/{size}/{condition}/{key}')
    rule = plan['protocol']['continuation']
    checks = {}
    for size in SIZES:
        checks[f'perfect_all_fits_size_{size}'] = all(records[('fast_selective', seed, size, 'intact')]['accuracy2']
                                                   >= rule['accuracy2_each_fit_each_size'] for seed in SEEDS)
        checks[f'conditional_ce_size_{size}'] = all(
            records[('fast_selective', seed, size, 'intact')]['conditional_binary_cross_entropy']
            < rule['conditional_ce_each_fit_each_size_strictly_below'] for seed in SEEDS)
    candidate = means['fast_selective']['23']
    checks['longest_store_reset_drop'] = (candidate['intact']['accuracy2'] - candidate['reset_store']['accuracy2']
                                        >= rule['longest_size_store_reset_accuracy2_drop_minimum'])
    for control in rule['controls']:
        checks[f'longest_gain_over_{control}'] = (
            candidate['intact']['accuracy2'] - means[control]['23']['intact']['accuracy2']
            >= rule['longest_size_accuracy2_gain_each_control_minimum'])
    if checks != summary['continuation_checks'] or summary['continuation_passed'] is not all(checks.values()):
        raise ValueError('Frozen continuation gate does not replay')
    close(summary['training_seconds'], sum(r['training_seconds'] for r in summary['fits']), 'Training time', 1e-9)
    comparisons = verify_reference(run_dir, reference, reference_plan, reproduction, summary)
    return summary, records, comparisons


def figure(summary, records):
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.4))
    fig.subplots_adjust(left=.065, right=.975, top=.72, bottom=.32, wspace=.29)
    fig.suptitle('Associative memory solves the choice task; selective writes do not improve accuracy',
                 x=.065, y=.97, ha='left', fontsize=17, fontweight='bold', color='#233044')
    fig.text(.065, .917, 'Four-context supervised diagnostic  |  3 fits per mode  |  500 updates per fit  |  Forced forward paths',
             fontsize=10.5, color='#5c6878')
    fig.legend([Line2D([0], [0], color=c, linewidth=2.2) for c in COLORS], LABELS,
               loc='upper left', bbox_to_anchor=(.058, .88), ncol=2, frameon=False,
               fontsize=10.5, handlelength=2.2, columnspacing=2.5)
    panels = [('accuracy2', 'intact', 'Memory intact'),
              ('accuracy2', 'reset_store', 'Store reset before each observation'),
              ('conditional_binary_cross_entropy', 'intact', 'Conditional cross entropy (intact)')]
    for ax, (metric, condition, title) in zip(axes, panels, strict=True):
        for index, (mode, color) in enumerate(zip(MODES, COLORS, strict=True)):
            x = np.arange(3)+(index-1.5)*.07
            values = np.asarray([[records[(mode, seed, size, condition)][metric] for size in SIZES]
                                 for seed in SEEDS])
            if metric == 'accuracy2':
                values *= 100
            for row, offset in zip(values, [-.022, 0, .022], strict=True):
                ax.scatter(x+offset, row, color=color, s=25, alpha=.85, zorder=4)
            ax.plot(x, values.mean(0), color=color, linewidth=2.1, zorder=3)
        ax.set(xticks=range(3), xticklabels=['11 / 9\ntrained', '17 / 15', '23 / 21'],
               xlabel='Grid size / observations to fork')
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=14)
        if metric == 'accuracy2':
            ax.axhline(50, color='#8a96a5', linestyle=':', linewidth=1)
            ax.set(ylim=(0, 108), yticks=[0, 25, 50, 75, 100], ylabel='Correct left / right choices (%)')
        else:
            if any(records[(mode, seed, size, condition)][metric] <= 0
                   for mode in MODES for seed in SEEDS for size in SIZES):
                raise ValueError('Log-scale CE requires strictly positive observed values')
            ax.set_yscale('log')
            ax.set(ylabel='Cross entropy (nats; lower is better)')
        audit.style_axis(ax)
    failed = [key for key, passed in summary['continuation_checks'].items() if not passed]
    if failed != ['longest_gain_over_fast_global']:
        raise ValueError('This figure annotation requires the observed selective/global continuation failure')
    fig.text(.065, .19, 'Continuation gate failed: selective memory did not beat global memory in longest-path accuracy.',
             fontsize=11.2, fontweight='bold', color='#8e3d35')
    fig.text(.065, .142, 'Every fit is shown; lines are means. Horizontal offsets separate overlaps. '
             '50% is the balanced constant-turn baseline.', fontsize=9.4, color='#5c6878')
    fig.text(.065, .104, 'Probabilities / loss normalize over left and right. Seven-action argmax agrees in all saved results. '
             'Resetting all state gives 50% for every fit.', fontsize=9.4, color='#5c6878')
    fig.text(.065, .067, 'Store reset is a no-op for GRU / adapter controls. Longer paths reuse the same four context seeds; '
             'no learned navigation, RL or novelty result.', fontsize=9.4, color='#26364a')
    if (not all(r['accuracy7'] == r['accuracy2'] for r in records.values())
            or not all(r['accuracy2'] == .5 for identity, r in records.items() if identity[-1] == 'reset_all')
            or not all(records[(mode, seed, size, 'intact')]['accuracy2'] == 1.
                       for mode in ('fast_global', 'fast_selective') for seed in SEEDS for size in SIZES)
            or not all(records[(mode, seed, size, 'intact')] == records[(mode, seed, size, 'reset_store')]
                       for mode in ('gru', 'feedforward') for seed in SEEDS for size in SIZES)):
        raise ValueError('Figure scope annotations do not match verified records')
    return fig


def render(summary, records):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.hashsalt': 'associative-candidate-v1'})
    fig = figure(summary, records)
    content = {}
    for suffix in ('png', 'svg'):
        buffer = io.BytesIO()
        metadata = {'Software': 'OpenJev verified associative diagnostic'} if suffix == 'png' else {'Date': None}
        fig.savefig(buffer, format=suffix, dpi=170, facecolor='white', metadata=metadata)
        content[f'associative-candidate.{suffix}'] = buffer.getvalue()
    plt.close(fig)
    return content


def publish(args):
    if args.out.exists():
        raise FileExistsError('Figure output directory must be fresh')
    summary, records, comparisons = verify(args.run, args.plan, args.reference_run,
                                           args.reference_plan, args.reproduction)
    content = render(summary, records)
    args.out.mkdir(parents=True, exist_ok=False)
    for filename, value in content.items():
        with (args.out/filename).open('xb') as stream:
            stream.write(value)
    receipt = {
        'kind': 'verified_supervised_associative_candidate_figures',
        'summary_sha256': sha(args.run/'summary.json'), 'plan_sha256': sha(args.plan),
        'completed_sha256': sha(args.run/'completed.json'),
        'publisher_sha256': sha(Path(__file__)), 'verification_helper_sha256': sha(Path(audit.__file__)),
        'reference_summary_sha256': sha(args.reference_run/'summary.json'),
        'reference_plan_sha256': sha(args.reference_plan), 'reproduction_sha256': sha(args.reproduction),
        'verified_fits': 12, 'verified_optimizer_log_rows': 6000, 'verified_evaluation_results': 108,
        'verified_context_predictions': 432, 'verified_telemetry_trajectories': 36,
        'verified_telemetry_steps': 540, 'verified_telemetry_context_steps': 2160,
        'frozen_source_dependency_data_signature_verified': True, 'native_partial_observation_paths_replayed': True,
        'four_unique_balanced_contexts_each_result': True, 'checkpoint_prediction_replay': True,
        'telemetry_checkpoint_replay': True, 'baseline_exact_tensor_reproduction': comparisons,
        'accuracy7_equals_accuracy2_in_recorded_results': True,
        'continuation_passed': summary['continuation_passed'], 'continuation_checks': summary['continuation_checks'],
        'mean_metrics': summary['averages'],
        'artifacts': {filename: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
                      for filename, value in content.items()},
        'new_training': False, 'rl_training': False, 'gameplay_evaluated': False, 'unseen_unique_map_test': False,
        'novelty_established': False, 'independent_confirmation': False, 'scope': summary['protocol']['scope'],
    }
    with (args.out/'associative-candidate-figures.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in ('mean_metrics', 'baseline_exact_tensor_reproduction')}, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--reference-plan', type=Path, required=True)
    parser.add_argument('--reproduction', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    publish(parser.parse_args())


if __name__ == '__main__':
    main()
