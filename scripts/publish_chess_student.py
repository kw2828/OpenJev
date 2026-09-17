"""Audit and package original chess students, or redraw their completed report.

No training, engine calls, model inference, checkpoint selection or external model
redistribution occurs here. Original weights, JSON and JSONL files are copied
byte-for-byte; manifests record their new locations and unchanged hashes.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import chess
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_chess_student_publisher_study',
                                             ROOT/'scripts/train_chess_student.py')
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)
MODES = ('circuit', 'rewired', 'gru')
SEEDS = (17, 29, 43)
COLORS = {'circuit': '#176c78', 'rewired': '#b57935', 'gru': '#6557a5'}
METRICS = ('top1_teacher_agreement', 'target_nll', 'value_mae', 'value_mse')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('Nonfinite JSON number')

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def close(actual, expected, label):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f'Inconsistent {label}')


def validate_summary(summary):
    if (summary['status'] != 'completed' or summary['protocol'] != STUDY.PROTOCOL
            or summary['novelty_established'] is not False or summary['elo_estimate'] is not None
            or summary['data_role'] != 'generated development set, not untouched confirmation'):
        raise ValueError('Expected the completed, scoped chess-student-v1 development report')
    expected = {(mode, seed) for mode in MODES for seed in SEEDS}
    fits = summary['fits']
    if len(fits) != 9 or {(row['mode'], row['seed']) for row in fits} != expected:
        raise ValueError('Expected all nine distinct final fits')
    for row in fits:
        if row['examples'] != STUDY.PROTOCOL['dev_examples']:
            raise ValueError('Incorrect development example count')
        for metric in METRICS:
            if not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError('Invalid fit metric')
        if row['top1_teacher_agreement'] > 1 or row['value_mae'] > 2 or row['value_mse'] > 4:
            raise ValueError('Fit metric exceeds its valid range')
    for mode in MODES:
        for metric in METRICS:
            close(summary['mean_by_mode'][mode][metric],
                  np.mean([row[metric] for row in fits if row['mode'] == mode]), f'{mode} mean {metric}')
    if summary['continuation_gate'] != STUDY.continuation_gate(fits):
        raise ValueError('Continuation gate differs from the frozen rule')
    for name, key in [('uniform_random', 'expected_top1_teacher_agreement'),
                      ('training_move_frequency', 'top1_teacher_agreement')]:
        value = summary['baselines'][name][key]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid baseline agreement')
    return summary


def load_report(report):
    report = Path(report)
    receipt = read_json(report/'completed.json')
    if (receipt['status'] != 'completed' or receipt['summary_sha256'] != sha(report/'summary.json')
            or (report/'failed.json').exists()):
        raise ValueError('Report is incomplete or its summary checksum changed')
    summary = validate_summary(read_json(report/'summary.json'))
    if receipt['plan_sha256'] != summary['plan_sha256']:
        raise ValueError('Report plan identity mismatch')
    return summary, receipt


def audit(plan_path, execution, report, weights=None):
    """Verify archived evidence without requiring the original Stockfish binary."""
    plan_path, execution = Path(plan_path), Path(execution)
    weights = execution if weights is None else Path(weights)
    plan, plan_hash = read_json(plan_path), sha(plan_path)
    summary, report_receipt = load_report(report)
    if plan['protocol'] != STUDY.PROTOCOL or summary['plan_sha256'] != plan_hash:
        raise ValueError('Frozen protocol or plan identity differs')
    if set(plan['sources']) != set(STUDY.SOURCES):
        raise ValueError('Incomplete frozen source coverage')
    for source, digest in plan['sources'].items():
        if sha(ROOT/source) != digest:
            raise ValueError(f'Frozen training source changed: {source}')
    complete = read_json(execution/'completed.json')
    started = read_json(execution/'started.json')
    if (complete['status'] != 'completed' or complete['plan_sha256'] != plan_hash
            or report_receipt['execution_receipt_sha256'] != sha(execution/'completed.json')
            or started['plan_sha256'] != plan_hash or started['protocol'] != plan['protocol']
            or (execution/'failed.json').exists()):
        raise ValueError('Execution completion chain mismatch')
    expected_names = {f'{mode}-{seed}' for mode in MODES for seed in SEEDS}
    if (complete['fits'] != 9 or set(complete['fit_receipts']) != expected_names
            or {f'{mode}-{seed}' for mode, seed in complete['fit_order']} != expected_names
            or len(complete['fit_order']) != 9):
        raise ValueError('Execution does not contain exactly nine final fits')
    data = STUDY.validate_data(execution/'data')
    data_receipt = read_json(execution/'data/completed.json')
    if summary['teacher_cost'] != data_receipt:
        raise ValueError('Summary teacher costs differ from data-generation receipts')
    data_hash = sha(execution/'data/completed.json')
    if summary['baselines'] != STUDY.simple_baselines(data['train'], data['dev']):
        raise ValueError('Baselines do not reproduce from the published data')
    expected_updates = STUDY.PROTOCOL['epochs']*math.ceil(
        STUDY.PROTOCOL['train_examples']/STUDY.PROTOCOL['batch_size'])
    training_seconds = 0.
    for fit in summary['fits']:
        mode, seed = fit['mode'], fit['seed']
        name = f'{mode}-{seed}'
        directory = execution/name
        receipt = read_json(directory/'completed.json')
        if (sha(directory/'completed.json') != complete['fit_receipts'][name]
                or receipt['status'] != 'completed' or receipt['mode'] != mode or receipt['seed'] != seed
                or receipt['plan_sha256'] != plan_hash or receipt['data_receipt_sha256'] != data_hash
                or receipt['updates'] != expected_updates
                or receipt['examples_seen'] != STUDY.PROTOCOL['epochs']*STUDY.PROTOCOL['train_examples']
                or (directory/'failed.json').exists()
                or set(receipt['files']) != {'weights.pt', 'learning.jsonl', 'evaluation.json'}):
            raise ValueError(f'Invalid final fit receipt: {name}')
        for filename, digest in receipt['files'].items():
            artifact = weights/name/filename if filename == 'weights.pt' else directory/filename
            if sha(artifact) != digest:
                raise ValueError(f'Changed fit artifact: {name}/{filename}')
        if fit['checkpoint_sha256'] != receipt['files']['weights.pt']:
            raise ValueError('Summary checkpoint mismatch')
        model = STUDY.ChessStudent.load(weights/name/'weights.pt', expected_plan_sha256=plan_hash)
        if (model.mode != mode or model.seed != seed or model.parameter_count() != receipt['parameter_count']
                or model.parameter_count() != fit['parameter_count']
                or model.parameter_count() != plan['model_parameters'][mode]
                or any(not torch.isfinite(value).all() for value in model.state_dict().values())):
            raise ValueError('Checkpoint identity, tensor or parameter count mismatch')
        logs = STUDY.jsonl(directory/'learning.jsonl')
        if len(logs) != expected_updates:
            raise ValueError('Incomplete training log')
        for update, row in enumerate(logs, start=1):
            if (row['update'] != update or row['examples_seen'] != update*STUDY.PROTOCOL['batch_size']
                    or row['epoch'] != (update-1)//64+1
                    or any(not math.isfinite(row[key]) or row[key] < 0
                           for key in ('policy_ce', 'value_mse', 'loss', 'gradient_norm'))
                    or abs(row['loss']-row['policy_ce']-.5*row['value_mse']) > 1e-5):
                raise ValueError('Training log is inconsistent with the fixed budget/loss')
        evaluation = read_json(directory/'evaluation.json')
        predictions = evaluation['predictions']
        if len(predictions) != len(data['dev']):
            raise ValueError('Incomplete development predictions')
        for prediction, row in zip(predictions, data['dev'], strict=True):
            if (prediction['id'] != row['id'] or prediction['game_id'] != row['game_id']
                    or prediction['target'] != row['target_uci']
                    or prediction['target_value'] != row['target_value']
                    or prediction['correct'] is not (prediction['choice'] == prediction['target'])
                    or not 0 <= prediction['target_probability'] <= 1
                    or not math.isfinite(prediction['value']) or not -1 <= prediction['value'] <= 1
                    or chess.Move.from_uci(prediction['choice']) not in chess.Board(row['fen']).legal_moves):
                raise ValueError('Invalid saved development prediction')
        for metric, value in STUDY.summarize_predictions(predictions).items():
            close(evaluation['metrics'][metric], value, f'{name} saved evaluation {metric}')
            close(fit[metric], value, f'{name} summary {metric}')
        close(fit['training_wall_seconds'], receipt['training_wall_seconds'], 'fit training time')
        training_seconds += receipt['training_wall_seconds']
    close(complete['total_updates'], expected_updates*9, 'total training updates')
    close(complete['training_wall_seconds'], training_seconds, 'execution training time')
    close(summary['training_wall_seconds'], training_seconds, 'summary training time')
    return summary


def figure(summary):
    validate_summary(summary)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'chess-student-v1'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={'width_ratios': [1.65, 1]})
    fig.patch.set_facecolor('#fbfcfe')
    fig.subplots_adjust(left=.065, right=.975, bottom=.24, top=.77, wspace=.30)
    fig.text(.065, .935, 'OpenJev chess students', fontsize=21, weight='bold', color='#192b3c')
    fig.text(.065, .869, 'Stockfish top-move agreement | 1,024 generated development positions',
             fontsize=11, color='#536272')
    for ax in (left, right):
        ax.set_facecolor('#fbfcfe')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#c8d1da')
        ax.tick_params(color='#c8d1da', labelcolor='#354657')
        ax.set_axisbelow(True)
    values = [summary['mean_by_mode'][mode]['top1_teacher_agreement']*100 for mode in MODES]
    values += [summary['baselines']['uniform_random']['expected_top1_teacher_agreement']*100,
               summary['baselines']['training_move_frequency']['top1_teacher_agreement']*100]
    left.bar(range(5), values, color=[*(COLORS[mode] for mode in MODES), '#ccd3db', '#9aa7b6'], width=.66,
             alpha=.8, zorder=2)
    markers = ('o', 's', '^')
    fit_map = {(row['mode'], row['seed']): row for row in summary['fits']}
    for x, mode in enumerate(MODES):
        for offset, seed, marker in zip((-.18, 0, .18), SEEDS, markers, strict=True):
            left.scatter(x+offset, fit_map[mode, seed]['top1_teacher_agreement']*100,
                         s=42, marker=marker, facecolor='white', edgecolor='#1d3342', linewidth=1.1, zorder=4)
    for index, value in enumerate(values):
        top = (max(fit_map[MODES[index], seed]['top1_teacher_agreement']*100 for seed in SEEDS)
               if index < 3 else value)
        left.text(index, top+.72,
                  f'{value:.2f}%', ha='center', fontsize=10, weight='bold', color='#273d50')
    left.set_xticks(range(5), ['Circuit', 'Rewired', 'GRU', 'Uniform\nexpectation', 'Train-label\nfrequency'])
    left.set_ylim(0, max(20, max(values)+4))
    left.set_ylabel('Teacher agreement (%)')
    left.grid(axis='y', alpha=.2)
    left.set_title('All three final fits per learned model', loc='left', fontsize=11, pad=13)
    paired = [(fit_map['circuit', seed]['top1_teacher_agreement']
               -fit_map['rewired', seed]['top1_teacher_agreement'])*100 for seed in SEEDS]
    threshold = summary['protocol']['continuation_gate']['circuit_mean_agreement_gain_over_rewired']*100
    mean_gain = float(np.mean(paired))
    right.axvline(0, color='#a7b1bc', linewidth=1)
    right.axvline(threshold, color='#bd5147', linestyle='--', linewidth=1.3)
    for y, gain, marker in zip((3, 2, 1), paired, markers, strict=True):
        right.plot([0, gain], [y, y], color=COLORS['circuit'], linewidth=2)
        right.scatter(gain, y, s=49, marker=marker, color=COLORS['circuit'], zorder=3)
    right.scatter(mean_gain, 0, s=90, marker='D', color='#192b3c', zorder=3)
    right.text(threshold-.07, 3.66, f'Required +{threshold:.0f} pp', ha='right', color='#a43c35', fontsize=9)
    right.set_yticks([3, 2, 1, 0], ['Seed 17', 'Seed 29', 'Seed 43', 'Mean'])
    right.set_xlim(min(-.8, min(paired)-.3), max(threshold+.4, max(paired)+.4))
    right.set_ylim(-.5, 4.1)
    right.set_xlabel('Circuit minus rewired (percentage points)')
    right.set_title('Declared continuation comparison', loc='left', fontsize=11, pad=13)
    passed = summary['continuation_gate']['passed']
    fig.text(.065, .13,
             f'Circuit continuation gate: {"PASSED" if passed else "FAILED"}. '
             f'Mean gain {mean_gain:+.2f} pp; required +{threshold:.2f} pp.',
             fontsize=11, weight='bold', color='#176c78' if passed else '#a43c35')
    fig.text(.065, .074, 'Markers show seeds 17 / 29 / 43; bars show means. '
             'Shared data and training budget, unequal parameter counts.', fontsize=9, color='#536272')
    fig.text(.065, .035, 'Initial supervised development study. '
             'Teacher agreement does not establish gameplay strength, Elo or a circuit advantage.',
             fontsize=9, color='#536272')
    left.legend(handles=[Line2D([], [], marker=marker, color='#1d3342', linestyle='',
                                markerfacecolor='white', label=str(seed))
                         for marker, seed in zip(markers, SEEDS, strict=True)],
                title='Fit seed', frameon=False, fontsize=8, title_fontsize=8,
                loc='upper left', ncols=3, handletextpad=.3, columnspacing=.7)
    return fig


def save_figure(summary, report, prefix):
    prefix = Path(prefix)
    paths = {extension: prefix.with_suffix('.'+extension) for extension in ('png', 'svg', 'json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Figure outputs must be fresh')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(summary)
    fig.savefig(paths['png'], dpi=155, facecolor=fig.get_facecolor(), metadata={'Software': 'OpenJev'})
    fig.savefig(paths['svg'], facecolor=fig.get_facecolor(), metadata={'Date': None})
    plt.close(fig)
    svg = paths['svg'].read_text()
    paths['svg'].write_text('\n'.join(line.rstrip() for line in svg.splitlines())+'\n')
    STUDY.write_new(paths['json'], {
        'status': 'completed', 'protocol': 'chess-student-v1',
        'summary_sha256': sha(Path(report)/'summary.json'), 'report_receipt_sha256': sha(Path(report)/'completed.json'),
        'publisher_sha256': sha(__file__), 'training_source_sha256': sha(ROOT/'scripts/train_chess_student.py'),
        'plots': {name: sha(paths[name]) for name in ('png', 'svg')},
        'all_fit_seeds': list(SEEDS), 'gate_passed': summary['continuation_gate']['passed'],
        'scope': summary['data_role'], 'matplotlib': matplotlib.__version__,
    })
    return paths


def publish(plan, execution, report, out, models, figure_prefix):
    plan, execution, report, out, models = map(Path, (plan, execution, report, out, models))
    if out.exists() or models.exists():
        raise FileExistsError('Publication and model directories must be fresh')
    summary = audit(plan, execution, report)
    copies = [(plan, out/'plan.json')]
    copies += [(report/name, out/'report'/name) for name in ('summary.json', 'completed.json')]
    copies += [(execution/name, out/'execution'/name) for name in ('started.json', 'completed.json', 'engine.json')]
    copies += [(execution/'data'/name, out/'execution/data'/name) for name in
               ('started.json', 'completed.json', 'train.jsonl', 'dev.jsonl', 'games.jsonl', 'analyses.jsonl')]
    for mode in MODES:
        for seed in SEEDS:
            name = f'{mode}-{seed}'
            copies += [(execution/name/file, out/'execution'/name/file)
                       for file in ('started.json', 'completed.json', 'learning.jsonl', 'evaluation.json')]
            copies.append((execution/name/'weights.pt', models/name/'weights.pt'))
    out.mkdir(parents=True)
    models.mkdir(parents=True)
    mapping = []
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = sha(source)
        if sha(destination) != digest:
            raise ValueError('Publication changed an original file')
        mapping.append({'source': str(source), 'destination': str(destination),
                        'sha256': digest, 'bytes': destination.stat().st_size})
    shutil.copyfile(ROOT/'LICENSE', models/'LICENSE')
    (models/'README.md').write_text(
        '# OpenJev chess student checkpoints\n\n'
        'Nine original final checkpoints trained from random initialization by OpenJev. '
        'These weights are released under the included MIT license. '
        'They do not contain ChessFly, FlyWire, LFM, Qwen or Stockfish model assets.\n\n'
        'Modes: `circuit`, `rewired`, `gru`; seeds: `17`, `29`, `43`. '
        'Each `<mode>-<seed>/weights.pt` preserves the original training output bytes.\n\n'
        'The synthetic circuit continuation gate failed. All fits are included without selection. '
        'The GRU has 206,833 parameters; each circuit has 182,289. '
        'Training used bounded Stockfish-generated move/value labels, not biological data.\n\n'
        'See [usage and results](../../docs/chess-student.md) and '
        '[audit records](../../evidence/chess-student-v1/results/manifest.json). '
        '`checksums.json` records every weight hash.\n'
    )
    STUDY.write_new(models/'checksums.json', {
        'license': 'MIT', 'original_weights': True, 'plan_sha256': sha(plan),
        'files': {str(destination.relative_to(models)): sha(destination)
                  for _, destination in copies if destination.is_relative_to(models)},
    })
    plots = save_figure(summary, out/'report', figure_prefix)
    manifest = {
        'status': 'completed', 'protocol': 'chess-student-v1', 'original_files_unchanged': True,
        'scope': summary['scope'], 'data_role': summary['data_role'], 'novelty_established': False,
        'plan_sha256': sha(plan), 'publisher_sha256': sha(__file__), 'frozen_sources': read_json(plan)['sources'],
        'checkpoint_count': 9, 'checkpoint_license': 'MIT', 'external_assets_included': False,
        'files': mapping, 'additional_files': {
            str(path): sha(path) for path in [models/'LICENSE', models/'README.md', models/'checksums.json',
                                            *plots.values()]},
        'audits': ['completed receipt chain and all source/file hashes', 'nine original checkpoint identities',
                   'game-separated, globally state-deduplicated teacher data', 'all 192 updates per fit',
                   'all 1024 saved predictions per fit and recomputed metrics',
                   'recomputed baselines, averages and frozen continuation gate'],
        'inference_or_training_performed': False,
    }
    STUDY.write_new(out/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    package = commands.add_parser('publish')
    for argument in ('plan', 'execution', 'report', 'out', 'models', 'figure-prefix'):
        package.add_argument('--'+argument, type=Path, required=True)
    plot = commands.add_parser('plot')
    plot.add_argument('--report', type=Path, required=True)
    plot.add_argument('--out', type=Path, required=True, help='Fresh figure prefix, without extension')
    verify = commands.add_parser('audit', help='Read-only audit of the published, split-location package')
    for argument in ('plan', 'execution', 'report', 'models'):
        verify.add_argument('--'+argument, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'publish':
        result = publish(args.plan, args.execution, args.report, args.out, args.models, args.figure_prefix)
        print(json.dumps({'checkpoints': result['checkpoint_count'], 'status': result['status']}))
    elif args.command == 'plot':
        summary, _ = load_report(args.report)
        print(save_figure(summary, args.report, args.out))
    else:
        summary = audit(args.plan, args.execution, args.report, weights=args.models)
        print(json.dumps({'status': 'verified', 'fits': len(summary['fits']),
                          'continuation_gate_passed': summary['continuation_gate']['passed']}))


if __name__ == '__main__':
    main()
