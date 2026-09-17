"""Audit and publish the completed zero-critic initialization comparison.

No policy inference or training. Reuses the original publisher's numerical
validators, then checks all paired controls, initialization receipts and gates.
The compact public report includes a lossless archive of bound raw evidence.
"""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import random
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_script('_zero_publisher_audit', 'scripts/publish_associative_ppo.py')
study = load_script('_zero_publisher_study', 'scripts/zero_critic_ppo_study.py')
read, sha, same_tree = audit.read, audit.sha, audit.same_tree
ARMS, SEEDS, SIZES = audit.ARMS, audit.SEEDS, audit.SIZES
LABELS = ['GRU', 'GRU +\nadapter', 'Global\nwrite', 'Selective\nwrite']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_zero(report, execution, plan_path):
    plan = study.verify(plan_path)
    protocol = plan['protocol']
    done, reported = read(execution/'completed.json'), read(report/'completed.json')
    core_run, core_report = execution/'core', report/'core'
    core_done, core_reported = read(core_run/'completed.json'), read(core_report/'completed.json')
    summary, comparison = read(core_report/'summary.json'), read(report/'comparison.json')
    require(all(row['status'] == 'completed' and row['plan_sha256'] == sha(plan_path)
                for row in (done, reported, core_done, core_reported)), 'Incomplete or mismatched terminal receipts')
    require(done['fits'] == core_done['fits'] == 12 and done['evaluations'] == core_done['evaluations'] == 55,
            'Incomplete zero-critic execution coverage')
    require(done['core_receipt_sha256'] == sha(core_run/'completed.json') and
            reported['comparison_sha256'] == sha(report/'comparison.json') and
            reported['zero_summary_sha256'] == core_reported['summary_sha256'] == sha(core_report/'summary.json'),
            'Outer/core report hashes differ')
    require(read(core_report/'plan.json') == plan and summary['protocol'] == protocol and
            summary['plan_sha256'] == sha(plan_path), 'Core report is not bound to the frozen plan')
    require(summary['novelty_established'] is False and summary['independent_confirmation'] is False,
            'Claim boundary changed')
    start, core_start = read(execution/'started.json'), read(core_run/'started.json')
    require(start['plan_sha256'] == core_start['plan_sha256'] == sha(plan_path) and
            start['started_unix'] <= core_start['started_unix'] <= core_done['finished_unix'] <= done['finished_unix'],
            'Execution start/completion sequence differs')
    training = read(core_run/'training-completed.json')
    require(training['status'] == 'completed' and training['fits'] == summary['training'], 'Training receipt differs')
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    fits = {(row['arm'], row['seed']): row for row in summary['training']}
    require(len(summary['training']) == 12 and set(fits) == expected, 'Fit identities differ')
    order = [(arm, seed) for arm in ARMS for seed in SEEDS]
    random.Random(protocol['fit_order_seed']).shuffle(order)
    require(read(core_run/'fit_order.json') == [list(identity) for identity in order] and
            [(row['arm'], row['seed']) for row in training['fits']] == order, 'Fit order differs')
    require(set(done['initialization_sha256']) == {f'{arm}-{seed}' for arm, seed in expected},
            'Initialization receipt membership differs')
    models, logs = {}, {}
    for identity, row in fits.items():
        arm, seed = identity
        model = study.MODEL(arm, study.reference.OBS_SIZE, protocol['hidden'])
        logs[identity] = audit.validate_training(core_run, row, protocol, model)
        models[identity] = model
        name = f'{arm}-{seed}'
        path = core_run/'fits'/name/'initialization.json'
        initial = read(path)
        require(sha(path) == done['initialization_sha256'][name] and initial['arm'] == arm and
                initial['seed'] == seed and initial['smoke'] is False and initial['protocol'] == protocol['version'],
                'Initialization receipt identity or hash differs')
        for key in ('only_value_parameters_changed', 'policy_logits_and_states_identical', 'rng_unchanged',
                    'zero_value_predictions', 'zero_value_head_trainable'):
            require(initial[key] is True, f'Initialization intervention invariant failed: {key}')
    require(summary['interactions'] == sum(row['interactions'] for row in fits.values()) == 12582912 and
            sum(len(rows) for rows in logs.values()) == 12288 and
            sum(row['gradient_steps'] for row in fits.values()) == 49152, 'Full training budget differs')
    audit.close(summary['training_seconds'], sum(row['training_seconds'] for row in fits.values()), 'Training time')
    paths = sorted((core_run/'evaluation').glob('*.json'))
    records = [read(path) for path in paths]
    expected_records = {(arm, seed, mode) for arm, seed in expected for mode in study.reference.modes(arm)}
    expected_records.add(('random', None, 'intact'))
    require(len(records) == 55 and records == summary['evaluation'] and
            {(row['arm'], row.get('seed'), row['mode']) for row in records} == expected_records,
            'Evaluation coverage or lossless report records differ')
    grouped, results, identities = {}, {}, {}
    episodes, write_audits = 0, 0
    for record in records:
        arm, seed, mode = record['arm'], record.get('seed'), record['mode']
        identities[(arm, seed, mode)] = record
        if arm != 'random':
            require(record['checkpoint_sha256'] == fits[(arm, seed)]['checkpoint_sha256'], 'Checkpoint binding differs')
        require(len(record['results']) == 3 and {row['size'] for row in record['results']} == set(SIZES),
                'Evaluation size coverage differs')
        for row in record['results']:
            aggregate = audit.validate_result(row, arm, protocol, models.get((arm, seed)))
            results[(arm, seed, mode, row['size'])] = aggregate
            grouped.setdefault(arm, {}).setdefault(mode, {}).setdefault(str(row['size']), []).append(aggregate)
            episodes += len(row['episodes'])
            write_audits += int(row['write_audit'] is not None)
    require(len(results) == 165 and episodes == 21120 and write_audits == 90, 'Episode or telemetry count differs')
    averages = {arm: {mode: {size: {key: float(np.mean([row[key] for row in rows])) for key in audit.METRICS}
                            for size, rows in sizes.items()} for mode, sizes in modes.items()}
                for arm, modes in grouped.items()}
    same_tree(summary['averages'], averages, 'Zero-head averages')
    same_tree(summary['per_fit'], grouped, 'Zero-head per-fit aggregates')
    checks = audit.recompute_gates(protocol, averages, grouped)
    require(summary['checks'] == checks and summary['continuation_passed'] is all(checks.values()),
            'Selective-writing gate differs')
    paired, episode_identity = [], []
    for arm, seed in sorted(expected):
        intact = identities[(arm, seed, 'intact')]
        for mode in study.reference.modes(arm)[1:]:
            changed = {row['size']: row for row in identities[(arm, seed, mode)]['results']}
            for before in intact['results']:
                after = changed[before['size']]
                pairs = list(zip(before['episodes'], after['episodes'], strict=True))
                require(all(a['seed'] == b['seed'] for a, b in pairs), 'Intervention seeds differ')
                both = sum(not a['timeout'] and not b['timeout'] for a, b in pairs)
                reversed_branch = sum(not a['timeout'] and not b['timeout'] and a['success'] != b['success']
                                      for a, b in pairs)
                paired.append({'arm': arm, 'seed': seed, 'mode': mode, 'size': before['size'],
                               'assigned_pairs': len(pairs), 'both_reach_branch': both,
                               'reversed_branch': reversed_branch,
                               'success_difference': before['success']-after['success']})
                episode_identity.append({'arm': arm, 'seed': seed, 'mode': mode, 'size': before['size'],
                                         'assigned_pairs': len(pairs),
                                         'identical_saved_episode_records': sum(a == b for a, b in pairs)})
    require(len(paired) == 126, 'Intervention comparison coverage differs')
    same_tree(summary['paired_interventions'], paired, 'Zero-head interventions')
    bindings = plan['bindings']['paths']
    control, control_results, _, _ = audit.verify(ROOT/bindings['control_report'], ROOT/bindings['control_run'],
                                                  ROOT/bindings['control_plan'])
    expected_comparison = {
        'plan_sha256': sha(plan_path), 'control_summary_sha256': sha(ROOT/bindings['control_report']/'summary.json'),
        'zero_summary_sha256': sha(core_report/'summary.json'), **study.comparison(control, summary),
    }
    same_tree(comparison, expected_comparison, 'All 162 paired initialization comparisons and both gates')
    require(reported['repeated_random_episode_identity_verified'] is True, 'Random replay identity missing')
    return plan, summary, control, comparison, results, control_results, episode_identity


def make_figure(zero, comparison, results, control_results):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.hashsalt': 'zero-critic-ppo-v1'})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4))
    fig.subplots_adjust(left=.075, right=.98, bottom=.17, top=.82, hspace=.44, wspace=.24)
    fig.suptitle('Zero critic initialization: mixed optimization effects, no useful cue-memory evidence',
                 x=.075, y=.975, ha='left', fontsize=16.5, fontweight='bold', color='#233044')
    fig.text(.075, .934, 'All 12 new fits paired with 12 original controls | Same previously scored development seeds',
             fontsize=10.7, color='#536171')
    fig.legend([Line2D([0], [0], marker='o', color='#8b97a5', markerfacecolor='white', linestyle=''),
                Line2D([0], [0], marker='o', color='#147d87', linestyle='')],
               ['Original value head', 'Zero-initialized value head'],
               loc='upper left', bbox_to_anchor=(.068, .912), ncol=2, frameon=False, fontsize=10)
    for ax, size in zip(axes.flat, SIZES, strict=False):
        for index, arm in enumerate(ARMS):
            for offset, seed in zip([-.20, 0, .20], SEEDS, strict=True):
                original = 100*control_results[(arm, seed, 'intact', size)]['success']
                current = 100*results[(arm, seed, 'intact', size)]['success']
                ax.plot([index+offset]*2, [original, current], color='#bdc6cf', linewidth=1.2, zorder=2)
                ax.scatter(index+offset, original, s=43, facecolors='white', edgecolors='#8b97a5', zorder=3)
                ax.scatter(index+offset, current, s=25, color='#147d87', zorder=4)
        ax.axhline(50, color='#d9dfe5', linewidth=.8, linestyle=':')
        ax.set(xticks=range(4), xticklabels=LABELS, ylabel='Successful episodes (%)', ylim=(-4, 104),
               yticks=[0, 25, 50, 75, 100])
        ax.set_title(f'Size {size}'+(' (training size)' if size == 11 else ' (longer corridor)'),
                     loc='left', fontweight='bold', pad=10)
        audit.style_axis(ax)
    ax = axes[1, 1]
    for offset, mode, marker, color in [(-.16, 'reset_all', 's', '#d8a15f'),
                                       (0, 'cue_swapped', 'o', '#526e9a'),
                                       (.16, 'reset_store', '^', '#8e65ad')]:
        for index, arm in enumerate(ARMS):
            if mode == 'reset_store' and not arm.startswith('fast_'):
                continue
            values = [100*(results[(arm, seed, 'intact', 11)]['success']-
                           results[(arm, seed, mode, 11)]['success']) for seed in SEEDS]
            ax.scatter([index+offset-.04, index+offset, index+offset+.04], values,
                       marker=marker, color=color, s=28, zorder=3)
    ax.axhline(0, color='#bbc4ce', linewidth=1)
    ax.set(xticks=range(4), xticklabels=LABELS, ylabel='Intact minus intervention (pp)', ylim=(-7, 62),
           yticks=[0, 20, 40, 60])
    ax.set_title('Zero head: cue and state interventions, size 11', loc='left', fontweight='bold', pad=10)
    ax.legend([Line2D([0], [0], marker=m, linestyle='', color=c) for m, c in
               [('s', '#d8a15f'), ('o', '#526e9a'), ('^', '#8e65ad')]],
              ['Reset all', 'Swap cue', 'Reset store'], frameon=False, loc='upper left', fontsize=8.5)
    audit.style_axis(ax)
    status = 'PASSED' if comparison['optimization_continuation_passed'] else 'FAILED'
    selective = 'PASSED' if zero['continuation_passed'] else 'FAILED'
    fig.text(.075, .105, f'Optimization continuation gate: {status}. Selective-writing gate: {selective}. '
             'Every final fit is shown.', fontsize=11, fontweight='bold', color='#963f36')
    fig.text(.075, .067, 'Each marker is one training seed; links pair the same seed across initializations. '
             'Three seeds per architecture, 128 episodes per size.', fontsize=9.2, color='#536171')
    fig.text(.075, .032, 'One global-store reset causes timeouts. Cue swaps and selective-store resets show no '
             'success loss. Reused development panel; no novelty claim.', fontsize=9.2, color='#233044')
    return fig


def raw_archive(paths):
    """Deterministic, lossless bundle; source-relative names allow exact recovery."""
    content = io.BytesIO()
    manifest = {}
    with (gzip.GzipFile(fileobj=content, mode='wb', filename='', mtime=0) as compressed,
          tarfile.open(fileobj=compressed, mode='w') as archive):
        for path in sorted({Path(path).resolve() for path in paths}):
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
            manifest[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    value = content.getvalue()
    # Verify every archive member in memory before writing the public packet.
    with tarfile.open(fileobj=io.BytesIO(value), mode='r:gz') as archive:
        require(set(archive.getnames()) == set(manifest), 'Archive membership differs')
        for member in archive:
            require(hashlib.sha256(archive.extractfile(member).read()).hexdigest() == manifest[member.name]['sha256'],
                    'Lossless archive byte verification failed')
    return value, manifest


def csv_bytes(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def publish(plan_path, execution, report, out):
    require(not out.exists(), 'Publication output must be fresh')
    plan, zero, original, comparison, results, control_results, identity = verify_zero(report, execution, plan_path)
    files = {}
    figure = make_figure(zero, comparison, results, control_results)
    for extension in ('png', 'svg'):
        stream = io.BytesIO()
        metadata = {'Software': 'OpenJev audited zero-critic report'} if extension == 'png' else {'Date': None}
        figure.savefig(stream, format=extension, dpi=160, facecolor='white', metadata=metadata)
        files[f'zero-critic-comparison.{extension}'] = stream.getvalue()
    plt.close(figure)
    aggregate = []
    for condition, summary in [('original', original), ('zero', zero)]:
        for arm, modes in summary['averages'].items():
            for mode, sizes in modes.items():
                for size, metrics in sizes.items():
                    aggregate.append({'initialization': condition, 'arm': arm, 'mode': mode,
                                      'size': int(size), **metrics})
    files['aggregate-results.csv'] = csv_bytes(aggregate)
    files['paired-fit-results.csv'] = csv_bytes(comparison['paired_fits'])
    files['episode-identity-audit.csv'] = csv_bytes(identity)
    files['comparison.json'] = (report/'comparison.json').read_bytes()
    files['report-completed.json'] = (report/'completed.json').read_bytes()
    paths = [path for directory in (execution, report) for path in directory.rglob('*') if path.is_file()]
    paths += [plan_path, ROOT/'uv.lock']
    paths += [ROOT/path for path in plan['bindings']['artifact_sha256']]
    paths += [ROOT/path for path in plan['source_hashes']]
    files['raw-evidence.tar.gz'], raw_manifest = raw_archive(paths)
    files['raw-evidence-manifest.json'] = (json.dumps(raw_manifest, indent=2)+'\n').encode()
    summary = {
        'status': 'completed', 'plan_sha256': sha(plan_path), 'scope': comparison['scope'],
        'new_fits': 12, 'reused_control_fits': 12, 'new_evaluation_files': 55, 'new_assigned_episodes': 21120,
        'new_training_interactions': zero['interactions'], 'new_optimizer_steps': 49152,
        'new_training_seconds': zero['training_seconds'], 'original_training_seconds': original['training_seconds'],
        'original_averages': original['averages'], 'zero_averages': zero['averages'],
        'mean_success_gain': comparison['mean_success_gain'],
        'optimization_checks': comparison['optimization_checks'],
        'optimization_continuation_passed': comparison['optimization_continuation_passed'],
        'original_selective_checks': comparison['original_selective_checks'],
        'zero_selective_checks': comparison['zero_selective_checks'],
        'zero_selective_continuation_passed': zero['continuation_passed'],
        'cue_swap_identical_records': sum(row['identical_saved_episode_records'] for row in identity
                                          if row['mode'] == 'cue_swapped'),
        'cue_swap_assigned_pairs': sum(row['assigned_pairs'] for row in identity if row['mode'] == 'cue_swapped'),
        'selective_store_reset_identical_records': sum(row['identical_saved_episode_records'] for row in identity
                                                       if row['arm'] == 'fast_selective' and row['mode'] == 'reset_store'),
        'selective_store_reset_assigned_pairs': sum(row['assigned_pairs'] for row in identity
                                                    if row['arm'] == 'fast_selective' and row['mode'] == 'reset_store'),
        'novelty_established': False, 'independent_confirmation': False, 'new_inference_or_training': False,
    }
    files['summary.json'] = (json.dumps(summary, indent=2, allow_nan=False)+'\n').encode()
    receipt = {
        'status': 'completed', 'kind': 'zero_critic_ppo_lossless_publication',
        'publisher_sha256': sha(Path(__file__)),
        'audit_helper_sha256': sha(ROOT/'scripts/publish_associative_ppo.py'),
        'plan_sha256': sha(plan_path), 'report_receipt_sha256': sha(report/'completed.json'),
        'execution_receipt_sha256': sha(execution/'completed.json'),
        'verified_new_fits': 12, 'verified_reused_control_fits': 12,
        'verified_new_log_rows': 12288, 'verified_new_gradient_steps': 49152,
        'verified_new_evaluation_files': 55, 'verified_new_size_conditions': 165,
        'verified_new_episodes': 21120, 'verified_new_write_audits': 90,
        'verified_initialization_receipts': 12, 'paired_initialization_rows': 162,
        'paired_intervention_rows': 126, 'new_inference_or_training': False,
        'raw_archive_member_count': len(raw_manifest), 'raw_archive_bytes_verified': True,
        'raw_scope': 'Complete new execution/report plus every frozen reused-control/probe artifact and study source',
        'artifacts': {name: {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
                      for name, value in files.items()},
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        with (out/name).open('xb') as handle:
            handle.write(value)
    with (out/'publication.json').open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--execution', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    summary = publish(args.plan, args.execution, args.report, args.out)
    print(json.dumps({key: value for key, value in summary.items() if 'averages' not in key}, indent=2))


if __name__ == '__main__':
    main()
