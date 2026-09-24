"""Publish one stopped convex-head attempt, without numerical re-evaluation.

Only stdlib JSON/CSV/SVG and opaque byte hashing/archival are used. Sources,
upstream proof, original qualification and failed producer closures, and the
exact stopped inventory are authenticated before reading solver reports. Those
reports remain producer outputs, not independently numerically audited results.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import math
import re
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STUDY_NAME = 'finite-convex-readout-v1'
VERSION = 'finite-convex-readout-stopped-publication-v1'
ARCHIVE = 'finite-convex-readout-stopped-v1.tar.gz'
PARENTS = ('factorized', 'matched_free', 'dense_free')
SEEDS = (426261001, 426261002, 426261003)
CONFIG = {'dev_seed_namespace': 427260924, 'dev_attempts': 128, 'dev_horizon': 8,
          'batch_size': 64, 'parent_seeds': list(SEEDS), 'parent_arms': list(PARENTS),
          'train_namespace': 426260924, 'train_horizon': 2,
          'solver_maxiter': 2000, 'solver_ftol': 1e-12, 'solver_max_objective_calls': 10000,
          'solver_gap_tolerance': 1e-8, 'solver_repair_tolerance': 1e-10,
          'solver_nonincrease_tolerance': 1e-12}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve() == path,
            'absolute regular original file: ' + str(path))
    return path


def descriptor(path):
    path = regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(regular(path).read_text())


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder, 'original phase directory')
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'no symlinks in original output')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = descriptor(path)
    return result


def finite_tree(value):
    if type(value) is float:
        require(math.isfinite(value), 'finite publication scalar')
    elif type(value) is dict:
        for child in value.values():
            finite_tree(child)
    elif type(value) is list:
        for child in value:
            finite_tree(child)


def safe_relative(name):
    require(type(name) is str and name and '\\' not in name, 'portable relative member')
    require(not PurePosixPath(name).is_absolute()
            and all(part not in ('', '.', '..') for part in name.split('/')), 'no traversal or empty member component')
    return name


def closed(plan, plan_path, phase, *, success, worker):
    spec = plan['phases'][phase]
    folder, launch_path = Path(spec['output']), regular(Path(spec['supervision']))
    require(folder.is_relative_to(plan_path.parent) and launch_path.is_relative_to(plan_path.parent), 'registered phase paths')
    require(launch_path.name.endswith('.launch.json'), 'original launch suffix')
    terminal_path = launch_path.with_name(launch_path.name.removesuffix('.launch.json') + '.terminal.json')
    receipt_path, log_path = Path(str(folder) + '.receipt.json'), launch_path.with_name(launch_path.name.removesuffix('.launch.json') + '.log')
    receipt, launch, terminal = read(receipt_path), read(launch_path), read(terminal_path)
    regular(log_path)
    require(receipt['phase'] == phase and receipt['status'] == ('PASS' if success else 'FAILED')
            and receipt['plan'] == str(plan_path) and receipt['plan_sha256'] == descriptor(plan_path)['sha256']
            and receipt['output'] == str(folder) and receipt['supervision'] == str(launch_path), 'original receipt identity')
    require(receipt['sources_before'] == plan['sources'], 'original source entry pins')
    if success:
        require(receipt['sources_after'] == plan['sources'], 'successful original source closure')
    else:
        require('sources_after' not in receipt and 'result' not in receipt,
                'failed producer did not reach success/source-after recording')
    require(receipt['launch'] == launch and all(terminal[key] == value for key, value in launch.items()), 'exact original process join')
    require(terminal['status'] == ('completed' if success else 'failed')
            and type(terminal['returncode']) is int and (terminal['returncode'] == 0 if success else terminal['returncode'] > 0)
            and terminal['timed_out'] is False and terminal['group_absent'] is True
            and terminal['cleanup']['reaped'] is True and terminal['cleanup']['group_absent'] is True
            and terminal['cleanup']['errors'] == [] and terminal['error'] is None
            and terminal['clock_error'] is None and terminal['timing_available'] is True,
            'ordinary completed or failed child, no timeout or remaining group')
    require(terminal['cwd'] == str(ROOT) and terminal['cap_seconds'] == spec['cap_seconds']
            and terminal['started_ns'] < terminal['finished_ns'] < terminal['deadline_ns']
            and terminal['deadline_ns'] == terminal['started_ns'] + spec['cap_seconds'] * 10**9
            and math.isclose(terminal['wall_seconds'], (terminal['finished_ns'] - terminal['started_ns']) / 1e9,
                             rel_tol=0, abs_tol=1e-9), 'registered original native-clock bounds')
    command = terminal['command']
    expected = {'--plan': str(plan_path), '--plan-sha256': descriptor(plan_path)['sha256'],
                '--phase': phase, '--supervision': str(launch_path), '--output': str(folder)}
    require(len(command) == 12 and command[0] == plan['runtime']['executable']
            and (ROOT / command[1]).resolve() == ROOT / worker and len(set(command[2::2])) == 5,
            'exact original executable, worker and unique flags')
    observed = dict(zip(command[2::2], command[3::2], strict=True))
    require(set(observed) == set(expected), 'exact flag roster')
    for flag in ('--plan', '--supervision', '--output'):
        observed[flag] = str((ROOT / observed[flag]).resolve())
    require(observed == expected and terminal['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py']['sha256']
            and terminal['clock_source_sha256'] == plan['sources']['src/openjev/research/suspend_clock.py']['sha256'],
            'registered command and original supervisor sources')
    require(inventory(folder) == receipt['files'], 'every original phase payload matches receipt')
    return {'directory': folder, 'receipt': receipt, 'terminal': terminal,
            'receipt_path': receipt_path, 'launch_path': launch_path, 'terminal_path': terminal_path, 'log_path': log_path}


def authenticate(study, registration_sha256):
    require(study == ROOT / 'output' / STUDY_NAME, 'exact current study directory')
    plan_path = regular(study / 'study-registration.json')
    require(re.fullmatch('[0-9a-f]{64}', registration_sha256) is not None
            and descriptor(plan_path)['sha256'] == registration_sha256, 'external registered identity')
    plan = read(plan_path)
    require(plan['version'] == STUDY_NAME and plan['mode'] == 'study' and plan['root'] == str(ROOT)
            and plan['config'] == CONFIG and len(plan['sources']) == 37, 'exact stopped study and 37-source closure')
    require({name: spec['cap_seconds'] for name, spec in plan['phases'].items()}
            == {'qualify': 300, 'fit': 600, 'audit': 600}, 'registered phase limits')
    for name, pin in plan['sources'].items():
        require(descriptor(ROOT / safe_relative(name)) == pin, 'unchanged registered source ' + name)
    upstream = plan['upstream']
    upstream_folder = ROOT / 'output/finite-factorized-dynamics-v1'
    require(upstream['folder'] == str(upstream_folder) and inventory(upstream_folder) == upstream['files'],
            'complete opaque historical parent inventory')
    upstream_path = regular(Path(upstream['registration']['path']))
    require(upstream_path == upstream_folder / 'study-registration.json'
            and upstream['registration'] == {'path': str(upstream_path), **descriptor(upstream_path)}, 'original parent registration')
    upstream_plan = read(upstream_path)
    require(upstream_plan['version'] == 'finite-factorized-dynamics-v1' and upstream_plan['mode'] == 'study'
            and upstream_plan['root'] == str(ROOT)
            and upstream['run'] == upstream_plan['phases']['fit']['output'], 'original parent producer path')
    for name, pin in upstream_plan['sources'].items():
        require(plan['sources'].get(name) == pin, 'current closure contains frozen parent source')
    # Parent payloads are opaque; original parent process records remain joined.
    old_fit, old_audit = (closed(upstream_plan, upstream_path, phase, success=True,
                               worker='scripts/finite_factorized_dynamics_worker.py') for phase in ('fit', 'audit'))
    require(old_fit['terminal']['finished_ns'] <= old_audit['terminal']['started_ns']
            and old_audit['receipt']['producer_receipt'] == descriptor(old_fit['receipt_path'])
            and old_audit['receipt']['producer_terminal'] == descriptor(old_fit['terminal_path']), 'original parent audit join')
    old_q = upstream_plan['qualification']
    require(descriptor(Path(old_q['path'])) == old_q['descriptor'], 'parent qualification bytes')
    old_q_plan_path = regular(Path(read(Path(old_q['path']))['plan']))
    old_q_plan = read(old_q_plan_path)
    require(old_q_plan['sources'] == upstream_plan['sources'] and old_q_plan['mode'] == 'engineering'
            and old_q_plan['config'] == upstream_plan['config'], 'qualified original parent sources')
    old_qualify = closed(old_q_plan, old_q_plan_path, 'qualify', success=True,
                         worker='scripts/finite_factorized_dynamics_worker.py')
    require(descriptor(old_qualify['terminal_path']) == old_q['terminal']
            and old_qualify['terminal']['finished_ns'] <= old_fit['terminal']['started_ns'], 'original parent qualification closure')
    q = plan['qualification']
    require(descriptor(Path(q['path'])) == q['descriptor'], 'current qualification receipt bytes')
    q_plan_path = regular(Path(read(Path(q['path']))['plan']))
    q_plan = read(q_plan_path)
    require(q_plan['version'] == STUDY_NAME and q_plan['mode'] == 'engineering'
            and q_plan['config'] == CONFIG and q_plan['sources'] == plan['sources']
            and q_plan['upstream'] == upstream, 'same qualified current configuration and inputs')
    qualify = closed(q_plan, q_plan_path, 'qualify', success=True, worker='scripts/finite_convex_readout_worker.py')
    require(qualify['receipt_path'] == Path(q['path']) and descriptor(qualify['terminal_path']) == q['terminal'],
            'exact current original qualification closure')
    commands = qualify['receipt']['commands']
    expected_commands = [[str(ROOT / '.venv/bin/ruff'), 'check', *q_plan['lint_sources']],
                         [q_plan['runtime']['executable'], '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *q_plan['tests']]]
    require(len(commands) == 2 and set(qualify['receipt']['files']) == {'command-0.log', 'command-1.log'}, 'complete engineering logs')
    for row, command in zip(commands, expected_commands, strict=True):
        require(row['command'] == command and row['returncode'] == 0
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'original successful lint and fabricated-test commands')
    fit = closed(plan, plan_path, 'fit', success=False, worker='scripts/finite_convex_readout_worker.py')
    require(qualify['terminal']['finished_ns'] <= fit['terminal']['started_ns'], 'qualification closes before stopped fit')
    expected_files = {'config.json', 'train.npz', 'solves.jsonl', 'failure.json'}
    expected_files |= {f'{prefix}-{parent}-{seed}.{suffix}' for parent in PARENTS for seed in SEEDS
                       for prefix, suffix in (('states-train', 'npz'), ('solve', 'json'))}
    require(len(expected_files) == 22 and set(fit['receipt']['files']) == expected_files,
            'exact stopped 22-file roster, no DEV, solve barrier or success summary')
    audit_spec = plan['phases']['audit']
    audit_launch = Path(audit_spec['supervision'])
    for path in (Path(audit_spec['output']), Path(audit_spec['output'] + '.receipt.json'), audit_launch,
                 audit_launch.with_name(audit_launch.name.removesuffix('.launch.json') + '.terminal.json'),
                 audit_launch.with_name(audit_launch.name.removesuffix('.launch.json') + '.log')):
        require(not path.exists(), 'scientific audit was never started')
    require(read(fit['directory'] / 'config.json') == CONFIG, 'saved configuration unchanged')
    return plan_path, plan, q_plan_path, {'qualify': qualify, 'fit': fit}


def stopped_records(plan, phases):
    # The caller completes every source, process and payload check first.
    folder = phases['fit']['directory']
    failure = read(folder / 'failure.json')
    journal = [json.loads(line) for line in regular(folder / 'solves.jsonl').read_text().splitlines()]
    order = [(parent, seed) for index, seed in enumerate(SEEDS)
             for parent in (PARENTS[index % 3:] + PARENTS[:index % 3])]
    require(len(journal) == 9 and [(row['parent'], row['seed']) for row in journal] == order, 'complete rotated nine-parent journal')
    parent_summary = read(Path(plan['upstream']['run']) / 'summary.json')
    parents = {(row['arm'], row['seed']): row for row in parent_summary['fits']}
    require(len(parents) == 9 and set(parents) == set(order), 'original nine-parent metadata roster')
    require(descriptor(folder / 'train.npz') == parent_summary['files']['train.npz'], 'byte-identical reused TRAIN only')
    n = parent_summary['dataset_counts']['train']['retained']
    require(type(n) is int and n > 0, 'original nonempty retained TRAIN support')
    rows = []
    for row in journal:
        parent, seed = row['parent'], row['seed']
        require(read(folder / f'solve-{parent}-{seed}.json') == row, 'per-solve JSON equals durable journal')
        finite_tree(row)
        original = parents[parent, seed]
        checkpoint = Path(plan['upstream']['run']) / f'{parent}-{seed}.npz'
        require(row['source_checkpoint'] == {'path': str(checkpoint), **descriptor(checkpoint)}
                and descriptor(checkpoint) == {name: original['checkpoint'][name] for name in ('sha256', 'bytes')}, 'original checkpoint identity')
        require(row['source_state_sha256'] == row['model_state_after'] == original['final_state_sha256']
                and row['frozen_weights'] is True and row['optimizer_weight_updates'] == 0,
                'reported all-parameter identity before and after each solve')
        require(row['training_cases'] == n and row['horizon'] == 2, 'unchanged TRAIN-only objective population')
        state_name = f'states-train-{parent}-{seed}.npz'
        require(row['train_state_file'] == {'path': state_name, **descriptor(folder / state_name)}, 'opaque saved TRAIN-state identity')
        result, certificate = row['solver_result'], row['solver_result']['certificate']
        require(result['version'] == STUDY_NAME and result['solver']['success'] is True
                and result['solver']['status'] == 0, 'all original SLSQP calls reported success')
        require(type(result['complete']) is bool and certificate['passed'] == result['complete']
                and result['status'] == ('SOLVE_PASS' if result['complete'] else 'SOLVE_FAIL')
                and result['failure_reasons'] == ([] if result['complete'] else ['final_numerical_certificate']),
                'only the numerical certificate stopped these reported solves')
        require(-1e-12 <= certificate['fw_gap'] and certificate['simplex_violation'] <= 1e-12
                and result['raw_feasibility'] <= 1e-10 and result['projection']['applied'] is True
                and result['projection']['max_abs'] <= 1e-10
                and certificate['objective'] <= result['objective_initial'] + 1e-12
                and (certificate['fw_gap'] <= 1e-8) == result['complete'],
                'reported certificate scalars agree with frozen pass/fail rule, without recomputing them')
        work = result['work']
        require(work['simplex_projection_calls'] == 1 and work['direct_objective_gradient_passes'] == 3
                and 0 < work['objective_calls'] == work['gradient_calls'] <= 10000
                and 0 <= result['solver']['iterations'] <= 2000, 'single reported bounded attempt and projection')
        for key in ('load_seconds', 'extraction_seconds', 'solve_seconds'):
            require(type(row[key]) in (int, float) and row[key] >= 0, 'reported nonnegative nested time')
        rows.append({'parent': parent, 'seed': seed, 'scipy_success': True,
            'certificate_passed': result['complete'], 'reported_fw_gap': certificate['fw_gap'], 'threshold': 1e-8,
            'reported_objective_initial': result['objective_initial'], 'reported_objective_final': certificate['objective'],
            'reported_simplex_violation': certificate['simplex_violation'], 'reported_projection_max_abs': result['projection']['max_abs'],
            'load_seconds': row['load_seconds'], 'extraction_seconds': row['extraction_seconds'], 'solve_seconds': row['solve_seconds'],
            'objective_calls': work['objective_calls'], 'iterations': result['solver']['iterations']})
    passed = sum(row['certificate_passed'] for row in rows)
    require(passed == 3 and len(rows) - passed == 6, 'this original stopped attempt has three reported passes and six failures')
    counts = failure['counts']
    batches = (n + 63) // 64
    expected_counts = {'train_array_decodes': 1, 'checkpoint_decodes': 9, 'model_constructions': 9,
        'oracle_model_constructions': 0, 'original_head_exports': 9, 'original_head_softmaxes': 9,
        'train_blind_calls': 9 * batches, 'train_observed_calls': 9 * batches,
        'train_readout_validation_products': 18 * batches, 'train_readout_validation_rows': 36 * n,
        'solver_calls': 9, 'completed_solves': 3, 'dev_generation_count': 0,
        'oracle_blind_rollouts': 0, 'oracle_observed_rollouts': 0,
        'evaluation_blind_calls': 0, 'evaluation_observed_calls': 0, 'evaluation_shuffled_calls': 0,
        'head_matrix_products': 0, 'head_matrix_rows': 0, 'evaluation_case_views': 0,
        'checkpoint_writes': 0, 'optimizer_steps': 0, 'external_model_calls': 0, 'teacher_calls': 0, 'native_calls': 0}
    reason = "ValueError('every TRAIN-only solve must succeed before DEV, no replacement or fallback')"
    require(failure['version'] == STUDY_NAME and failure['completed_solve_records'] == 9
            and counts == expected_counts and failure['error'] == phases['fit']['receipt']['error'] == reason
            and failure['stage'] == f'head solve {order[-1][0]} {order[-1][1]}', 'original stop reason and exact no-DEV counters')
    finite_tree(failure)
    return rows, failure, journal


def chart(rows):
    width, height = 1200, 545
    left, right, top, step = 260, 815, 90, 39
    positive = [row['reported_fw_gap'] for row in rows if row['reported_fw_gap'] > 0]
    low = min(-10, math.floor(math.log10(min(positive))) - 1)
    high = max(-6, math.ceil(math.log10(max(positive))))

    def x(value):
        exponent = math.log10(value) if value > 0 else low
        return left + (max(low, exponent) - low) * (right - left) / (high - low)

    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Stopped convex-readout run: all nine reported certificate gaps</title>',
        '<desc id="desc">Producer-reported values, not independently numerically audited. Dashed line is the fixed one times ten to the minus eight threshold. Three pass and six fail.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial,sans-serif" fill="#17212b">',
        '<text x="28" y="30" font-size="20" font-weight="bold">Stopped before DEV: 3 of 9 certificate checks passed</text>',
        '<text x="28" y="55" font-size="13">SLSQP success was insufficient. These are producer-reported values, not a numerical audit.</text>']
    for exponent in range(low, high + 1):
        px = x(10. ** exponent)
        elements += [f'<line x1="{px:.2f}" y1="75" x2="{px:.2f}" y2="{top + step * 9 - 12}" stroke="#e3e7eb"/>',
                     f'<text x="{px:.2f}" y="{top + step * 9 + 12}" font-size="12" text-anchor="middle">1e{exponent}</text>']
    threshold = x(1e-8)
    elements.append(f'<line x1="{threshold:.2f}" y1="73" x2="{threshold:.2f}" y2="{top + step * 9 - 10}" stroke="#333" stroke-dasharray="5 4"/>')
    for i, row in enumerate(rows):
        y, color = top + step * i, '#247b52' if row['certificate_passed'] else '#b53f33'
        label = html.escape(f"{row['parent']} | {row['seed']}")
        elements += [f'<text x="245" y="{y + 5}" font-size="13" text-anchor="end">{label}</text>',
            f'<line x1="{left}" y1="{y}" x2="{x(row["reported_fw_gap"]):.2f}" y2="{y}" stroke="{color}" stroke-width="7"/>',
            f'<circle cx="{x(row["reported_fw_gap"]):.2f}" cy="{y}" r="4" fill="{color}"/>',
            f'<text x="835" y="{y + 5}" font-size="12" fill="{color}">{row["reported_fw_gap"]:.17g} | {"PASS" if row["certificate_passed"] else "FAIL"}</text>']
    elements += ['<text x="538" y="490" font-size="12" text-anchor="middle">Reported Frank-Wolfe gap (log scale); fixed threshold 1e-8</text>',
                 f'<text x="28" y="514" font-size="12">Nonpositive gaps are plotted at 1e{low} only; labels and CSV retain the reported value, including zero.</text>',
                 '<text x="28" y="536" font-size="12">No retries. No DEV generation or independent scientific audit. All nine outputs retained.</text>', '</g></svg>']
    return '\n'.join(elements) + '\n'


def document(rows, failure, phases):
    lines = ['# Convex-readout diagnostic stopped before DEV', '',
        '**STOPPED: three of nine reported certificate checks passed.** All nine SLSQP calls reported success, but six did not meet the fixed Frank-Wolfe gap threshold of 1e-8. The registered all-solves gate stopped the run before DEV generation. There was no retry, fallback success or independent scientific audit.', '',
        '![All nine producer-reported certificate gaps and the fixed threshold](certificate-gaps.svg)', '',
        '**These certificate values are producer outputs, not independently numerically audited results.** This publisher verifies original source/process/payload identity and consistency of the saved status fields. It does not decode latent states or recompute any loss, gradient or certificate.', '',
        '| Parent | Seed | Reported FW gap | Fixed threshold | Reported certificate | Original TRAIN loss | Reported final TRAIN loss |',
        '|---|---:|---:|---:|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['parent']} | {row['seed']} | {row['reported_fw_gap']:.9g} | 1e-8 | "
                     f"{'PASS' if row['certificate_passed'] else 'FAIL'} | {row['reported_objective_initial']:.9g} | {row['reported_objective_final']:.9g} |")
    lines += ['', 'The objective remains original blind cost MSE plus observed cost MSE on H1/H2 TRAIN states. Solver termination tolerance and the registered direct-residual certificate requirement are different conditions. A successful SLSQP status did not establish the required numerical certificate. The original recurrent models stayed frozen according to every saved before/after state hash; this publisher checks those hashes against original metadata without decoding checkpoints.', '',
        'The domain is the closed per-state probability simplex, with cost matrix C = 0.25 - P. It permits boundary heads that finite softmax logits only approach. Even a completed solve would therefore not isolate optimizer choice from this domain extension.', '',
        '## Recorded cost and stop boundary', '',
        '| Parent | Seed | Load seconds | TRAIN extraction seconds | Build/solve/certificate seconds | Objective calls | Iterations |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['parent']} | {row['seed']} | {row['load_seconds']:.6f} | {row['extraction_seconds']:.6f} | "
                     f"{row['solve_seconds']:.6f} | {row['objective_calls']} | {row['iterations']} |")
    lines += ['', (f"Original qualification duration: {phases['qualify']['terminal']['wall_seconds']:.3f} seconds. "
              f"Original stopped producer duration: {phases['fit']['terminal']['wall_seconds']:.3f} seconds. "
              'These supervisor durations include process cleanup and already contain the nested timers. Prior recurrent-model training is excluded.'), '',
        (f"Recorded solver calls: {failure['counts']['solver_calls']}; qualifying solves: {failure['counts']['completed_solves']}; "
        'DEV generation, evaluation calls, new checkpoint writes, model-weight optimizer steps, teacher calls and native environment calls: zero. '
        'The exact 22-file stopped inventory has nine saved TRAIN-state archives, nine solve records, the journal, copied TRAIN, configuration and failure receipt. No solve barrier, DEV file, success summary or scientific-audit launch exists.'), '',
        '## Evidence and limits', '',
        ('[Frozen protocol](https://github.com/kw2828/OpenJev/blob/main/research/finite-convex-readout-protocol.md) · '
        '[All solve rows](solves.csv) · '
        '[Manifest](https://github.com/kw2828/OpenJev/releases/download/finite-convex-readout-v1/manifest.json) · '
        f'[Complete stopped evidence archive](https://github.com/kw2828/OpenJev/releases/download/finite-convex-readout-v1/{ARCHIVE})'), '',
        'The archive contains all 37 registered sources, current original process records and payloads, and every explicitly pinned parent-study member as opaque historical provenance. Only original TRAIN and the nine original checkpoints were reused scientific inputs; previous DEV arrays are historical proof and were not decoded for this experiment. Absolute source paths are provenance, not a portable installation contract.', '',
        'The failed worker did not reach its final sources-after field. Current source and upstream hashes are verified against the registered pins before and after publication; that check is not relabeled as a historical worker-end attestation.', '',
        'No fresh decision-performance result, architecture improvement, latent-information conclusion, calibration result, native transfer or novelty claim follows from this stopped attempt. Earlier study verdicts remain unchanged. A repaired solver would require a separate prospective registration, not a silent rerun of this attempt.', '']
    return '\n'.join(lines)


def publish(study, output, *, registration_sha256):
    study, output = Path(study).resolve(), Path(output)
    require(output.is_absolute() and output.resolve() == output and not output.exists(), 'exclusive absolute publication directory')
    require(not output.is_relative_to(study) and not output.is_relative_to(ROOT / 'output/finite-factorized-dynamics-v1'),
            'publication cannot mutate either evidence inventory')
    plan_path, plan, q_plan_path, phases = authenticate(study, registration_sha256)
    rows, failure, journal = stopped_records(plan, phases)
    members, origins = {}, {}

    def add(name, path, expected=None):
        name, path = safe_relative(name), regular(path)
        require(path.is_relative_to(ROOT) and name != 'MANIFEST.json', 'repository-bound evidence and nonreserved name')
        pin = descriptor(path)
        require(expected is None or pin == expected, 'explicit archive input identity')
        if name in members:
            require(origins[name] == path and members[name] == {**pin, 'original_path': str(path)}, 'consistent duplicate binding')
            return
        members[name], origins[name] = {**pin, 'original_path': str(path)}, path

    for name, pin in plan['sources'].items():
        add('sources/' + safe_relative(name), ROOT / name, pin)
    for name, pin in plan['upstream']['files'].items():
        add('upstream/finite-factorized-dynamics-v1/' + safe_relative(name), Path(plan['upstream']['folder']) / name, pin)
    for path in (plan_path, q_plan_path):
        add('study/' + path.relative_to(study).as_posix(), path)
    require(sorted(study.glob('engineering-registration-[0-9][0-9].json')) == [q_plan_path],
            'this publication preserves the single original qualification; unexpected attempts require explicit handling')
    for phase in phases.values():
        for name, pin in phase['receipt']['files'].items():
            path = phase['directory'] / safe_relative(name)
            add('study/' + path.relative_to(study).as_posix(), path, pin)
        for key in ('receipt_path', 'launch_path', 'terminal_path', 'log_path'):
            path = phase[key]
            add('study/' + path.relative_to(study).as_posix(), path)
    add('publication/publish_finite_convex_readout_stop.py', Path(__file__).resolve())
    add('publication/LICENSE', ROOT / 'LICENSE')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.md').write_text(document(rows, failure, phases))
    (output / 'certificate-gaps.svg').write_text(chart(rows))
    with (output / 'solves.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ('report.md', 'solves.csv', 'certificate-gaps.svg'):
        add('publication/' + name, output / name)
    manifest = {'version': VERSION, 'status': 'STOPPED_BEFORE_DEV',
        'registration': {'path': str(plan_path), **descriptor(plan_path)}, 'files': members,
        'reported_solves': rows, 'reported_complete_solve_records': journal, 'reported_failure': failure,
        'scientific_numerical_audit_performed': False, 'producer_values_recomputed': False,
        'original_phase_seconds': {name: phase['terminal']['wall_seconds'] for name, phase in phases.items()},
        'original_processes': {name: {key: {'path': str(phase[key + '_path']), **descriptor(phase[key + '_path'])}
                                     for key in ('receipt', 'launch', 'terminal', 'log')} for name, phase in phases.items()},
        'source_after_scope': 'Failed producer has no sources_after. Current pins verified before and after publication only.',
        'upstream': plan['upstream'],
        'upstream_scope': 'Every explicit pin retained as opaque historical proof. Earlier DEV data was not decoded or reused in this experiment.',
        'manifest_coverage': 'Every tar member except MANIFEST.json; that member equals this external manifest byte for byte.',
        'archive_metadata': 'Sorted regular files, mode0644, uid/gid0, empty owner names, timestamps0; gzip filename empty.',
        'reproduction_limit': 'Evidence package, not an admitted retry or portable environment installer.',
        'model_selection': False, 'architecture_claim': False, 'novelty_claim': False,
        'counts': dict.fromkeys(('array_decodes', 'checkpoint_decodes', 'model_calls', 'generator_calls',
                                'optimizer_calls', 'teacher_calls', 'native_calls', 'certificate_recomputations'), 0)}
    raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    (output / 'manifest.json').write_bytes(raw_manifest)
    with (output / ARCHIVE).open('xb') as stream, gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as zipped, \
            tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted([*members, 'MANIFEST.json']):
            info = tarfile.TarInfo(name)
            info.size = len(raw_manifest) if name == 'MANIFEST.json' else members[name]['bytes']
            info.mode, info.uid, info.gid, info.mtime = 0o644, 0, 0, 0
            info.uname = info.gname = ''
            if name == 'MANIFEST.json':
                archive.addfile(info, io.BytesIO(raw_manifest))
            else:
                with origins[name].open('rb') as source:
                    archive.addfile(info, source)
    with tarfile.open(output / ARCHIVE, 'r:gz') as archive:
        entries = archive.getmembers()
        require(len(entries) == len(members) + 1 and {entry.name for entry in entries} == set(members) | {'MANIFEST.json'}, 'complete archive roster')
        for entry in entries:
            require(entry.isfile() and entry.mode == 0o644 and entry.uid == entry.gid == entry.mtime == 0, 'regular deterministic member')
            source, digest = archive.extractfile(entry), hashlib.sha256()
            for block in iter(lambda source=source: source.read(1024**2), b''):
                digest.update(block)
            pin = {'sha256': hashlib.sha256(raw_manifest).hexdigest(), 'bytes': len(raw_manifest)} if entry.name == 'MANIFEST.json' else members[entry.name]
            require(entry.size == pin['bytes'] and digest.hexdigest() == pin['sha256'], 'opaque member roundtrip')
    for name, path in origins.items():
        require(descriptor(path) == {key: members[name][key] for key in ('sha256', 'bytes')}, 'all bytes unchanged after publication')
    require(inventory(Path(plan['upstream']['folder'])) == plan['upstream']['files'], 'upstream inventory unchanged')
    for phase in phases.values():
        require(inventory(phase['directory']) == phase['receipt']['files'], 'original phase inventory unchanged')
    receipt = {'version': VERSION, 'status': 'PUBLICATION_COMPLETE_FOR_STOPPED_RUN',
        'study_status': 'STOPPED_BEFORE_DEV', 'scientific_numerical_audit_performed': False,
        'registration': {'path': str(plan_path), **descriptor(plan_path)},
        'publisher': {'path': str(Path(__file__).resolve()), **descriptor(Path(__file__).resolve())},
        'files': {name: descriptor(output / name) for name in ('report.md', 'solves.csv', 'certificate-gaps.svg', 'manifest.json', ARCHIVE)},
        'opaque_roundtrip': True, 'sources_unchanged': True, 'members_checked': len(members) + 1,
        'counts': manifest['counts'], 'visual_review_required': True}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(descriptor(output / name)['sha256'] + '  ' + name + '\n'
                                             for name in (*receipt['files'], 'receipt.json')))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.output, registration_sha256=args.registration_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
