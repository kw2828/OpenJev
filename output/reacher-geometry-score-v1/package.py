"""Prepare complete lossless publication assets only after completed audit authorization.

No experiment imports, model calls, native replay, uploads or reads at import time.
Archives use deterministic headers, streamed verification and byte-exact parts.
Failed engineering preparation attempts are preserved in a separately labeled archive.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'reacher-geometry-score-v1'
EXPECTED_PLAN = '94c90f7585303f4edd88e1b0cb0dc8a488a2e2f07237aae4cfb489ade2068a2b'
EXPECTED_READINESS = 'd5ed5822d5948ede133e2dbab9af5502a882dda913db1bbd53caa5e835b88e35'
EXPECTED_LAUNCH = '132460958820e4e9845b1a104cc94a46f2cf4c94ecb155310f80734950976d4d'
MAX_ASSET_BYTES = 1_400_000_000
PART_BYTES = 1_400_000_000
CHUNK_BYTES = 8 * 1024 * 1024
ARMS = ('residual_gru', 'cached_mlp')
PAIRS = ('pair0', 'pair1', 'pair2')
MODES = ('learned', 'geometry')
PANELS = ('full', 'ordinary', 'shift')
REFERENCES = ('known_state', 'particle', 'zero', 'uniform', 'public_kinematic')
REPORT_FILES = {'report.json', 'tables.md', 'native-costs.png', 'utility-vs-cost.png',
                'gate-checks.png', 'fixed-case-replay.gif', 'replay-preview.png'}
PREPARATION_TREES = ('output/reacher-geometry-capacity-v1', 'output/reacher-geometry-rehearsal-v1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'Nonfinite JSON number')
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def digest(value):
    require(isinstance(value, str) and len(value) == 64
            and set(value) <= set('0123456789abcdef'), 'External SHA-256 required')
    return value


def child(folder, name):
    require(isinstance(name, str) and name and '\\' not in name, 'Safe relative path')
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and all(part not in ('', '.', '..') for part in rel.parts)
            and rel.as_posix() == name, 'Canonical relative member path')
    result = Path(folder).joinpath(*rel.parts)
    require(not Path(folder).is_symlink() and not any(parent.is_symlink()
            for parent in [result, *result.parents] if parent != parent.parent), 'No symlink paths')
    return result


def checked(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing regular member: ' + str(path))
    require(sha(path) == digest(expected), 'Member hash changed: ' + str(path))
    return path


def files(folder):
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(), 'Required real directory: ' + str(folder))
    result = {}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'No symlinks in artifact tree')
        require(path.is_file() or path.is_dir(), 'No special artifact files')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = path
    return result


def bind_members(folder, members, extras=()):
    require(isinstance(members, dict), 'Member hash mapping required')
    actual = files(folder)
    require(set(actual) == set(members) | set(extras), 'Exact artifact membership: ' + str(folder))
    for name, expected in members.items():
        checked(child(folder, name), expected)
    return set(actual.values())


def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, label)
    return value


def verify_archive(archive, manifest):
    expected = {row['path']: row for row in manifest}
    require(len(expected) == len(manifest), 'Unique archived member paths')
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            require(member.isfile() and member.name in expected and member.name not in seen,
                    'Exact regular archive member')
            require(member.name == PurePosixPath(member.name).as_posix()
                    and not PurePosixPath(member.name).is_absolute()
                    and '..' not in PurePosixPath(member.name).parts, 'Safe archive path')
            row = expected[member.name]
            require(member.size == row['bytes'] and member.mode == 0o644 and member.mtime == 0
                    and member.uid == member.gid == 0 and member.uname == member.gname == '',
                    'Archive member length and deterministic header')
            with tar.extractfile(member) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256'],
                        'Archive member bytes')
            seen.add(member.name)
    require(seen == set(expected), 'Complete archive membership')


def bound_argument(value):
    require(isinstance(value, str) and '=' in value, 'Expected repository-relative path=SHA256')
    name, expected = value.rsplit('=', 1)
    return name, digest(expected)


def write_archive(root, paths, archive):
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        require(child(root, relative) == path and path.is_file(), 'Repository-contained regular input')
        rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    with (
        archive.open('xb') as raw,
        gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=1) as gz,
        tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar,
    ):
        for row in rows:
            path = root / row['path']
            require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                    'Input changed before archiving')
            info = tarfile.TarInfo(row['path'])
            info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with path.open('rb') as handle:
                tar.addfile(info, handle)
    verify_archive(archive, rows)
    return rows


def split_assets(archive, out, *, max_asset_bytes=MAX_ASSET_BYTES, part_bytes=PART_BYTES):
    require(0 < part_bytes <= max_asset_bytes < 2_000_000_000, 'Every asset below 2 GB')
    archive_bytes, archive_sha = archive.stat().st_size, sha(archive)
    if archive_bytes <= max_asset_bytes:
        return [{'name': archive.name, 'bytes': archive_bytes, 'sha256': archive_sha}], False
    parts = []
    with archive.open('rb') as source:
        while True:
            first = source.read(min(CHUNK_BYTES, part_bytes))
            if not first:
                break
            destination = out / f'{archive.name}.part{len(parts):03d}'
            count, remaining = 0, part_bytes
            with destination.open('xb') as handle:
                data = first
                while data:
                    handle.write(data)
                    count += len(data)
                    remaining -= len(data)
                    if remaining == 0:
                        break
                    data = source.read(min(CHUNK_BYTES, remaining))
            parts.append({'name': destination.name, 'bytes': count, 'sha256': sha(destination)})
    joined = hashlib.sha256()
    total = 0
    for part in parts:
        path = out / part['name']
        require(0 < path.stat().st_size == part['bytes'] <= part_bytes and sha(path) == part['sha256'],
                'Part identity and size')
        with path.open('rb') as handle:
            while data := handle.read(CHUNK_BYTES):
                joined.update(data)
                total += len(data)
    require(total == archive_bytes and joined.hexdigest() == archive_sha, 'Concatenated parts exactly recover archive')
    return parts, True


def fit_names():
    return [f'{arm}-{pair}' for pair in PAIRS for arm in ARMS]


def expected_members():
    """Independent filename-only coverage; no scientific module import."""
    names = {'started.json', 'random-streams.json', 'inheritance.json', 'all-models-restored.json',
             'evaluation-started.json', 'diagnostic-completed.json', 'control-completed.json',
             'final-models.json', 'costs.json', 'inherited/train.npz', 'inherited/train.json'}
    names |= {f'inherited/source-{part}.json' for part in ('plan', 'audit', 'completed', 'summary')}
    for pair in PAIRS:
        names |= {f'inherited/{folder}/{pair}.{ext}' for folder in ('initializations', 'orders')
                  for ext in ('pt', 'json')}
    for fit in fit_names():
        names |= {f'inherited/fits/{fit}/{name}' for name in
                  ('initial-weights.pt', 'weights.pt', 'checkpoint.pt', 'training.jsonl', 'completed.json')}
        names |= {f'model-states/{fit}-{stage}.pt' for stage in ('before', 'after')}
    names |= {f'innovations/control/{step:03d}.{ext}' for step in range(50) for ext in ('npz', 'json')}
    for panel in PANELS:
        names |= {f'inherited/control/{panel}/residual_gru-pair0/episodes.{ext}' for ext in ('npz', 'json')}
        learned = {f'{fit}--{mode}' for fit in fit_names() for mode in MODES}
        for label in learned | set(REFERENCES):
            prefix = f'control/{panel}/{label}'
            names |= {f'{prefix}/{name}' for name in ('episodes.npz', 'episodes.json', 'timings.json')}
            if label in learned:
                names |= {f'{prefix}/{name}' for name in ('executed_predictions.npz', 'states.npz', 'state-work.json')}
                names |= {f'{prefix}/{folder}/{step:03d}.{ext}' for folder in ('decisions', 'scoring')
                          for step in range(50) for ext in ('npz', 'json')}
            else:
                names.add(f'{prefix}/planning.npz')
                if label == 'public_kinematic':
                    names.add(f'{prefix}/observer-final.json')
        for case in range(8):
            for step in (12, 32):
                prefix = f'diagnostic/{panel}/{case:03d}/{step:03d}'
                names |= {f'{prefix}/{name}' for name in ('innovations.npz', 'innovations.json', 'history.npz',
                    'union.npz', 'native-root.npz', 'root.json', 'native.npz', 'timings.json')}
                for fit in fit_names():
                    names.add(f'{prefix}/states/{fit}.npz')
                    names |= {f'{prefix}/{folder}/{fit}--{mode}.{ext}' for folder in ('search', 'search-scoring', 'scores')
                              for mode in MODES for ext in ('npz', 'json')}
    return names


def validate_scored(root, audit_sha):
    # The external audit identity and successful terminal state precede all result reads.
    audit_dir = root / f'evidence/{STUDY}/audit'
    audit = read(checked(audit_dir / 'receipt.json', audit_sha))
    require(audit['status'] == 'completed' and audit['engineering'] is False
            and audit['saved_output_only'] is True and audit['version'] == STUDY,
            'Completed independent scored audit required first')
    protocol = root / f'evidence/{STUDY}/protocol'
    plan = read(checked(protocol / 'plan.json', EXPECTED_PLAN))
    readiness = read(checked(protocol / 'readiness.json', EXPECTED_READINESS))
    require(set(files(protocol)) == {'plan.json', 'readiness.json'}, 'Exact plan/readiness files')
    require(plan['study'] == STUDY and plan['engineering'] is False and len(plan['sources']) == 80,
            'Exact scored study and80 frozen source files')
    require(plan['arms'] == list(ARMS) and plan['pairs'] == list(PAIRS) and plan['score_modes'] == list(MODES)
            and plan['panels'] == list(PANELS) and plan['references'] == list(REFERENCES)
            and plan['control_episodes'] == 64 and plan['steps'] == 50
            and plan['diagnostic_cases'] == 8 and plan['diagnostic_root_steps'] == [12, 32]
            and plan['diagnostic_branches'] == 4, 'Exact full-size prospective coverage')
    require(readiness['status'] == 'prepared_before_scored_execution'
            and readiness['plan_sha256'] == EXPECTED_PLAN
            and readiness['scored_random_samples_drawn'] is False
            and readiness['models_selected_by_capacity_performance'] is False
            and readiness['frozen_execution_cap_seconds'] == plan['cap_seconds'] == 2400
            and readiness['frozen_audit_cap_seconds'] == plan['audit_cap_seconds'] == 600,
            'Readiness and fixed budgets are unchanged')
    paths = {checked(child(root, name), value) for name, value in plan['sources'].items()}
    paths |= set(files(protocol).values())
    execution = root / f'runs/{STUDY}/attempt'
    completed = read(checked(execution / 'completed.json', audit['execution_completed_sha256']))
    require(completed['status'] == 'completed' and completed['study'] == STUDY
            and completed['plan_sha256'] == audit['plan_sha256'] == EXPECTED_PLAN
            and audit['source_sha256'] == plan['sources'] and audit['runtime'] == plan['runtime'],
            'Completed execution/audit/source identity')
    require(completed['new_fits'] == completed['astra_calls'] == 0
            and completed['restored_models'] == 6 and completed['control_rows'] == 51
            and completed['diagnostic_roots'] == 48, 'All inherited models and every row/root completed')
    require(set(completed['files']) == expected_members() and audit['execution_members'] == completed['files'],
            'Exact full trace and six inherited checkpoint membership')
    paths |= bind_members(execution, completed['files'], ('completed.json',))
    require(set(audit['files']) == {'summary.json', 'README.md'}, 'Exact successful audit outputs')
    paths |= bind_members(audit_dir, audit['files'], ('receipt.json',))
    summary = read(audit_dir / 'summary.json')
    require(summary['status'] == 'completed' and summary['version'] == STUDY
            and summary['engineering'] is False and summary['saved_output_only'] is True
            and summary['plan_sha256'] == EXPECTED_PLAN
            and summary['execution_completed_sha256'] == audit['execution_completed_sha256']
            and summary['costs'] == audit['costs']
            and summary['new_model_calls'] == summary['new_policy_calls'] == summary['new_fits'] == 0,
            'Summary is the authenticated saved-output audit')
    require(set(summary['inherited_fits']) == set(fit_names()) and set(summary['control']) == set(PANELS),
            'All six fits and all panels')
    expected_rows = {f'{fit}--{mode}' for fit in fit_names() for mode in MODES} | set(REFERENCES)
    for panel in PANELS:
        require(set(summary['control'][panel]) == expected_rows, 'All17 control rows in each panel')
        require(all(len(row['episode_costs']) == 64 for row in summary['control'][panel].values()),
                'All64 native cases in every control row')
    roots = {(row['panel'], row['case_index'], row['step']) for row in summary['diagnostic']}
    require(len(summary['diagnostic']) == len(roots) == 48
            and roots == {(panel, case, step) for panel in PANELS for case in range(8) for step in (12, 32)},
            'All48 prespecified diagnostic roots')
    for row in summary['diagnostic']:
        require(row['timing']['identity_slots'] == 76 and 1 <= row['timing']['unique_sequences'] <= 76
                and row['native_replay']['transitions'] == row['timing']['unique_sequences'] * 4 * 12,
                'Complete finite union and four full native branches')
    control_count = 51 * 64 * 50
    diagnostic_count = sum(row['native_replay']['transitions'] for row in summary['diagnostic'])
    require(summary['native_control_transitions_checked'] == control_count
            and summary['native_diagnostic_transitions_checked'] == diagnostic_count
            and summary['native_transitions_checked'] == control_count + diagnostic_count,
            'Complete audited native replay count')
    require(summary['phase_boundary']['restored_models_before_diagnostic'] == 6
            and summary['phase_boundary']['diagnostic_before_fresh_controls'] is True
            and summary['phase_boundary']['new_fits'] == summary['phase_boundary']['new_optimizer_updates'] == 0,
            'Original zero-fit phase boundary')
    checks = summary['continuation_gate']['checks']
    require(len(checks) == 25 and len({row['name'] for row in checks}) == 25
            and all(type(row['passed']) is bool for row in checks)
            and type(summary['continuation_gate']['passed']) is bool
            and summary['continuation_gate']['passed'] == all(row['passed'] for row in checks),
            'Preserve all25 checks whether the gate passes or fails')
    require(positive(completed['wall_seconds'], 'Execution wall') <= plan['cap_seconds']
            and positive(summary['costs']['audit_validation_wall_seconds'], 'Audit wall') <= plan['audit_cap_seconds'],
            'Successful terminal completion within frozen caps')
    started = read(execution / 'started.json')
    require(readiness['prepared_unix_time'] <= started['unix_time'], 'Readiness predates execution')
    launch_path = checked(root / f'output/{STUDY}/launch.json', EXPECTED_LAUNCH)
    launch = read(launch_path)
    require(launch['record_type'] == 'launch_metadata_recorded_after_process_start'
            and launch['frozen_plan_sha256'] == EXPECTED_PLAN
            and launch['runner_started'] == started
            and launch['runner_started_sha256'] == sha(execution / 'started.json')
            and launch['execution_cap_seconds'] == plan['cap_seconds']
            and launch['audit_cap_seconds'] == plan['audit_cap_seconds']
            and launch['recorded_unix_time'] >= started['unix_time'],
            'Retrospective launch metadata is labeled and bound to the original start')
    paths.add(launch_path)
    return plan, audit, summary, readiness, paths


def validate_reporting(root, directory, expected, plan, audit, summary, audit_sha):
    receipt = read(checked(directory / 'receipt.json', expected))
    inputs = {'plan_sha256': EXPECTED_PLAN, 'audit_receipt_sha256': audit_sha,
              'audit_summary_sha256': audit['files']['summary.json'],
              'execution_completed_sha256': audit['execution_completed_sha256'],
              'execution_member_count': len(audit['execution_members']), 'frozen_source_count': 80}
    require(receipt['status'] == 'completed' and receipt['study'] == STUDY
            and receipt['engineering'] is False
            and receipt['scope'] == 'saved-artifact reporting and fixed-case schematic replay only'
            and receipt['inputs'] == inputs and receipt['source_sha256'] == plan['sources']
            and receipt['new_model_calls'] == receipt['new_policy_calls'] == receipt['new_native_calls'] == 0,
            'Final completed reporting binds exact scientific inputs')
    require(set(receipt['files']) == REPORT_FILES and receipt['control_rows'] == 51
            and receipt['learned_rows'] == 36 and receipt['reference_rows'] == 15,
            'All reporting figures, raw report and schematic replay retained')
    gate = summary['continuation_gate']
    require(receipt['gate_passed'] == gate['passed']
            and receipt['checks_passed'] == sum(row['passed'] for row in gate['checks']), 'Report gate unchanged')
    members = bind_members(directory, receipt['files'], ('receipt.json',))
    members.add(checked(root / f'output/{STUDY}/render.py', receipt['renderer_source_sha256']))
    report = read(directory / 'report.json')
    require(report['inputs'] == inputs and report['continuation_gate'] == gate
            and report['costs'] == summary['costs'] and report['replay'] == receipt['replay'], 'Report arithmetic provenance')
    replay = receipt['replay']
    fixed = {f'{arm}-pair0--{mode}' for arm in ARMS for mode in MODES}
    require(set(replay['models']) == fixed and replay['frames'] == 50
            and replay['frame_duration_ms'] == 100 and replay['playback_seconds'] == 5
            and replay['native_episode_seconds'] == 1 and replay['native_pixels'] is False,
            'All four fixed policies and complete50-step schematic')
    for label, row in replay['models'].items():
        require(row['case_index'] == 0 and row['pair'] == 'pair0' and row['panel'] == 'ordinary',
                'Replay selection fixed before outcomes')
        for key, extension in (('episodes_npz_sha256', 'npz'), ('episodes_json_sha256', 'json')):
            require(row[key] == audit['execution_members'][f'control/ordinary/{label}/episodes.{extension}'],
                    'Replay bytes are part of audited raw execution')
        require(math.isclose(row['native_cost'], summary['control']['ordinary'][label]['episode_costs'][0],
                             rel_tol=1e-12, abs_tol=1e-12), 'Replay cost is the selected actual case')
    return members


def preparation_files(root, readiness, sources):
    """Retain both preparation attempts, including the two failed originals."""
    paths = set()
    for tree in PREPARATION_TREES:
        paths |= set(files(child(root, tree)).values())
    for name, expected in readiness['inputs'].items():
        path = checked(child(root, name), expected)
        require(path in paths, 'Every readiness dependency retained in preparation archive')
    required = {
        'output/reacher-geometry-capacity-v1/attempt/execution/failed.json': 'failed',
        'output/reacher-geometry-capacity-v1/attempt-02/execution/completed.json': 'completed',
        'output/reacher-geometry-capacity-v1/attempt-02/audit/completed.json': 'completed',
        'output/reacher-geometry-rehearsal-v1/attempt-01/audit/failed.json': 'failed',
        'output/reacher-geometry-rehearsal-v1/attempt-02/execution/completed.json': 'completed',
        'output/reacher-geometry-rehearsal-v1/attempt-02/audit/receipt.json': 'completed',
    }
    for name, status in required.items():
        require(read(checked(child(root, name), readiness['inputs'][name]))['status'] == status,
                'Original preparation terminal states remain visible')
    # Rebind complete successful manifests, rather than merely carrying their receipts.
    terminals = (
        'output/reacher-geometry-capacity-v1/attempt-02/execution/completed.json',
        'output/reacher-geometry-capacity-v1/attempt-02/audit/completed.json',
        'output/reacher-geometry-rehearsal-v1/attempt-01/execution/completed.json',
        'output/reacher-geometry-rehearsal-v1/attempt-02/execution/completed.json',
        'output/reacher-geometry-rehearsal-v1/attempt-02/audit/receipt.json',
    )
    for name in terminals:
        receipt_path = child(root, name)
        terminal = read(checked(receipt_path, readiness['inputs'][name]))
        mapping = terminal['files']
        hashes = {key: value['sha256'] if isinstance(value, dict) else value for key, value in mapping.items()}
        require(bind_members(receipt_path.parent, hashes, (receipt_path.name,)) <= paths,
                'Every successful preparation member is retained')
        for key, value in mapping.items():
            if isinstance(value, dict):
                require(child(receipt_path.parent, key).stat().st_size == value['bytes'],
                        'Preparation byte count remains exact')
    for stem in ('attempt-01', 'attempt-02'):
        folder = root / f'output/reacher-geometry-rehearsal-v1/{stem}'
        plan = read(folder / 'plan.json')
        paths.add(checked(root / 'tests/reacher_geometry_fixture.py', plan['fixture_source_sha256']))
        for name, expected in plan['sources'].items():
            historical = child(folder / 'source-snapshot', name)
            source = historical if historical.is_file() else child(root, name)
            paths.add(checked(source, expected))
    capacity_plan = read(root / 'output/reacher-geometry-capacity-v1/attempt-02/execution/capacity-plan.json')
    for name, expected in capacity_plan['sources'].items():
        paths.add(checked(child(root, name), expected))
    repair = read(root / 'output/reacher-geometry-capacity-v1/repaired-launch.json')
    require(repair['source_content_changed'] is False and repair['scientific_run_started'] is False,
            'Import repair is not a scientific restart')
    for key in ('preserved_source', 'active_source'):
        value = repair[key]
        require(checked(child(root, value['path']), value['sha256']) in paths, 'Exact repaired source bytes retained')
    paths |= {checked(child(root, name), expected) for name, expected in sources.items()}
    paths.add(root / f'evidence/{STUDY}/protocol/readiness.json')
    return paths


def archive_bundle(root, paths, out, name, scope):
    archive = out / name
    rows = write_archive(root, paths, archive)
    assets, split = split_assets(archive, out)
    for asset in assets:
        asset['bundle'] = name
        asset['scope'] = scope
    return {'name': name, 'scope': scope, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
            'member_count': len(rows), 'members': rows, 'all_members_reopened_and_verified': True,
            'split_for_release': split, 'concatenation_verified': split,
            'ordered_release_assets': [row['name'] for row in assets]}, assets


def package(args):
    require(args.completed_authorized is True, 'Explicit completion and packaging authorization required before reads')
    out = child(ROOT, args.out)
    protected = [ROOT / f'runs/{STUDY}/attempt', ROOT / f'evidence/{STUDY}/audit',
                 ROOT / f'evidence/{STUDY}/protocol', child(ROOT, args.reporting_directory),
                 *(child(ROOT, path) for path in PREPARATION_TREES)]
    require(not out.exists() and not any(out.is_relative_to(path) for path in protected),
            'Exclusive publication output must not be inside input artifact trees')
    out.mkdir(parents=True, exist_ok=False)
    stage = 'authorized-preflight'
    try:
        write(out / 'packaging-started.json', {
            'scope': 'Authorized packaging attempt; scientific inputs not yet authenticated; no new scientific calls or uploads',
            'requested_plan_sha256': args.expected_plan_sha256,
            'requested_readiness_sha256': args.expected_readiness_sha256,
            'requested_audit_receipt_sha256': args.expected_audit_receipt_sha256,
            'requested_reporting_receipt_sha256': args.expected_reporting_receipt_sha256,
            'packager_sha256': sha(Path(__file__)), 'started_utc': datetime.datetime.now(datetime.UTC).isoformat()})
        require(args.expected_plan_sha256 == EXPECTED_PLAN and args.expected_readiness_sha256 == EXPECTED_READINESS,
                'Pinned prospective plan and readiness required')
        audit_sha, report_sha = digest(args.expected_audit_receipt_sha256), digest(args.expected_reporting_receipt_sha256)
        plan, audit, summary, readiness, scientific = validate_scored(ROOT, audit_sha)
        scientific |= validate_reporting(ROOT, child(ROOT, args.reporting_directory), report_sha,
                                         plan, audit, summary, audit_sha)
        preparation = preparation_files(ROOT, readiness, plan['sources'])
        scientific |= {ROOT / name for name in ('LICENSE', 'pyproject.toml', 'uv.lock')}
        scientific |= {Path(__file__).resolve(), ROOT / f'output/{STUDY}/publish_release.py'}
        for declaration in args.reporting_file:
            name, expected = bound_argument(declaration)
            scientific.add(checked(child(ROOT, name), expected))
        stage = 'archive-scored-evidence'
        scored_bundle, scored_assets = archive_bundle(ROOT, scientific, out,
            'reacher-geometry-scored-execution-and-audit.tar.gz', 'completed_scored_study_all_results')
        stage = 'archive-engineering-preparation'
        prep_bundle, prep_assets = archive_bundle(ROOT, preparation, out,
            'reacher-geometry-engineering-preparation-all-attempts.tar.gz', 'engineering_only_no_effectiveness_claim')
        stage = 'verify-manifest-and-sidecars'
        bundles, assets = [scored_bundle, prep_bundle], [*scored_assets, *prep_assets]
        require(len({row['name'] for row in assets}) == len(assets), 'Unique release asset names')
        manifest = {'study': STUDY, 'archives': bundles,
            'historical_dependency_limit': 'Both full local attempts and all copied inherited checkpoint bytes are retained. Older upstream releases remain separate authenticated lineage dependencies.',
            'supervision_limit': 'The scored runner and audit were launched directly. Original phase/terminal receipts are retained; no retrospective pre-run supervision is invented.'}
        write(out / 'manifest.json', manifest)
        # Recheck original bytes after both streamed archive verifications.
        for bundle in bundles:
            for row in bundle['members']:
                path = child(ROOT, row['path'])
                require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Original changed during packaging')
        gate = summary['continuation_gate']
        receipt = {'status': 'verified', 'study': STUDY, 'plan_sha256': EXPECTED_PLAN,
            'readiness_sha256': EXPECTED_READINESS, 'audit_receipt_sha256': audit_sha,
            'reporting_receipt_sha256': report_sha, 'execution_completed_sha256': audit['execution_completed_sha256'],
            'source_sha256': plan['sources'], 'package_source_sha256': sha(Path(__file__)),
            'manifest_sha256': sha(out / 'manifest.json'), 'archives': [
                {key: value for key, value in bundle.items() if key != 'members'} for bundle in bundles],
            'member_count': sum(bundle['member_count'] for bundle in bundles), 'release_assets': assets,
            'raw_coverage': {'inherited_fits': 6, 'new_fits': 0, 'control_rows': 51,
                             'control_cases_per_row': 64, 'diagnostic_roots': 48, 'frozen_source_count': 80},
            'scientific_status': {'continuation_passed': gate['passed'],
                'checks_passed': sum(row['passed'] for row in gate['checks']), 'checks_total': 25,
                'packaging_changes_gate': False},
            'new_model_calls': 0, 'new_native_calls': 0, 'new_optimizer_steps': 0,
            'uploaded': False, 'release_created': False, 'verified_utc': datetime.datetime.now(datetime.UTC).isoformat()}
        write(out / 'receipt.json', receipt)
        (out / 'README.md').write_text(
            '# Verified geometry study assets\n\n'
            'Every scored result is retained whether its scientific continuation gate passes or fails. '
            'The separate engineering archive includes the failed and successful preparation attempts, repair source, '
            'and original terminal receipts; these are not effectiveness results. No upload has occurred.\n\n'
            'For each split archive, concatenate its ordered_release_assets from receipt.json in order. '
            'These are consecutive byte segments, not standalone tar files. Verify the combined archive SHA-256 '
            'and then open the recovered gzip tar. Every archive member was reopened, streamed and checked against '
            'manifest.json. Each release asset is at most1,400,000,000 bytes, below2GB.\n\n'
            'All80 frozen source bytes, six inherited checkpoints,51 complete controller rows,48 diagnostic roots, '
            'independent audit and final report/replay are included. Older upstream lineage dependencies remain '
            'separate releases. Packaging changes no scientific claim.\n')
        sidecars = [out / name for name in ('packaging-started.json', 'manifest.json', 'receipt.json', 'README.md')]
        (out / 'SHA256SUMS').write_text(''.join(f"{sha(path)}  {path.name}\n" for path in
                                              [*sidecars, *(out / row['name'] for row in assets)]))
        return receipt
    except BaseException as error:
        if (out / 'receipt.json').exists():
            (out / 'receipt.json').rename(out / 'incomplete-receipt.json')
        write(out / 'packaging-failed.json', {'status': 'failed', 'stage': stage, 'error': repr(error),
              'scope': 'Packaging only; preserve all partial bytes and original attempts',
              'plan_sha256': args.expected_plan_sha256, 'new_model_calls': 0, 'new_native_calls': 0,
              'uploaded': False, 'retry_permitted': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-authorized', action='store_true')
    parser.add_argument('--expected-plan-sha256', default=EXPECTED_PLAN)
    parser.add_argument('--expected-readiness-sha256', default=EXPECTED_READINESS)
    parser.add_argument('--expected-audit-receipt-sha256', required=True)
    parser.add_argument('--expected-reporting-receipt-sha256', required=True)
    parser.add_argument('--reporting-directory', default=f'evidence/{STUDY}/report')
    parser.add_argument('--reporting-file', action='append', default=[], help='Extra final prose/receipt: relative-path=SHA256')
    parser.add_argument('--out', default=f'output/{STUDY}/publication-v1')
    args = parser.parse_args()
    receipt = package(args)
    print(json.dumps({'status': receipt['status'], 'release_assets': receipt['release_assets'],
                      'receipt_sha256': sha(child(ROOT, args.out) / 'receipt.json')}))


if __name__ == '__main__':
    main()
