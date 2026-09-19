"""Package final, authenticated saved evidence only after explicit completion notice.

This module performs no reads at import time and never imports experiment code.
It does not run an audit, learned model, native replay, training or upload.
Example, only after root confirms completion and the reporting receipt exists:
  python output/reacher-memory-ablation-v1/package_scored.py \
    --completed-authorized \
    --expected-audit-receipt-sha256 FINAL_AUDIT_SHA \
    --expected-reporting-receipt-sha256 FINAL_REPORT_SHA

Large archives are split as consecutive byte segments. Concatenate parts in
receipt order to recover the exact gzip archive before opening it with tar.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'reacher-memory-ablation-v1'
EXPECTED_PLAN = '05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786'
MAX_ASSET_BYTES = 1_900_000_000
PART_BYTES = 900 * 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024
ARMS = ('residual_gru', 'current_gru', 'bounded_gru', 'packet_mlp')
PAIRS = ('pair0', 'pair1', 'pair2')
PANELS = ('full', 'ordinary', 'shift')
REFERENCES = ('known_state', 'particle', 'zero', 'uniform', 'public_kinematic')
ENGINEERING_RECEIPTS = {
    'reacher-memory-capacity-v1': '7dda298707ff70cdc73f7b67dcc29f7cf9714d54f66c862e37efa39f9a532717',
    'reacher-memory-rehearsal-v1': '41c79a817c3e9cfd4b07bbae7f783ad5326b5f284b550391da55a20908d8f1a0',
}


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


def validate_scored(root, expected_plan, expected_audit):
    protocol = root / f'evidence/{STUDY}/protocol'
    plan_path = checked(protocol / 'plan.json', expected_plan)
    plan = read(plan_path)
    freeze = read(protocol / 'freeze.json')
    require(set(files(protocol)) == {'plan.json', 'freeze.json'}, 'Exact frozen protocol membership')
    require(plan['study'] == STUDY and plan['engineering'] is False and len(plan['sources']) == 58,
            'Exact scored study and 58 sources')
    require(freeze['status'] == 'frozen_before_scored_execution'
            and freeze['plan_sha256'] == expected_plan and freeze['sources'] == plan['sources'],
            'Frozen source/plan identity')
    sources = {checked(child(root, name), value) for name, value in plan['sources'].items()}
    execution = root / f'runs/{STUDY}/attempt'
    audit = root / f'evidence/{STUDY}/audit'
    receipt_path = checked(audit / 'receipt.json', expected_audit)
    receipt, completed = read(receipt_path), read(execution / 'completed.json')
    require(receipt['status'] == completed['status'] == 'completed'
            and receipt['engineering'] is False and receipt['saved_output_only'] is True,
            'Completed scored execution and saved-output audit required')
    require(receipt['plan_sha256'] == completed['plan_sha256'] == expected_plan
            and receipt['source_sha256'] == plan['sources'] and receipt['runtime'] == plan['runtime'],
            'Audit/execution/protocol bindings')
    checked(execution / 'completed.json', receipt['execution_completed_sha256'])
    require(receipt['execution_members'] == completed['files'], 'Exact execution manifest equality')
    members = bind_members(execution, completed['files'], ('completed.json',))
    members |= bind_members(audit, receipt['files'], ('receipt.json',))
    summary = read(audit / 'summary.json')
    require(summary['status'] == 'completed' and summary['engineering'] is False
            and summary['plan_sha256'] == expected_plan
            and summary['execution_completed_sha256'] == receipt['execution_completed_sha256']
            and summary['saved_output_only'] is True and summary['costs'] == receipt['costs'],
            'Summary provenance matches completed audit')
    require(summary['new_model_calls'] == summary['new_policy_calls'] == summary['new_fits'] == 0,
            'Audit performs no learned calls or fits')
    expected_fits = {f'{arm}-{pair}' for arm in ARMS for pair in PAIRS}
    expected_rows = expected_fits | set(REFERENCES)
    require(set(summary['fits']) == set(summary['prediction']) == expected_fits
            and set(summary['control']) == set(PANELS), 'All 12 fits and prediction rows')
    for panel in PANELS:
        require(set(summary['control'][panel]) == expected_rows, 'All 17 rows per panel')
        for row in summary['control'][panel].values():
            require(len(row['episode_costs']) == 64, 'All 64 control cases retained')
    coverage = summary['coverage']
    require(completed['fits'] == coverage['fits'] == 12
            and completed['control_rows'] == coverage['control_rows'] == 51
            and completed['prediction_episodes'] == plan['prediction_episodes'] == 96
            and coverage['learned_control_rows'] == 36 and coverage['reference_rows'] == 15
            and coverage['control_cases_per_row'] == plan['control_episodes'] == 64
            and plan['steps'] == 50 and coverage['optimizer_updates_per_fit'] == 1152
            and coverage['total_optimizer_updates'] == 13824 and completed['astra_calls'] == 0,
            'Exact full-size scored coverage')
    require(summary['native_transitions_checked'] == 768*50 + 96*50 + 51*64*50 == 206400,
            'Complete native replay coverage')
    require(summary['phase_boundary']['all_twelve_fits_restored_before_evaluation'] is True,
            'Actual-class restoration before evaluation')
    checks = summary['continuation_gate']['checks']
    require(len(checks) == 25 and all(type(row['passed']) is bool for row in checks)
            and type(summary['continuation_gate']['passed']) is bool
            and summary['continuation_gate']['passed'] == all(row['passed'] for row in checks),
            'Complete gate retained regardless of pass or failure')
    require(positive(completed['wall_seconds'], 'Execution wall') <= plan['cap_seconds'] == 3600
            and positive(summary['costs']['audit_validation_wall_seconds'], 'Audit wall')
            <= plan['audit_cap_seconds'] == 300, 'Frozen caps respected')
    supervision = root / f'output/{STUDY}/supervision'
    require(set(files(supervision)) == {'started.json', 'completed.json', 'execution.log', 'audit.log'},
            'Exact closed supervision files, no failed attempt')
    started, terminal = read(supervision / 'started.json'), read(supervision / 'completed.json')
    launcher = checked(root / f'output/{STUDY}/launch.py', started['launcher_sha256'])
    require(started['plan_sha256'] == terminal['plan_sha256'] == expected_plan
            and started['retries'] == 0 and terminal['status'] == 'completed'
            and started['execution'] == execution.relative_to(root).as_posix()
            and started['audit'] == audit.relative_to(root).as_posix()
            and started['frozen_execution_cap_seconds'] == plan['cap_seconds']
            and started['frozen_audit_cap_seconds'] == plan['audit_cap_seconds']
            and terminal['audit_receipt_sha256'] == expected_audit,
            'Closed supervision binds the two successful subprocesses')
    require(terminal['continuation_passed'] == summary['continuation_gate']['passed']
            and terminal['passed_checks'] == sum(row['passed'] for row in checks)
            and terminal['native_transitions_checked'] == summary['native_transitions_checked'],
            'Supervisor preserves audited gate')
    positive(terminal['whole_supervision_seconds'], 'Supervisor wall')
    members |= sources | set(files(protocol).values()) | set(files(supervision).values()) | {launcher}
    return plan, receipt, summary, members


def validate_reporting(root, folder, expected, plan_sha, audit_sha, audit):
    receipt_path = checked(folder / 'receipt.json', expected)
    receipt = read(receipt_path)
    require(receipt['status'] == 'completed' and receipt['engineering'] is False
            and receipt['scope'] == 'completed_scored_development_study',
            'Final scored reporting scope')
    require(receipt['inputs'] == {
        'plan_sha256': plan_sha, 'audit_receipt_sha256': audit_sha,
        'audit_summary_sha256': audit['files']['summary.json'],
        'execution_completed_sha256': audit['execution_completed_sha256'],
        'execution_member_count': len(audit['execution_members']), 'frozen_source_count': 58,
    }, 'Reporting binds exact final plan, audit, summary and execution')
    require(receipt['new_model_calls'] == receipt['new_simulator_calls'] == 0,
            'Reporting performs no new model/simulator calls')
    positive(receipt['wall_seconds'], 'Reporting wall')
    require(set(receipt['files']) == {'report.json', 'tables.md', 'native-costs.png', 'utility-vs-cost.png'},
            'Complete final report, tables and figure')
    members = bind_members(folder, receipt['files'], ('receipt.json',))
    members.add(checked(root / f'output/{STUDY}/report_results.py', receipt['source_sha256']))
    return members


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
            require(member.size == row['bytes'], 'Archive member length')
            with tar.extractfile(member) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256'],
                        'Archive member bytes')
            seen.add(member.name)
    require(seen == set(expected), 'Complete archive membership')


def engineering_assets(root):
    result, source_files = [], set()
    for name, expected_receipt in ENGINEERING_RECEIPTS.items():
        folder = root / 'evidence' / name
        receipt_path = checked(folder / 'receipt.json', expected_receipt)
        receipt = read(receipt_path)
        require(receipt['status'] == 'completed', 'Completed engineering package')
        source_files |= bind_members(folder, receipt['files'], ('receipt.json', 'SHA256SUMS'))
        archive_info = receipt['archive']
        archive = checked(child(root, archive_info['path']), archive_info['sha256'])
        require(archive.stat().st_size == archive_info['bytes'] < MAX_ASSET_BYTES,
                'Small engineering asset remains separate')
        manifest = read(folder / 'bundle-manifest.json')['members']
        rows = [{'path': path, **value} for path, value in sorted(manifest.items())]
        verify_archive(archive, rows)
        result.append({'scope': 'engineering_only_not_scientific_result', 'source': archive,
                       'name': archive.name, 'bytes': archive_info['bytes'], 'sha256': archive_info['sha256'],
                       'receipt_sha256': expected_receipt, 'members_verified': len(rows)})
    return result, source_files


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


def package(args):
    require(args.completed_authorized, 'Explicit root completion authorization required before reading outcomes')
    root = ROOT
    require(args.expected_plan_sha256 == EXPECTED_PLAN, 'Exact frozen scored plan required')
    expected_audit = digest(args.expected_audit_receipt_sha256)
    expected_reporting = digest(args.expected_reporting_receipt_sha256)
    plan, audit, summary, paths = validate_scored(root, args.expected_plan_sha256, expected_audit)
    figures = child(root, args.reporting_directory)
    paths |= validate_reporting(root, figures, expected_reporting, args.expected_plan_sha256, expected_audit, audit)
    engineering, evidence_paths = engineering_assets(root)
    paths |= evidence_paths
    paths |= {root / 'LICENSE', root / 'pyproject.toml', root / 'uv.lock', Path(__file__).resolve()}
    for name in args.reporting_file:
        path = child(root, name)
        require(path.is_file(), 'Additional final report file required')
        paths.add(path)
    out = child(root, args.out)
    require(out.is_relative_to(root) and not out.exists(), 'Exclusive repository-contained publication directory')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'packaging-started.json', {
        'scope': 'Final saved-byte release packaging; no new scientific calls or uploads',
        'plan_sha256': args.expected_plan_sha256, 'audit_receipt_sha256': expected_audit,
        'reporting_receipt_sha256': expected_reporting, 'packager_sha256': sha(Path(__file__)),
        'started_at_utc': datetime.datetime.now(datetime.UTC).isoformat()})
    try:
        archive = out / 'reacher-memory-scored-execution-and-audit.tar.gz'
        rows = write_archive(root, paths, archive)
        manifest = {'scope': 'Complete scored attempt, saved audit, 58 frozen sources, protocol/freeze, closed supervision, final reporting and engineering evidence sidecars.',
                    'historical_dependency_limit': 'External upstream artifacts remain separate release dependencies; their digests are retained in protocol and audit.',
                    'members': rows, 'files': len(rows), 'bytes': sum(row['bytes'] for row in rows)}
        write(out / 'manifest.json', manifest)
        assets, split = split_assets(archive, out)
        for item in engineering:
            destination = out / item['name']
            require(not destination.exists(), 'Unique release asset filename')
            shutil.copyfile(item['source'], destination)
            require(sha(destination) == item['sha256'], 'Engineering asset byte copy')
            assets.append({key: value for key, value in item.items() if key != 'source'})
        require(all(asset['bytes'] < 2_000_000_000 for asset in assets), 'Every release asset under 2 GB')
        # Final byte rechecks, not re-execution of any scientific validator.
        for row in rows:
            require(sha(root / row['path']) == row['sha256'], 'Archived input changed during packaging')
        gate = summary['continuation_gate']
        result = {'status': 'verified', 'study': STUDY, 'plan_sha256': args.expected_plan_sha256,
                  'audit_receipt_sha256': expected_audit, 'reporting_receipt_sha256': expected_reporting,
                  'execution_completed_sha256': audit['execution_completed_sha256'],
                  'source_sha256': plan['sources'], 'package_source_sha256': sha(Path(__file__)),
                  'manifest_sha256': sha(out / 'manifest.json'), 'member_count': len(rows),
                  'archive': {'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
                              'all_members_reopened_and_verified': True, 'split_for_release': split,
                              'concatenation_verified': split},
                  'release_assets': assets, 'raw_coverage': {'fits': 12, 'control_rows': 51, 'control_cases_per_row': 64},
                  'scientific_status': {'continuation_passed': gate['passed'],
                                        'checks_passed': sum(row['passed'] for row in gate['checks']),
                                        'checks_total': 25, 'packaging_changes_gate': False},
                  'new_model_calls': 0, 'new_native_calls': 0, 'new_optimizer_steps': 0,
                  'uploaded': False, 'release_created': False,
                  'verified_at_utc': datetime.datetime.now(datetime.UTC).isoformat()}
        write(out / 'receipt.json', result)
        description = '# Verified release assets\n\n'
        description += 'This package preserves the completed scored study and its original scientific gate unchanged. Engineering archives are separate assets and are not scientific results. No upload or release creation has occurred.\n\n'
        if split:
            description += 'The `.partNNN` files are consecutive byte segments, not individually readable tar archives. Concatenate them in the exact `release_assets` order for the scored archive, verify its SHA-256 from `receipt.json`, then open the recovered gzip tar archive.\n\n'
        description += 'Archive, part, engineering-asset and manifest hashes are recorded in `receipt.json`. `manifest.json` binds every original member. Upstream historical lineage artifacts may still be required to rerun the saved-output audit.\n'
        (out / 'README.md').write_text(description)
        sidecars = [out / name for name in ('packaging-started.json', 'manifest.json', 'receipt.json', 'README.md')]
        asset_paths = [out / row['name'] for row in assets]
        (out / 'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.name}\n' for path in [*sidecars, *asset_paths]))
        return result
    except BaseException as error:
        if (out / 'receipt.json').exists():
            (out / 'receipt.json').rename(out / 'incomplete-receipt.json')
        write(out / 'packaging-failed.json', {'status': 'failed', 'error': repr(error),
              'scope': 'Packaging only; preserve partial package and original scientific attempt',
              'plan_sha256': args.expected_plan_sha256})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-authorized', action='store_true', help='Root has explicitly confirmed completed execution/audit and authorized packaging.')
    parser.add_argument('--expected-plan-sha256', default=EXPECTED_PLAN)
    parser.add_argument('--expected-audit-receipt-sha256', required=True)
    parser.add_argument('--expected-reporting-receipt-sha256', required=True)
    parser.add_argument('--reporting-directory', default=f'evidence/{STUDY}/figures')
    parser.add_argument('--reporting-file', action='append', default=[], help='Additional final repository-relative prose/report, preserved by hash.')
    parser.add_argument('--out', default=f'output/{STUDY}/publication-v1')
    args = parser.parse_args()
    print(json.dumps(package(args), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
