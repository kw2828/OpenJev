"""Publish one completed byte-verified package through the existing gh account.

No import-time reads or API calls. A draft stays unpublished until every remote
asset reports the expected upload state, size and SHA-256. No retries/clobbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/reacher-geometry-memory-v1'
PACKAGE_SHA = '632afd1eeac40d651d54166285308ddcb5f6dbf10983214cf26cdcd41dbd50c7'
PLAN_SHA = '23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee'
STUDY = 'reacher-geometry-memory-v1'
REPO = 'kw2828/OpenJev'
TAG = 'research-reacher-geometry-memory-v1'
SIDECARS = ('packaging-started.json', 'manifest.json', 'receipt.json', 'README.md', 'SHA256SUMS')
ARCHIVES = ('reacher-geometry-memory-scored-execution-and-audit.tar.gz',
            'reacher-geometry-memory-engineering-preparation-all-attempts.tar.gz')
RAW_COVERAGE = {'execution_manifest_members': 9505, 'inherited_fits': 12, 'new_fits': 0,
    'control_rows': 51, 'control_cases_per_row': 64, 'diagnostic_roots': 0, 'frozen_source_count': 90,
    'native_control_transitions_checked': 163200, 'native_nominal_candidate_transitions_checked': 78741504,
    'native_nominal_selected_transitions_checked': 28800, 'public_observer_transitions_checked': 310464}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def api(*args):
    result = subprocess.run(['gh', 'api', *args], capture_output=True, text=True, check=True, timeout=120)
    return json.loads(result.stdout)


def load_package():
    path = BASE / 'package.py'
    require(path.is_file() and not any(item.is_symlink() for item in (path, *path.parents)),
            'Exact reviewed package source required')
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == PACKAGE_SHA, 'Exact reviewed package source hash required')
    module = ModuleType('geometry_memory_publication_bytes')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)  # noqa: S102 - exact SHA-pinned reviewed source
    require(module.STUDY == STUDY and module.EXPECTED_PLAN == PLAN_SHA, 'Pinned package identity')
    return module


def authenticate_package(args):
    """Authenticate a completed package without reopening large archive payloads.

    The externally pinned packaging receipt records the prior streamed archive
    verification. Every release asset is hashed here before any network action.
    """
    helper = load_package()
    folder = args.package.resolve()
    package = helper.read(helper.checked(folder / 'receipt.json', args.expected_package_receipt_sha256))
    require(package['status'] == 'verified' and package['study'] == STUDY
            and package['plan_sha256'] == PLAN_SHA and package['readiness_sha256'] == helper.EXPECTED_READINESS
            and package['audit_receipt_sha256'] == helper.digest(args.expected_audit_receipt_sha256)
            and package['package_source_sha256'] == PACKAGE_SHA
            and package['new_model_calls'] == package['new_native_calls'] == package['new_optimizer_steps'] == 0
            and package['uploaded'] is False and package['release_created'] is False,
            'Verified unpublished exact package')
    for name in ('reporting_receipt_sha256', 'terminal_verification_sha256',
                 'previous_release_verification_sha256', 'execution_completed_sha256'):
        helper.digest(package[name])
    require(package['raw_coverage'] == RAW_COVERAGE and len(package['source_sha256']) == 90,
            'Complete twelve-checkpoint/51-row/native-work package coverage')
    for name, expected in package['source_sha256'].items():
        helper.checked(helper.child(ROOT, name), expected)
    manifest = helper.read(helper.checked(folder / 'manifest.json', package['manifest_sha256']))
    require(manifest['study'] == STUDY
            and [row['name'] for row in manifest['archives']] == list(ARCHIVES)
            and manifest['previous_release_dependency'] == package['previous_release_dependency'],
            'Both exact archives and historical dependency retained')
    require([{k: v for k, v in row.items() if k != 'members'} for row in manifest['archives']]
            == package['archives'], 'Manifest/archive identity binding')
    archived = [row for archive in manifest['archives'] for row in archive['members']]
    require(sum(archive['member_count'] for archive in manifest['archives']) == package['member_count']
            == len(archived), 'Complete archive member counts')
    for source in (Path(__file__).resolve(), BASE / 'package.py'):
        matches = [row for row in archived if row['path'] == source.relative_to(ROOT).as_posix()]
        require(len(matches) == 1 and matches[0]['sha256'] == sha(source)
                and matches[0]['bytes'] == source.stat().st_size, 'Exact current publication sources must be packaged')
    scientific = package['scientific_status']
    require(set(scientific) == {'checks_total', 'checks_passed', 'continuation_passed', 'packaging_changes_gate'}
            and scientific['checks_total'] == 25 and type(scientific['checks_passed']) is int
            and 0 <= scientific['checks_passed'] <= 25 and type(scientific['continuation_passed']) is bool
            and scientific['continuation_passed'] == (scientific['checks_passed'] == 25)
            and scientific['packaging_changes_gate'] is False, 'No publication change to scientific gate')
    assets = package['release_assets']
    expected = {row['name']: row for row in assets}
    require(len(expected) == len(assets), 'Unique release assets')
    require([name for archive in package['archives'] for name in archive['ordered_release_assets']]
            == [row['name'] for row in assets], 'Exact ordered archive asset membership')
    for archive in package['archives']:
        require(archive['all_members_reopened_and_verified'] is True
                and type(archive['split_for_release']) is bool
                and archive['concatenation_verified'] is archive['split_for_release'],
                'Previously streamed verification and concatenation required')
        selected = [expected[name] for name in archive['ordered_release_assets']]
        require(selected and sum(row['bytes'] for row in selected) == archive['bytes']
                and all(row['bundle'] == archive['name'] and row['scope'] == archive['scope'] for row in selected),
                'Complete archive byte and scope accounting')
        helper.digest(archive['sha256'])
        require((len(selected) > 1) is archive['split_for_release'], 'Explicit split archive membership')
        if not archive['split_for_release']:
            require(selected[0]['name'] == archive['name'] and selected[0]['sha256'] == archive['sha256'],
                    'Unsplit archive identity')
        else:
            require([row['name'] for row in selected] ==
                    [f"{archive['name']}.part{index:03d}" for index in range(len(selected))], 'Consecutive split parts')
    for name in SIDECARS:
        require(name not in expected, 'Sidecar name collision')
        path = helper.child(folder, name)
        expected[name] = {'name': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
    require(set(helper.files(folder)) == set(expected) | set(ARCHIVES),
            'Exact completed package membership, without failed or partial attempts')
    for name, row in expected.items():
        require(Path(name).name == name and type(row['bytes']) is int and 0 < row['bytes'] <= helper.MAX_ASSET_BYTES,
                'Safe bounded asset name/size')
        path = helper.checked(helper.child(folder, name), row['sha256'])
        require(path.stat().st_size == row['bytes'], 'Asset size changed before upload')
    sums = {}
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        value, name = line.split('  ', 1)
        require(name not in sums, 'Duplicate SHA256SUMS entry')
        sums[name] = helper.digest(value)
    require(set(sums) == set(expected) - {'SHA256SUMS'}
            and all(value == expected[name]['sha256'] for name, value in sums.items()), 'Exact asset/sidecar checksum list')
    started = helper.read(folder / 'packaging-started.json')
    require(started['requested_audit_receipt_sha256'] == package['audit_receipt_sha256']
            and started['package_source_sha256'] == PACKAGE_SHA, 'Packaging attempt identity')
    return helper, package, expected


def verify_committed_helpers(commit):
    for path in (Path(__file__).resolve(), BASE / 'package.py'):
        name = path.relative_to(ROOT).as_posix()
        result = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=ROOT,
                                capture_output=True, check=True, timeout=120)
        require(hashlib.sha256(result.stdout).hexdigest() == sha(path), 'Pushed target must contain exact publication sources')


def tag_target():
    values = api(f'repos/{REPO}/git/matching-refs/tags/{TAG}')
    values = [item for item in values if item['ref'] == 'refs/tags/' + TAG]
    require(len(values) <= 1, 'Unique release tag')
    if not values:
        return None
    value = values[0]['object']
    for _ in range(8):
        if value['type'] == 'commit':
            return value['sha']
        require(value['type'] == 'tag', 'Tag must resolve to a commit')
        value = api(f"repos/{REPO}/git/tags/{value['sha']}")['object']
    raise ValueError('Excessive annotated tag nesting')


def release_absent():
    # A transient auth/network failure is not interpreted as absence.
    result = subprocess.run(['gh', 'api', f'repos/{REPO}/releases/tags/{TAG}'],
                            capture_output=True, text=True, timeout=120, check=False)
    require(result.returncode != 0, 'Release already exists; do not retry or overwrite')
    try:
        response = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError('Release absence was not established') from error
    require(str(response.get('status')) == '404' and response.get('message') == 'Not Found',
            'Only explicit authenticated404 establishes no existing release')


def verify(release, expected, commit):
    require(release['tag_name'] == TAG and release['target_commitish'] == commit
            and release.get('prerelease') is False, 'Release target identity')
    pages = api(f"repos/{REPO}/releases/{release['id']}/assets?per_page=100", '--paginate', '--slurp')
    assets = [row for page in pages for row in page]
    rows = {row['name']: row for row in assets}
    require(len(rows) == len(assets) and set(rows) == set(expected), 'Exact remote asset membership')
    for name, row in rows.items():
        require(row['state'] == 'uploaded' and row['size'] == expected[name]['bytes']
                and row.get('digest') == 'sha256:' + expected[name]['sha256'],
                'Remote upload state/size/hash mismatch: ' + name)
    return rows


def publish(args):
    require(args.publish_authorized, 'Explicit final publication authorization required before reads/API calls')
    require(not any(path.is_symlink() for value in (args.package, args.out, args.release_notes)
                    for path in (Path(value), *Path(value).parents)), 'No publication input/output symlink paths')
    package_path, out = args.package.resolve(), args.out.resolve()
    require(package_path.is_relative_to(ROOT) and out.is_relative_to(ROOT)
            and not out.is_relative_to(package_path) and not package_path.is_relative_to(out)
            and not out.is_relative_to(ROOT / 'runs') and not out.exists(), 'Exclusive separate publication receipt path')
    out.mkdir(parents=True, exist_ok=False)
    release_id, stage = None, 'authorized-local-preflight'
    try:
        publisher_sha = sha(Path(__file__))
        write(out / 'publication-started.json', {
            'scope': 'Authorized publication attempt; package not yet authenticated and no remote mutation',
            'package_receipt_sha256': args.expected_package_receipt_sha256,
            'audit_receipt_sha256': args.expected_audit_receipt_sha256,
            'target_commit': args.target_commit, 'publisher_sha256': publisher_sha,
            'requested_release_notes_sha256': args.expected_release_notes_sha256,
            'started_utc': datetime.now(UTC).isoformat(), 'retries': 0})
        require(len(args.target_commit) == 40 and set(args.target_commit) <= set('0123456789abcdef'), 'Exact full commit SHA')
        helper, package, expected = authenticate_package(args)
        scientific = package['scientific_status']
        verify_committed_helpers(args.target_commit)
        notes = helper.checked(args.release_notes.resolve(), args.expected_release_notes_sha256)
        stage = 'remote-read-only-preflight'
        require(api('user')['login'] == 'kw2828', 'Use the intended authenticated GitHub account')
        require(api(f'repos/{REPO}/commits/{args.target_commit}')['sha'] == args.target_commit, 'Remote commit must exist')
        require(api(f'repos/{REPO}/git/ref/heads/main')['object']['sha'] == args.target_commit,
                'Explicit target must be the exact pushed main commit')
        require(tag_target() in (None, args.target_commit), 'Existing tag targets another commit')
        release_absent()
        stage = 'copy-verified-sidecars'
        for name in SIDECARS:
            shutil.copyfile(package_path / name, out / name)
        write(out / 'publication-preflight.json', {'package_receipt_sha256': args.expected_package_receipt_sha256,
            'audit_receipt_sha256': args.expected_audit_receipt_sha256, 'target_commit': args.target_commit,
            'publisher_sha256': sha(Path(__file__)), 'release_notes_sha256': sha(notes),
            'assets': list(expected.values()), 'verified_utc': datetime.now(UTC).isoformat(), 'remote_mutations': 0})
        stage = 'create-draft'
        request = out / 'create-request.json'
        write(request, {'tag_name': TAG, 'target_commitish': args.target_commit, 'draft': True,
                        'prerelease': False, 'name': args.title, 'body': notes.read_text(), 'make_latest': 'false'})
        created = api('--method', 'POST', f'repos/{REPO}/releases', '--input', str(request))
        write(out / 'created.json', created)
        release_id = created['id']
        require(created['draft'] is True and created['tag_name'] == TAG
                and created['target_commitish'] == args.target_commit and created['prerelease'] is False,
                'New exact-target draft required')
        stage = 'upload-once'
        argv = ['gh', 'release', 'upload', TAG, *[str(package_path / name) for name in expected], '--repo', REPO]
        write(out / 'upload-command.json', {'argv': argv, 'replacement_permitted': False, 'retries': 0})
        subprocess.run(argv, check=True, timeout=7200)
        endpoint = f'repos/{REPO}/releases/{release_id}'
        stage = 'verify-draft'
        draft = api(endpoint)
        require(draft['draft'] is True, 'Keep release private until every upload is verified')
        verify(draft, expected, args.target_commit)
        require(tag_target() in (None, args.target_commit), 'Tag changed during upload')
        require(draft['name'] == args.title and draft['body'] == notes.read_text(), 'Draft public wording unchanged')
        helper.checked(package_path / 'receipt.json', args.expected_package_receipt_sha256)
        helper.checked(notes, args.expected_release_notes_sha256)
        require(sha(Path(__file__)) == publisher_sha and sha(BASE / 'package.py') == PACKAGE_SHA,
                'Publication helpers unchanged before public transition')
        write(out / 'draft-verified.json', draft)
        stage = 'publish'
        request = out / 'publish-request.json'
        write(request, {'draft': False, 'make_latest': 'false'})
        api('--method', 'PATCH', endpoint, '--input', str(request))
        stage = 'verify-public'
        final = api(endpoint)
        require(final['draft'] is False and final['published_at'] and final['name'] == args.title
                and final['body'] == notes.read_text(), 'Public completed release and wording required')
        rows = verify(final, expected, args.target_commit)
        require(tag_target() == args.target_commit, 'Actual release tag must resolve to the exact commit')
        downloaded = []
        with tempfile.TemporaryDirectory(prefix='openjev-geometry-memory-public-sidecars-') as temporary:
            for name in ('receipt.json', 'manifest.json', 'SHA256SUMS'):
                path = Path(temporary) / name
                require(rows[name]['browser_download_url'] ==
                        f'https://github.com/{REPO}/releases/download/{TAG}/{name}', 'Exact public sidecar location')
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '120',
                                '--output', str(path), rows[name]['browser_download_url']], check=True, timeout=130)
                require(path.stat().st_size == expected[name]['bytes'] and sha(path) == expected[name]['sha256'],
                        'Public sidecar bytes differ from uploaded identity')
                downloaded.append({'name': name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        result = {'status': 'published_and_verified', 'repository': REPO, 'release_tag': TAG,
            'release_id': release_id, 'release_url': final['html_url'], 'target_commit': args.target_commit,
            'resolved_tag_commit': args.target_commit, 'published_at': final['published_at'],
            'verified_utc': datetime.now(UTC).isoformat(), 'package_receipt_sha256': args.expected_package_receipt_sha256,
            'audit_receipt_sha256': args.expected_audit_receipt_sha256, 'scientific_status': scientific,
            'publisher_sha256': publisher_sha, 'package_source_sha256': PACKAGE_SHA,
            'plan_sha256': PLAN_SHA, 'execution_completed_sha256': package['execution_completed_sha256'],
            'terminal_verification_sha256': package['terminal_verification_sha256'],
            'new_model_calls': 0, 'new_native_calls': 0, 'new_optimizer_steps': 0,
            'assets': [{**expected[name], 'asset_id': rows[name]['id'],
                        'browser_download_url': rows[name]['browser_download_url']} for name in expected],
            'public_sidecar_downloads': downloaded,
            'verification_scope': 'Every remote asset upload state, size and SHA-256 checked. Three public sidecars downloaded and hashed. Large parts verified through GitHub digests, not downloaded again; local archives were streamed and reopened by packaging.'}
        write(out / 'release-verification.json', result)
        print(json.dumps({'status': result['status'], 'url': result['release_url'], 'assets': len(rows)}), flush=True)
        return result
    except BaseException as error:
        try:
            write(out / 'publication-failed.json', {'status': 'failed', 'stage': stage, 'release_id': release_id,
                  'error': repr(error), 'recorded_utc': datetime.now(UTC).isoformat(),
                  'scope': 'Publication only. Preserve current remote draft/public state, all local receipts and scientific evidence. No automatic retry.'})
        except BaseException as preservation_error:  # noqa: BLE001 - preserve the original publication failure
            error.add_note(f'Failure receipt could not be preserved: {preservation_error!r}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-authorized', action='store_true')
    parser.add_argument('--package', type=Path, default=BASE / 'publication-v1')
    parser.add_argument('--expected-package-receipt-sha256', required=True)
    parser.add_argument('--expected-audit-receipt-sha256', required=True)
    parser.add_argument('--target-commit', required=True)
    parser.add_argument('--release-notes', type=Path, required=True)
    parser.add_argument('--expected-release-notes-sha256', required=True)
    parser.add_argument('--title', default='OpenJev: geometry-scored Reacher memory comparison')
    parser.add_argument('--out', type=Path, default=ROOT / 'evidence/reacher-geometry-memory-v1/scored-publication')
    args = parser.parse_args()
    publish(args)


if __name__ == '__main__':
    main()
