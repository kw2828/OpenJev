"""Publish one completed byte-verified package through the existing gh account.

No import-time reads or API calls. A draft stays unpublished until every remote
asset reports the expected upload state, size and SHA-256. No retries/clobbers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/reacher-geometry-score-v1'
REPO = 'kw2828/OpenJev'
TAG = 'research-reacher-geometry-score-v1'
SIDECARS = ('packaging-started.json', 'manifest.json', 'receipt.json', 'README.md', 'SHA256SUMS')


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
    require(release['tag_name'] == TAG and release['target_commitish'] == commit, 'Release target identity')
    pages = api(f"repos/{REPO}/releases/{release['id']}/assets?per_page=100", '--paginate', '--slurp')
    assets = [row for page in pages for row in page]
    rows = {row['name']: row for row in assets}
    require(len(rows) == len(assets) and set(rows) == set(expected), 'Exact remote asset membership')
    for name, row in rows.items():
        require(row['state'] == 'uploaded' and row['size'] == expected[name]['bytes']
                and row.get('digest') == 'sha256:' + expected[name]['sha256'],
                'Remote upload state/size/hash mismatch: ' + name)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-authorized', action='store_true')
    parser.add_argument('--package', type=Path, default=BASE / 'publication-v1')
    parser.add_argument('--expected-package-receipt-sha256', required=True)
    parser.add_argument('--expected-audit-receipt-sha256', required=True)
    parser.add_argument('--target-commit', required=True)
    parser.add_argument('--release-notes', type=Path, required=True)
    parser.add_argument('--expected-release-notes-sha256', required=True)
    parser.add_argument('--title', default='OpenJev: frozen-model Reacher geometry-scoring study')
    parser.add_argument('--out', type=Path, default=ROOT / 'evidence/reacher-geometry-score-v1/scored-publication')
    args = parser.parse_args()
    require(args.publish_authorized, 'Explicit final publication authorization required before reads/API calls')
    package_path, out = args.package.resolve(), args.out.resolve()
    require(package_path.is_relative_to(ROOT) and out.is_relative_to(ROOT)
            and not out.is_relative_to(package_path) and not out.exists(), 'Exclusive separate publication receipt path')
    out.mkdir(parents=True, exist_ok=False)
    release_id, stage = None, 'authorized-local-preflight'
    try:
        write(out / 'publication-started.json', {
            'scope': 'Authorized publication attempt; package not yet authenticated and no remote mutation',
            'package_receipt_sha256': args.expected_package_receipt_sha256,
            'audit_receipt_sha256': args.expected_audit_receipt_sha256,
            'target_commit': args.target_commit, 'publisher_sha256': sha(Path(__file__)),
            'requested_release_notes_sha256': args.expected_release_notes_sha256,
            'started_utc': datetime.now(UTC).isoformat(), 'retries': 0})
        require(len(args.target_commit) == 40 and set(args.target_commit) <= set('0123456789abcdef'), 'Exact full commit SHA')
        # Reuse only the new packager's strict JSON/path/hash helpers, never research code.
        module_path = BASE / 'package.py'
        spec = importlib.util.spec_from_file_location('geometry_publication_bytes', module_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        package = helper.read(helper.checked(package_path / 'receipt.json', args.expected_package_receipt_sha256))
        require(package['status'] == 'verified' and package['study'] == helper.STUDY
                and package['plan_sha256'] == helper.EXPECTED_PLAN
                and package['readiness_sha256'] == helper.EXPECTED_READINESS
                and package['audit_receipt_sha256'] == helper.digest(args.expected_audit_receipt_sha256)
                and package['package_source_sha256'] == sha(module_path)
                and package['new_model_calls'] == package['new_native_calls'] == package['new_optimizer_steps'] == 0
                and package['uploaded'] is False and package['release_created'] is False, 'Verified unpublished exact package')
        manifest = helper.read(helper.checked(package_path / 'manifest.json', package['manifest_sha256']))
        require(manifest['study'] == helper.STUDY and len(manifest['archives']) == len(package['archives']) == 2,
                'Both scientific and complete preparation archives')
        require([{k: v for k, v in row.items() if k != 'members'} for row in manifest['archives']]
                == package['archives'], 'Manifest/archive identity binding')
        publisher_members = [row for archive in manifest['archives'] for row in archive['members']
                             if row['path'] == Path(__file__).resolve().relative_to(ROOT).as_posix()]
        require(len(publisher_members) == 1 and publisher_members[0]['sha256'] == sha(Path(__file__)),
                'Current publication code must be the exact packaged source')
        scientific = package['scientific_status']
        require(scientific['checks_total'] == 25 and type(scientific['checks_passed']) is int
                and 0 <= scientific['checks_passed'] <= 25 and type(scientific['continuation_passed']) is bool
                and scientific['continuation_passed'] == (scientific['checks_passed'] == 25)
                and scientific['packaging_changes_gate'] is False, 'No publication change to scientific gate')
        expected = {row['name']: row for row in package['release_assets']}
        require(len(expected) == len(package['release_assets']), 'Unique release assets')
        for name in SIDECARS:
            require(name not in expected, 'Sidecar name collision')
            path = helper.child(package_path, name)
            expected[name] = {'name': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
        for name, row in expected.items():
            require(Path(name).name == name and 0 < row['bytes'] < 2_000_000_000, 'Safe bounded asset name/size')
            path = helper.checked(helper.child(package_path, name), row['sha256'])
            require(path.stat().st_size == row['bytes'], 'Asset size changed before upload')
        sums = {}
        for line in (package_path / 'SHA256SUMS').read_text().splitlines():
            value, name = line.split('  ', 1)
            require(name not in sums, 'Duplicate SHA256SUMS entry')
            sums[name] = helper.digest(value)
        require(set(sums) == set(expected) - {'SHA256SUMS'}
                and all(value == expected[name]['sha256'] for name, value in sums.items()), 'Exact asset/sidecar checksum list')
        notes = helper.checked(args.release_notes.resolve(), args.expected_release_notes_sha256)
        stage = 'remote-read-only-preflight'
        require(api('user')['login'] == 'kw2828', 'Use the intended authenticated GitHub account')
        require(api(f'repos/{REPO}/commits/{args.target_commit}')['sha'] == args.target_commit, 'Remote commit must exist')
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
                        'name': args.title, 'body': notes.read_text(), 'make_latest': 'false'})
        created = api('--method', 'POST', f'repos/{REPO}/releases', '--input', str(request))
        write(out / 'created.json', created)
        release_id = created['id']
        require(created['draft'] is True and created['tag_name'] == TAG, 'New draft required')
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
        write(out / 'draft-verified.json', draft)
        stage = 'publish'
        request = out / 'publish-request.json'
        write(request, {'draft': False, 'make_latest': 'false'})
        api('--method', 'PATCH', endpoint, '--input', str(request))
        stage = 'verify-public'
        final = api(endpoint)
        require(final['draft'] is False and final['published_at'], 'Public completed release required')
        rows = verify(final, expected, args.target_commit)
        require(tag_target() == args.target_commit, 'Actual release tag must resolve to the exact commit')
        downloaded = []
        with tempfile.TemporaryDirectory(prefix='openjev-geometry-public-sidecars-') as temporary:
            for name in ('receipt.json', 'manifest.json', 'SHA256SUMS'):
                path = Path(temporary) / name
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
            'publisher_sha256': sha(Path(__file__)), 'new_model_calls': 0, 'new_native_calls': 0,
            'assets': [{**expected[name], 'asset_id': rows[name]['id'],
                        'browser_download_url': rows[name]['browser_download_url']} for name in expected],
            'public_sidecar_downloads': downloaded,
            'verification_scope': 'Every remote asset upload state, size and SHA-256 checked. Three public sidecars downloaded and hashed. Large parts verified through GitHub digests, not downloaded again; local archives were streamed and reopened by packaging.'}
        write(out / 'release-verification.json', result)
        print(json.dumps({'status': result['status'], 'url': result['release_url'], 'assets': len(rows)}), flush=True)
    except BaseException as error:
        write(out / 'publication-failed.json', {'status': 'failed', 'stage': stage, 'release_id': release_id,
              'error': repr(error), 'recorded_utc': datetime.now(UTC).isoformat(),
              'scope': 'Publication only. Preserve current remote draft/public state, all local receipts and scientific evidence. No automatic retry.'})
        raise


if __name__ == '__main__':
    main()
