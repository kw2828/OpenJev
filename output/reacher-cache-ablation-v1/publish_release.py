"""Create one draft, upload verified saved assets, then publish after byte checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/reacher-cache-ablation-v1'
PACKAGE = BASE / 'publication-v1'
OUT = ROOT / 'evidence/reacher-cache-ablation-v1/scored-publication'
REPO = 'kw2828/OpenJev'
TAG = 'research-reacher-cache-ablation-v1'
COMMIT = '1e91a2e2cb10692ee7f1baa4cd5c04d8628da55b'
AUDIT = 'd1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790'
SIDECARS = ('packaging-started.json', 'manifest.json', 'receipt.json', 'README.md', 'SHA256SUMS')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def api(*args):
    result = subprocess.run(['gh', 'api', *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def tag_target():
    refs = api(f'repos/{REPO}/git/matching-refs/tags/{TAG}')
    refs = [row for row in refs if row['ref'] == 'refs/tags/' + TAG]
    assert len(refs) <= 1
    if not refs:
        return None
    obj = refs[0]['object']
    for _ in range(8):
        if obj['type'] == 'commit':
            return obj['sha']
        assert obj['type'] == 'tag'
        obj = api(f"repos/{REPO}/git/tags/{obj['sha']}")['object']
    raise ValueError('Excessive annotated tag nesting')


def verify(release, expected):
    assert release['tag_name'] == TAG and release['target_commitish'] == COMMIT
    rows = {row['name']: row for row in release['assets']}
    assert len(rows) == len(release['assets']) and set(rows) == set(expected)
    for name, row in rows.items():
        assert row['state'] == 'uploaded' and row['size'] == expected[name]['bytes'], name
        assert row['digest'] == 'sha256:' + expected[name]['sha256'], name
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-package-receipt-sha256', required=True)
    args = parser.parse_args()
    assert sha(PACKAGE / 'receipt.json') == args.expected_package_receipt_sha256
    package = json.loads((PACKAGE / 'receipt.json').read_text())
    assert package['status'] == 'verified' and package['audit_receipt_sha256'] == AUDIT
    assert sha(PACKAGE / 'manifest.json') == package['manifest_sha256']
    assert package['scientific_status'] == {'continuation_passed': False,
        'checks_passed': 15, 'checks_total': 28, 'packaging_changes_gate': False}
    expected = {row['name']: row for row in package['release_assets']}
    for name in SIDECARS:
        path = PACKAGE / name
        assert name not in expected
        expected[name] = {'name': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
    for name, row in expected.items():
        assert Path(name).name == name
        path = PACKAGE / name
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        assert row['bytes'] < 2_000_000_000
    sum_lines = (PACKAGE / 'SHA256SUMS').read_text().splitlines()
    sums = {}
    for line in sum_lines:
        digest, name = line.split('  ', 1)
        assert name not in sums
        sums[name] = digest
    assert set(sums) == set(expected) - {'SHA256SUMS'}
    assert all(digest == expected[name]['sha256'] for name, digest in sums.items())
    assert api(f'repos/{REPO}/commits/{COMMIT}')['sha'] == COMMIT
    assert tag_target() in (None, COMMIT), 'Existing tag points to another commit'
    OUT.mkdir(parents=True, exist_ok=False)
    for name in SIDECARS:
        shutil.copyfile(PACKAGE / name, OUT / name)
    write(BASE / 'upload-started.json', {'assets': list(expected.values()),
        'package_receipt_sha256': args.expected_package_receipt_sha256,
        'target_commit': COMMIT, 'started_at_utc': datetime.now(UTC).isoformat()})
    release_id = None
    stage = 'create-draft'
    try:
        request = BASE / 'release-create-request.json'
        write(request, {'tag_name': TAG, 'target_commitish': COMMIT, 'draft': True,
            'name': 'Reacher cache study: recurrent superiority not established',
            'body': (BASE / 'release-notes.md').read_text(), 'make_latest': 'false'})
        created = api('--method', 'POST', f'repos/{REPO}/releases', '--input', str(request))
        write(BASE / 'release-created.json', created)
        release_id = created['id']
        assert created['draft'] and created['tag_name'] == TAG
        stage = 'upload-assets'
        print(json.dumps({'stage': stage, 'release_id': release_id, 'assets': len(expected)}), flush=True)
        command = ['gh', 'release', 'upload', TAG, *[str(PACKAGE / name) for name in expected], '--repo', REPO]
        write(BASE / 'upload-command.json', {'argv': command, 'replacement_permitted': False})
        subprocess.run(command, check=True, timeout=1200)
        endpoint = f'repos/{REPO}/releases/{release_id}'
        stage = 'verify-draft'
        draft = api(endpoint)
        assert draft['draft'] is True
        verify(draft, expected)
        assert tag_target() in (None, COMMIT), 'Tag changed before publication'
        write(BASE / 'draft-upload-verification.json', draft)
        stage = 'publish'
        request = BASE / 'publish-request.json'
        write(request, {'draft': False, 'make_latest': 'false'})
        api('--method', 'PATCH', endpoint, '--input', str(request))
        stage = 'verify-public'
        final = api(endpoint)
        assert not final['draft'] and final['published_at']
        rows = verify(final, expected)
        assert tag_target() == COMMIT, 'Published tag does not resolve to intended commit'
        downloaded = []
        for name in ('receipt.json', 'manifest.json', 'SHA256SUMS'):
            with urllib.request.urlopen(rows[name]['browser_download_url'], timeout=60) as response:
                data = response.read()
            assert len(data) == expected[name]['bytes']
            assert hashlib.sha256(data).hexdigest() == expected[name]['sha256']
            downloaded.append({'name': name, 'bytes': len(data), 'sha256': expected[name]['sha256']})
        result = {'status': 'published_and_verified', 'repository': REPO,
            'release_id': release_id, 'release_tag': TAG, 'release_url': final['html_url'],
            'target_commit': COMMIT, 'published_at': final['published_at'],
            'resolved_release_tag_commit': COMMIT,
            'verified_at_utc': datetime.now(UTC).isoformat(),
            'verification_scope': 'Every GitHub asset upload state, size and SHA-256 digest matches its local asset. Three public sidecars separately downloaded and hashed. Large raw archives were reopened and verified locally, not downloaded again.',
            'assets': [{**expected[name], 'asset_id': rows[name]['id'],
                       'browser_download_url': rows[name]['browser_download_url']} for name in expected],
            'public_sidecar_downloads': downloaded,
            'package_receipt_sha256': args.expected_package_receipt_sha256,
            'audit_receipt_sha256': AUDIT,
            'scientific_gate': {'passed': False, 'checks_passed': 15, 'checks_total': 28},
            'new_model_calls': 0, 'new_native_calls': 0,
            'publisher_sha256': sha(Path(__file__))}
        write(OUT / 'release-verification.json', result)
        (OUT / 'PUBLICATION.md').write_text('# Publication status\n\nAll assets were published and verified. See release-verification.json for remote sizes, digests and public sidecar downloads. The copied receipt.json and README.md are immutable pre-upload packaging records; their uploaded:false wording describes that earlier stage. Publication changes no scientific result.\n')
        print(json.dumps({'status': result['status'], 'url': result['release_url'], 'assets': len(rows)}), flush=True)
    except BaseException as error:
        write(BASE / 'publication-failed.json', {'status': 'failed', 'stage': stage,
            'release_id': release_id, 'error': repr(error), 'time_utc': datetime.now(UTC).isoformat(),
            'scope': 'Publication only. Preserve draft or published state and original scientific evidence.'})
        raise


if __name__ == '__main__':
    main()
