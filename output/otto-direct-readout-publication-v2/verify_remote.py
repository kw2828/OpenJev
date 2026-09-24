"""Read-only GitHub release/blob integrity check after the original upload closes."""
from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
COMMIT = '4c3bb706dc7202ff0dee8dd156cad867c44e78d7'
REPO = 'kw2828/OpenJev'
TAG = 'otto-direct-readout-v1'


def api(endpoint):
    return json.loads(subprocess.run(['gh', 'api', endpoint], check=True, capture_output=True, text=True).stdout)


def main():
    release = api(f'repos/{REPO}/releases/tags/{TAG}')
    assert not release['draft'] and release['tag_name'] == TAG
    obj = api(f'repos/{REPO}/git/ref/tags/{TAG}')['object']
    while obj['type'] == 'tag':
        obj = api(f'repos/{REPO}/git/tags/{obj["sha"]}')['object']
    assert obj['type'] == 'commit' and obj['sha'] == COMMIT
    names = ['otto-direct-readout-v1.tar.gz', 'archive-manifest.json', 'archive-verification.json', 'SHA256SUMS']
    assets = {a['name']: a for a in release['assets']}
    assert set(assets) == set(names)
    verified = []
    for name in names:
        raw, asset = (BASE/name).read_bytes(), assets[name]
        digest = hashlib.sha256(raw).hexdigest()
        assert asset['size'] == len(raw) and asset['digest'] == 'sha256:' + digest and asset['state'] == 'uploaded'
        verified.append({'name': name, 'bytes': len(raw), 'sha256': digest,
                         'github_asset_id': asset['id'], 'url': asset['browser_download_url']})
    local = json.loads((BASE/'readme-local-check-01.json').read_text())
    paths = ['README.md'] + [a['path'] for a in local['embedded_images']]

    def verify_blob(path):
        data = (ROOT/path).read_bytes()
        digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        remote = api(f'repos/{REPO}/contents/{urllib.parse.quote(path)}?ref={COMMIT}')
        assert remote['sha'] == digest and remote['size'] == len(data) and remote['type'] == 'file'
        return {'path': path, 'bytes': len(data), 'git_blob_sha': digest,
                'sha256': hashlib.sha256(data).hexdigest(), 'url': remote['html_url']}

    with ThreadPoolExecutor(max_workers=4) as pool:
        blobs = list(pool.map(verify_blob, paths))
    assert blobs[0]['sha256'] == local['readme_sha256']
    assert all(blob['sha256'] == im['sha256'] for blob, im in zip(blobs[1:], local['embedded_images'], strict=True))
    result = {'status': 'passed', 'checked_at_utc': datetime.now(UTC).isoformat(),
              'release_url': release['html_url'], 'release_id': release['id'], 'tag': TAG,
              'target_commit': COMMIT, 'assets': verified, 'readme_and_embedded_image_blobs': blobs,
              'scientific_calls': 0,
              'scope': 'Remote release asset SHA256/size, exact tag commit, README and all four image Git blobs. File integrity, not browser animation playback.'}
    with (BASE/'release-verification-01.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'release_url': result['release_url'],
                      'assets': len(verified), 'readme_images': len(blobs)-1}), flush=True)


if __name__ == '__main__':
    main()
