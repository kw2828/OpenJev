"""Verify release digests and committed report/media bytes, without UI claims.

Run only after the owner supplies the completed closure and publication commit.
All GitHub operations are reads. No scientific array or model is decoded.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPOSITORY = 'kw2828/OpenJev'
TAG = 'otto-cost-information-v1'
ASSETS = {TAG + '.tar.gz', 'archive-manifest.json', 'archive-verification.json', 'SHA256SUMS'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path], cwd=ROOT, text=True, timeout=60))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(path):
    require(path.is_file() and not path.is_symlink(), 'regular local artifact')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def images(readme):
    references = [a or b for a, b in re.findall(r'!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))', readme)]
    references += re.findall(r'<img\b[^>]*\bsrc=[\'"]([^\'"]+)[\'"]', readme)
    names = []
    for reference in references:
        parsed = urlsplit(reference)
        require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment,
                'repository-local embedded image: ' + reference)
        name = PurePosixPath(unquote(parsed.path))
        require(not name.is_absolute() and '..' not in name.parts, 'safe image path')
        if name.as_posix() not in names:
            names.append(name.as_posix())
    require('research/otto-cost-information-results/benchmark.png' in names, 'new benchmark embedded in README')
    require(any(name.lower().endswith('.gif') for name in names), 'existing README GIF remains embedded')
    return names


def verify(commit, closure_sha):
    require(Path.cwd() == ROOT and re.fullmatch('[0-9a-f]{40}', commit)
            and re.fullmatch('[0-9a-f]{64}', closure_sha), 'root, full publication commit and external closure pin')
    closure_path = ROOT / 'output' / TAG / 'closure-01.json'
    require(descriptor(closure_path)['sha256'] == closure_sha, 'externally supplied completed closure')
    closure = json.loads(closure_path.read_text())
    require(closure['version'] == TAG and closure['technical_complete'] is True
            and closure['independent_audit_passed'] is True and closure['new_training'] is False
            and closure['new_execution_admitted'] is False and closure['old_test_admitted'] is False,
            'closed diagnostic before publication verification')
    release = api(f'repos/{REPOSITORY}/releases/tags/{TAG}')
    require(release['tag_name'] == TAG and release['target_commitish'] == commit
            and release['draft'] is False, 'published release targets requested commit')
    target = api(f'repos/{REPOSITORY}/git/ref/tags/{TAG}')['object']
    for _ in range(4):
        if target['type'] != 'tag':
            break
        target = api(f'repos/{REPOSITORY}/git/tags/' + target['sha'])['object']
    require(target['type'] == 'commit' and target['sha'] == commit, 'actual peeled remote tag commit')
    require(api(f'repos/{REPOSITORY}/git/commits/{commit}')['sha'] == commit, 'remote commit object')
    assets = {value['name']: value for value in release['assets']}
    require(len(release['assets']) == len(assets) == 4 and set(assets) == ASSETS, 'exact four release assets')
    asset_checks = []
    for name in sorted(ASSETS):
        expected = descriptor(OUT / name)
        remote = assets[name]
        require(remote['state'] == 'uploaded' and remote['size'] == expected['bytes']
                and remote['digest'] == 'sha256:' + expected['sha256'], 'server digest and size: ' + name)
        asset_checks.append({'name': name, **expected, 'remote_digest': remote['digest'],
                             'download_url': remote['browser_download_url']})
    sums = {name: digest for digest, name in
            (line.split('  ', 1) for line in (OUT / 'SHA256SUMS').read_text().splitlines())}
    require(set(sums) == ASSETS - {'SHA256SUMS'}
            and all(descriptor(OUT / name)['sha256'] == digest for name, digest in sums.items()), 'checksum roster')
    manifest = json.loads((OUT / 'archive-manifest.json').read_text())
    archive_check = json.loads((OUT / 'archive-verification.json').read_text())
    require(manifest['study'] == TAG and archive_check['status'] == 'passed'
            and archive_check['opaque_byte_roundtrip'] is True
            and manifest['publication_commit'] == archive_check['publication_commit'] == commit
            and manifest['closure'] == archive_check['closure'] == descriptor(closure_path)
            and archive_check['archive'] == descriptor(OUT / (TAG + '.tar.gz'))
            and archive_check['manifest'] == descriptor(OUT / 'archive-manifest.json'), 'local closed archive verification')
    local_readme = (ROOT / 'README.md').read_bytes()
    media = images(local_readme.decode())
    publication = manifest['committed_publication_files']
    required = {'README.md', 'research/experiment-index.md', 'research/jev-architecture-source-review.md',
                'research/otto-cost-information-protocol.md', 'research/otto-cost-information-results.md',
                'research/otto-cost-information-next.md',
                'research/otto-cost-information-results/summary.json',
                'research/otto-cost-information-results/benchmark.png',
                'research/otto-cost-information-results/receipt.json', *media}
    require(set(publication) == required, 'complete declared publication and README media roster')
    verified = []
    from PIL import Image, ImageSequence

    for name in sorted(required):
        entries = subprocess.check_output(['git', 'ls-tree', '-z', commit, '--', name], cwd=ROOT, timeout=30).split(b'\0')
        entries = [entry for entry in entries if entry]
        require(len(entries) == 1, 'single committed publication entry')
        header, tree_name = entries[0].split(b'\t', 1)
        mode, kind, blob = header.decode().split()
        require(tree_name.decode() == name and kind == 'blob' and mode in ('100644', '100755'), 'regular committed file')
        remote = api(f'repos/{REPOSITORY}/git/blobs/{blob}')
        require(remote['encoding'] == 'base64' and remote['sha'] == blob, 'exact remote Git blob')
        data = base64.b64decode(''.join(remote['content'].split()), validate=True)
        require(remote['size'] == len(data) and data == (ROOT / name).read_bytes()
                and hashlib.sha1(f'blob {len(data)}\0'.encode() + data, usedforsecurity=False).hexdigest() == blob,
                'remote Git content equals commit and local publication')
        expected = manifest['files'][name]
        require(sha(data) == expected['sha256'] and len(data) == expected['bytes']
                and publication[name] == {'blob': blob, 'sha256': sha(data), 'bytes': len(data)},
                'committed publication belongs to evidence archive')
        row = {'path': name, 'blob': blob, 'bytes': len(data), 'sha256': sha(data)}
        if name in media:
            with Image.open(io.BytesIO(data)) as picture:
                frame_count = getattr(picture, 'n_frames', 1)
                row.update(format=picture.format, size=list(picture.size), frames=frame_count)
                require(all(v > 0 for v in picture.size), 'positive image dimensions')
                content_hashes, durations = set(), []
                decoded = 0
                for frame in ImageSequence.Iterator(picture):
                    frame.load()
                    content_hashes.add(sha(frame.convert('RGBA').tobytes()))
                    durations.append(int(frame.info.get('duration', 0)))
                    decoded += 1
                require(decoded == frame_count, 'every remote image frame decoded')
                row.update(distinct_decoded_frames=len(content_hashes), duration_ms=sum(durations))
                if name.lower().endswith('.gif'):
                    require(row['format'] == 'GIF' and frame_count > 1 and len(content_hashes) > 1
                            and sum(durations) > 0, 'animated GIF contains changing decoded frames')
        verified.append(row)
    return {'status': 'passed', 'release': release['html_url'], 'commit': commit,
        'closure': descriptor(closure_path), 'peeled_tag_commit': target['sha'],
        'assets': asset_checks, 'publication_files': verified,
        'readme_images': [row for row in verified if row['path'] in media],
        'scope': 'GitHub server asset digests and exact commit blobs for report, summary, receipts, protocol, README and embedded images; every remote image frame decoded locally. This does not verify browser rendering, animation playback, or GitHub UI appearance.',
        'array_or_checkpoint_decodes': 0, 'new_model_or_native_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--closure-sha256', required=True)
    parser.add_argument('--output', type=Path, default=OUT / 'release-verification-01.json')
    args = parser.parse_args()
    require(args.output.is_absolute() and args.output.parent == OUT and not args.output.exists(), 'fresh local verification record')
    try:
        result = verify(args.commit, args.closure_sha256)
    except BaseException as error:
        with args.output.open('x') as stream:
            json.dump({'status': 'failed', 'commit': args.commit, 'closure_sha256': args.closure_sha256,
                       'error_type': type(error).__name__,
                       'error': str(error), 'new_model_or_native_calls': 0}, stream, sort_keys=True, indent=2)
            stream.write('\n')
        raise
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'release': result['release'], 'assets': len(result['assets']),
                      'publication_files': len(result['publication_files']), 'readme_images': len(result['readme_images'])}))
