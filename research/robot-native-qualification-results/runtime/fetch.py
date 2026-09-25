"""Pinned official Rust downloads with normal HTTPS verification, no installer execution."""
import hashlib
import json
import platform
import re
import ssl
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOST = 'aarch64-apple-darwin'
BASE = 'https://static.rust-lang.org/dist/'


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def pin(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def download(url, destination, receipt):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'https' or parts.hostname != 'static.rust-lang.org':
        raise ValueError('Only verified official HTTPS endpoints admitted')
    context = ssl.create_default_context()
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise ValueError('TLS verification must remain enabled')
    started = time.monotonic()
    with urllib.request.urlopen(url, context=context, timeout=120) as response:
        final_url = response.geturl()
        final = urllib.parse.urlsplit(final_url)
        if final.scheme != 'https' or final.hostname != 'static.rust-lang.org':
            raise ValueError('Unexpected download redirect')
        with destination.open('xb') as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
    row = {'url': url, 'final_url': final_url, 'path': str(destination),
           'seconds': time.monotonic() - started, **pin(destination),
           'tls': {'check_hostname': True, 'verify_mode': 'CERT_REQUIRED'}}
    receipt['downloads'].append(row)
    return row


def main():
    phase = sys.argv[1]
    if phase not in ('metadata', 'archives'):
        raise ValueError('Known phase required')
    started = time.monotonic()
    receipt = {'phase': phase, 'argv': sys.argv, 'status': 'RUNNING', 'downloads': [],
               'started_utc': datetime.now(timezone.utc).isoformat(),
               'platform': platform.platform(), 'machine': platform.machine(),
               'host': HOST, 'python': sys.version}
    try:
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise ValueError('Exact native Apple ARM64 host required')
        if phase == 'metadata':
            target = HERE / 'channel-rust-stable.toml'
            download(BASE + target.name, target, receipt)
            checksum = HERE / (target.name + '.sha256')
            download(BASE + checksum.name, checksum, receipt)
            expected = checksum.read_text().split()[0]
            if not re.fullmatch('[0-9a-f]{64}', expected) or pin(target)['sha256'] != expected:
                raise ValueError('Official metadata checksum mismatch; stop')
            manifest = tomllib.loads(target.read_text())
            version = manifest['pkg']['rustc']['version'].split()[0]
            if not re.fullmatch(r'\d+\.\d+\.\d+', version):
                raise ValueError('Stable numeric version required')
            selected = {'version': version, 'host': HOST, 'manifest_date': manifest['date'],
                        'metadata': pin(target), 'metadata_checksum': pin(checksum), 'components': {}}
            for component in ('rustc', 'rust-std'):
                package = manifest['pkg'][component]
                target_package = package['target'][HOST]
                if not target_package['available'] or package['version'].split()[0] != version:
                    raise ValueError('Matching available native component required')
                selected['components'][component] = {'url': target_package['xz_url'],
                                                     'sha256': target_package['xz_hash']}
            selected['prefix'] = '/Users/kevinwu/.cache/openjev-rust/' + version
            write(HERE / 'selected.json', selected)
            receipt['selected'] = selected
        else:
            selected = json.loads((HERE / 'selected.json').read_text())
            if pin(HERE / 'channel-rust-stable.toml') != selected['metadata']:
                raise ValueError('Admitted metadata changed')
            for component, item in selected['components'].items():
                name = Path(urllib.parse.urlsplit(item['url']).path).name
                if name != f'{component}-{selected["version"]}-{HOST}.tar.xz':
                    raise ValueError('Unexpected component filename')
                destination = HERE / name
                row = download(item['url'], destination, receipt)
                checksum = HERE / (name + '.sha256')
                download(item['url'] + '.sha256', checksum, receipt)
                expected = checksum.read_text().split()[0]
                if row['sha256'] != item['sha256'] or expected != item['sha256']:
                    raise ValueError('Official component checksum mismatch; stop')
                row['official_sha256_verified'] = True
        receipt['status'] = 'PASS'
    except BaseException as error:
        receipt['status'] = 'FAILED'
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        write(HERE / (phase + '-01.json'), receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'downloads'}), flush=True)


if __name__ == '__main__':
    main()
