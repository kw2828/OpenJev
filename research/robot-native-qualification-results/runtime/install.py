"""Install reviewed, checksum-verified official components into a fresh private prefix."""
import ctypes
import hashlib
import json
import os
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def pin(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')


def process(label, argv, cwd, env):
    started = time.monotonic()
    receipt = {'argv': argv, 'cwd': str(cwd), 'started_utc': datetime.now(timezone.utc).isoformat()}
    log = HERE / (label + '.log')
    try:
        with log.open('xb') as handle:
            completed = subprocess.run(argv, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       timeout=180, check=False)
        receipt['returncode'] = completed.returncode
    except BaseException as error:
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        if log.exists():
            receipt['log'] = {'path': str(log), **pin(log)}
        write(HERE / (label + '.json'), receipt)
    if completed.returncode:
        raise RuntimeError('Original process failed: ' + label)
    return receipt


def main():
    selected = json.loads((HERE / 'selected.json').read_text())
    inspected = json.loads((HERE / 'archive-inspection-01.json').read_text())
    prefix = Path(selected['prefix'])
    expected = Path('/Users/kevinwu/.cache/openjev-rust') / selected['version']
    if prefix != expected or prefix.exists() or prefix.resolve() != expected:
        raise ValueError('Fresh exact private prefix required')
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(Path.home()), 'LANG': 'C'}
    extraction = HERE / 'extracted'
    extraction.mkdir(exist_ok=False)
    write(HERE / 'installer-review-01.json', {
        'status': 'REVIEWED', 'installers': inspected,
        'scope': 'Reviewed generic shell installer, component manifests and archive paths before execution.',
        'findings': ['No network download or shell-profile/PATH mutation in installer.',
                     'All bindir/libdir/etc/share paths derive from explicit isolated prefix.',
                     'ldconfig explicitly disabled; no sudo invocation.',
                     'Fresh prefix has no preexisting component manifests to uninstall.',
                     'Option parsing uses eval; only fixed trusted version/path arguments supplied.',
                     'Both installers differ only in their success message.'],
        'prefix': str(prefix), 'global_environment_changed': False})
    receipts = []
    for component in ('rustc', 'rust-std'):
        item = selected['components'][component]
        archive_path = HERE / Path(item['url']).name
        if pin(archive_path)['sha256'] != item['sha256']:
            raise ValueError('Admitted archive changed; stop')
        with tarfile.open(archive_path, 'r:xz') as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ValueError('Unsupported or unsafe archive entry')
                if member.name.endswith('/manifest.in'):
                    for line in archive.extractfile(member).read().decode().splitlines():
                        kind, name = line.split(':', 1)
                        subpath = PurePosixPath(name)
                        if kind not in ('file', 'dir') or subpath.is_absolute() or '..' in subpath.parts:
                            raise ValueError('Unsafe install destination')
            archive.extractall(extraction, filter='data')
        directory = extraction / inspected[component]['top_level'][0]
        installer = directory / 'install.sh'
        if pin(installer)['sha256'] != inspected[component]['installer_sha256']:
            raise ValueError('Inspected installer changed')
        receipts.append(process('install-' + component + '-01', ['/bin/bash', str(installer),
                                '--prefix=' + str(prefix), '--disable-ldconfig'], directory, env))
        installed_log = prefix / 'lib/rustlib/install.log'
        (HERE / (component + '-installed-manifest.log')).write_bytes(installed_log.read_bytes())
    compiler = prefix / 'bin/rustc'
    receipts.append(process('compiler-version-01', [str(compiler), '--version', '--verbose'], HERE, env))
    version = (HERE / 'compiler-version-01.log').read_text()
    if 'release: ' + selected['version'] not in version or 'host: ' + selected['host'] not in version:
        raise ValueError('Installed compiler identity mismatch')
    smoke = HERE / 'arithmetic.rs'
    smoke.write_text('#[no_mangle]\npub extern "C" fn openjev_arithmetic(a: f32, b: f32) -> f32 { a * b + 1.0 }\n')
    library = HERE / 'libarithmetic.dylib'
    receipts.append(process('compile-arithmetic-01', [str(compiler), '--edition=2021', '--crate-type=cdylib',
                            '-C', 'opt-level=2', str(smoke), '-o', str(library)], HERE, env))
    module = ctypes.CDLL(str(library))
    function = module.openjev_arithmetic
    function.argtypes = [ctypes.c_float, ctypes.c_float]
    function.restype = ctypes.c_float
    answer = function(3.0, 7.0)
    if answer != 22.0:
        raise ValueError('Arithmetic smoke failed')
    write(HERE / 'arithmetic-smoke-01.json', {'status': 'PASS', 'input': [3., 7.], 'expected': 22., 'actual': answer,
          'source': pin(smoke), 'library': pin(library), 'scope': 'Trivial arithmetic only, no model or benchmark'})
    inventory = {str(path.relative_to(prefix)): pin(path) for path in sorted(prefix.rglob('*')) if path.is_file()}
    write(HERE / 'installed-manifest.json', {'prefix': str(prefix), 'files': inventory})
    write(HERE / 'runtime-setup-01.json', {'status': 'PASS', 'version': selected['version'], 'host': selected['host'],
          'prefix': str(prefix), 'compiler': {'path': str(compiler), **pin(compiler)},
          'official_components': selected['components'], 'processes': receipts, 'arithmetic_smoke': 'PASS',
          'installed_manifest': pin(HERE / 'installed-manifest.json'), 'cargo_installed': False,
          'system_or_profile_modified': False, 'measured_data_access': False, 'benchmark_executed': False})
    print(json.dumps({'status': 'PASS', 'compiler': str(compiler), 'version': selected['version'], 'smoke': answer}), flush=True)


if __name__ == '__main__':
    main()
