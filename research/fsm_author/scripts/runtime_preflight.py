"""Verify pinned source, installed dependencies and CPU float64 configuration."""
# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from pathlib import Path


def pin(path):
    data = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    component = Path(__file__).resolve().parents[1]
    root = component.parents[1]
    for name in ('pyproject.toml', 'uv.lock'):
        (output/name).write_bytes((component/name).read_bytes())
    error, details = None, {}
    try:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        import freq_statespace as fs

        dist = importlib.metadata.distribution('freq-statespace')
        direct = json.loads(dist.read_text('direct_url.json'))
        assert direct['vcs_info']['commit_id'] == 'a79e8c567b018a6c9462528fc1e10b77fd19b3e2'
        source = root/'output/fsm-author-engineering-v1/vendor/freq-statespace/src/freq_statespace'
        installed = Path(fs.__file__).parent
        matches = {}
        for path in sorted(source.rglob('*.py')):
            relative = path.relative_to(source)
            assert pin(path) == pin(installed/relative), f'installed source drift: {relative}'
            matches[str(relative)] = pin(path)
        assert matches
        assert jax.config.jax_enable_x64 and jnp.ones((1,), dtype=jnp.float64).dtype.name == 'float64'
        devices = [str(d) for d in jax.devices()]
        assert all(d.platform == 'cpu' for d in jax.devices())
        details = {'python': platform.python_version(), 'platform': platform.platform(),
                   'executable': sys.executable, 'jax_x64': bool(jax.config.jax_enable_x64),
                   'devices': devices, 'upstream_direct_url': direct, 'installed_source_matches': matches,
                   'versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                   'lock': pin(component/'uv.lock'), 'script': pin(__file__),
                   'scope': 'Import/source/runtime qualification only; no dataset loader, pretrained weights or fitting'}
    except Exception as exc:  # noqa: BLE001 - preserve failed runtime admission
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    result = {'status': 'PASS' if error is None else 'FAIL', 'error': error, **details}
    (output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'error': error,
                      'versions': details.get('versions'), 'devices': details.get('devices')}, indent=2))
    return 0 if error is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    raise SystemExit(run(parser.parse_args().output))
