"""Fabricated path checks only; never execute either publication entrypoint."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
PREFIX = 'https://raw.githubusercontent.com/kw2828/OpenJev/main/'
NAMES = (
    'docs/assets/chess-candidate-game-001.gif',
    'docs/assets/jepa-policy-181000-preview.gif',
    'evidence/reacher-geometry-memory-v1/report/fixed-case-replay.gif',
    'research/otto-conditional-label-results/benchmark.png',
)


def descriptor(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    sys.dont_write_bytecode = True
    paths = [OUT / name for name in ('archive.py', 'verify-remote-01.py', Path(__file__).name)]
    before = {p.name: descriptor(p) for p in paths}
    archive = load('fabricated_publication_archive', paths[0])
    verifier = load('fabricated_publication_verifier', paths[1])
    checks = []
    valid = [
        (NAMES[0], NAMES[0]),
        (PREFIX + NAMES[0], NAMES[0]),
        ('docs/assets/a%20b.png', 'docs/assets/a b.png'),
        (PREFIX + 'docs/assets/a%20b.png', 'docs/assets/a b.png'),
    ]
    invalid = [
        '', '/docs/a.png', '//raw.githubusercontent.com/kw2828/OpenJev/main/docs/a.png',
        '../docs/a.png', 'docs/../a.png', 'docs/%2e%2e/a.png',
        'docs/%252e%252e/a.png', 'docs/%2Fa.png', 'docs/./a.png', 'docs//a.png',
        'docs\\a.png', 'docs/%5ca.png', 'docs/a.png?x=1', 'docs/a.png?',
        'docs/a.png#x', 'docs/a.png#', 'docs/a%3fx.png', 'docs/a%23x.png',
        ' docs/a.png', 'docs/a.png ', 'docs/\ta.png', 'docs/%00a.png',
        PREFIX, PREFIX + '../a.png', PREFIX + '%2e%2e/a.png',
        PREFIX + NAMES[0] + '?raw=1', PREFIX + NAMES[0] + '#frame',
        PREFIX.replace('https:', 'http:') + NAMES[0],
        PREFIX.replace('raw.githubusercontent.com', 'example.com') + NAMES[0],
        PREFIX.replace('raw.githubusercontent.com', 'raw.githubusercontent.com.evil.test') + NAMES[0],
        PREFIX.replace('raw.githubusercontent.com', 'user@raw.githubusercontent.com') + NAMES[0],
        PREFIX.replace('raw.githubusercontent.com', 'raw.githubusercontent.com:443') + NAMES[0],
        PREFIX.replace('/kw2828/', '/other/') + NAMES[0],
        PREFIX.replace('/OpenJev/', '/Other/') + NAMES[0],
        PREFIX.replace('/main/', '/other/') + NAMES[0],
        PREFIX.replace('/main/', '/mainly/') + NAMES[0],
        PREFIX.replace('https:', 'HTTPS:') + NAMES[0],
        'data:image/png;base64,AA', 'file:///docs/a.png',
    ]
    for label, module in (('archive', archive), ('verifier', verifier)):
        for reference, expected in valid:
            if module.repository_image_path(reference).as_posix() != expected:
                raise AssertionError((label, reference, expected))
            checks.append({'parser': label, 'reference': reference, 'accepted': True})
        for reference in invalid:
            try:
                module.repository_image_path(reference)
            except ValueError:
                pass
            else:
                raise AssertionError(('unexpected acceptance', label, reference))
            checks.append({'parser': label, 'reference': reference, 'accepted': False})
    with tempfile.TemporaryDirectory(prefix='openjev-image-parser-') as directory:
        root = Path(directory).resolve()
        archive.ROOT = root
        for name in NAMES:
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'fabricated path fixture, not an image')
        for form in ('relative', 'raw', 'mixed'):
            refs = [PREFIX + name if form == 'raw' or (form == 'mixed' and i % 2) else name
                    for i, name in enumerate(NAMES)]
            readme = '\n'.join(f'![fixture]({ref})' if i % 2 else f'<img src="{ref}">'
                               for i, ref in enumerate(refs))
            if archive.embedded_images(readme) != [root / NAMES[i] for i in (1, 3, 0, 2)]:
                raise AssertionError(('archive complete roster', form))
            if set(verifier.images(readme)) != set(NAMES):
                raise AssertionError(('verifier complete roster', form))
            checks.append({'complete_roster_form': form, 'passed': True})
        valid_readme = '\n'.join(f'![fixture]({PREFIX + name})' for name in NAMES)
        for label, parse in (('archive', archive.embedded_images), ('verifier', verifier.images)):
            try:
                parse(valid_readme + '\n<img src="https://example.com/other.png">')
            except ValueError:
                pass
            else:
                raise AssertionError(('unsafe extra image accepted', label))
            checks.append({'parser': label, 'unsafe_extra_rejected': True})
    after = {p.name: descriptor(p) for p in paths}
    if before != after:
        raise AssertionError('source changed during fabricated checks')
    result = {
        'version': 'otto-conditional-label-image-parser-checks-v1',
        'status': 'passed', 'scope': 'fabricated paths and temporary files only',
        'checks': checks, 'check_count': len(checks),
        'sources_before': before, 'sources_after': after, 'sources_unchanged': True,
        'archive_executions': 0, 'release_executions': 0, 'remote_calls': 0,
        'scientific_array_decodes': 0, 'checkpoint_decodes': 0,
        'model_calls': 0, 'native_calls': 0, 'teacher_calls': 0,
    }
    with (OUT / 'image-parser-checks-02.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'check_count': len(checks)}))


if __name__ == '__main__':
    main()
