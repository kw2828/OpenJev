"""Preserve this closed pilot and its registered source/native-input bytes."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path.cwd()
STUDY = ROOT / 'output/otto-action-latent-v1'
OUT = ROOT / 'output/otto-action-latent-publication-v1'

def desc(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}

plan = json.loads((STUDY / 'registration-01.json').read_text())
closure = json.loads((STUDY / 'closure-01.json').read_text())
assert closure['technical_complete'] and closure['independent_audit_passed']
paths = {ROOT / name for name in plan['sources']} | {ROOT / name for name in plan['inputs']}
paths.update(p for p in STUDY.rglob('*') if p.is_file() and 'pytest-temp' not in p.parts)
paths.update(p for p in (ROOT / 'research/otto-action-latent-results').rglob('*') if p.is_file())
paths.update(ROOT / p for p in ('README.md', 'LICENSE', 'research/experiment-index.md',
    'research/otto-action-latent-results.md', 'research/otto-action-latent-next.md',
    'research/otto-action-latent-proposal.md', 'scripts/report_otto_action_latent.py'))
paths.update(p for p in (OUT / 'render-01').rglob('*') if p.is_file())
paths.update([OUT / 'archive.py', OUT / 'renderer-01.py'])
for name, expected in {**plan['sources'], **plan['inputs']}.items():
    assert desc(ROOT / name) == expected, name
files = {str(p.relative_to(ROOT)): desc(p) for p in sorted(paths)}
manifest = {'study': 'otto-action-latent-v1', 'files': files, 'closure': desc(STUDY / 'closure-01.json'),
    'scope': 'All current-run scientific evidence, checkpoints, registered source and direct native inputs. Fabricated pytest temporary directories excluded.',
    'external_dependencies': ['Qualified Python/Torch and original TensorFlow environments are not bundled.',
        'Ancestral authentication plans reference earlier study receipts beyond the direct inputs in this snapshot; use the repository and earlier evidence releases.',
        'Registered absolute paths describe the original workstation; this is an evidence snapshot, not a portable application installer.'],
    'model_or_native_calls': 0, 'array_decodes': 0}
(OUT / 'archive-manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
archive = OUT / 'otto-action-latent-v1.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    for name in files:
        tar.add(ROOT / name, arcname=name, recursive=False)
with tarfile.open(archive, 'r:gz') as tar:
    assert {m.name for m in tar.getmembers()} == set(files)
    for member in tar.getmembers():
        digest = hashlib.sha256()
        stream = tar.extractfile(member)
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
        assert member.size == files[member.name]['bytes'] and digest.hexdigest() == files[member.name]['sha256']
assert all(desc(ROOT / name) == expected for name, expected in files.items())
verification = {'status': 'passed', 'members_checked': len(files), 'archive': desc(archive),
                'manifest': desc(OUT / 'archive-manifest.json'), 'opaque_byte_roundtrip': True}
(OUT / 'archive-verification.json').write_text(json.dumps(verification, sort_keys=True, indent=2) + '\n')
(OUT / 'SHA256SUMS').write_text('\n'.join(desc(OUT / name)['sha256'] + '  ' + name for name in
    ('otto-action-latent-v1.tar.gz', 'archive-manifest.json', 'archive-verification.json')) + '\n')
print(json.dumps(verification))
