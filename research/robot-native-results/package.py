"""Opaque byte-copy publication. No numerical imports, array decoding or replay."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/robot-native-results'
STUDY = ROOT / 'output/robot-native-benchmark-v1'
AUDIT = ROOT / 'output/robot-native-audit-v1'
ENGINEERING = ROOT / 'output/robot-native-benchmark-engineering-v1'
AUDIT_FIXTURE = ROOT / 'output/robot-native-audit-engineering-v1'
AUDIT_SHA = '05f4aec0d41ae225372b8e05e8c4947cb14c3f83284d3412173ed2379060ca63'
PLAN_SHA = '83f53f9ac995a93117968ad048350dccfbe3bcc0e570ec764919fbbb231420e1'


def pin(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), str(path)
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def inventory(folder):
    paths = list(folder.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    return {str(p.relative_to(folder)): pin(p) for p in paths if p.is_file()}


assert pin(AUDIT / 'audit.json')['sha256'] == AUDIT_SHA
audit = read(AUDIT / 'audit.json')
assert audit['status'] == 'PASS' and audit['agreement'] is True
assert audit['benchmark_status'] == 'FAILED' and audit['parity_passed'] == 0 and audit['parity_total'] == 30
assert audit['counts']['npz_decodes'] == 30 and audit['counts']['array_loads'] == 420
assert audit['counts']['timed_pairs'] == audit['counts']['timing_records'] == audit['counts']['warmup_pairs'] == 0
assert read(AUDIT / 'manifest.json') == {'files': {'audit.json': pin(AUDIT / 'audit.json')}}
assert set(inventory(AUDIT)) == {'audit.json', 'manifest.json'}
study_manifest = read(STUDY / 'manifest.json')['files']
assert inventory(STUDY) == {**study_manifest, 'manifest.json': pin(STUDY / 'manifest.json'), 'receipt.json': pin(STUDY / 'receipt.json')}
receipt = read(STUDY / 'receipt.json')
assert receipt['status'] == 'FAILED' and receipt['parity_comparisons'] == 30 and receipt['timed_slots'] == 0
assert receipt['manifest'] == pin(STUDY / 'manifest.json')
assert all(audit['inputs'][str(STUDY / name)] == value for name, value in inventory(STUDY).items())
summary = read(STUDY / 'summary.json')
assert summary['status'] == 'FAILED' and summary['inputs_unchanged'] is True
assert len(summary['parity']) == 30 and all(not row['passed'] for row in summary['parity'])
assert summary['timings'] == [] and summary['result'] is None
check_keys = {batch + '/' + metric for batch in ('batch22', 'batch1')
              for metric in ('physical', 'standardized', 'physical_final_state', 'standardized_final_state')}
assert all(set(row['checks']) == check_keys and not row['errors'] for row in summary['parity'])
assert len([name for name in study_manifest if name.endswith('.npz')]) == 30
assert not any(name.startswith('timing-') or name == 'parity-barrier.json' for name in study_manifest)
plan_path = ROOT / 'research/robot-native-registration.json'
assert pin(plan_path)['sha256'] == PLAN_SHA
plan = read(plan_path)
assert pin(ROOT / 'research/robot-native-protocol.md') == {k: plan['protocol'][k] for k in ('bytes', 'sha256')}
launch = read(ENGINEERING / 'run-launch-01.json')
process = read(ENGINEERING / 'run-process-01.json')
assert all(process[k] == value for k, value in launch.items())
assert process['registration'] == pin(plan_path) and process['command'] == plan['command']
assert process['returncode'] == 1 and process['external_timeout'] is False and process['inputs_unchanged'] is True
assert process['manifest'] == pin(STUDY / 'manifest.json') and process['receipt'] == pin(STUDY / 'receipt.json')
assert process['log'] == pin(ENGINEERING / 'run-process-01.log')
audit_process = read(ENGINEERING / 'audit-process-01.json')
assert audit_process['returncode'] == 0 and audit_process['source_unchanged'] is True
assert audit_process['audit'] == pin(AUDIT / 'audit.json')
assert audit_process['log'] == pin(ENGINEERING / 'audit-process-01.log')
assert audit_process['script'] == pin(ROOT / 'scripts/audit_robot_native.py')
assert audit_process['command'] == ['.venv/bin/python', 'scripts/audit_robot_native.py',
    '--study', 'output/robot-native-benchmark-v1', '--process',
    'output/robot-native-benchmark-engineering-v1/run-process-01.json', '--registration',
    'research/robot-native-registration.json', '--output', 'output/robot-native-audit-v1/audit.json']
assert pin(AUDIT_FIXTURE / 'tests-01.json') == {'bytes': 714, 'sha256': 'b7bd6ff56acb00cd561bbe375821f45d42d9636c82622ef8c026f3c28f8fee0e'}
aq = read(AUDIT_FIXTURE / 'tests-01.json')
assert aq['returncode'] == 0 and aq['log'] == pin(AUDIT_FIXTURE / 'tests-01.log')
assert '24 passed' in (AUDIT_FIXTURE / 'tests-01.log').read_text()
for name, value in aq['sources'].items():
    assert pin(ROOT / name) == value
fixture = ENGINEERING / 'fixture-01'
assert set(inventory(fixture)) == {'definition.json', 'process.json', 'pytest.log'}
for desc in plan['helper_qualification'].values():
    assert pin(desc['path']) == {k: desc[k] for k in ('bytes', 'sha256')}
for name in ('scripts/benchmark_robot_native.py', 'tests/test_benchmark_robot_native.py'):
    assert pin(ROOT / name) == plan['sources'][name]

copies = {}
def add(source, destination):
    assert destination not in copies and not Path(destination).is_absolute() and '..' not in Path(destination).parts
    copies[destination] = {'source': str(source.relative_to(ROOT)), **pin(source)}


for folder, target in ((STUDY, 'study'), (AUDIT, 'audit'), (fixture, 'benchmark-fixture'), (AUDIT_FIXTURE, 'audit-fixture')):
    if folder == AUDIT_FIXTURE:
        assert set(inventory(folder)) == {'tests-01.json', 'tests-01.log'}
    for name in sorted(inventory(folder)):
        add(folder / name, target + '/' + name)
for name in ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log', 'audit-process-01.json', 'audit-process-01.log'):
    add(ENGINEERING / name, 'processes/' + name)
for name in ('robot-native-registration.json', 'robot-native-protocol.md'):
    add(ROOT / 'research' / name, 'registration/' + name)
for name in ('scripts/benchmark_robot_native.py', 'tests/test_benchmark_robot_native.py',
             'scripts/audit_robot_native.py', 'tests/test_audit_robot_native.py'):
    add(ROOT / name, 'sources/' + name)
add(Path(__file__), 'package.py')
assert not any(Path(name).suffix in ('.dylib', '.so', '.dll', '.mat') for name in copies)
external = {name: pin(ROOT / name) for name in (
    'research/robot-native-qualification-results/manifest.json',
    'research/robot-native-qualification-results/provenance.json')}
OUT.mkdir(parents=True, exist_ok=False)
for name, value in copies.items():
    source = ROOT / value['source']
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(source.read_bytes())
    assert pin(target) == {k: value[k] for k in ('bytes', 'sha256')}
readme = '''# Native robot comparison: preserved failed result

**Benchmark FAILED. Independent saved-output evidence audit PASS. No timing was collected.**

All 30 selected fit/DEV cases failed the fixed numerical parity requirement. Every case is retained for both batch sizes 22 and 1, with standardized and physical forecasts and both complete native final-state comparisons. The package includes all 30 opaque forecast archives, 30 per-case records, the original aggregate results, and the original successful audit of the failed result. The audit independently checked 420 saved arrays and 240 comparisons; it did not rerun inference.

The numerical gate stopped execution before warmups or timed pairs. This provides no latency or speedup result and does not revise the parent study's quality rule. The earlier 110-test fabricated native qualification passed; those fixtures did not establish parity for these trained weights.

- `study/`: complete original benchmark folder, copied byte for byte.
- `audit/`: original independent audit and its manifest.
- `processes/`: original run launch, failed run closure/log and successful audit closure/log.
- `registration/`: frozen benchmark registration and protocol.
- `benchmark-fixture/` and `audit-fixture/`: original 22-test and 24-test qualification receipts/logs.
- `sources/`: the exact benchmark, auditor and corresponding tests.
- `provenance.json`: source-to-copy byte identities and scope.
- `manifest.json`: every published file except this manifest, with SHA-256 and byte count.

The [native qualification package](../robot-native-qualification-results/README.md) preserves both native qualification attempts and the source-derived repair. Its manifest and provenance hashes are recorded here as external references. Compiled libraries, source measurement arrays, targets and trained checkpoints are not included. Original dependency paths and hashes remain in the preserved receipts. Re-executing the benchmark requires the separately retained parent study and its measurement permissions; this package alone supports inspection of the saved parity evidence.

Publication performed only JSON/metadata checks and opaque byte copies, followed by independent byte verification. It did not decode forecast arrays, load models, refit, score targets or collect new timings.
'''
(OUT / 'README.md').write_text(readme)
for name, value in copies.items():
    assert pin(ROOT / value['source']) == pin(OUT / name) == {k: value[k] for k in ('bytes', 'sha256')}
assert all(pin(ROOT / name) == value for name, value in external.items())
write(OUT / 'provenance.json', {'version': 'robot-native-byte-publication-v1',
    'benchmark_status': 'FAILED', 'audit_status': 'PASS', 'parity_passed': 0, 'parity_total': 30,
    'timed_pairs': 0, 'warmup_pairs': 0, 'source_copy_count': len(copies), 'copies': copies,
    'external_references': external, 'all_copies_byte_equal': True, 'all_sources_unchanged': True,
    'array_decodes': 0, 'model_calls': 0, 'source_measurements_included': False,
    'compiled_libraries_included': False})
write(OUT / 'manifest.json', {'files': inventory(OUT)})
manifest = read(OUT / 'manifest.json')['files']
assert inventory(OUT) == {**manifest, 'manifest.json': pin(OUT / 'manifest.json')}
write(Path(__file__).parent / 'receipt.json', {'status': 'PASS', 'output': str(OUT),
    'manifest': pin(OUT / 'manifest.json'), 'files': len(manifest)+1,
    'copied_files': len(copies), 'bytes': sum(p['bytes'] for p in inventory(OUT).values()),
    'all_copies_byte_equal': True, 'array_decodes': 0, 'model_calls': 0})
print(json.dumps(read(Path(__file__).parent / 'receipt.json')))
