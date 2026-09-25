"""Opaque temporary-file checks; no measured files or numerical imports."""
import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

REAL_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REAL_ROOT/'scripts/package_fsm_author_factorial.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def expect_failure(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('Expected an explicit rejection')


def fixture(root):
    spec = importlib.util.spec_from_file_location('fabricated_package', SOURCE)
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    p.ROOT = root

    def save(name, value):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True)+'\n').encode()
        path.write_bytes(data)
        return p.descriptor(path)

    for name in p.HELD:
        save(name, b'# fabricated source\n')
    fake_auditor = (
        'import json\nfrom pathlib import Path\n'
        'def authenticate(study, process, freeze):\n'
        '    d=json.loads((Path(__file__).resolve().parents[1]/"admitted.json").read_text())\n'
        '    return d["plan"],d["inputs"],{k:Path(v) for k,v in d["paths"].items()},{},{},{}\n'
    ).encode()
    save(p.AUDITOR, fake_auditor)
    p.HELD = {name: p.descriptor(root/name)['sha256'] for name in p.HELD}
    own = save('scripts/package_fsm_author_factorial.py', SOURCE.read_bytes())
    p.__file__ = own['path']
    raw = b'forbidden combined measurement sentinel'
    old_reg = save('research/old-registration.json', {'data_sha256': digest(raw)})
    parent_bank = save('output/parent-study/bank.npz', b'not an NPZ: inherited bytes never decoded')
    parent_manifest = {'files': {'output/parent-study/bank.npz': p.short(parent_bank)}}
    parent_pin = save(p.PARENT_MANIFEST, parent_manifest)
    p.PARENT_MANIFEST_SHA = parent_pin['sha256']
    p.PARENT_ARCHIVE_SHA = 'a'*64
    save(p.PARENT_RECEIPT, {'status': 'PASS', 'manifest_sha256': parent_pin['sha256'], 'archive_sha256': 'a'*64})
    save(p.LICENSE, b'CC BY 4.0 fabricated notice')
    prod = save('producer.py', b'# producer never executed')
    budget_plan = {'source_sha256': {'producer.py': prod['sha256']}, 'prerequisites': {
        'parent_bank': {'path': 'output/parent-study/bank.npz', 'sha256': parent_bank['sha256']}}}
    newreg = save('research/new-registration.json', budget_plan)
    budget_study = root/'output/new-study'
    newbank = save('output/new-study/fit-data.npz', b'not an NPZ: fabricated FIT-only cache')
    new_inputs = {'files': {'fit-data.npz': p.short(newbank)}}
    na = save('output/new-audit/audit.json', {'status': 'PASS', 'agreement': True,
             'study': str(budget_study), 'inputs': new_inputs})
    study = root/'output/child-study'
    bank = save('output/child-study/bank.npz', b'not an NPZ: fabricated forecast')
    summary = save('output/child-study/summary.json', {'status': 'FACTORIAL_INCOMPLETE'})
    engineering = root/p.ENGINEERING[1]
    process = save(str((engineering/'original-process-01/process.json').relative_to(root)), {'status': 'failed'})
    freeze = save(str((engineering/'auditor-freeze.json').relative_to(root)), {})
    qual = save(str((engineering/'qualification.json').relative_to(root)), {})
    for name in p.ENGINEERING:
        save(name+'/qualification-01/receipt.json', {'status': 'FAIL', 'original': True})
        save(name+'/qualification-02/receipt.json', {'status': 'PASS'})
    plan = {'source_sha256': {'producer.py': prod['sha256']}, 'prerequisites': {
        'new_registration': {'path': 'research/new-registration.json', 'sha256': newreg['sha256']},
        'old_registration': {'path': 'research/old-registration.json', 'sha256': old_reg['sha256']},
        'new_audit': {'path': 'output/new-audit/audit.json', 'sha256': na['sha256']}}}
    reg = save('research/registration.json', plan)
    inputs = {'registration': reg, 'process': process, 'freeze': freeze, 'qualification': qual,
        'files': {'bank.npz': p.short(bank), 'summary.json': p.short(summary)}, 'parents': {'new': new_inputs}}
    audit_path = root/'output/child-audit/audit.json'
    audit = {'status': 'PASS', 'agreement': True, 'study': str(study), 'scientific_status': 'REFERENCE_INCOMPLETE',
             'inputs': inputs, 'results': {'matrix_status': 'FACTORIAL_INCOMPLETE'}}
    ap = save(str(audit_path.relative_to(root)), audit)
    closure = {'state': 'EXITED', 'observed_exit_code': 0, 'success': True, 'audit_status': 'PASS', 'agreement': True,
        'sources_unchanged': True, 'inputs_unchanged': True, 'error': None, 'closure_error': None, 'evidence_errors': {},
        'audit_output': ap, 'scientific_status': audit['scientific_status'], 'sources_before': {
            name: p.descriptor(root/name) for name in p.HELD}, 'inputs_before': {'bank': bank},
        'admission_before': inputs, 'admission_after': copy.deepcopy(inputs), 'registration': reg,
        'freeze': freeze, 'evaluation_process': process, 'qualification': qual,
        'helper': save(str((engineering/'audit-original.py').relative_to(root)), b'# wrapper'),
        'log': save(str((engineering/'audit-process.log').relative_to(root)), b'PASS'),
        'timeout_seconds': 3600, 'rss_cap_bytes': 32*1024**3,
        'command': ['.venv/bin/python', '-u', p.AUDITOR, '--study', str(study.relative_to(root)),
                    '--process', str(Path(process['path']).relative_to(root)), '--freeze', str(Path(freeze['path']).relative_to(root)),
                    '--output', str(audit_path.relative_to(root))]}
    closure['sources_after'] = copy.deepcopy(closure['sources_before'])
    closure['inputs_after'] = copy.deepcopy(closure['inputs_before'])
    closure_path = engineering/'audit-process.json'
    save(str(closure_path.relative_to(root)), closure)
    save('admitted.json', {'plan': plan, 'inputs': inputs, 'paths': {
        'new_audit': na['path'], 'new_registration': newreg['path'], 'old_registration': old_reg['path']}})
    return p, (study, audit_path, closure_path, root/'output/absent-plots'), closure, save, raw


def main():
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        p, args, closure, save, raw = fixture(root)
        result = p.run(*args, root/'release-one')
        again = p.run(*args, root/'release-two')
        assert result['scientific_status'] == 'REFERENCE_INCOMPLETE'
        assert result['matrix_status'] == 'FACTORIAL_INCOMPLETE'
        assert result['archive']['sha256'] == again['archive']['sha256']
        checks.extend(['full_opaque_admission_and_roundtrip', 'incomplete_status_preserved', 'deterministic_archive'])
        manifest = p.read(root/'release-one/evidence-manifest.json')
        assert 'output/parent-study/bank.npz' not in manifest['files']
        assert set(manifest['parent_dependency']['required_direct_files']) == {'output/parent-study/bank.npz'}
        assert all(name+'/qualification-01/receipt.json' in manifest['files'] for name in p.ENGINEERING)
        checks.extend(['parent_bytes_not_rebundled', 'failed_qualification_retained'])
        expect_failure(lambda: p.run(*args, root/'release-one'))
        checks.append('exclusive_output')
        for key, value in [('state', 'running'), ('success', False)]:
            bad = copy.deepcopy(closure); bad[key] = value
            save(str(args[2].relative_to(root)), bad)
            expect_failure(lambda: p.gather(*args))
            save(str(args[2].relative_to(root)), closure)
        checks.append('unclosed_or_failed_audit_rejected')
        bad = copy.deepcopy(closure); bad['admission_after'] = {}
        save(str(args[2].relative_to(root)), bad)
        expect_failure(lambda: p.gather(*args))
        save(str(args[2].relative_to(root)), closure)
        checks.append('admission_tree_tamper')
        bank = root/'output/child-study/bank.npz'; original = bank.read_bytes(); bank.write_bytes(b'tampered')
        expect_failure(lambda: p.gather(*args)); bank.write_bytes(original)
        checks.append('child_bank_hash_tamper')
        extra = root/p.ENGINEERING[0]/'combined_data.npz'; extra.write_bytes(raw)
        expect_failure(lambda: p.gather(*args)); extra.unlink()
        extra = root/p.ENGINEERING[0]/'renamed.npz'; extra.write_bytes(raw)
        expect_failure(lambda: p.gather(*args)); extra.unlink()
        checks.extend(['raw_name_exclusion', 'raw_content_hash_exclusion'])
        extra = root/p.ENGINEERING[0]/'link'; extra.symlink_to(bank)
        expect_failure(lambda: p.gather(*args)); extra.unlink()
        checks.append('symlink_exclusion')
        bank.write_bytes(b'drift after manifest')
        expect_failure(lambda: p.archive(manifest, root/'partial-release'))
        assert (root/'partial-release/evidence.tar.gz').exists()
        bank.write_bytes(original)
        checks.append('midarchive_drift_retains_partial_attempt')
        for name in ('output/vendor/unread.bin', 'output/examples/u_300mV_train.npy', 'outside/../bad'):
            expect_failure(lambda name=name: p.regular(root/name))
        checks.append('forbidden_trees_and_traversal')
        assert p.gather(*args) == manifest
        checks.append('fixture_restored_byte_exact')
    print(json.dumps({'status': 'PASS', 'checks': checks, 'count': len(checks),
                      'scope': 'Only fabricated JSON and intentionally invalid NPZ bytes; no scientific or measured imports.'}, indent=2))


if __name__ == '__main__':
    main()
