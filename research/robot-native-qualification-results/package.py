"""Metadata-only public copies of closed native qualification evidence."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/robot-native-qualification-results'
ENGINEERING = ROOT / 'output/native-robot-transition-engineering-v1'
RUNTIME = ROOT / 'output/robot-native-runtime-setup-v1'


def pin(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), path
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    attempts = [json.loads((ENGINEERING / f'attempt-0{i}/qualification.json').read_text()) for i in (1, 2)]
    assert attempts[0]['status'] == 'FAILED' and attempts[0]['counts'] == {'tests': 90, 'failed': 2, 'errors': 0, 'skipped': 0}
    assert attempts[1]['status'] == 'PASS' and attempts[1]['counts'] == {'tests': 110, 'failed': 0, 'errors': 0, 'skipped': 0}
    assert pin(ENGINEERING / 'attempt-01/qualification.json')['sha256'] == '69ee45ce8cf3e70c83b68ae9a95a4ee91f83a2a70e5d71f09a8ed5e6ab1c5732'
    assert pin(ENGINEERING / 'attempt-02/qualification.json')['sha256'] == '7bcdd4a41316b086ac494e50832266fb4aae461f9f6a132b4afce1af38338113'
    current = {name: pin(ROOT / name) for name in attempts[1]['sources_after']}
    assert current == {name: {k: item[k] for k in ('sha256', 'bytes')} for name, item in attempts[1]['sources_after'].items()}
    native_files = sorted(p for p in ENGINEERING.rglob('*') if p.is_file())
    runtime_files = sorted(p for p in RUNTIME.rglob('*') if p.is_file())
    original = {str(p.relative_to(ROOT)): pin(p) for p in native_files + runtime_files}
    selected, excluded = {}, {}
    for p in native_files:
        relative = p.relative_to(ENGINEERING)
        if p.suffix == '.dylib' or p.name == 'SumKernel.cpp':
            excluded[str(p.relative_to(ROOT))] = {**pin(p), 'reason': 'Compiled library; rebuild required' if p.suffix == '.dylib' else 'Downloaded third-party Torch source; URL/hash retained'}
        else:
            selected['qualification/' + str(relative)] = p
    safe_logs = {'compile-arithmetic-01.log', 'compiler-version-01.log',
                 'install-rust-std-01.log', 'install-rustc-01.log'}
    safe_sources = {'arithmetic.rs', 'fetch.py', 'install.py'}
    for p in runtime_files:
        relative = p.relative_to(RUNTIME)
        if len(relative.parts) == 1 and (p.suffix == '.json' or p.name in safe_logs | safe_sources):
            selected['runtime/' + str(relative)] = p
        else:
            excluded[str(p.relative_to(ROOT))] = {**pin(p), 'reason': 'Third-party toolchain/download/extraction or redundant installation listing; retained locally'}
    definition = json.loads((ENGINEERING / 'reduction-diagnostic-01/definition.json').read_text())
    external = {'torch_sum_kernel': {
        'url': definition['source_url'],
        'path': 'output/native-robot-transition-engineering-v1/reduction-diagnostic-01/SumKernel.cpp',
        **pin(ENGINEERING / 'reduction-diagnostic-01/SumKernel.cpp')},
        'torch_neon_header': {
            'url': 'https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h',
            'path': '.venv/lib/python3.12/site-packages/torch/include/ATen/cpu/vec/vec128/vec128_float_neon.h',
            **pin(ROOT / '.venv/lib/python3.12/site-packages/torch/include/ATen/cpu/vec/vec128/vec128_float_neon.h')}}
    for value in external.values():
        assert definition['pins'][value['path']] == value['sha256']
    setup = json.loads((RUNTIME / 'runtime-setup-01.json').read_text())
    installed = json.loads((RUNTIME / 'installed-manifest.json').read_text())
    installed_excluded = {str(Path(setup['prefix']) / name): {**item, 'basis': 'Original installed-manifest.json; not recopied or reinstalled'}
                          for name, item in installed['files'].items()}
    OUT.mkdir(parents=True, exist_ok=False)
    provenance = {}
    for relative, source in sorted(selected.items()):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert pin(destination) == original[str(source.relative_to(ROOT))]
        provenance[relative] = {'original_path': str(source.relative_to(ROOT)), **pin(destination)}
    shutil.copyfile(__file__, OUT / 'package.py')
    provenance['package.py'] = {'original_path': str(Path(__file__).relative_to(ROOT)), **pin(__file__)}
    write(OUT / 'exclusions.json', {
        'files': excluded, 'installed_toolchain_files': installed_excluded,
        'external_source_references': external,
        'scope': 'Exact omitted files within the two evidence roots, plus installed toolchain members from the original setup manifest. Other Python/runtime packages are external prerequisites, not bundled.',
        'rebuild_required': True,
    })
    write(OUT / 'provenance.json', {'copied_files': provenance, 'current_qualified_sources': current,
        'source_paths_are_repository_relative': True,
        'original_receipts_unmodified': True, 'original_absolute_paths_retained_in_receipts': True,
        'scope': 'Closed fabricated qualification and toolchain setup only. No measured robot arrays/checkpoints, training, or speed measurements.'})
    (OUT / 'README.md').write_text('''# Native robot inference qualification

**Attempt 02 passed 110 fabricated tests with zero skips at unchanged float32 tolerances, rtol=atol=1e-5.** This package establishes tested inference parity on the recorded host. It does not establish a speedup, measured-data accuracy, or an architecture improvement.

| Original attempt | Scope | Outcome |
| --- | --- | --- |
| [01](qualification/attempt-01/qualification.json) | Four structured families | 88 passed, 2 failed, 0 skips |
| [02](qualification/attempt-02/qualification.json) | Four structured families plus GRU32 | 110 passed, 0 failed, 0 skips |

The two original failures were initialized Householder rollouts at horizons 128 and 512. Scalar accumulation differed from Torch's four-lane CPU reduction, accumulating enough error to exceed the frozen criterion. The [source-derived diagnostic](qualification/reduction-diagnostic-01/result.json) matched Torch on 1,024 fabricated reductions and all 16 initial reflection reductions with the four-lane order. Only the Householder dot-product accumulation was changed for the repair. The second attempt also added the separately implemented GRU reference. Neither the tolerance nor the original failure was removed.

The tests cover predictions and final states, active bounds, saturated gates, finite affine overflow through saturating activations, invalid inputs, causal/chunked execution, live parameter export, explicit copies, empty horizons, and ownership. This is a hybrid implementation: Python validates/prepares structured weights on every request, including dense spectral norms; Rust executes the sequential rollout. No prepared-weight cache is used. Storage/work counters describe explicit numeric payloads, not peak process memory.

## Evidence and exclusions

- [File manifest](manifest.json): every public payload, SHA-256 and byte count, excluding the manifest itself.
- [Provenance](provenance.json): original source paths and the 17 qualified current source pins.
- [Exclusions](exclusions.json): exact omitted filenames, hashes, reasons, and external source URLs.
- [Runtime setup](runtime/runtime-setup-01.json): Rust 1.98.1, aarch64-apple-darwin, original installation/version/smoke metadata.
- [Build definition](qualification/attempt-02/build/definition.json): source pins and exact compiler commands. Optimization was `opt-level=3`, with no fast-math or target-CPU override.
- [Original process receipts](qualification/attempt-02/qualification.json): lint, build, and test closure joins. Their absolute paths describe the original machine and have not been rewritten.

Compiled libraries, compiler archives/installations, downloaded installer scripts, and downloaded Torch implementation files are intentionally absent. Their exact hashes remain in the receipts and exclusion index. Torch sources remain available at the primary URLs recorded there. Repo-owned build/qualification/diagnostic sources are included. The runtime's installed-file manifest records the excluded toolchain contents.

## Rebuilding

A compatible Python/Torch environment and an explicit Rust compiler are external prerequisites. Restore the repository-owned sources at the paths recorded in provenance, then run the included `scripts/build_robot_native.py` from that repository with `--rustc /absolute/path/to/rustc --output /new/exclusive/build-folder`. Set `ROBOT_TRANSITION_LIBRARY` to the rebuilt library and run both native test files with the five thread variables in the attempt-02 preflight set to `1`.

Do not rerun the historical attempt scripts against their original output directories. A rebuilt library needs its own fresh qualification receipt. The reduction repair follows the recorded Torch 2.14 CPU/ARM implementation; portability and bitwise binary reproducibility across hosts or toolchains are not claimed. No future benchmark result is included here.
''')
    all_payloads = {str(p.relative_to(OUT)): pin(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write(OUT / 'manifest.json', {'version': 'robot-native-qualification-public-package-v1',
        'files': all_payloads, 'file_count_excluding_manifest': len(all_payloads),
        'bytes_excluding_manifest': sum(p['bytes'] for p in all_payloads.values()),
        'manifest_self_excluded': True})
    actual = {str(p.relative_to(OUT)): pin(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    assert actual == all_payloads
    assert all(pin(ROOT / name) == value for name, value in current.items())
    assert all(pin(ROOT / name) == value for name, value in original.items())
    write(Path(__file__).parent / 'receipt.json', {
        'status': 'PASS', 'public_folder': str(OUT.relative_to(ROOT)),
        'manifest': pin(OUT / 'manifest.json'), 'all_original_evidence_unchanged': True,
        'current_qualified_sources_unchanged': True, 'public_copies_byte_identical': True,
        'public_files': len(all_payloads) + 1,
        'public_bytes': sum(p['bytes'] for p in all_payloads.values()) + (OUT / 'manifest.json').stat().st_size,
        'excluded_evidence_files': len(excluded), 'excluded_installed_toolchain_files': len(installed_excluded),
        'compiled_libraries_included': 0, 'measured_data_read': False})
    print((Path(__file__).parent / 'receipt.json').read_text())


if __name__ == '__main__':
    main()
