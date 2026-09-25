# SPDX-License-Identifier: GPL-3.0-or-later
"""A fresh, larger-budget FIT-only attempt from the exact original initialization."""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import fit_nllfr as base

ROOT = base.ROOT
VERSION = 'fsm-author-nllfr-budget-study-v1'
EXPERIMENT = {**base.EXPERIMENT, 'max_iter': 100000}
ENV = base.ENV
require, read, sha, pin, write, event = base.require, base.read, base.sha, base.pin, base.write, base.event
PARENT_FILES = {'parent_initial_zip': 'initial.zip', 'parent_initial_npz': 'initial.npz',
                'parent_original_bla': 'original-bla.npz', 'parent_fit_data': 'fit-data.npz',
                'parent_solver_trace': 'solver-trace.npz'}
REQUIRED = {*PARENT_FILES, 'runtime_preflight', 'parent_registration', 'parent_process',
            'parent_evaluation_process', 'parent_audit', 'parent_audit_process',
            'producer_qualification', 'source_review'}
MIN_SOURCES = {'research/fsm_author/scripts/fit_nllfr_budget.py',
               'research/fsm_author/scripts/run_nllfr_budget_study.py',
               'research/fsm_author/tests/test_fit_nllfr_budget.py',
               'research/fsm_author/scripts/fit_nllfr.py', 'scripts/audit_fsm_author_nllfr.py',
               'research/fsm_author/src/openjev_fsm_author/nllfr.py',
               'research/fsm_author/src/openjev_fsm_author/benchmark.py',
               'research/fsm_author/uv.lock', 'research/fsm_author/pyproject.toml'}
FIT_KEYS = ('raw_fit_u', 'raw_fit_y', 'u', 'y', 'U', 'Y', 'f_idx', 'training_x0', 'independent_x0')
TRACE_KEYS = ('loss_history', 'iter_times', 'iter_count', 'author_stop_flag', 'wall_time')


def relative(value):
    require(isinstance(value, str) and not Path(value).is_absolute() and '\\' not in value
            and all(p not in ('', '.', '..') for p in value.split('/')), 'canonical repository-relative path')
    return ROOT/value


def prerequisite(cfg, name):
    return relative(cfg['prerequisites'][name]['path'])


def parent_admission(cfg):
    """Reuse only the held auditor's opaque metadata admission, never its replay."""
    sys.path.insert(0, str(ROOT/'scripts'))
    import audit_fsm_author_nllfr as parent
    audit = read(prerequisite(cfg, 'parent_audit'))
    process = prerequisite(cfg, 'parent_process')
    _, inputs, _, terminal, evaluation = parent.authenticate(Path(audit['study']), process,
        prerequisite(cfg, 'parent_evaluation_process'))
    require(audit['status'] == 'PASS' and audit['agreement'] is True and audit['inputs'] == inputs,
            'original independent parent audit admission')
    require(terminal['status'] == evaluation['status'] == 'completed'
            and terminal['observed_exit_code'] == evaluation['observed_exit_code'] == 0,
            'closed original parent fit/evaluation')
    require(inputs['registration'] == pin(prerequisite(cfg, 'parent_registration'))
            and inputs['process'] == pin(process), 'parent registration/process role')
    closure = read(prerequisite(cfg, 'parent_audit_process'))
    require(closure['state'] == 'EXITED' and closure['observed_exit_code'] == 0 and closure['success'] is True
            and closure['sources_unchanged'] is True and closure['inputs_unchanged'] is True
            and closure['error'] is None and closure['closure_error'] is None
            and closure['audit_output'] == pin(prerequisite(cfg, 'parent_audit'))
            and closure['fit_process'] == pin(process)
            and closure['evaluation_process'] == inputs['evaluation_process'], 'original parent audit closure')
    require(closure['sources_before'] == closure['sources_after']
            and closure['inputs_before'] == closure['inputs_after'], 'parent audit closure identities')
    for value in (*closure['sources_before'].values(), *closure['inputs_before'].values(), closure['log']):
        require(pin(value['path']) == value, 'parent audit original input/log drift')
    require(closure['sources_before']['scripts/audit_fsm_author_nllfr.py'] == inputs['source'], 'parent audit source join')
    require(audit['results']['fit_status'] == 'iteration_cap_reached'
            and read(Path(audit['study'])/'fit.json')['iterations'] == 10000, 'original10000-step capped fit required')
    for key, name in PARENT_FILES.items():
        path = prerequisite(cfg, key)
        require(path.resolve() == (Path(audit['study'])/name).resolve(), 'exact original input role: '+key)
        require({k: pin(path)[k] for k in ('bytes', 'sha256')} == inputs['files'][name], 'audited parent payload: '+key)
    return inputs


def metadata_admission(registration, output=None):
    cfg = read(registration)
    require(cfg['version'] == VERSION and cfg['experiment'] == EXPERIMENT, 'budget-only recipe drift')
    require(MIN_SOURCES <= set(cfg['source_sha256']) and REQUIRED <= set(cfg['prerequisites']), 'required frozen inputs')
    require(set(cfg) == {'version', 'experiment', 'source_sha256', 'prerequisites', 'output', 'process_directory'},
            'FIT-only registration schema; no data/evaluation/selection phase')
    for name, expected in cfg['source_sha256'].items():
        require(sha(relative(name)) == expected, 'source drift: '+name)
    for value in cfg['prerequisites'].values():
        require(sha(relative(value['path'])) == value['sha256'], 'prerequisite drift')
    if output is not None:
        require(output.resolve() == relative(cfg['output']).resolve(), 'registered output required')
    for name in ('runtime_preflight', 'producer_qualification', 'source_review'):
        require(read(prerequisite(cfg, name))['status'] == 'PASS', 'failed prerequisite: '+name)
    q = read(prerequisite(cfg, 'producer_qualification'))
    definition = read(q['definition']['path'])
    require(q['definition'] == pin(q['definition']['path'])
            and q['sources_before'] == q['sources_after'] == definition['sources']
            and MIN_SOURCES <= set(q['sources_before']), 'qualification source closure')
    for name, value in q['sources_before'].items():
        require(value == pin(relative(name)) and value['sha256'] == cfg['source_sha256'][name], 'qualified source drift')
        snapshot = pin(Path(q['definition']['path']).parent/'source'/name)
        require(all(snapshot[k] == value[k] for k in ('sha256', 'bytes')), 'qualified source snapshot')
    require([r['command'] for r in q['commands']] == definition['commands']
            and any('pytest' in r['command'] and 'research/fsm_author/tests/test_fit_nllfr_budget.py' in r['command']
                    for r in q['commands']), 'qualified command scope')
    for row in q['commands']:
        require(row['returncode'] == 0 and pin(row['log']['path']) == row['log'], 'qualification original log')
    review = read(prerequisite(cfg, 'source_review'))
    require(review['qualification'] == cfg['prerequisites']['producer_qualification']
            and review['reviewed_source_sha256'] == cfg['source_sha256'], 'peer-reviewed source identity')
    parent_admission(cfg)
    return cfg


def identities(cfg):
    # This inherited helper validates exact installed packages, source bytes,
    # Python executable, CPU/x64 and the fixed pre-interpreter environment.
    return base.identities(cfg)


def load_arrays(path, names, np):
    with np.load(path, allow_pickle=False) as source:
        require(len(source.files) == len(set(source.files)) and set(source.files) == set(names), 'exact saved array roster')
        return {key: source[key].copy(order='K') for key in names}


def prefix_comparison(current, original, np, *, required=10000):
    require(current.ndim == original.ndim == 1 and original.shape == (required,), 'loss prefix geometry')
    require(current.dtype == original.dtype == np.float64, 'float64 loss prefix')
    compared = min(len(current), required)
    equal = current[:compared] == original[:compared]
    positions = np.flatnonzero(~equal)
    with np.errstate(over='ignore', invalid='ignore'):
        difference = float(np.max(np.abs(current[:compared]-original[:compared]))) if compared else None
    return {'required': required, 'compared': compared,
            'exact': compared == required and bool(equal.all()),
            'first_difference': int(positions[0]) if len(positions) else None,
            'max_absolute_difference': difference if difference is None or np.isfinite(difference) else None}


def fit_status(author_stop, prefix):
    optimizer_status = 'complete' if author_stop else 'iteration_cap_reached'
    return ('prefix_mismatch' if not prefix['exact'] else optimizer_status), optimizer_status


def run(registration, output):
    cfg = metadata_admission(registration, output)
    import equinox as eqx
    import freq_statespace as fss
    import jax
    import numpy as np
    import optimistix as optx
    from freq_statespace import _nonlin_lfr

    from openjev_fsm_author import nllfr
    from openjev_fsm_author.benchmark import NAMES as BLA_NAMES
    from openjev_fsm_author.benchmark import export_model

    before = identities(cfg)
    write(output/'identity-before.json', before)
    allowed = {prerequisite(cfg, key).resolve() for key in PARENT_FILES}
    def guard(name, args):
        if name == 'socket.connect':
            raise RuntimeError('network forbidden during FIT-only budget attempt')
        if name == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix.lower() in ('.npz', '.npy', '.mat', '.zip'):
                require(path in allowed or path.is_relative_to(output), 'unregistered numerical file access')
    sys.addaudithook(guard)
    # Copy original bytes, never reinitialize and never load the parent final.
    for key, name in PARENT_FILES.items():
        target = 'parent-solver-trace.npz' if name == 'solver-trace.npz' else name
        (output/target).write_bytes(prerequisite(cfg, key).read_bytes())
    initial = fss.load_model(output/'initial.zip')
    require(type(initial.ts) is type(initial._bla.ts) is float, 'both sample intervals must remain Python float')
    live = base.raw_arrays(initial, np)
    np.savez_compressed(output/'initial-loaded.npz', **live)
    initial_arrays = nllfr.export(initial)
    original = export_model(initial._bla)
    expected = load_arrays(output/'initial.npz', nllfr.NAMES, np)
    expected_bla = load_arrays(output/'original-bla.npz', BLA_NAMES, np)
    for actual, parent in ((initial_arrays, expected), (original, expected_bla)):
        for key in parent:
            np.testing.assert_array_equal(actual[key], parent[key])
    require(nllfr.validate(initial_arrays) == (28, 16, 8, 64), 'original full architecture required')
    data = load_arrays(output/'fit-data.npz', FIT_KEYS, np)
    require(data['raw_fit_u'].shape == data['raw_fit_y'].shape == (8192, 3, 6, 2), 'original FIT geometry')
    require(data['raw_fit_u'].dtype == data['raw_fit_y'].dtype == np.float64, 'original float64 FIT required')
    require(all(np.isfinite(v).all() for v in data.values()), 'finite original FIT arrays')
    np.testing.assert_array_equal(data['f_idx'], np.arange(1, 3840))
    author_data = fss.create_data_object(data['raw_fit_u'], data['raw_fit_y'], data['f_idx'], 6400.)
    for key in ('u_mean', 'u_std', 'y_mean', 'y_std'):
        np.testing.assert_array_equal(getattr(author_data.norm, key), initial_arrays[key])
    for name, actual in (('u', author_data.time.u), ('y', author_data.time.y), ('U', author_data.freq.U), ('Y', author_data.freq.Y)):
        np.testing.assert_array_equal(np.asarray(actual), data[name])
    theta, args = _nonlin_lfr._prepare_nonlin_optimization(author_data, initial, 820, False)
    leaves = jax.tree_util.tree_leaves(eqx.filter(theta, eqx.is_inexact_array))
    require(len(leaves) == 14 and sum(v.size for v in leaves) == 7473
            and all(v.dtype == np.float64 for v in leaves), 'original14-leaf/7473-parameter partition')
    np.testing.assert_array_equal(np.asarray(args.x0), data['training_x0'])
    np.testing.assert_array_equal(base.independent_x0(original, data['U'], np), data['independent_x0'])
    require(args.u.shape == (9012, 3, 6) and data['Y'].shape == (4097, 3, 6), 'original full objective geometry')
    parent_trace = load_arrays(output/'parent-solver-trace.npz', TRACE_KEYS, np)
    require(int(parent_trace['iter_count']) == 10000 and not bool(parent_trace['author_stop_flag'])
            and parent_trace['loss_history'].shape == (10000,), 'original capped trace required')
    initial_loss = base.independent_loss(initial_arrays, data['u'], data['Y'], data['independent_x0'], nllfr, np)
    write(output/'fresh-start.json', {'original_initial_zip': pin(prerequisite(cfg, 'parent_initial_zip')),
        'original_initial_arrays': pin(prerequisite(cfg, 'parent_initial_npz')), 'all_initial_arrays_exact': True,
        'original_bla_exact': True, 'fit_normalization_spectra_and_x0_exact': True,
        'trainable_leaves': 14, 'trainable_scalars': 7473, 'optimizer_initialization': 'fresh unchanged BFGS',
        'parent_final_loaded': False, 'dev_evaluation_calls': 0, 'model_selection_calls': 0,
        'initial_fit_loss': initial_loss})
    write(output/'admission.json', {'input_files': {k: pin(prerequisite(cfg, k)) for k in PARENT_FILES},
        'raw_archive_decodes': 0, 'fit_records': 12, 'dev_evaluation_calls': 0,
        'scope': 'Only previously audited FIT tensors and original model initialization are numerically loaded'})
    del theta, args, leaves
    event(output, 'optimization_started', initial_fit_loss=initial_loss, max_iter=100000)
    start = time.perf_counter()
    final, solve = fss.nonlin.optimize(initial, author_data, solver=optx.BFGS(rtol=1e-3, atol=1e-5),
        freq_weighting=False, max_iter=100000, print_every=-1, return_solve_details=True,
        offset=None, device='cpu')
    fss.save_model(final, output/'final.zip')
    np.savez_compressed(output/'final.npz', **base.raw_arrays(final, np))
    np.savez_compressed(output/'solver-trace.npz', loss_history=np.asarray(solve.loss_history),
        iter_times=np.asarray(solve.iter_times), iter_count=np.asarray(solve.iter_count),
        author_stop_flag=np.asarray(solve.converged), wall_time=np.asarray(solve.wall_time))
    elapsed = time.perf_counter()-start
    prefix = prefix_comparison(np.asarray(solve.loss_history), parent_trace['loss_history'], np)
    write(output/'prefix.json', prefix)
    arrays = nllfr.export(final)
    require(solve.loss_history.shape == solve.iter_times.shape == (solve.iter_count,)
            and 1 <= solve.iter_count <= 100000 and np.isfinite(solve.loss_history).all()
            and np.isfinite(solve.iter_times).all(), 'finite returned solver trace')
    for key in original:
        np.testing.assert_array_equal(export_model(final._bla)[key], original[key])
    for key in ('u_mean', 'u_std', 'y_mean', 'y_std', 'ts'):
        np.testing.assert_array_equal(arrays[key], original[key])
    reloaded = fss.load_model(output/'final.zip')
    np.savez_compressed(output/'roundtrip.npz', **base.raw_arrays(reloaded, np))
    np.savez_compressed(output/'roundtrip-original-bla.npz', **export_model(reloaded._bla))
    for actual, expected in ((nllfr.export(reloaded), arrays), (export_model(reloaded._bla), original)):
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    write(output/'serialization.json', {'all_live_arrays_bitwise_equal': True, 'original_bla_arrays_bitwise_equal': True,
        'files': {name: pin(output/name) for name in ('final.zip', 'final.npz', 'original-bla.npz',
                                                   'roundtrip.npz', 'roundtrip-original-bla.npz')}})
    final_loss = base.independent_loss(arrays, data['u'], data['Y'], data['independent_x0'], nllfr, np)
    require(final_loss <= initial_loss, 'independent FIT objective worsened')
    status, optimizer_status = fit_status(bool(solve.converged), prefix)
    fit = {'status': status, 'optimizer_status': optimizer_status, 'budget_only_attributable': prefix['exact'],
        'author_stop_flag': bool(solve.converged), 'iterations': int(solve.iter_count),
        'initial_fit_loss': initial_loss, 'final_fit_loss': final_loss, 'trainable_scalars': 7473,
        'optimization_and_preservation_compile_inclusive_seconds': elapsed,
        'author_reported_seconds': float(solve.wall_time), 'discarded_vendor_warmup_steps': 1,
        'stop_meaning': 'Author BFGS small-change boolean only; no stationarity/global-optimum certificate',
        'zip_roundtrip_all_numeric_arrays_bitwise_equal': True,
        'peak_ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'initial': pin(output/'initial.npz'), 'final': pin(output/'final.npz'), 'prefix': pin(output/'prefix.json')}
    write(output/'fit.json', fit)
    after = identities(cfg)
    require(before == after, 'runtime/source identity drift')
    write(output/'identity-after.json', after)
    summary = {'status': 'FIT_ONLY_COMPLETE' if status == 'complete' else 'FIT_ONLY_INCOMPLETE', 'fit': fit,
        'dev_evaluation_calls': 0, 'model_selection_calls': 0, 'raw_archive_decodes': 0,
        'scope': 'Fresh original-initialization optimizer; only max_iter increased. Exact prefix required for attribution; no performance claim.'}
    write(output/'summary.json', summary)
    event(output, 'fit_closed', status=summary['status'], iterations=fit['iterations'], prefix_exact=prefix['exact'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    cfg = metadata_admission(args.registration, output)
    require(all(os.environ.get(k) == v for k, v in ENV.items()), 'fixed environment required before interpreter start')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'started.json', {'registration': pin(args.registration), 'pid': os.getpid(), 'time_ns': time.time_ns()})
    for name in cfg['source_sha256']:
        dest = output/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(relative(name).read_bytes())
    try:
        run(args.registration, output)
    except Exception as exc:
        write(output/'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        if not (output/'summary.json').exists():
            write(output/'summary.json', {'status': 'FIT_ONLY_INCOMPLETE', 'error': f'{type(exc).__name__}: {exc}',
                'dev_evaluation_calls': 0, 'model_selection_calls': 0, 'raw_archive_decodes': 0})
        raise


if __name__ == '__main__':
    main()
