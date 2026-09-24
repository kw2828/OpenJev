"""Independently audit saved public linear-function reference evidence.

No producer, reference learner, model, world generator or RNG is imported.
Private matrices and function indices authenticate labels only. Predictive
routing is reconstructed exclusively from saved public examples and fitted
maps. Full-rank recovery uses QR; rank diagnostics and minimum-norm recovery
use an explicit truncated SVD, separately from the producer's lstsq routine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'function-reuse-reference-audit-v1'
GROUPS = (1, 3, 8, 16)
COHORTS, EPISODES, QUERIES, DIMENSION = 5, 64, 8, 8
DEMONSTRATIONS, FEWSHOTS, CHOICES = 16, 4, 6
RTOL = ATOL = 1e-10
FLOAT_FIELDS = {
    'matrices', 'basis_x', 'basis_y', 'fewshot_x', 'fewshot_y', 'query_x',
    'target', 'cost_vectors', 'maps', 'singular_values', 'block_residuals',
    'lookup_prediction', 'fewshot_prediction', 'fewshot_singular_values',
    'lookup_fit_seconds', 'lookup_query_seconds', 'fewshot_query_seconds',
}
INTEGER_FIELDS = {'true_index', 'ranks', 'selected_index', 'fewshot_ranks',
                  'lookup_choice', 'fewshot_choice'}
CONFIG = {'namespace': 440260924, 'cohorts': 5, 'blocks': [1, 3, 8, 16], 'cases': 64,
          'dimensions': 8, 'basis_examples': 16, 'fewshot_examples': 4, 'queries': 8, 'choices': 6}
SOURCES = {'scripts/function_reuse_reference_study.py', 'scripts/audit_function_reuse_reference.py',
           'src/openjev/research/function_reuse_reference.py', 'tests/test_function_reuse_reference.py',
           'tests/test_function_reuse_reference_study.py', 'tests/test_function_reuse_reference_audit.py',
           'research/function-reuse-reference-protocol.md'}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'ordinary evidence file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def close(left, right, np, message):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and np.isfinite(left).all()
            and np.isfinite(right).all()
            and np.allclose(left, right, rtol=RTOL, atol=ATOL), message)


def independent_fit(x, y, np):
    """Explicit rank rule; QR for full column rank, SVD minimum norm otherwise."""
    require(x.ndim == y.ndim == 2 and x.shape[0] == y.shape[0], 'fit row alignment')
    u, singular, vh = np.linalg.svd(x, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(x.shape) * singular[0]
    keep = singular > cutoff
    rank = int(np.count_nonzero(keep))
    if rank == x.shape[1]:
        q, r = np.linalg.qr(x, mode='reduced')
        coefficients = np.linalg.solve(r, q.T @ y)
        method = 'qr'
    elif rank:
        coefficients = vh[keep].T @ ((u[:, keep].T @ y) / singular[keep, None])
        method = 'svd_minimum_norm'
    else:
        coefficients = np.zeros((x.shape[1], y.shape[1]), np.float64)
        method = 'svd_zero_rank'
    require(np.isfinite(coefficients).all(), 'finite independently recovered coefficients')
    return coefficients, singular, rank, method


def public_query(maps, x, y, query, np):
    """Match the frozen public routing reduction, without private identity."""
    error = np.einsum('fd,kdo->kfo', x, maps) - y[None]
    residuals = np.mean(error ** 2, axis=(1, 2))
    selected = int(np.argmin(residuals))
    return selected, residuals, query @ maps[selected]


def validate_arrays(data, k, np, *, episodes=EPISODES, queries=QUERIES):
    e, q, d, b, f, c = episodes, queries, DIMENSION, DEMONSTRATIONS, FEWSHOTS, CHOICES
    shapes = {
        'matrices': (e, k, d, d), 'basis_x': (e, k, b, d), 'basis_y': (e, k, b, d),
        'fewshot_x': (e, q, f, d), 'fewshot_y': (e, q, f, d),
        'query_x': (e, q, d), 'target': (e, q, d), 'cost_vectors': (e, q, c, d),
        'true_index': (e, q), 'maps': (e, k, d, d), 'singular_values': (e, k, d),
        'ranks': (e, k), 'selected_index': (e, q), 'block_residuals': (e, q, k),
        'lookup_prediction': (e, q, d), 'fewshot_prediction': (e, q, d),
        'fewshot_ranks': (e, q), 'fewshot_singular_values': (e, q, f),
        'lookup_choice': (e, q), 'fewshot_choice': (e, q),
        'lookup_fit_seconds': (e,), 'lookup_query_seconds': (e,), 'fewshot_query_seconds': (e,),
    }
    require(set(data) == FLOAT_FIELDS | INTEGER_FIELDS == set(shapes), 'complete exact saved array schema')
    for name, shape in shapes.items():
        value = data[name]
        require(isinstance(value, np.ndarray) and value.shape == shape
                and value.dtype == np.dtype('float64' if name in FLOAT_FIELDS else 'int64')
                and np.isfinite(value).all(), 'shape/dtype/finiteness: ' + name)
    for name in ('true_index', 'selected_index'):
        require(((data[name] >= 0) & (data[name] < k)).all(), 'valid block index: ' + name)
    for name in ('lookup_choice', 'fewshot_choice'):
        require(((data[name] >= 0) & (data[name] < c)).all(), 'valid decision index: ' + name)
    require(((data['ranks'] >= 0) & (data['ranks'] <= d)).all()
            and ((data['fewshot_ranks'] >= 0) & (data['fewshot_ranks'] <= f)).all(), 'rank range')
    for name in ('singular_values', 'fewshot_singular_values'):
        require((data[name] >= 0).all() and (np.diff(data[name], axis=-1) <= 0).all(), 'ordered singular values')
    require((data['block_residuals'] >= 0).all(), 'nonnegative public residuals')
    for name in ('lookup_fit_seconds', 'lookup_query_seconds', 'fewshot_query_seconds'):
        require((data[name] >= 0).all(), 'recorded elapsed time is nonnegative')
    close(np.sum(data['cost_vectors'] ** 2, axis=-1), np.ones((e, q, c)), np, 'unit cost vectors')


def reconstruct(data, k, np, *, check=lambda: None):
    """Validate every equation, fit, public route, prediction and decision."""
    episodes, queries = data['query_x'].shape[:2]
    validate_arrays(data, k, np, episodes=episodes, queries=queries)
    methods = {'qr': 0, 'svd_minimum_norm': 0, 'svd_zero_rank': 0}
    exact_ties = 0
    for episode in range(episodes):
        check()
        independent = []
        for block in range(k):
            x, y = data['basis_x'][episode, block], data['basis_y'][episode, block]
            close(y, x @ data['matrices'][episode, block], np, 'private generated demonstration equation')
            matrix, singular, rank, method = independent_fit(x, y, np)
            independent.append(matrix); methods[method] += 1
            close(data['maps'][episode, block], matrix, np, 'independent public block coefficients')
            close(data['singular_values'][episode, block], singular, np, 'independent block singular values')
            require(data['ranks'][episode, block] == rank, 'independent block rank')
        for query in range(queries):
            x, y, request = (data[name][episode, query] for name in ('fewshot_x', 'fewshot_y', 'query_x'))
            # The private identity below only checks supplied labels. It never
            # enters public_query, either independent fit, or a decision.
            truth = data['matrices'][episode, data['true_index'][episode, query]]
            close(y, x @ truth, np, 'private few-shot label equation')
            close(data['target'][episode, query], request @ truth, np, 'private query target equation')
            selected, residuals, prediction = public_query(data['maps'][episode], x, y, request, np)
            close(data['block_residuals'][episode, query], residuals, np, 'public block residuals')
            require(data['selected_index'][episode, query] == selected, 'public lowest-index argmin routing')
            close(data['lookup_prediction'][episode, query], prediction, np, 'saved-map lookup prediction')
            close(prediction, request @ independent[selected], np, 'independent reconstructed lookup prediction')
            exact_ties += int(np.count_nonzero(residuals == residuals[selected]) > 1)
            matrix, singular, rank, method = independent_fit(x, y, np)
            methods[method] += 1
            close(data['fewshot_prediction'][episode, query], request @ matrix, np, 'independent minimum-norm prediction')
            close(data['fewshot_singular_values'][episode, query], singular, np, 'few-shot singular values')
            require(data['fewshot_ranks'][episode, query] == rank, 'few-shot numerical rank')
            for arm in ('lookup', 'fewshot'):
                costs = data['cost_vectors'][episode, query] @ data[arm + '_prediction'][episode, query]
                require(data[arm + '_choice'][episode, query] == int(np.argmin(costs)), 'public predicted cost decision')
    return {'block_fits': episodes*k, 'fewshot_fits': episodes*queries,
            'queries': episodes*queries, 'methods': methods, 'exact_residual_tie_queries': exact_ties}


def metrics(data, np):
    """Recompute all-query means without selecting favorable ranks or cases."""
    true_costs = np.einsum('eqcd,eqd->eqc', data['cost_vectors'], data['target'])
    true_choice = np.argmin(true_costs, axis=-1)
    best = np.min(true_costs, axis=-1)
    row = {}
    for arm in ('lookup', 'fewshot'):
        error = data[arm + '_prediction'] - data['target']
        chosen_cost = np.take_along_axis(true_costs, data[arm + '_choice'][..., None], axis=-1)[..., 0]
        row[arm + '_mse'] = float(np.mean(error ** 2))
        row[arm + '_regret'] = float(np.mean(chosen_cost-best))
        row[arm + '_action_accuracy'] = float(np.mean(data[arm + '_choice'] == true_choice))
    row['retrieval_accuracy'] = float(np.mean(data['selected_index'] == data['true_index']))
    singular = data['singular_values']
    ratios = np.divide(singular[..., -1], singular[..., 0], out=np.zeros_like(singular[..., 0]), where=singular[..., 0] > 0)
    row['min_relative_singular_value'] = float(np.min(ratios))
    row['full_rank'] = bool(np.all(data['ranks'] == DIMENSION))
    return row


def classify(rows):
    require(len(rows) == 20 and {(row['cohort'], row['blocks']) for row in rows}
            == {(cohort, blocks) for cohort in range(COHORTS) for blocks in GROUPS}, 'all20 groups required for classification')
    conditions = {}
    for row in rows:
        prefix = f"cohort-{row['cohort']:02d}/k-{row['blocks']:02d}/"
        checks = row_conditions(row)
        for key, value in checks.items():
            require(prefix+key not in conditions, 'unique case-group condition')
            conditions[prefix+key] = bool(value)
    return {'status': 'LINEAR_TASK_SOLVED_BY_CLASSICAL_REFERENCE' if all(conditions.values()) else 'REFERENCE_NOT_SOLVED',
            'conditions': conditions}


def row_conditions(row):
    return {'full_rank': row['full_rank'] is True, 'numerical_mse': row['lookup_mse'] <= 1e-12,
            'retrieval_exact': row['retrieval_accuracy'] == 1., 'numerical_regret': row['lookup_regret'] <= 1e-10}


def summarize(data, cohort, blocks, np):
    row = {'cohort': cohort, 'blocks': blocks, 'cases': len(data['target']),
           'queries': int(data['target'].shape[0]*data['target'].shape[1]), **metrics(data, np)}
    for key in ('lookup_fit_seconds', 'lookup_query_seconds', 'fewshot_query_seconds'):
        row[key] = float(np.sum(data[key]))
    row['map_array_bytes_per_context'] = blocks*DIMENSION*DIMENSION*8
    row['raw_basis_array_bytes_per_context'] = 2*blocks*DEMONSTRATIONS*DIMENSION*8
    row['diagnostic_array_bytes_per_context'] = blocks*DIMENSION*8+blocks*8+blocks
    row['conditions'] = row_conditions(row)
    return row


def compare(left, right, name='summary'):
    if isinstance(right, dict):
        require(isinstance(left, dict) and left.keys() == right.keys(), name + ' keys')
        for key in right:
            compare(left[key], right[key], name + '/' + key)
    elif isinstance(right, list):
        require(isinstance(left, list) and len(left) == len(right), name + ' list')
        for index, value in enumerate(right):
            compare(left[index], value, name + '/' + str(index))
    elif type(right) is float:
        require(type(left) in (float, int) and math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=RTOL, abs_tol=ATOL), name + ' scalar')
    else:
        require(type(left) is type(right) and left == right, name + ' exact value')


def admit(folder):
    """Authenticate byte inventories and fixed configuration before array reads."""
    folder = Path(folder).resolve()
    registration = json.loads((folder/'registration.json').read_text())
    require(registration['config'] == CONFIG and registration['seconds_cap'] == 120
            and registration['run_attempts'] == 1 and registration['data_selection'] == 'none'
            and registration['training_updates'] == 0 and set(registration['sources']) == SOURCES,
            'complete fixed prospective registration')
    for name, pin in registration['sources'].items():
        require(descriptor(ROOT/name) == descriptor(folder/'sources'/name) == pin, 'original frozen source and snapshot: ' + name)
    actual_snapshot = {str(p.relative_to(folder/'sources')): descriptor(p)
                       for p in (folder/'sources').rglob('*') if p.is_file()}
    require(actual_snapshot == registration['sources'], 'exact source snapshot inventory')
    runtime = registration['runtime']
    require(runtime['python'] == sys.version and runtime['executable'] == sys.executable
            and runtime['platform'] == platform.platform()
            and runtime['threads'] == {key: '1' for key in THREADS}
            and all(os.environ.get(key) == '1' for key in THREADS), 'unchanged registered runtime/thread environment')
    receipt = json.loads((folder/'run/receipt.json').read_text())
    require(receipt['status'] == 'COMPLETE' and receipt['registration'] == descriptor(folder/'registration.json')
            and type(receipt['elapsed_seconds']) in (float, int)
            and 0 < receipt['elapsed_seconds'] <= 120, 'original complete run within its fixed cap')
    files = {f'run/cohort-{cohort:02d}-k-{blocks:02d}.npz' for cohort in range(COHORTS) for blocks in GROUPS}
    files.update(('run/started.json', 'run/summary.json'))
    require(set(receipt['files']) == files, 'exact20-group and metadata receipt roster')
    actual = {str(p.relative_to(folder)) for p in (folder/'run').iterdir()}
    require(actual == files | {'run/receipt.json'} and all((folder/name).is_file() for name in actual), 'no omitted/failed/additional run payload')
    for name, pin in receipt['files'].items():
        require(descriptor(folder/name) == pin, 'original producer payload bytes: ' + name)
    started = json.loads((folder/'run/started.json').read_text())
    require(started['registration'] == descriptor(folder/'registration.json'), 'original start registration join')
    return {'registration': descriptor(folder/'registration.json'), 'plan': registration,
            'producer_receipt': descriptor(folder/'run/receipt.json'), 'receipt': receipt}


def audit(folder, *, check=lambda: None):
    """Audit saved evidence once; the caller retains original process closure."""
    folder = Path(folder).resolve()
    require(not (folder/'audit.json').exists() and not (folder/'audit.receipt.json').exists(), 'exclusive saved-output audit')
    started = time.perf_counter()
    admission = admit(folder)
    import numpy as np
    require(np.__version__ == admission['plan']['runtime']['numpy'], 'registered NumPy runtime')
    summary = json.loads((folder/'run/summary.json').read_text())
    require(set(summary) == {'config', 'rows', 'training_updates', 'external_model_calls', 'status'}
            and summary['config'] == CONFIG and summary['training_updates'] == summary['external_model_calls'] == 0,
            'exact producer summary without training or external models')
    require(len(summary['rows']) == 20 and [(r['cohort'], r['blocks']) for r in summary['rows']]
            == [(cohort, blocks) for cohort in range(COHORTS) for blocks in GROUPS], 'complete fixed group/cohort order')
    rows, counts = [], {'npz_decodes': 0, 'array_loads': 0, 'block_fits': 0, 'fewshot_fits': 0, 'queries': 0,
                       'qr_fits': 0, 'svd_minimum_norm_fits': 0, 'svd_zero_rank_fits': 0,
                       'exact_residual_tie_queries': 0, 'model_calls': 0, 'generator_calls': 0,
                       'optimizer_calls': 0, 'rng_replays': 0}
    for saved in summary['rows']:
        check()
        cohort, blocks = saved['cohort'], saved['blocks']
        with np.load(folder/f'run/cohort-{cohort:02d}-k-{blocks:02d}.npz', allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)) and set(archive.files) == FLOAT_FIELDS | INTEGER_FIELDS, 'unique complete archive array keys')
            data = {name: archive[name].copy() for name in archive.files}
        counts['npz_decodes'] += 1; counts['array_loads'] += len(data)
        validate_arrays(data, blocks, np)
        work = reconstruct(data, blocks, np, check=check)
        for name in ('block_fits', 'fewshot_fits', 'queries', 'exact_residual_tie_queries'):
            counts[name] += work[name]
        for method, count in work['methods'].items():
            counts[method+'_fits'] += count
        expected = summarize(data, cohort, blocks, np)
        compare(saved['conditions'], row_conditions(saved), 'producer metrics and decisions self-consistent')
        compare(saved, expected, 'independent all-case row')
        rows.append(expected)
    result = classify(rows)
    require(summary['status'] == result['status'] and len(result['conditions']) == 80, 'unchanged all80-condition scientific classification')
    require(counts['npz_decodes'] == 20 and counts['array_loads'] == 460
            and counts['block_fits'] == 8960 and counts['fewshot_fits'] == counts['queries'] == 10240,
            'complete fixed reconstruction work')
    require(admit(folder) == admission, 'all frozen original evidence unchanged after audit')
    output = {'version': VERSION, 'agreement': True, 'status': result['status'], 'config': CONFIG,
              'rows': rows, 'conditions': result['conditions'], 'counts': counts,
              'registration': admission['registration'], 'producer_receipt': admission['producer_receipt'],
              'producer_elapsed_seconds': admission['receipt']['elapsed_seconds'],
              'audit_elapsed_seconds': time.perf_counter()-started,
              'tolerance': {'absolute': ATOL, 'relative': RTOL},
              'scope': 'Independent QR/SVD recovery, public saved-map routing and all-case metric reconstruction. Private matrices/indices check labels and score recovery only; never route predictions. Rank-deficient groups are retained and cannot pass full-rank recovery. No generation or neural/model/optimizer calls. Per-case timings are producer attestations, not replay timings. Original process exit/closure is external evidence, not inferred from this saved receipt.',
              'external_process_closure_required': True, 'architecture_claim': False}
    with (folder/'audit.json').open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    audit_receipt = {'version': VERSION, 'status': 'COMPLETE', 'agreement': True,
                     'scientific_status': result['status'], 'files': {'audit.json': descriptor(folder/'audit.json')},
                     'registration': admission['registration'], 'producer_receipt': admission['producer_receipt']}
    with (folder/'audit.receipt.json').open('x') as stream:
        json.dump(audit_receipt, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
