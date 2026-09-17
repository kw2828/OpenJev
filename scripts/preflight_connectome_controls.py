"""Graph-only feasibility on all packed descending neurons; no training."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from openjev.research.connectome_graph import (
    GROUPS,
    RewireBudgetError,
    graph_sha256,
    induced_subgraph,
    load_pinned_graph,
    rewire_signed_degrees,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (151, 163, 179)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(assets, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    full = load_pinned_graph(assets)
    graph = induced_subgraph(full, full.node_ids[full.groups == GROUPS['descending']])
    n, e = len(graph.node_ids), len(graph.sources)
    whole_incident = np.count_nonzero(
        (full.groups[full.sources] == GROUPS['descending'])
        | (full.groups[full.destinations] == GROUPS['descending'])
    )
    del full
    report = {
        'status': 'started', 'scope': 'Structural feasibility only, no model or training-subgraph selection.',
        'node_selection': 'All and only nodes with packed coarse group descending (3).',
        'training_run': False, 'model_inference_run': False, 'weights_loaded': False,
        'chess_positions_or_labels_read': False,
        'nodes': n, 'connections': e, 'boundary_connections_removed_by_induction': int(whole_incident - e),
        'positive_connections': int(np.count_nonzero(graph.signs > 0)),
        'negative_connections': int(np.count_nonzero(graph.signs < 0)),
        'isolated_nodes': int(np.count_nonzero(
            np.bincount(graph.sources, minlength=n) + np.bincount(graph.destinations, minlength=n) == 0)),
        'dense_float32_adjacency_bytes': 4 * n * n,
        'source_graph_sha256': graph_sha256(graph),
        'provenance': dict(graph.provenance),
        'protocol': {'seeds': list(SEEDS), 'accepted_swaps_per_edge': 10, 'attempts_per_requested_swap': 100,
                     'minimum_change_gate': None,
                     'selection': 'Every listed seed; no retry or replacement controls.'},
        'controls': [],
        'sources': {p: sha(ROOT / p) for p in ('scripts/preflight_connectome_controls.py',
            'src/openjev/research/connectome_graph.py', 'tests/test_connectome_graph.py')},
    }
    for seed in SEEDS:
        begun = time.perf_counter()
        try:
            result = rewire_signed_degrees(graph, seed=seed, accepted_swaps=10 * e, max_attempts=1000 * e)
            # Independent endpoint histograms supplement the implementation's assertions.
            for sign in (-1, 1):
                for field in ('sources', 'destinations'):
                    before = np.bincount(getattr(graph, field)[graph.signs == sign], minlength=n)
                    after = np.bincount(getattr(result.graph, field)[result.graph.signs == sign], minlength=n)
                    if not np.array_equal(before, after):
                        raise ValueError('Independent signed-degree validation failed')
            report['controls'].append({
                'status': 'completed', 'seed': seed, 'graph_sha256': graph_sha256(result.graph),
                'attempted_swaps': result.attempted_swaps, 'accepted_swaps': result.accepted_swaps,
                'changed_edge_fraction': result.changed_edge_fraction,
                'signed_degrees_preserved': True,
                'anatomical_strengths_preserved': False,
                'wall_seconds': time.perf_counter() - begun,
            })
        except RewireBudgetError as error:
            report['controls'].append({
                'status': 'failed', 'seed': seed, 'error': str(error),
                'attempted_swaps': error.attempted_swaps, 'accepted_swaps': error.accepted_swaps,
                'wall_seconds': time.perf_counter() - begun,
            })
    report['status'] = ('completed' if all(r['status'] == 'completed' for r in report['controls']) else 'failed')
    report['wall_seconds'] = time.perf_counter() - started
    report['created_unix'] = time.time()
    report['limits'] = ('No chess performance measured. This group-induced graph omits all connections '
                        'to other groups and is not a complete functional circuit. Swap counts and overlap '
                        'do not establish uniform graph sampling. No graph byte arrays or node lists are published.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.assets, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'nodes', 'connections', 'controls', 'wall_seconds')}))
