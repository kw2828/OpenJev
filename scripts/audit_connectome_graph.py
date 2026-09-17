"""Graph-only structural receipt; no weights, chess labels or neural inference."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from openjev.research.connectome_graph import GROUPS, graph_sha256, load_pinned_graph

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(assets, out):
    """Retain aggregate statistics only, without redistributing graph bytes."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    graph = load_pinned_graph(assets)
    n, e = len(graph.node_ids), len(graph.sources)
    pairs = graph.sources * n + graph.destinations
    group_pair = graph.groups[graph.sources] * 4 + graph.groups[graph.destinations]
    records = []
    for source_name, source_group in GROUPS.items():
        for target_name, target_group in GROUPS.items():
            keep = group_pair == source_group * 4 + target_group
            records.append({
                'source_group': source_name, 'destination_group': target_name,
                'connections': int(keep.sum()),
                'positive': int(np.count_nonzero(keep & (graph.signs > 0))),
                'negative': int(np.count_nonzero(keep & (graph.signs < 0))),
                'synapses': int(graph.anatomical_counts[keep].sum()),
            })
    report = {
        'status': 'completed', 'scope': 'Anatomical graph-byte and structure audit only.',
        'training_run': False, 'model_inference_run': False, 'weights_loaded': False,
        'chess_positions_or_labels_read': False,
        'created_unix': time.time(), 'graph_sha256': graph_sha256(graph),
        'provenance': dict(graph.provenance),
        'nodes': n, 'connections': e,
        'positive_connections': int(np.count_nonzero(graph.signs > 0)),
        'negative_connections': int(np.count_nonzero(graph.signs < 0)),
        'anatomical_synapses': int(graph.anatomical_counts.sum()),
        'self_connections': int(np.count_nonzero(graph.sources == graph.destinations)),
        'duplicate_directed_pairs': e - len(np.unique(pairs)),
        'node_groups': {name: int(np.count_nonzero(graph.groups == group))
                        for name, group in GROUPS.items()},
        'group_connections': records,
        'dense_float32_adjacency_bytes': n * n * 4,
        'sources': {name: sha(ROOT / name) for name in (
            'src/openjev/research/connectome_graph.py', 'tests/test_connectome_graph.py',
            'scripts/audit_connectome_graph.py')},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'wall_seconds': time.perf_counter() - started,
        'limits': 'No subgraph or model selected. Counts are for the pinned packed asset. '
                  'Packed indices are not original FlyWire root IDs. '
                  'No graph byte arrays, coordinates or selected-node lists are distributed in this receipt.',
    }
    assert sum(r['connections'] for r in records) == e
    assert sum(r['synapses'] for r in records) == report['anatomical_synapses']
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
    result = audit(args.assets, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'nodes', 'connections',
                                           'self_connections', 'duplicate_directed_pairs', 'wall_seconds')}))
