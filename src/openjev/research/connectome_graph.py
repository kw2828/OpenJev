"""Graph-only primitives for prospective connectome controls.

The pinned loader reads anatomical assets, never pretrained weights. It retains
self edges and parallel edges exactly; rewiring deliberately requires a simple,
loop-free induced graph. Nothing here writes graph-derived assets to disk.

Packed FlyWire node indices are identities, not original FlyWire root IDs. The
four coarse group labels are not complete cell-type or transmitter annotations.
Graph assets and derivatives retain their external noncommercial terms; this
module does not grant an MIT license to them.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

SPACE_REVISION = '375374bf58f1c828ad7d33905b64f1fa6ac4906f'
META_SHA256 = '4405c948ffaa490eeb5ce4577e6c2ce9fc06113c014732f3491a62248eb3b693'
ASSET_SHA256 = {
    'connectome.bin.gz': 'a90aadf847f6939b306b2fe707134dc22213431b6eec3d5a7b4c45d62c98ff8a',
    'neurons.bin.gz': '4d801ed09ee0c4114ca7cae5a18b6ed2decc84dd115c9c30ae8f4a46a4294842',
}
GROUPS = {'other': 0, 'optic': 1, 'input': 2, 'descending': 3}
Scalar = str | int | float | bool | None


def _vector(values, name: str, *, minimum: int, maximum: int,
            dtype=np.int64) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in 'iu' and array.size:
        raise ValueError(f'{name} must be a one-dimensional integer vector')
    if array.dtype.kind == 'b':
        raise ValueError(f'{name} must contain integers, not booleans')
    if array.size and (np.any(array < minimum) or np.any(array > maximum)):
        raise ValueError(f'{name} values must be between {minimum} and {maximum}')
    # An immutable bytes backing also prevents callers from re-enabling writes.
    return np.frombuffer(array.astype(dtype, copy=False).tobytes(), dtype=dtype)


@dataclass(frozen=True, eq=False)
class SignedGraph:
    """Immutable snapshot with local edge indices and original packed node IDs.

    ``anatomical_counts`` stores positive synapse-count magnitudes, separately
    from ``signs``. An absent count vector means topology only. Coarse groups
    retain their original identity and consequently any group-defined I/O sets.
    No additional input/readout mapping is implied. Self/parallel edges may be
    represented so loading and induced extraction never silently alter anatomy.
    """

    node_ids: np.ndarray | Sequence[int]
    groups: np.ndarray | Sequence[int]
    sources: np.ndarray | Sequence[int]
    destinations: np.ndarray | Sequence[int]
    signs: np.ndarray | Sequence[int]
    anatomical_counts: np.ndarray | Sequence[int] | None = None
    provenance: Mapping[str, Scalar] | None = None

    def __post_init__(self):
        nodes = _vector(self.node_ids, 'node_ids', minimum=0,
                        maximum=np.iinfo(np.int64).max)
        if not len(nodes) or len(np.unique(nodes)) != len(nodes):
            raise ValueError('node_ids must be nonempty and unique')
        groups = _vector(self.groups, 'groups', minimum=0, maximum=3, dtype=np.int8)
        sources = _vector(self.sources, 'sources', minimum=0, maximum=len(nodes)-1)
        destinations = _vector(self.destinations, 'destinations', minimum=0,
                               maximum=len(nodes)-1)
        signs = _vector(self.signs, 'signs', minimum=-1, maximum=1, dtype=np.int8)
        if len(groups) != len(nodes):
            raise ValueError('groups must have one entry per node')
        if len(destinations) != len(sources) or len(signs) != len(sources):
            raise ValueError('Edge vectors must have equal lengths')
        if np.any(signs == 0):
            raise ValueError('Edge signs must be -1 or +1')
        counts = None
        if self.anatomical_counts is not None:
            counts = _vector(self.anatomical_counts, 'anatomical_counts', minimum=1,
                             maximum=np.iinfo(np.int64).max)
            if len(counts) != len(sources):
                raise ValueError('anatomical_counts must have one entry per edge')
        if self.provenance is not None and not isinstance(self.provenance, Mapping):
            raise ValueError('provenance must be a mapping of scalar values')
        provenance = dict(self.provenance or {})
        for key, value in provenance.items():
            if (not isinstance(key, str)
                    or type(value) not in (str, int, float, bool, type(None))
                    or isinstance(value, float) and not math.isfinite(value)):
                raise ValueError('provenance requires string keys and finite scalar values')
        for name, value in (('node_ids', nodes), ('groups', groups), ('sources', sources),
                            ('destinations', destinations), ('signs', signs),
                            ('anatomical_counts', counts),
                            ('provenance', MappingProxyType(provenance))):
            object.__setattr__(self, name, value)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.astype('<i8', copy=False).tobytes()).hexdigest()


def graph_sha256(graph: SignedGraph) -> str:
    """Canonical content identity, including counts but excluding provenance."""
    if not isinstance(graph, SignedGraph):
        raise TypeError('Expected a SignedGraph')
    digest = hashlib.sha256(b'openjev-signed-graph-v1\0')
    for name in ('node_ids', 'groups', 'sources', 'destinations', 'signs',
                 'anatomical_counts'):
        array = getattr(graph, name)
        digest.update(name.encode()+b'\0')
        if array is None:
            digest.update(b'absent\0')
        else:
            digest.update(struct.pack('<Q', len(array)))
            digest.update(array.astype('<i8', copy=False).tobytes())
    return digest.hexdigest()


def _checked_raw(asset_dir: Path, metadata: dict, name: str) -> bytes:
    path = asset_dir/name
    expected = metadata['files'][name]
    if not path.is_file() or path.stat().st_size != expected['bytes']:
        raise ValueError(f'Pinned asset size mismatch: {name}')
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != ASSET_SHA256[name]:
        raise ValueError(f'Pinned compressed asset hash mismatch: {name}')
    # The authenticated compressed bytes are trusted, but bound decompression too.
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as handle:
        raw = handle.read(expected['rawBytes']+1)
    if (len(raw) != expected['rawBytes']
            or hashlib.sha256(raw).hexdigest() != expected['rawSha256']):
        raise ValueError(f'Pinned raw asset hash mismatch: {name}')
    return raw


def load_pinned_graph(asset_dir: str | Path) -> SignedGraph:
    """Load only pinned meta/connectome/neuron files; never open model weights.

    The neuron file is authenticated in full. Its coordinates remain in the
    source asset; this topology object exposes only packed IDs and coarse groups.
    """
    asset_dir = Path(asset_dir)
    meta_path = asset_dir/'meta.json'
    if not meta_path.is_file() or meta_path.stat().st_size != 1274:
        raise ValueError('Pinned metadata size mismatch')
    meta_bytes = meta_path.read_bytes()
    if hashlib.sha256(meta_bytes).hexdigest() != META_SHA256:
        raise ValueError('Metadata does not match the pinned ChessFly revision')
    metadata = json.loads(meta_bytes)
    raw = _checked_raw(asset_dir, metadata, 'connectome.bin.gz')
    neuron_raw = _checked_raw(asset_dir, metadata, 'neurons.bin.gz')
    if len(raw) < 12 or raw[:4] != b'CFLY':
        raise ValueError('Invalid connectome header')
    _, nodes, edges = struct.unpack_from('<III', raw)
    if (nodes != metadata['neurons'] or edges != metadata['edges']
            or len(raw) != 12+4*(nodes+1)+6*edges or len(neuron_raw) != 13*nodes):
        raise ValueError('Connectome dimensions disagree with pinned metadata')
    rows = np.frombuffer(raw, dtype='<u4', count=nodes+1, offset=12)
    destinations = np.frombuffer(raw, dtype='<u4', count=edges, offset=12+4*(nodes+1))
    counts = np.frombuffer(raw, dtype='<i2', count=edges,
                           offset=12+4*(nodes+1)+4*edges)
    if (rows[0] != 0 or rows[-1] != edges or np.any(rows[1:] < rows[:-1])
            or np.any(destinations >= nodes) or np.any(counts == 0)):
        raise ValueError('Invalid connectome row offsets, endpoints or counts')
    groups = np.frombuffer(neuron_raw, dtype=np.uint8, count=nodes, offset=12*nodes)
    if (metadata['groups'] != GROUPS or np.any(groups > 3)
            or np.count_nonzero(groups == GROUPS['input']) != metadata['inputs']
            or np.count_nonzero(groups == GROUPS['descending']) != metadata['descending']):
        raise ValueError('Neuron groups disagree with pinned metadata')
    magnitudes = np.abs(counts.astype(np.int64))
    if int(magnitudes.sum()) != metadata['synapses']:
        raise ValueError('Synapse counts disagree with pinned metadata')
    return SignedGraph(
        node_ids=np.arange(nodes), groups=groups,
        sources=np.repeat(np.arange(nodes), np.diff(rows).astype(np.int64)),
        destinations=destinations, signs=np.where(counts < 0, -1, 1),
        anatomical_counts=magnitudes,
        provenance={
            'dataset': metadata['dataset'], 'space_revision': SPACE_REVISION,
            'meta_sha256': META_SHA256,
            'graph_raw_sha256': metadata['files']['connectome.bin.gz']['rawSha256'],
            'neurons_raw_sha256': metadata['files']['neurons.bin.gz']['rawSha256'],
            'external_license': 'FlyWire non-commercial terms; model card license: other',
            'node_identity': 'Original packed indices, not FlyWire root IDs',
            'operation': 'pinned_graph_only_load',
        },
    )


def induced_subgraph(graph: SignedGraph, node_ids: Sequence[int] | np.ndarray) -> SignedGraph:
    """Retain exactly the requested nodes and all edges between them.

    IDs are original packed identities, returned in ascending order. Edge order,
    multiplicity, self edges, signs and counts are preserved. No data-dependent
    choice of nodes is performed, and no anatomical region is inferred.
    """
    if not isinstance(graph, SignedGraph):
        raise TypeError('Expected a SignedGraph')
    selected = _vector(node_ids, 'selected node_ids', minimum=0,
                       maximum=np.iinfo(np.int64).max)
    if not len(selected) or len(np.unique(selected)) != len(selected):
        raise ValueError('Selected node_ids must be nonempty and unique')
    selected = np.sort(selected)
    order = np.argsort(graph.node_ids)
    locations = np.searchsorted(graph.node_ids[order], selected)
    if (np.any(locations == len(order))
            or not np.array_equal(graph.node_ids[order[locations]], selected)):
        raise ValueError('Selected node_ids must exist in the source graph')
    original_indices = order[locations]
    local = np.full(len(graph.node_ids), -1, dtype=np.int64)
    local[original_indices] = np.arange(len(selected))
    source = local[graph.sources]
    destination = local[graph.destinations]
    keep = (source >= 0) & (destination >= 0)
    return SignedGraph(
        selected, graph.groups[original_indices], source[keep], destination[keep],
        graph.signs[keep],
        None if graph.anatomical_counts is None else graph.anatomical_counts[keep],
        {**graph.provenance, 'operation': 'exact_induced_subgraph',
         'source_graph_sha256': graph_sha256(graph)},
    )


@dataclass(frozen=True)
class RewireResult:
    graph: SignedGraph
    attempted_swaps: int
    accepted_swaps: int
    changed_edge_fraction: float
    seed: int


class RewireBudgetError(ValueError):
    """The fixed attempt budget could not produce the requested accepted swaps."""

    def __init__(self, message: str, *, attempted_swaps: int, accepted_swaps: int,
                 requested_swaps: int, max_attempts: int):
        super().__init__(message)
        self.attempted_swaps = attempted_swaps
        self.accepted_swaps = accepted_swaps
        self.requested_swaps = requested_swaps
        self.max_attempts = max_attempts


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _degree_signature(graph: SignedGraph) -> tuple[np.ndarray, np.ndarray]:
    nodes = len(graph.node_ids)
    degrees = np.stack([
        np.bincount(endpoints[graph.signs == sign], minlength=nodes)
        for sign in (-1, 1) for endpoints in (graph.sources, graph.destinations)
    ])
    bucket = (graph.groups[graph.sources]*4+graph.groups[graph.destinations])*2+(graph.signs > 0)
    return degrees, np.bincount(bucket, minlength=32)


def rewire_signed_degrees(graph: SignedGraph, *, seed: int, accepted_swaps: int,
                          max_attempts: int) -> RewireResult:
    """Swap destinations within (source group, destination group, sign) buckets.

    PCG64 chooses uniformly among buckets containing at least two edges, then
    uniformly among ordered distinct edge pairs. Sources and signs stay fixed;
    new self edges or duplicate directed pairs are rejected. The successful
    result preserves per-node signed degrees and group mixing, not anatomical
    in-strength or synapse-count assignment. Its count vector is deliberately
    absent, with the original counts' hash retained in provenance.

    ``changed_edge_fraction`` counts original signed directed edge triples absent
    from the final topology. Accepted swaps can undo earlier swaps, so neither
    success nor the requested swap count guarantees a useful amount of change.
    This function is intended for small induced subgraphs, not a dense full-brain
    control. It never returns an incomplete shuffle as a successful result.
    """
    if not isinstance(graph, SignedGraph):
        raise TypeError('Expected a SignedGraph')
    seed = _nonnegative_integer(seed, 'seed')
    target = _nonnegative_integer(accepted_swaps, 'accepted_swaps')
    ceiling = _nonnegative_integer(max_attempts, 'max_attempts')
    if seed > 2**64-1:
        raise ValueError('seed must fit in uint64')
    nodes = len(graph.node_ids)
    source, signs = graph.sources, graph.signs
    destination = graph.destinations.copy()
    if np.any(source == destination):
        raise ValueError('Rewiring requires a loop-free graph; self edges are not dropped')
    original_codes = source*nodes+destination
    if len(np.unique(original_codes)) != len(original_codes):
        raise ValueError('Rewiring requires a simple graph; parallel/duplicate pairs are not dropped')
    edge_set = set(map(int, original_codes))
    original_signed = set(map(int, original_codes*2+(signs > 0)))
    bucket_ids = (graph.groups[source]*4+graph.groups[destination])*2+(signs > 0)
    buckets = [np.flatnonzero(bucket_ids == key) for key in range(32)]
    buckets = [indices for indices in buckets if len(indices) >= 2]
    rng = np.random.Generator(np.random.PCG64(seed))
    attempted = accepted = 0
    while accepted < target and attempted < ceiling and buckets:
        attempted += 1
        bucket = buckets[int(rng.integers(len(buckets)))]
        first, second = map(int, rng.choice(bucket, size=2, replace=False))
        a, b = int(source[first]), int(destination[first])
        c, d = int(source[second]), int(destination[second])
        if a == c or b == d or a == d or c == b:
            continue
        new_first, new_second = a*nodes+d, c*nodes+b
        if new_first in edge_set or new_second in edge_set:
            continue
        edge_set.remove(a*nodes+b)
        edge_set.remove(c*nodes+d)
        edge_set.update((new_first, new_second))
        destination[first], destination[second] = d, b
        accepted += 1
    if accepted < target:
        raise RewireBudgetError(
            f'Accepted {accepted}/{target} swaps in {attempted}/{ceiling} attempts; '
            'no successful control returned', attempted_swaps=attempted,
            accepted_swaps=accepted, requested_swaps=target, max_attempts=ceiling,
        )
    final_signed = set(map(int, (source*nodes+destination)*2+(signs > 0)))
    fraction = len(original_signed-final_signed)/len(original_signed) if original_signed else 0.0
    result = SignedGraph(
        graph.node_ids, graph.groups, source, destination, signs, None,
        {**graph.provenance, 'operation': 'signed_degree_rewire',
         'source_graph_sha256': graph_sha256(graph),
         'source_anatomical_counts_sha256': (
             None if graph.anatomical_counts is None else _array_sha256(graph.anatomical_counts)),
         'seed': seed, 'attempted_swaps': attempted, 'accepted_swaps': accepted,
         'requested_swaps': target, 'max_attempts': ceiling,
         'changed_edge_fraction': fraction, 'anatomical_strengths_preserved': False,
         'sampler': 'PCG64; uniform eligible bucket, then uniform ordered distinct edge pair'},
    )
    before, after = _degree_signature(graph), _degree_signature(result)
    if any(not np.array_equal(left, right) for left, right in zip(before, after, strict=True)):
        raise RuntimeError('Internal error: rewiring violated signed-degree or group-mixing invariants')
    return RewireResult(result, attempted, accepted, fraction, seed)
