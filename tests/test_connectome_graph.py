"""Independent topology controls; all inputs are small synthetic graphs."""
from collections import Counter

import numpy as np
import pytest

from openjev.research.connectome_graph import (
    SignedGraph,
    induced_subgraph,
    load_pinned_graph,
    rewire_signed_degrees,
)


def fixture_graph():
    edges = [(0, 2), (1, 3), (0, 4), (1, 5), (2, 6), (3, 7),
             (4, 6), (5, 7), (2, 4), (3, 5), (4, 3), (5, 2)]
    return SignedGraph(
        node_ids=np.arange(100, 108), groups=np.array([2, 2, 0, 0, 0, 0, 3, 3]),
        sources=np.array([a for a, _ in edges]),
        destinations=np.array([b for _, b in edges]),
        signs=np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1]),
        anatomical_counts=np.arange(1, len(edges) + 1),
    )


def invariants(graph):
    incoming, outgoing, mixing = Counter(), Counter(), Counter()
    # Counts are derived from individual edges, independent of control internals.
    for source, target, sign in zip(graph.sources, graph.destinations, graph.signs, strict=True):
        incoming[(int(target), int(sign), int(graph.groups[source]))] += 1
        outgoing[(int(source), int(sign), int(graph.groups[target]))] += 1
        mixing[(int(graph.groups[source]), int(graph.groups[target]), int(sign))] += 1
    return incoming, outgoing, mixing


@pytest.mark.parametrize('seed', [0, 1, 13, 97, 109, 127])
def test_controls_preserve_degrees_group_mixing_and_node_identity(seed):
    original = fixture_graph()
    before = (original.sources.copy(), original.destinations.copy(), original.signs.copy())
    result = rewire_signed_degrees(original, seed=seed, accepted_swaps=19, max_attempts=1000)
    control = result.graph
    assert invariants(control) == invariants(original)
    np.testing.assert_array_equal(control.node_ids, original.node_ids)
    np.testing.assert_array_equal(control.groups, original.groups)
    assert len(control.sources) == len(original.sources)
    pairs = list(zip(control.sources.tolist(), control.destinations.tolist()))
    assert len(set(pairs)) == len(pairs)
    assert all(a != b for a, b in pairs)
    assert result.accepted_swaps == 19
    assert 19 <= result.attempted_swaps <= 1000
    old_edges = set(zip(original.sources.tolist(), original.destinations.tolist(), original.signs.tolist()))
    new_edges = set(zip(control.sources.tolist(), control.destinations.tolist(), control.signs.tolist()))
    assert result.changed_edge_fraction == pytest.approx(len(old_edges - new_edges) / len(old_edges))
    assert control.anatomical_counts is None  # Counts are not invented for new edges.
    for current, snapshot in zip((original.sources, original.destinations, original.signs), before, strict=True):
        np.testing.assert_array_equal(current, snapshot)


def test_rewiring_reproducible_without_global_rng_changes():
    np.random.seed(751)
    state = np.random.get_state()
    a = rewire_signed_degrees(fixture_graph(), seed=912, accepted_swaps=23, max_attempts=1000)
    b = rewire_signed_degrees(fixture_graph(), seed=912, accepted_swaps=23, max_attempts=1000)
    for name in ('sources', 'destinations', 'signs', 'node_ids', 'groups'):
        np.testing.assert_array_equal(getattr(a.graph, name), getattr(b.graph, name))
    assert a.attempted_swaps == b.attempted_swaps
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_induced_graph_uses_original_ids_and_preserves_exact_edges():
    original = fixture_graph()
    selected = induced_subgraph(original, [107, 100, 103, 101])
    np.testing.assert_array_equal(selected.node_ids, [100, 101, 103, 107])
    np.testing.assert_array_equal(selected.groups, [2, 2, 0, 3])
    rows = {(int(selected.node_ids[a]), int(selected.node_ids[b]), int(s)): int(c)
            for a, b, s, c in zip(selected.sources, selected.destinations, selected.signs,
                                 selected.anatomical_counts, strict=True)}
    assert rows == {(101, 103, 1): 2, (103, 107, 1): 6}
    assert invariants(original) == invariants(fixture_graph())


@pytest.mark.parametrize('selected', [[99], [100, 100], [100, 108], [100.5], [True]])
def test_induced_graph_rejects_unknown_or_ambiguous_ids(selected):
    with pytest.raises((ValueError, TypeError)):
        induced_subgraph(fixture_graph(), selected)


def test_graph_copies_arrays_and_prevents_accidental_mutation():
    ids, groups = np.arange(3), np.zeros(3, dtype=int)
    source, target, signs = np.array([0, 1]), np.array([1, 2]), np.array([1, -1])
    graph = SignedGraph(ids, groups, source, target, signs)
    ids[0], groups[0], source[0], target[0], signs[0] = 90, 3, 2, 0, -1
    np.testing.assert_array_equal(graph.node_ids, [0, 1, 2])
    np.testing.assert_array_equal(graph.groups, [0, 0, 0])
    np.testing.assert_array_equal(graph.sources, [0, 1])
    np.testing.assert_array_equal(graph.destinations, [1, 2])
    np.testing.assert_array_equal(graph.signs, [1, -1])
    for value in (graph.node_ids, graph.groups, graph.sources, graph.destinations, graph.signs):
        assert not value.flags.writeable


@pytest.mark.parametrize('field,value', [
    ('node_ids', [1, 1, 2]), ('node_ids', [-1, 1, 2]),
    ('node_ids', [0., 1., 2.]), ('node_ids', [True, False, True]),
    ('groups', [0, 1]), ('groups', [0, 1, 4]), ('groups', [0, -1, 2]),
    ('sources', [-1, 1]), ('sources', [0, 3]), ('sources', [[0, 1]]),
    ('destinations', [1]), ('signs', [1, 0]), ('signs', [1, 2]),
    ('signs', [1., -1.]), ('anatomical_counts', [0, 2]),
    ('anatomical_counts', [1]), ('anatomical_counts', [-1, 2]),
])
def test_graph_rejects_malformed_identity_edges_and_counts(field, value):
    args = {'node_ids': np.arange(3), 'groups': np.zeros(3, dtype=int), 'sources': np.array([0, 1]),
                'destinations': np.array([1, 2]), 'signs': np.array([1, -1])}
    args[field] = np.array(value)
    with pytest.raises((ValueError, TypeError)):
        SignedGraph(**args)


def test_immobile_graph_fails_instead_of_returning_unmatched_control():
    graph = SignedGraph(np.arange(3), np.zeros(3, dtype=int),
                        np.array([0, 1]), np.array([1, 0]), np.array([1, 1]))
    with pytest.raises(ValueError) as error:
        rewire_signed_degrees(graph, seed=1, accepted_swaps=1, max_attempts=17)
    assert error.value.accepted_swaps == 0
    assert error.value.attempted_swaps <= 17


@pytest.mark.parametrize('source,target', [([0, 1], [0, 2]), ([0, 0], [1, 1])])
def test_rewiring_rejects_loops_or_parallel_edges_without_silent_filtering(source, target):
    graph = SignedGraph(np.arange(3), np.zeros(3, dtype=int), np.array(source), np.array(target),
                        np.array([1, 1]))
    with pytest.raises(ValueError):
        rewire_signed_degrees(graph, seed=1, accepted_swaps=1, max_attempts=17)


@pytest.mark.parametrize('kwargs', [
    {'seed': True, 'accepted_swaps': 1, 'max_attempts': 10},
    {'seed': 1, 'accepted_swaps': -1, 'max_attempts': 10},
    {'seed': 1, 'accepted_swaps': 1, 'max_attempts': -1},
    {'seed': 1.2, 'accepted_swaps': 1, 'max_attempts': 10},
    {'seed': 1, 'accepted_swaps': 1.5, 'max_attempts': 10},
])
def test_invalid_swap_requests_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        rewire_signed_degrees(fixture_graph(), **kwargs)


def test_loader_rejects_unpinned_metadata_before_graph_or_weights(tmp_path):
    (tmp_path / 'meta.json').write_text('{"neurons": 3, "edges": 2}')
    with pytest.raises(ValueError, match='[Mm]etadata|[Pp]inned|[Cc]hecksum|[Hh]ash'):
        load_pinned_graph(tmp_path)
