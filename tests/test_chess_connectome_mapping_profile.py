import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('mapping_profile', Path(__file__).parents[1]/'scripts/chess_connectome_mapping_profile.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_fixture_has_exact_shape_no_self_or_duplicate_edges_and_no_teacher_values():
    graph = profile.synthetic_graph()
    assert len(graph.node_ids) == 1409 and len(graph.sources) == 44090
    assert not np.any(graph.sources == graph.destinations)
    assert len(set(zip(graph.sources, graph.destinations, strict=True))) == 44090
    assert graph.anatomical_counts is None
    batch = profile.synthetic_batch()
    assert batch['legal_mask'].shape == (128, 128)
    assert batch['legal_mask'].sum(1).tolist() == [20]*128
    assert batch['targets'].sum().item() == 0 and batch['values'].abs().sum().item() == 0


def test_projection_charges_all_updates_proposals_topologies_and_seeds():
    cases = [{'mode': m, 'policy': p, 'mean_seconds': 1 if p == 'ordinary' else 3}
             for m in ('sparse', 'node_local') for p in ('ordinary', 'fixed', 'learned')]
    assert profile.projected_seconds(cases) == 30*(1216+320*3)
    for invalid in (cases[:-1], cases+[cases[0]], [dict(r, mean_seconds=math.nan) for r in cases]):
        with pytest.raises(ValueError):
            profile.projected_seconds(invalid)
