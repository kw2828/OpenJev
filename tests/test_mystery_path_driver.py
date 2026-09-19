"""Pure reference-route and byte-binding tests; no environment imports or resets."""

import ast
import hashlib
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjev.research.mystery_path_observation import Observation

DRIVER = Path(__file__).resolve().parents[1] / 'scripts/run_mystery_path_qualification.py'
DELTAS = ((0, -1), (-1, 0), (0, 1), (1, 0))


def actual_function(name, namespace):
    """Use actual driver code while excluding all environment imports/entrypoints."""
    tree = ast.parse(DRIVER.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(DRIVER), 'exec'), namespace)  # noqa: S102
    return namespace[name]


@pytest.fixture
def reference():
    return actual_function('reference_action', {'deque': deque, 'DELTAS': DELTAS})


@pytest.mark.parametrize('heading', range(4))
@pytest.mark.parametrize(('relative', 'expected'), ((0, 3), (1, 1), (2, 1), (3, 2)))
def test_adjacent_goal_charges_turns_and_uses_stable_half_turn(reference, heading, relative, expected):
    direction = (heading + relative) % 4
    dx, dy = DELTAS[direction]
    goal = (3 + dx, 3 + dy)
    assert reference(Observation(3, 3, heading, False), {(3, 3), goal}, goal) == expected


def follow(reference, safe, start, goal, limit):
    observation = start
    actions = []
    for _ in range(limit):
        if (observation.x, observation.y) == goal:
            break
        action = reference(observation, safe, goal)
        actions.append(action)
        x, y, h = observation.x, observation.y, observation.heading
        if action == 3:
            dx, dy = DELTAS[h]
            x, y = x + dx, y + dy
            assert (x, y) in safe
        elif action == 1:
            h = (h + 1) % 4
        elif action == 2:
            h = (h - 1) % 4
        else:
            pytest.fail('Reachable non-goal route must not wait')
        observation = Observation(x, y, h, False)
    assert (observation.x, observation.y) == goal
    return actions


def test_heading_breaks_equal_grid_distance_in_favor_of_fewer_primitive_actions(reference):
    # Both sides are four grid moves away; east costs six calls, west costs eight.
    safe = frozenset({(3, 3), (4, 3), (4, 2), (4, 1), (3, 1), (2, 3), (2, 2), (2, 1)})
    assert follow(reference, safe, Observation(3, 3, 3, False), (3, 1), 6) == [3, 1, 3, 3, 1, 3]


def test_boundary_route_uses_no_outside_grid_shortcut_and_preserves_inputs(reference):
    safe = {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)}
    before = safe.copy()
    assert follow(reference, safe, Observation(0, 0, 0, False), (2, 2), 7) == [1, 1, 3, 3, 1, 3, 3]
    assert safe == before


@pytest.mark.parametrize('heading', range(4))
def test_goal_waits_and_off_path_return_ignores_route_lookup(reference, heading):
    assert reference(Observation(2, 1, heading, False), {(2, 1)}, (2, 1)) == 0
    # Not containers: any attempted hidden route access would fail this test.
    assert reference(Observation(6, 6, heading, True), object(), object()) == 0


def test_disconnected_route_is_an_explicit_failure(reference):
    with pytest.raises(ValueError, match='disconnected'):
        reference(Observation(0, 0, 0, False), {(0, 0), (6, 6)}, (6, 6))


@pytest.fixture
def binding_fixture(tmp_path):
    root = tmp_path / 'repo'
    base = root / 'binding'
    installed = tmp_path / 'installed'
    base.mkdir(parents=True)
    installed.mkdir()
    source = root / 'controller.py'
    upstream = installed / 'sprite.py'
    source.write_bytes(b'local frozen source\n')
    upstream.write_bytes(b'upstream frozen source\n')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    binding = {'sources': {'controller.py': sha(source)},
               'upstream_python': {'sprite.py': sha(upstream)},
               'packages': {'pygame': '2.4.0'}, 'python': 'fake exact runtime'}
    (base / 'bindings.json').write_text(json.dumps(binding))
    namespace = {'ROOT': root, 'BASE': base, 'Path': Path, 'json': json, 'digest': sha,
                 'memory_gym': SimpleNamespace(__file__=str(installed / '__init__.py')),
                 'importlib': SimpleNamespace(metadata=SimpleNamespace(version=lambda _: '2.4.0')),
                 'sys': SimpleNamespace(version='fake exact runtime')}
    return actual_function('verify_bindings', namespace), binding, source, upstream, namespace


def test_exact_source_upstream_package_and_python_binding(binding_fixture):
    verify, expected, *_ = binding_fixture
    assert verify() == expected


@pytest.mark.parametrize('mutation', ('source', 'upstream', 'missing_upstream', 'package', 'python'))
def test_binding_mutations_reject_before_environment_work(binding_fixture, mutation):
    verify, _expected, source, upstream, namespace = binding_fixture
    if mutation == 'source':
        source.write_bytes(b'changed local source')
    elif mutation == 'upstream':
        upstream.write_bytes(b'changed upstream source')
    elif mutation == 'missing_upstream':
        upstream.unlink()
    elif mutation == 'package':
        namespace['importlib'].metadata.version = lambda _: 'different'
    else:
        namespace['sys'].version = 'different'
    with pytest.raises((AssertionError, FileNotFoundError)):
        verify()
