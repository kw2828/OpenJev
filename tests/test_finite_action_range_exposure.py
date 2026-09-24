"""Hand timing and corrupted-metadata fixtures only, without numerical imports."""
from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import qualify_finite_action_range_exposure as p


def fixture():
    allocations, fits = {}, []
    for arm in p.ARMS:
        stages = []
        for index, (kind, target, start, stop) in enumerate((('prefix', 32, .25, .75), ('joint', 64, 1., 2.))):
            stages.append({'stage': index + 1, 'kind': kind, 'status': 'PASS', 'termination': 'completed_updates',
                'target_updates': target, 'accepted_updates': target, 'attempted_updates': target,
                'deadline_seconds': 30., 'overrun_seconds': 0., 'start_elapsed': start, 'stopped_elapsed': stop})
        allocations[arm] = {'status': 'PASS', 'termination': 'completed_updates', 'max_seconds': 30.,
            'prefix_updates': 32, 'joint_updates': 64, 'accepted_prefix_updates': 32,
            'accepted_joint_updates': 64, 'joint_cursor': 64, 'accepted_updates': 96,
            'attempted_updates': 96, 'stages': stages}
        fits.append({'arm': arm, 'seed': 948301, 'seconds': 2.5})
    return allocations, fits


def test_projection_uses_all_six_accepted_stages_and_full_nonstage_time():
    allocations, fits = fixture()
    original = copy.deepcopy((allocations, fits))
    rows = p.projections(allocations, list(reversed(fits)))
    assert [row['arm'] for row in rows] == list(p.ARMS)
    for row in rows:
        assert row['stage_seconds'] == [.5, 1.]
        assert row['nonstage_seconds'] == 1.
        assert row['scale_factors'] == [32., 48.]
        assert row['safety_multiplier'] == 2.
        assert row['accepted_updates'] == [32, 64] and row['target_updates'] == [1024, 3072]
        assert row['projected_seconds'] == 129.  # 2*(.5*32 + 1*48) + 1.
    assert (allocations, fits) == original
    assert not p.feasible(rows)


def test_fixed_inclusive_ninety_second_boundary_requires_every_arm():
    rows = p.projections(*fixture())
    for row in rows:
        row['projected_seconds'] = 90.
    assert p.feasible(rows)
    rows[-1]['projected_seconds'] = 90.000001
    assert not p.feasible(rows)
    rows[-1]['projected_seconds'] = 89.999999
    assert p.feasible(rows)
    assert p.EXPOSURE['maximum_projected_seconds'] == 90.
    assert p.EXPOSURE['projection_comparison'] == '<='


@pytest.mark.parametrize('fault', ['duplicate_fit', 'missing_arm', 'wrong_seed', 'failed', 'short_prefix',
    'extra_attempt', 'wrong_cap', 'missing_stage', 'wrong_kind', 'stage_rejected', 'overrun',
    'zero_duration', 'negative_duration', 'overlap', 'late', 'nonfinite_stage', 'bool_stage',
    'nonfinite_fit', 'negative_overhead', 'full_time_shorter_than_last_stop'])
def test_unusable_or_incomplete_exposure_is_not_projected(fault):
    allocations, fits = fixture()
    first = allocations[p.ARMS[0]]
    stage = first['stages'][0]
    if fault == 'duplicate_fit':
        fits[-1] = copy.deepcopy(fits[0])
    elif fault == 'missing_arm':
        allocations.pop(p.ARMS[-1])
    elif fault == 'wrong_seed':
        fits[0]['seed'] += 1
    elif fault == 'failed':
        first['status'] = 'FAILED_TIMEOUT'
    elif fault == 'short_prefix':
        first['accepted_prefix_updates'] -= 1
    elif fault == 'extra_attempt':
        first['attempted_updates'] += 1
    elif fault == 'wrong_cap':
        first['max_seconds'] = 31.
    elif fault == 'missing_stage':
        first['stages'].pop()
    elif fault == 'wrong_kind':
        stage['kind'] = 'joint'
    elif fault == 'stage_rejected':
        stage['accepted_updates'] -= 1
    elif fault == 'overrun':
        stage['overrun_seconds'] = .1
    elif fault == 'zero_duration':
        stage['stopped_elapsed'] = stage['start_elapsed']
    elif fault == 'negative_duration':
        stage['stopped_elapsed'] = -.1
    elif fault == 'overlap':
        first['stages'][1]['start_elapsed'] = .5
    elif fault == 'late':
        first['stages'][1]['stopped_elapsed'] = 31.
    elif fault == 'nonfinite_stage':
        stage['stopped_elapsed'] = float('nan')
    elif fault == 'bool_stage':
        stage['start_elapsed'] = False
    elif fault == 'nonfinite_fit':
        fits[0]['seconds'] = float('inf')
    elif fault == 'negative_overhead':
        fits[0]['seconds'] = 1.
    else:
        fits[0]['seconds'] = 1.75
    with pytest.raises(ValueError):
        p.projections(allocations, fits)


def test_no_config_override_and_train_generation_only_are_explicit_source_contracts():
    assert p.CONFIG == {'seed_namespace': 948201, 'fit_seeds': [948301], 'train_attempts': 512,
        'dev_attempts': 8, 'batch_size': 64, 'learning_rate': .003, 'gradient_clip': 5.,
        'train_horizon': 2, 'dev_horizon': 8, 'prefix_updates': 32, 'joint_updates': 64, 'fit_cap_seconds': 30.}
    tree = ast.parse(Path(p.__file__).read_text())
    generation = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == 'generate_attempt_split']
    assert len(generation) == 1
    assert [ast.literal_eval(arg) for arg in generation[0].args] == [0, 512, 2]
    assert [(arg.arg, ast.literal_eval(arg.value)) for arg in generation[0].keywords] == [('seed_namespace', 948201)]
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {'predict', 'predict_prefix', 'make_reference'} for node in ast.walk(tree))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    assert [arg.arg for arg in run.args.args] == ['output']


def test_existing_path_is_not_overwritten_and_never_reaches_numerical_imports(tmp_path):
    marker = tmp_path / 'preserved.txt'
    marker.write_text('original')
    with pytest.raises(ValueError, match='exclusive'):
        p.run(tmp_path)
    assert marker.read_text() == 'original'
    assert list(tmp_path.iterdir()) == [marker]


def test_final_clock_failure_is_retained_without_retry_or_fabricated_duration():
    marker = RuntimeError('fabricated native clock failure')
    class BrokenClock:
        def __init__(self):
            self.calls = 0

        def now_ns(self):
            self.calls += 1
            raise marker

    clock = BrokenClock()
    seconds, error = p.final_clock_read(clock, 123)
    assert seconds is None and error is marker and clock.calls == 1
    assert p.final_clock_read(clock, None) == (None, None)
    assert clock.calls == 1  # No read if initial clock setup never succeeded.


def test_final_native_clock_duration_and_backward_time_guard():
    class Clock:
        def now_ns(self):
            return 3_000_000_000

    assert p.final_clock_read(Clock(), 1_000_000_000) == (2., None)
    seconds, error = p.final_clock_read(Clock(), 4_000_000_000)
    assert seconds is None and isinstance(error, ValueError)
