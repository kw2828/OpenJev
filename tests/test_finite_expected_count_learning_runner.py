"""Engineering-only integration of timed prefixes, common training and DEV barrier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_expected_count_learning as runner


def test_all_pretraining_and_joint_fits_precede_dev_with_matched_public_inputs(tmp_path, monkeypatch):
    folder = tmp_path / 'integrated'
    original_generate, original_factory = runner.generate_attempt_split, runner.learned_model
    generations, rollout_calls = [], []

    def generate(split, attempts, horizon, **kwargs):
        assert kwargs == {'seed_namespace': 938001}
        generations.append(split)
        if split:
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            assert barrier['fit_count'] == 3 and barrier['dev_generation_count'] == 0
            assert len(list(folder.glob('pretraining-final-*.npz'))) == 3
            assert all((folder / item['path']).is_file() for item in barrier['checkpoints'])
        return original_generate(split, attempts, horizon, **kwargs)

    def factory(arm, seed):
        assert arm == 'factorized' and seed == 938101 and generations == [0]
        model = original_factory(arm, seed)
        for name in ('blind_rollout', 'observed_rollout'):
            original = getattr(model, name)

            def hooked(*args, _original=original, **kwargs):
                assert not kwargs
                rollout_calls.append(True)
                return _original(*args, **kwargs)

            monkeypatch.setattr(model, name, hooked)
        return model

    monkeypatch.setattr(runner, 'generate_attempt_split', generate)
    monkeypatch.setattr(runner, 'learned_model', factory)
    config = {'seed_namespace': 938001, 'fit_seeds': [938101], 'train_attempts': 8,
              'dev_attempts': 8, 'epochs': 1, 'batch_size': 3, 'pretraining_seconds': 1.}
    result = runner.run(folder, config, lambda: None)
    assert generations == [0, 1] and rollout_calls
    assert len(result['rows']) == 12 and len(result['prefix_rows']) == 3
    fits, stages, counts = result['fits'], result['pretraining'], result['counts']
    assert {stage['method'] for stage in stages} == {'none', 'em', 'gradient'}
    assert len({stage['initial_state_sha256'] for stage in stages}) == 1
    assert len({fit['case_order_sha256'] for fit in fits}) == 1
    assert [fit['parameter_metadata']['parameter_count'] for fit in fits] == [352] * 3
    assert counts['checkpoint_writes'] == 3 and counts['pretraining_checkpoint_writes'] == 6
    assert counts['pretraining_runs'] == 3 and counts['optimizer_steps'] == 9
    assert counts['training_attempt_exposures'] == 24
    assert all(counts[key] == 0 for key in ('checkpoint_decodes', 'array_decodes',
               'external_model_calls', 'native_calls', 'teacher_calls'))
    for stage, fit in zip(stages, fits, strict=True):
        assert stage['arm'] == fit['arm'] and stage['seed'] == fit['seed']
        assert stage['final_state_sha256'] == fit['initial_state_sha256']
        with np.load(folder / stage['initial_checkpoint']['path'], allow_pickle=False) as initial, \
                np.load(folder / stage['final_checkpoint']['path'], allow_pickle=False) as final:
            np.testing.assert_array_equal(initial['cost_logits'], final['cost_logits'])
            if stage['method'] == 'none':
                assert all(np.array_equal(initial[name], final[name]) for name in initial.files)
        if stage['method'] != 'none':
            trace = stage['result']['trace']
            assert stage['result']['accepted_updates'] >= 1
            assert all(row['completed_elapsed'] <= 1. for row in trace if row['accepted'])
            assert sum(row['rolled_back'] for row in trace) <= 1
        assert fit['readout_initial'] == fits[0]['readout_initial']
    with pytest.raises(FileExistsError):
        runner.run(folder, config, lambda: None)


@pytest.mark.parametrize('budget', [0., -1., float('nan'), float('inf'), 100., 1, True])
def test_invalid_pretraining_budgets_rejected_before_generation(tmp_path, monkeypatch, budget):
    def forbidden(*args, **kwargs):
        raise AssertionError('No arrays before settings validation')
    monkeypatch.setattr(runner, 'generate_attempt_split', forbidden)
    with pytest.raises(ValueError, match='budget'):
        runner.run(tmp_path / 'never-created', {'pretraining_seconds': budget}, lambda: None)
    assert not (tmp_path / 'never-created').exists()


@pytest.mark.parametrize('status', ['FAILED_UPDATE_CAP', 'PASS'])
def test_failed_or_zero_update_stage_is_saved_before_run_rejection(tmp_path, monkeypatch, status):
    from test_finite_expected_count_bridge import public_histories
    prefix, lengths = public_histories()
    model = runner.learned_model('factorized', 938101)
    attempted = {'status': status, 'head_unchanged': True, 'accepted_updates': 0,
                 'attempted_updates': 1, 'trace': [{'accepted': False, 'rolled_back': True}],
                 'timed_seconds': 1.1, 'overrun_seconds': .1}
    monkeypatch.setattr(runner, 'timed_fit', lambda *a, **k: attempted)
    counts = {'pretraining_checkpoint_writes': 0, 'pretraining_runs': 0}
    with pytest.raises(ValueError):
        runner.pretrain(model, 'em_prefix', 938101, {'prefix': prefix, 'lengths': lengths},
                        {'pretraining_seconds': 1.}, tmp_path, counts, lambda: None)
    row = json.loads((tmp_path / 'pretraining.jsonl').read_text())
    assert row['result'] == attempted
    assert (tmp_path / row['initial_checkpoint']['path']).is_file()
    assert (tmp_path / row['final_checkpoint']['path']).is_file()
    assert counts == {'pretraining_checkpoint_writes': 2, 'pretraining_runs': 1}
