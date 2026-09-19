"""Actor/label separation and batching checks for the dialogue development study."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    'study_dialogue_memory', Path(__file__).resolve().parents[1] / 'scripts/study_dialogue_memory.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def fixture():
    features = np.random.default_rng(44).normal(size=(10, 384)).astype('float32')
    queries = [{'text': 4, 'candidates': [5, 6, 7]}, {'text': 4, 'candidates': [5, 6]}]
    ds = [{'id': 'a', 'turns': [0, 1, 2], 'queries': [
        {'query': 0, 'time': 0, 'label': 0, 'bin': 'unmentioned_retention', 'unseen': False},
        {'query': 0, 'time': 2, 'label': 2, 'bin': 'first_assignment', 'unseen': False}]},
          {'id': 'b', 'turns': [3], 'queries': [
              {'query': 1, 'time': 0, 'label': 1, 'bin': 'revision', 'unseen': True}]}]
    return ds, queries, features


def test_labels_bins_and_unseen_flags_never_enter_actor():
    ds, queries, features = fixture()
    old_actor, _, _ = study.make_batch(ds, queries, features)
    changed = copy.deepcopy(ds)
    for d in changed:
        for q in d['queries']:
            q['label'] = 1
            q['bin'] = 'clear'
            q['unseen'] = not q['unseen']
    new_actor, _, _ = study.make_batch(changed, queries, features)
    assert all(torch.equal(a, b) for a, b in zip(old_actor, new_actor, strict=True))


def test_padding_does_not_become_a_training_target():
    ds, queries, features = fixture()
    actor, labels, bins = study.make_batch(ds, queries, features)
    turns, valid, _, candidates, times, mask = actor
    assert turns.shape == (2, 3, 384)
    assert valid.tolist() == [[True, True, True], [True, False, False]]
    assert labels.tolist() == [[0, 2], [1, -100]]
    assert bins.tolist() == [[0, 2], [2, 0]]
    assert mask[1, 1].tolist() == [True, False, False]
    assert times[1, 1] == 0
    assert torch.count_nonzero(candidates[1, 1]) == 0


def test_loss_bins_are_explicit_not_silent_fallback():
    assert [study.loss_bin(x) for x in ['unmentioned_retention', 'assigned_retention',
                                       'first_assignment', 'revision', 'clear']] == [0, 1, 2, 2, 2]
    with pytest.raises(ValueError):
        study.loss_bin('unknown')


@pytest.mark.parametrize('method', study.METHODS)
def test_actual_models_accept_batch_and_have_finite_training_gradients(method):
    ds, queries, features = fixture()
    actor, labels, _ = study.make_batch(ds, queries, features)
    torch.manual_seed(55)
    model = study.make_model(method)
    logits = model(*actor)
    valid = labels != -100
    loss = torch.nn.functional.cross_entropy(logits[valid], labels[valid])
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_failed_receipt_write_keeps_original_error(tmp_path):
    original = ValueError('original training error')
    study.preserve_failure(tmp_path / 'missing' / 'failed.json', {}, original)
    assert str(original) == 'original training error'
    assert any('Failure receipt also failed' in x for x in original.__notes__)


def test_matrix_initializations_are_paired():
    hashes = []
    for method in ['gated_delta', 'kalman', 'innovation_kalman']:
        torch.manual_seed(5)
        hashes.append(study.tensor_digest(study.make_model(method)))
    assert len(set(hashes)) == 1


def test_diagnostics_do_not_change_predictions(tmp_path):
    ds, queries, features = fixture()
    actor, _, _ = study.make_batch(ds, queries, features)
    model = study.make_model('innovation_kalman').eval()
    with torch.inference_mode():
        expected = model(*actor).softmax(-1)
    receipt = study.evaluate(model, ds, queries, features, tmp_path / 'predictions.npz')
    assert receipt['innovation_diagnostics']['real_turn_updates'] == 4
    with np.load(tmp_path / 'predictions.npz') as saved:
        i = 0
        for b, d in enumerate(ds):
            for q, row in enumerate(d['queries']):
                n = len(queries[row['query']]['candidates'])
                np.testing.assert_array_equal(saved['probabilities'][i, :n], expected[b, q, :n].numpy())
                i += 1


def test_references_use_public_prefix_without_gold(tmp_path):
    queries = [{'candidate_values': [None, None, 'red', 'blue'],
                'candidate_ids': ['reserved:NOT_MENTIONED', 'reserved:DONTCARE', 'value:red', 'value:blue']}]
    ds = [{'id': 'x', 'user_text': ['red please', 'thanks', 'blue instead'],
           'queries': [{'query': 0, 'time': t, 'label': 3, 'bin': 'revision', 'unseen': False}
                       for t in range(3)]}]
    study.reference_predictions(ds, queries, tmp_path / 'refs.npz')
    with np.load(tmp_path / 'refs.npz') as saved:
        assert saved['pred_literal'].tolist() == [2, 2, 3]
        assert saved['pred_none'].tolist() == [0, 0, 0]
