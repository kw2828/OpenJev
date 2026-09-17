"""Synthetic fixtures only: these tests never dispatch teachers or load a model."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('transformers', reason='Install research dependencies for text distillation')

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_test_codex_astra_distillation',
                                            ROOT / 'scripts' / 'codex_astra_distillation.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def dummy_rows(count, split):
    return [{'id': f'fixture:{split}:{index}', 'context': f'Synthetic {split} context {index}.',
             'question': 'Which dummy candidate?', 'task': 'boolq' if index % 2 else 'clinc_domains',
             'gold': 'a' if index % 3 else 'b', 'source_split': split,
             'candidates': [{'id': 'b', 'description': 'Second dummy'},
                            {'id': 'a', 'description': 'First dummy'}]} for index in range(count)]


def response_for(batch):
    return {'batch_id': batch['batch_id'], 'choices': [
        {'id': item['id'], 'choice': item['candidates'][0]['id']} for item in batch['items']]}


def test_batches_are_allowlisted_shuffled_and_cover_only_every_training_item():
    rows = dummy_rows(128, 'train')
    for row in rows:
        row['evaluation'] = {'gold': 'PROTECTED_EVAL_MARKER'}
        row['candidates'][0]['teacher_hint'] = 'PROTECTED_CANDIDATE_MARKER'
    batches, mapping = study.make_batches(rows)
    assert len(batches) == 4 and all(len(batch['items']) == 32 for batch in batches)
    assert {item['source_id'] for item in mapping.values()} == {row['id'] for row in rows}
    assert len(mapping) == 128
    assert [item['source_id'] for item in mapping.values()] != [row['id'] for row in rows]
    for batch in batches:
        assert set(batch) == {'batch_id', 'items'}
        for item in batch['items']:
            assert set(item) == {'id', 'context', 'question', 'candidates'}
            assert item['id'].startswith('item-') and ':' not in item['id']
            assert all(set(candidate) == {'id', 'description'} for candidate in item['candidates'])
            assert mapping[item['id']]['item_sha256'] == study.digest(item)
    assert 'PROTECTED_' not in json.dumps(batches)
    assert 'fixture:train:' not in json.dumps(batches)
    assert study.make_batches(rows) == (batches, mapping)
    changed = copy.deepcopy(rows)
    for row in changed:
        row.update(gold='MODIFIED_GOLD', task='MODIFIED_TASK', source_split='MODIFIED_METADATA')
        row['evaluation'] = None
    assert study.make_batches(changed) == (batches, mapping)
    with pytest.raises(ValueError, match='128 distinct'):
        study.make_batches(rows[:-1])
    with pytest.raises(ValueError, match='128 distinct'):
        study.make_batches([rows[0]] * 128)


def test_student_inputs_strip_parent_gold_metadata_and_nested_candidate_hints():
    row = dummy_rows(1, 'train')[0]
    row['candidates'][0]['gold_hint'] = True
    result = study.student_row(row)
    assert set(result) == {'id', 'context', 'question', 'candidates'}
    assert all(set(candidate) == {'id', 'description'} for candidate in result['candidates'])
    assert result['id'] == row['id']


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'reorder', 'candidate', 'extra',
                                  'wrong_batch', 'nonstring', 'root_extra'])
def test_worker_response_rejects_partial_extra_reordered_or_invalid_choices(change):
    batch = study.make_batches(dummy_rows(128, 'train'))[0][0]
    raw = response_for(batch)
    if change == 'missing':
        raw['choices'].pop()
    elif change == 'duplicate':
        raw['choices'][1] = raw['choices'][0]
    elif change == 'reorder':
        raw['choices'].reverse()
    elif change == 'candidate':
        raw['choices'][0]['choice'] = 'unlisted'
    elif change == 'extra':
        raw['choices'][0]['confidence'] = .99
    elif change == 'wrong_batch':
        raw['batch_id'] = 'batch-unknown'
    elif change == 'nonstring':
        raw['choices'][0]['choice'] = ['a']
    else:
        raw['reasoning'] = 'not allowed'
    with pytest.raises(ValueError):
        study.checked_choices(raw, batch)


def test_strict_response_json_rejects_duplicate_keys_and_nonfinite_numbers():
    with pytest.raises(ValueError, match='Duplicate JSON'):
        study.strict_json('{"choice":"a","choice":"b"}')
    with pytest.raises(ValueError, match='Nonfinite JSON'):
        study.strict_json('{"choice":NaN}')


@pytest.fixture
def prepared(tmp_path):
    packet_dir, study_dir = tmp_path / 'original' / 'packet', tmp_path / 'codex-study'
    packet = {'protocol': study.frozen.PROTOCOL,
              'train': dummy_rows(128, 'train'), 'evaluation': dummy_rows(172, 'evaluation')}
    manifest = {'packet_sha256': study.digest(packet),
                'code_sha256': {path.name: study.sha(path) for path in study.frozen.source_paths()}}
    study.durable_new(packet_dir / 'packet.json', packet)
    study.durable_new(packet_dir / 'manifest.json', manifest)
    for arm in ('untrained', 'gold'):
        directory = packet_dir.parent / arm
        study.durable_new(directory / 'completed.json', {
            'status': 'completed', 'arm': arm, 'packet_sha256': manifest['packet_sha256'],
            'fit_seeds': study.PROTOCOL['fit_seeds']})
        for seed in study.PROTOCOL['fit_seeds']:
            study.durable_new(directory / f'result-{seed}.json', {
                'arm': arm, 'seed': seed, 'packet_sha256': manifest['packet_sha256'], 'training_seconds': 0.,
                'predictions': [{'id': row['id'], 'task': row['task'], 'gold': row['gold'], 'choice': 'a',
                                 'probabilities': {'a': .7, 'b': .3}, 'latency_ms': 1.}
                                for row in packet['evaluation']]})
    study.prepare(packet_dir, study_dir)
    return study_dir, packet


def complete_batch(study_dir, batch_id, agent_id=None):
    study.start(study_dir, batch_id)
    batch = json.loads((study_dir / 'packets' / f'{batch_id}.json').read_text())
    study.durable_new(study_dir / 'responses' / f'{batch_id}.json', response_for(batch))
    study.accept(study_dir, batch_id, agent_id or f'synthetic-test-agent-{batch_id}')


def complete_teachers(study_dir):
    for index in range(4):
        complete_batch(study_dir, f'batch-{index:03d}')
    study.seal(study_dir)


def test_freeze_preserves_explicit_model_and_training_only_membership(prepared):
    study_dir, packet = prepared
    plan, verified_packet, mapping = study.verify(study_dir)
    assert verified_packet == packet and plan['protocol']['producer'] == 'codex_collaboration'
    assert {row['source_id'] for row in mapping.values()} == {row['id'] for row in packet['train']}
    assert not {row['source_id'] for row in mapping.values()} & {row['id'] for row in packet['evaluation']}
    for record in plan['batches']:
        path = study_dir / 'requests' / f"{record['batch_id']}.json"
        request = json.loads(path.read_text())
        assert (request['model'], request['reasoning_effort'], request['fork_turns']) == ('gpt-6-astra', 'low', 'none')
        assert 'groups of eight' in request['message']
        assert 'exactly 32 choices' in request['message']
    with pytest.raises(FileExistsError):
        study.prepare(Path(plan['source_packet_dir']), study_dir)
    path = study_dir / 'requests' / 'batch-000.json'
    request = json.loads(path.read_text())
    request['model'] = 'different-model'
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match='dispatch request changed'):
        study.verify(study_dir)


def test_started_or_failed_batch_cannot_retry_and_failure_is_preserved(prepared):
    study_dir, _ = prepared
    study.start(study_dir, 'batch-000')
    with pytest.raises(FileExistsError):
        study.start(study_dir, 'batch-000')
    study.durable_new(study_dir / 'responses' / 'batch-000.json', {'batch_id': 'batch-000', 'choices': []})
    with pytest.raises(ValueError, match='do not retry or train'):
        study.accept(study_dir, 'batch-000', 'synthetic-test-agent')
    receipt = json.loads((study_dir / 'receipts' / 'batch-000.json').read_text())
    assert receipt['status'] == 'failed' and receipt['accepted_items'] == 0
    assert receipt['api_response_id'] is None and receipt['api_token_usage'] is None
    assert receipt['api_cost_usd'] is None and receipt['direct_paid_api_calls'] == 0
    with pytest.raises(FileExistsError):
        study.accept(study_dir, 'batch-000', 'synthetic-test-agent')
    with pytest.raises((ValueError, FileNotFoundError)):
        study.seal(study_dir)
    assert not (study_dir / 'labels.json').exists()


def test_all_four_distinct_workers_are_required_and_responses_cannot_change(prepared):
    study_dir, _ = prepared
    complete_batch(study_dir, 'batch-000', 'synthetic-test-repeated-agent')
    with pytest.raises(FileNotFoundError):
        study.seal(study_dir)
    assert not (study_dir / 'labels.json').exists()
    complete_batch(study_dir, 'batch-001', 'synthetic-test-repeated-agent')
    with pytest.raises(ValueError, match='duplicate'):
        study.seal(study_dir)


def test_seal_validates_complete_source_mapping_and_rejects_later_tampering(prepared):
    study_dir, packet = prepared
    complete_teachers(study_dir)
    plan, _, mapping = study.verify(study_dir)
    labels = study.sealed_labels(study_dir, plan, mapping)
    assert labels == {row['id']: 'a' for row in packet['train']}
    assert not set(labels) & {row['id'] for row in packet['evaluation']}
    with pytest.raises(FileExistsError):
        study.seal(study_dir)
    path = study_dir / 'responses' / 'batch-000.json'
    changed = json.loads(path.read_text())
    changed['choices'][0]['choice'] = 'b'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='provenance'):
        study.sealed_labels(study_dir, plan, mapping)


@pytest.mark.parametrize('fault', ['short', 'updates', 'epoch', 'loss', 'time'])
def test_fit_history_must_prove_all_fixed_updates(fault):
    fit = {'history': [{'epoch': epoch, 'updates': 128, 'mean_loss': .1} for epoch in (1, 2, 3)],
           'training_seconds': 1.}
    study.validate_history(fit)
    if fault == 'short':
        fit['history'].pop()
    elif fault == 'time':
        fit['training_seconds'] = -1
    else:
        fit['history'][0][{'updates': 'updates', 'epoch': 'epoch', 'loss': 'mean_loss'}[fault]] = -1
    with pytest.raises(ValueError, match='three-epoch'):
        study.validate_history(fit)


def test_training_uses_teacher_labels_only_and_all_final_fits_precede_evaluation(prepared, monkeypatch, tmp_path):
    study_dir, packet = prepared
    complete_teachers(study_dir)
    fitted, evaluated = [], []

    class DummyStudent:
        def __init__(self, seed, device):
            self.seed = seed

        def save(self, folder):
            folder.mkdir()
            (folder / 'weights.pt').write_text(f'SYNTHETIC TEST CHECKPOINT {self.seed}')
            study.durable_new(folder / 'config.json', {'seed': self.seed})

        @classmethod
        def load(cls, folder, device):
            assert fitted == [17, 29, 43]
            return cls(json.loads((folder / 'config.json').read_text())['seed'], device)

        def decide_record(self, row):
            assert set(row) == {'id', 'context', 'question', 'candidates'}
            if ':evaluation:' in row['id']:
                assert fitted == [17, 29, 43]
                evaluated.append((self.seed, row['id']))
            return {'choice': 'a', 'probabilities': {'a': .7, 'b': .3}, 'latency_ms': 1.}

    def dummy_fit(model, rows, labels, seed):
        assert all(set(row) == {'id', 'context', 'question', 'candidates'} for row in rows)
        assert {row['id'] for row in rows} == {row['id'] for row in packet['train']}
        assert labels == {row['id']: 'a' for row in packet['train']}
        assert any(labels[row['id']] != row['gold'] for row in packet['train'])
        assert not evaluated and model.seed == seed
        fitted.append(seed)
        return [{'epoch': epoch, 'updates': 128, 'mean_loss': .1} for epoch in (1, 2, 3)], 1.

    monkeypatch.setattr(study, 'CandidateStudent', DummyStudent)
    monkeypatch.setattr(study.frozen, 'fit_student', dummy_fit)
    run_dir, out = tmp_path / 'synthetic-student', tmp_path / 'synthetic-aggregates'
    study.train(study_dir, run_dir, 'cpu')
    assert len(evaluated) == 172 * 3
    study.report(study_dir, run_dir, out)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['status'] == 'completed' and not summary['independent_confirmation']
    assert summary['arms']['gold']['reused_control'] and not summary['arms']['astra_codex']['reused_control']
    assert summary['codex_usage_cost'] == 'Unavailable' and not summary['novelty_established']
    assert 'Synthetic evaluation context' not in (out / 'summary.json').read_text()
    assert 'fixture:evaluation:' not in (out / 'summary.json').read_text()
    (run_dir / 'seed-17' / 'weights.pt').write_text('TAMPERED SYNTHETIC TEST WEIGHTS')
    with pytest.raises(ValueError, match='checkpoint'):
        study.report(study_dir, run_dir, tmp_path / 'rejected-aggregates')
