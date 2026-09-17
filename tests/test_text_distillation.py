import copy
import json
import os

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from openjev.research.text_distillation import (
    PROTOCOL,
    context_key,
    digest,
    file_hash,
    load_packet,
    select_rows,
    source_paths,
    write_new,
)
from openjev.research.text_student import CandidateStudent, categorical_metrics
from openjev.research.text_teacher import (
    build_payload,
    checked_result,
    durable_new,
    load_key,
    reserve,
    strict_json,
)


def record():
    return {'id': 'fixture:train:1', 'task': 'fixture', 'context': 'The door is blue.',
            'question': 'Is the door blue?', 'gold': 'yes',
            'candidates': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}]}


def test_teacher_has_no_gold_or_metadata_and_schema_restricts_choices():
    row = record()
    payload = build_payload(row)
    changed = {**row, 'gold': 'no', 'id': 'eval:SECRET', 'task': 'PRIVATE', 'source_split': 'evaluation'}
    assert payload == build_payload(changed)
    assert json.loads(payload['input'][1]['content']) == {k: v for k, v in row.items()
                                                         if k in ('context', 'question')} | {
        'candidates': sorted(row['candidates'], key=lambda c: c['description'])}
    assert payload['model'] == 'gpt-6-astra'
    assert payload['store'] is False and payload['tools'] == []
    assert payload['text']['format']['schema']['properties']['choice']['enum'] == ['no', 'yes']


def test_teacher_validation_rejects_incomplete_wrong_model_duplicate_or_invalid_labels():
    payload = build_payload(record())
    raw = {'id': 'response-fixture', 'model': 'gpt-6-astra', 'status': 'completed',
           'usage': {'input_tokens': 500, 'output_tokens': 40},
           'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"choice":"yes"}'}]}]}
    assert checked_result(raw, {'yes', 'no'}, reserve(payload))[0] == 'yes'
    for change in ({'status': 'incomplete'}, {'model': 'other'}, {'usage': {}},
                   {'usage': {'input_tokens': True, 'output_tokens': 1}}):
        with pytest.raises(ValueError):
            checked_result(raw | change, {'yes', 'no'}, reserve(payload))
    for text in ('{"choice":"yes","choice":"no"}', '{"choice":"maybe"}', '{"choice":"yes","reason":"x"}'):
        bad = copy.deepcopy(raw)
        bad['output'][0]['content'][0]['text'] = text
        with pytest.raises(ValueError):
            checked_result(bad, {'yes', 'no'}, reserve(payload))
    with pytest.raises(ValueError):
        strict_json('{"x":NaN}')


def test_receipts_are_exclusive_and_key_file_permissions_are_checked(tmp_path, monkeypatch):
    path = tmp_path / 'receipt.json'
    durable_new(path, {'ok': True})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        durable_new(path, {'ok': False})
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError):
        load_key()
    key = tmp_path / 'key.env'
    key.write_text('OPENAI_API_KEY=test-fixture-secret\n')
    key.chmod(0o644)
    with pytest.raises(ValueError):
        load_key(key)
    key.chmod(0o600)
    assert load_key(key) == 'test-fixture-secret'


def test_sampling_balances_classes_and_packet_rejects_overlap_and_tampering(tmp_path):
    rows = [record() | {'id': str(i), 'gold': 'yes' if i % 2 else 'no'} for i in range(20)]
    picked = select_rows(rows, {'yes': 3, 'no': 4}, 11)
    assert picked == select_rows(rows, {'yes': 3, 'no': 4}, 11)
    assert sum(r['gold'] == 'yes' for r in picked) == 3
    assert context_key(record()) == context_key(record() | {'context': 'THE DOOR IS BLUE. '})
    packet = {'protocol': PROTOCOL, 'train': [record()], 'evaluation': [record() | {'id': 'different'}]}
    write_new(tmp_path / 'packet.json', packet)
    manifest = {'packet_sha256': digest(packet), 'code_sha256': {p.name: file_hash(p) for p in source_paths()}}
    write_new(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='overlap'):
        load_packet(tmp_path)
    packet['train'][0]['gold'] = 'no'
    (tmp_path / 'packet.json').write_text(json.dumps(packet))
    with pytest.raises(ValueError, match='mismatch'):
        load_packet(tmp_path)


def test_probability_metrics_have_known_values_and_reject_invalid_choice():
    row = {'probabilities': {'yes': .75, 'no': .25}, 'gold': 'yes', 'choice': 'yes', 'latency_ms': 2.}
    result = categorical_metrics([row])
    assert result['accuracy'] == 1.
    assert result['nll'] == pytest.approx(.287682072)
    assert result['multiclass_brier'] == pytest.approx(.125)
    with pytest.raises(ValueError):
        categorical_metrics([row | {'choice': 'no'}])
    with pytest.raises(ValueError):
        categorical_metrics([row | {'probabilities': {'yes': .8, 'no': .3}}])


def test_teacher_stops_after_one_failure_without_retry_or_partial_training(tmp_path, monkeypatch):
    from openjev.research import text_teacher
    calls = []
    class FailureResponse:
        status_code = 429
        def __init__(self):
            self.headers = {'x-request-id': 'fixture-request'}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def iter_bytes(self):
            yield b'{"error":"fixture rate limit"}'
    class FixtureClient:
        def __init__(self, **kwargs):
            assert kwargs['follow_redirects'] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def stream(self, method, url, **kwargs):
            calls.append((method, url))
            return FailureResponse()
    monkeypatch.setattr(text_teacher, 'load_packet', lambda _: (
        {'train': [record(), record() | {'id': 'fixture:2'}]}, {'packet_sha256': 'fixture'}))
    monkeypatch.setattr(text_teacher, 'load_key', lambda _: 'fixture-secret')
    monkeypatch.setattr(text_teacher.httpx, 'Client', FixtureClient)
    with pytest.raises(RuntimeError, match='incomplete'):
        text_teacher.label(tmp_path, tmp_path / 'teacher')
    assert len(calls) == 1
    summary = json.loads((tmp_path / 'teacher/summary.json').read_text())
    assert summary['completed'] == 0 and summary['attempted'] == 1 and summary['planned'] == 2
    assert summary['unknown_cost_reserved_usd'] > 0
    with pytest.raises(ValueError, match='never silently retry'):
        text_teacher.label(tmp_path, tmp_path / 'teacher')
    assert len(calls) == 1


def test_paired_interval_preserves_matching_items_and_fits():
    import numpy as np

    from openjev.research.text_report import paired_interval
    a = np.ones((3, 4))
    result = paired_interval(a, a, ['yes', 'no', 'yes', 'no'])
    assert result == {'delta': 0., 'exploratory_95_percent_interval': [0., 0.]}


@pytest.mark.skipif(os.environ.get('OPENJEV_TEST_TEXT_ENCODER') != '1', reason='Requires pinned local encoder')
def test_real_encoder_gradient_candidate_order_and_checkpoint_roundtrip(tmp_path):
    model = CandidateStudent(seed=17, device='cpu')
    row = record()
    before = model.decide_record(row)
    swapped = model.decide_record(row | {'candidates': list(reversed(row['candidates']))})
    assert before['probabilities'] == swapped['probabilities']
    inputs, _candidates = model.tokenize(row)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(inputs).unsqueeze(0), torch.tensor([1]))
    loss.backward()
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    optimizer.step()
    after = model.decide_record(row)
    assert after['probabilities'] != before['probabilities']
    model.save(tmp_path / 'model')
    restored = CandidateStudent.load(tmp_path / 'model')
    assert restored.decide_record(row)['probabilities'] == after['probabilities']
    with pytest.raises(ValueError, match='no truncation'):
        model.tokenize(row | {'context': 'blue ' * 600})
