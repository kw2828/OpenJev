"""Synthetic-only mapping training, hostile saved-output audits and MPS smoke."""

import copy
import json
import random
from dataclasses import asdict

import chess
import chess.engine
import pytest
import torch

from openjev.research import chess_connectome_mapping_study as study
from openjev.research import chess_connectome_study as original
from openjev.research.chess_candidate import CandidateChess
from openjev.research.chess_candidate_eval import CachedPositions
from openjev.research.connectome_graph import SignedGraph

PLAN = 'a' * 64
DATA = 'c' * 64


@pytest.fixture(autouse=True)
def bounded_execution(monkeypatch):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)

    def forbidden(*args, **kwargs):
        raise AssertionError('Synthetic tests must never start an engine')

    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', forbidden)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture
def private_runs(tmp_path, monkeypatch):
    folder = tmp_path / 'runs'
    folder.mkdir()
    monkeypatch.setattr(study, 'RUNS_ROOT', folder)
    return folder


def graph():
    nodes = 131
    return SignedGraph(node_ids=list(range(nodes)), groups=[0] * nodes,
                       sources=list(range(nodes)), destinations=[(i + 1) % nodes for i in range(nodes)],
                       signs=[1 if i % 2 else -1 for i in range(nodes)],
                       provenance={'fixture': 'synthetic-only-no-biological-assets'})


def rows():
    fens = [chess.STARTING_FEN, 'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 12 20',
            '7k/8/8/8/3Pp3/8/8/7K b - d3 0 12', '7k/P7/8/8/8/8/8/7K w - - 7 19']
    return [{'id': f'synthetic-{i}', 'game_id': i, 'fen': fen,
             'target_uci': max(m.uci() for m in chess.Board(fen).legal_moves),
             'target_value': (-1) ** i * .2} for i, fen in enumerate(fens)]


def config(policy='learned', variant='biological', seed=97):
    return next(c for c in study.configurations() if c['mapping_policy'] == policy
                and c['variant'] == variant and c['seed'] == seed)


def settings(**overrides):
    return study.TrainingSettings(**({'train_examples': 4, 'epochs': 4, 'batch_size': 2,
                                      'device': 'cpu', 'warmup_updates': 1,
                                      'proposal_interval': 1} | overrides))


def backbone():
    return CandidateChess('direct', width=4, seed=97)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, records):
    path.write_text(''.join(json.dumps(item) + '\n' for item in records))


def fit_fixture(private_runs, *, policy='learned', device='cpu', directory='fit'):
    c, b, g, cache, s = config(policy), backbone(), graph(), CachedPositions(rows(), include_successors=False), settings(device=device)
    path = private_runs / directory
    result = study.fit(c, b, g, cache, path, plan_sha256=PLAN, data_sha256=DATA, settings=s)
    return path, c, b, g, cache, s, result


def audit_fixture(fixture):
    path, c, b, g, cache, s, _ = fixture
    return study.audit_training(path, c, PLAN, DATA, cache.metadata, b, g,
                                settings=s, training_rows=cache.rows)


def load_fixture(fixture):
    path, c, b, g, _, _, result = fixture
    return study.load_checkpoint(path / 'weights.pt', b, g, expected_plan_sha256=PLAN,
                                 expected_config=c, expected_backbone_sha256=study.state_sha256(b),
                                 expected_checkpoint_sha256=result['checkpoint_sha256'])


def refresh_logs(fixture):
    path, _, _, _, _, _, _ = fixture
    receipt = json.loads((path / 'training.json').read_text())
    for name in ('learning', 'proposals'):
        receipt[f'{name}_sha256'] = study.sha256(path / f'{name}.jsonl')
    (path / 'training.json').write_text(json.dumps(receipt))


def test_configs_and_default_schedules_preserve_original_minibatches_and_global_rng():
    configs = study.configurations()
    assert len(configs) == len({c['name'] for c in configs}) == 30
    assert {c['variant'] for c in configs} == set(study.VARIANTS)
    assert not any(c['variant'] == 'dense' for c in configs)
    torch_before, python_before = torch.random.get_rng_state().clone(), random.getstate()
    default = study.DEFAULT_TRAINING
    assert default.updates == 1536 and default.proposals == 320
    assert asdict(default) == asdict(original.DEFAULT_TRAINING) | {
        'warmup_updates': 256, 'proposal_interval': 4, 'proposal_seed_base': 11600041}
    for c in configs:
        old = study.legacy_config(c)
        assert old in original.configurations()
        assert study.schedule(c['seed']) == original.schedule(c['seed'])
        proposals = study.proposal_schedule(c['seed'])
        assert [p['before_update'] for p in proposals] == list(range(257, 1534, 4))
        assert [p['proposal_index'] for p in proposals] == list(range(1, 321))
        assert proposals == study.proposal_schedule(c['seed'])
        assert all(len(set(p['swap'])) == 2 and min(p['swap']) >= 0 and max(p['swap']) < 64 for p in proposals)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert python_before == random.getstate()


@pytest.mark.parametrize('overrides', [{'warmup_updates': -1}, {'warmup_updates': True},
                                      {'warmup_updates': 8}, {'proposal_interval': 0},
                                      {'proposal_interval': False}, {'proposal_seed_base': -1},
                                      {'train_examples': 0}, {'device': 'cuda'}])
def test_invalid_budgets_fail(overrides):
    with pytest.raises(ValueError):
        settings(**overrides)


@pytest.mark.parametrize('policy', ['fixed', 'learned'])
def test_real_synthetic_fit_has_complete_counts_and_saved_output_only_audit(private_runs, monkeypatch, policy):
    fixture = fit_fixture(private_runs, policy=policy)
    path, c, _, _, _, s, result = fixture
    assert result['updates'] == 8 and result['proposal_count'] == 7
    assert result['proposal_forward_calls'] == 14 and result['total_forward_calls'] == 22
    assert result['proposal_optimizer_updates'] == 0 and result['backward_calls'] == 8
    proposals = read_rows(path / 'proposals.jsonl')
    assert len(proposals) == s.proposals
    assert all(p['outer_wall_seconds'] >= p['wall_seconds'] >= 0 for p in proposals)
    if policy == 'fixed':
        assert result['accepted_proposals'] == 0 and result['final_permutation'] == list(range(64))
    else:
        # These fixed synthetic roots and deterministic initialization give a
        # genuine accepted proposal, not a fabricated receipt or forced loss.
        assert result['accepted_proposals'] > 0
        assert result['final_permutation'] != list(range(64))
    restored = load_fixture(fixture)
    assert restored.config['configuration'] == c
    assert restored.config['architecture'] == study.VERSION
    assert restored.square_permutation.tolist() == result['final_permutation']

    def no_forward(*args, **kwargs):
        raise AssertionError('Saved-output audit must not run inference')

    monkeypatch.setattr(study.MappingStudyAdapter, 'forward', no_forward)
    assert audit_fixture(fixture) == result


def test_fixed_and_learned_have_same_initial_state_work_and_swap_schedule(private_runs):
    fixed = fit_fixture(private_runs, policy='fixed', directory='fixed')
    learned = fit_fixture(private_runs, policy='learned', directory='learned')
    assert fixed[-1]['initial_state_sha256'] == learned[-1]['initial_state_sha256']
    for key in ('computation', 'proposal_computation', 'updates', 'proposal_count', 'parameter_counts'):
        assert fixed[-1][key] == learned[-1][key]
    a, b = (read_rows(f[0] / 'proposals.jsonl') for f in (fixed, learned))
    assert [(p['before_update'], p['swap'], p['indices_sha256']) for p in a] == [
        (p['before_update'], p['swap'], p['indices_sha256']) for p in b]


@pytest.mark.parametrize('fault', ['proposal_loss', 'proposal_accept', 'proposal_chain', 'proposal_swap',
                                  'proposal_update', 'proposal_boolean', 'proposal_nonfinite',
                                  'proposal_missing', 'proposal_duplicate', 'proposal_order',
                                  'update_permutation', 'update_link', 'update_loss', 'update_indices',
                                  'update_missing', 'stale_batch', 'time', 'counts'])
def test_audit_rejects_tampering_even_with_refreshed_journal_hashes(private_runs, fault):
    fixture = fit_fixture(private_runs)
    path = fixture[0]
    p, logs = read_rows(path / 'proposals.jsonl'), read_rows(path / 'learning.jsonl')
    if fault == 'proposal_loss':
        p[0]['proposed']['loss'] += .2
    elif fault == 'proposal_accept':
        p[0]['accepted'] = not p[0]['accepted']
    elif fault == 'proposal_chain':
        p[1]['starting_permutation'][0:2] = reversed(p[1]['starting_permutation'][0:2])
    elif fault == 'proposal_swap':
        p[0]['swap'].reverse()
    elif fault == 'proposal_update':
        p[0]['before_update'] += 1
    elif fault == 'proposal_boolean':
        p[0]['proposal_index'] = True
    elif fault == 'proposal_nonfinite':
        p[0]['proposed']['policy_ce'] = float('nan')
    elif fault == 'proposal_missing':
        p.pop()
    elif fault == 'proposal_duplicate':
        p[1] = copy.deepcopy(p[0])
    elif fault == 'proposal_order':
        p.reverse()
    elif fault == 'update_permutation':
        logs[0]['permutation'][0:2] = [1, 0]
    elif fault == 'update_link':
        logs[1]['proposal_sha256'] = 'e' * 64
    elif fault == 'update_loss':
        logs[0]['loss'] += .1
    elif fault == 'update_indices':
        logs[0]['indices_sha256'] = 'f' * 64
    elif fault == 'update_missing':
        logs.pop()
    elif fault == 'stale_batch':
        logs[1]['policy_ce'] += .25
        logs[1]['loss'] = float(torch.tensor(logs[1]['policy_ce'], dtype=torch.float32)
                               + .5 * torch.tensor(logs[1]['value_mse'], dtype=torch.float32))
    elif fault == 'time':
        p[0]['outer_wall_seconds'] = -1
    elif fault == 'counts':
        p[0]['computation_per_forward']['node_state_updates'] += 1
    # Bind changed proposal bytes in the update to exercise the semantic audit.
    for record in p:
        index = record['before_update'] - 1
        if 0 <= index < len(logs) and fault != 'update_link':
            logs[index]['proposal_sha256'] = study._digest(record) if fault != 'proposal_nonfinite' else 'a' * 64
    write_rows(path / 'proposals.jsonl', p)
    write_rows(path / 'learning.jsonl', logs)
    refresh_logs(fixture)
    with pytest.raises(ValueError):
        audit_fixture(fixture)


@pytest.mark.parametrize('tensor', ['node_slots', 'slot_occupancy', 'square_permutation',
                                   'original_node_slots', 'original_slot_occupancy', 'sources', 'signs',
                                   'indegree', 'node_ids', 'groups', 'backbone.input.weight'])
def test_checkpoint_rejects_changed_frozen_or_inconsistent_derived_tensors(private_runs, tensor):
    fixture = fit_fixture(private_runs)
    path, c, b, g, _, _, _ = fixture
    checkpoint = path / 'weights.pt'
    payload = torch.load(checkpoint, weights_only=True)
    if tensor == 'backbone.input.weight':
        tensor = next(k for k in payload['state_dict'] if k.startswith('backbone.') and payload['state_dict'][k].numel())
    payload['state_dict'][tensor].view(-1)[0] += 1
    payload['state_sha256'] = study._state_sha(payload['state_dict'])
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        study.load_checkpoint(checkpoint, b, g, expected_plan_sha256=PLAN, expected_config=c,
                              expected_backbone_sha256=study.state_sha256(b),
                              expected_checkpoint_sha256=study.sha256(checkpoint))


@pytest.mark.parametrize('field', ['configuration_sha256', 'source_sha256', 'plan_sha256',
                                  'graph_sha256', 'immutable_buffers_sha256'])
def test_checkpoint_rejects_changed_bindings(private_runs, field):
    fixture = fit_fixture(private_runs)
    path, c, b, g, _, _, _ = fixture
    checkpoint = path / 'weights.pt'
    payload = torch.load(checkpoint, weights_only=True)
    payload[field] = {} if field == 'source_sha256' else 'f' * 64
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        study.load_checkpoint(checkpoint, b, g, expected_plan_sha256=PLAN, expected_config=c,
                              expected_backbone_sha256=study.state_sha256(b),
                              expected_checkpoint_sha256=study.sha256(checkpoint))


def test_valid_checkpoint_mapping_disconnected_from_proposal_chain_is_rejected(private_runs):
    fixture = fit_fixture(private_runs)
    path, c, b, _, _, _, result = fixture
    model = load_fixture(fixture)
    model.restore_permutation()
    checkpoint = path / 'other.pt'
    changed = study.save_checkpoint(model, checkpoint, plan_sha256=PLAN, configuration=c,
                                    backbone_sha256=study.state_sha256(b))
    checkpoint.replace(path / 'weights.pt')
    result.update(changed)
    (path / 'training.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Final training state or mapping'):
        audit_fixture(fixture)


def test_local_boundary_and_exclusive_checkpoint_writes(private_runs, tmp_path):
    model = study.make_model(config(), backbone(), graph())
    kwargs = {'plan_sha256': PLAN, 'configuration': config(),
              'backbone_sha256': study.state_sha256(model.backbone)}
    with pytest.raises(ValueError):
        study.save_checkpoint(model, tmp_path / 'public.pt', **kwargs)
    out = private_runs / 'model.pt'
    study.save_checkpoint(model, out, **kwargs)
    digest = study.sha256(out)
    with pytest.raises(FileExistsError):
        study.save_checkpoint(model, out, **kwargs)
    assert study.sha256(out) == digest
    alias = private_runs / 'link.pt'
    alias.symlink_to(out)
    with pytest.raises(ValueError):
        study.save_checkpoint(model, alias, **kwargs)


@pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit, RuntimeError])
def test_baseexception_failure_receipt_keeps_completed_work_and_restores_runtime(private_runs, monkeypatch, exception):
    original_proposal = study.evaluate_square_swap
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise exception('synthetic deadline/failure')
        return original_proposal(*args, **kwargs)

    monkeypatch.setattr(study, 'evaluate_square_swap', fail)
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(exception):
        fit_fixture(private_runs)
    failed = json.loads((private_runs / 'fit' / 'failed.json').read_text())
    assert failed['updates_completed'] == 2 and failed['proposals_completed'] == 1
    assert failed['error_type'] == exception.__name__
    assert not (private_runs / 'fit' / 'training.json').exists()
    assert torch.get_num_threads() == previous_threads
    assert torch.are_deterministic_algorithms_enabled() == previous_determinism


def test_deadline_during_source_binding_is_preserved_before_first_update(private_runs, monkeypatch):
    def interrupted():
        raise KeyboardInterrupt('Synthetic setup deadline')

    monkeypatch.setattr(study, 'source_bindings', interrupted)
    with pytest.raises(KeyboardInterrupt):
        fit_fixture(private_runs)
    failed = json.loads((private_runs / 'fit' / 'failed.json').read_text())
    assert failed['updates_completed'] == failed['proposals_completed'] == 0
    assert failed['error_type'] == 'KeyboardInterrupt'


def legacy_roundtrip(fixture, destination):
    _, c, _, _, cache, _, result = fixture
    model = load_fixture(fixture)
    legacy = study.legacy_config(c)
    output = destination / 'predictions.jsonl'
    saved = original.evaluate(model, cache, output, configuration=legacy, plan_sha256=PLAN, batch_size=2)
    metrics, _ = original.audit_evaluation(saved, output, cache.rows, configuration=legacy,
                                           plan_sha256=PLAN, model=model, cache_metadata=cache.metadata)
    assert metrics['examples'] == 4
    assert saved['model']['architecture']['configuration'] == c
    assert saved['model']['state_sha256'] == result['state_sha256']
    timing = original.measure_latency(model, cache.rows, [0], configuration=legacy, plan_sha256=PLAN)
    assert timing['forward_calls'] == 4
    assert timing['records'][0]['decision']['mapping_policy'] == c['mapping_policy']
    return model


def test_cpu_checkpoint_retains_new_identity_through_legacy_evaluation_and_latency(private_runs):
    fixture = fit_fixture(private_runs)
    legacy_roundtrip(fixture, private_runs)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Actual MPS unavailable')
def test_mps_synthetic_fit_save_cpu_load_audit_and_legacy_eval_without_fallback(private_runs, monkeypatch):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    devices = []
    forward = study.MappingStudyAdapter.forward

    def traced(self, observations, *args, **kwargs):
        devices.append(str(observations.device))
        return forward(self, observations, *args, **kwargs)

    monkeypatch.setattr(study.MappingStudyAdapter, 'forward', traced)
    fixture = fit_fixture(private_runs, device='mps')
    assert len(devices) == 22 and set(devices) == {'mps:0'}
    assert fixture[-1]['accepted_proposals'] > 0
    assert audit_fixture(fixture) == fixture[-1]
    assert len(devices) == 22  # Audit added no forward calls.
    legacy_roundtrip(fixture, private_runs)
    assert set(devices[22:]) == {'cpu'}
