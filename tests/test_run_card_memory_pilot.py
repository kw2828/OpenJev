"""Fake-only pilot orchestration. No model, optimizer, native or RNG calls."""

import copy
import json

import card_memory_common as common
import numpy as np
import pytest
import run_card_memory_pilot as runner
import torch

from openjev.research.card_memory_training import _tensor_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def tensors(n):
    pos = np.broadcast_to(np.arange(3), (n, 3)).copy()
    targets = np.full((n, 3, 52), -1, dtype=np.int64)
    ages = np.full((n, 3, 52), -1, dtype=np.int32)
    mask = np.zeros((n, 3, 52), dtype=bool)
    targets[:, 2, 0], ages[:, 2, 0], mask[:, 2, 0] = 0, 1, True
    return {'positions': pos, 'ranks': pos.copy(), 'valid': np.ones((n, 3), dtype=bool),
            'targets': targets, 'ages': ages, 'target_mask': mask}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path
    paths = {name: root / name for name in ('protocol', 'inputs', 'bindings', 'data', 'preflight')}
    recipe = {'training': {'wall_cap_seconds': 1800, 'per_fit_wall_cap_seconds': 180,
                           'development_wall_cap_seconds': 300}}
    packet = {'fits': [{'pair': pair, 'seed': 410 + pair,
                        'orders': [list(range(128))] * 16, 'mode_order': list(runner.MODES)}
                       for pair in range(3)],
              'data': {part: [{'seed': index + offset, 'behavior_seed': index + offset + 1000}
                              for index in range(n)] for part, n, offset in (('train', 128, 0), ('dev', 32, 128))}}
    for name, value in (('protocol', recipe), ('inputs', packet), ('bindings', {'fake': True})):
        write(paths[name], value)
    events = []
    state = {'seed': None, 'fit_count': 0, 'mutate_after_fit': None, 'mutate_on_evaluate': None}

    for label, counts in (('data', {'train': 128, 'dev': 32}), ('preflight', {'engineering': 4})):
        folder = paths[label]
        folder.mkdir()
        start = {f'{name}_sha256': common.sha(paths[name]) for name in ('protocol', 'inputs', 'bindings')}
        write(folder / 'started.json', start)
        ledger, parts = [], {}
        for part, count in counts.items():
            np.savez(folder / f'{part}.npz', **tensors(count))
            parts[part] = common.sha(folder / f'{part}.npz')
            for index in range(count):
                file = folder / part / f'{index:03d}.npz'
                file.parent.mkdir(exist_ok=True)
                file.write_bytes(b'fake-native-archive')
                case = packet['data'][part][index] if label == 'data' else {'seed': index, 'behavior_seed': index}
                ledger.append({'part': part, 'index': index, **case, 'npz_path': str(file.relative_to(folder)),
                               'npz_sha256': common.sha(file), 'replay': 'passed'})
        write(folder / 'episodes.json', ledger)
        write(folder / 'completed.json', {'status': 'complete', 'episodes': sum(counts.values()),
              'started_sha256': common.sha(folder / 'started.json'),
              'episodes_sha256': common.sha(folder / 'episodes.json'), 'parts': parts,
              'work': {'native_steps_returned': sum(counts.values()), 'replay_steps_returned': sum(counts.values())}})

    class FakeModel:
        def __init__(self, mode):
            events.append(('construct', mode))
            self.mode = mode
            self.weights = {'shared': torch.tensor([float(state['seed'])])}

        def state_dict(self):
            return self.weights

        def configuration(self):
            return {'model_class': 'CardAssociativeMemory', 'mode': self.mode}

        def parameter_counts(self):
            return {'registered_parameters': 1, 'state_bytes_per_case': 4}

    def copy_initial(source, destination):
        destination.weights = {k: v.clone() for k, v in source.weights.items()}
        return ('shared',)

    def fake_fit(model, data, orders, out, *, lr, batch_size, wall_deadline):
        assert len(data['valid']) == 128 and orders.shape == (16, 128)
        assert lr == .003 and batch_size == 16 and wall_deadline > 0
        events.append(('fit', out.name))
        out.mkdir(parents=True)
        identity = {'initial_weights_sha256': _tensor_hash(model.weights),
                    'data_sha256': _tensor_hash({k: torch.as_tensor(v) for k, v in data.items()}),
                    'orders_sha256': _tensor_hash({'orders': torch.as_tensor(orders)})}
        model.weights = {'shared': model.weights['shared'] + 1}
        counts = {'successful_updates': 128, 'completed_epochs': 16}
        recipe = {'epochs': 16, 'expected_updates': 128}
        checkpoint = {'model_class': 'CardAssociativeMemory', 'resume_authorized': False,
                      'configuration': model.configuration(), 'identity': identity, 'recipe': recipe,
                      'counts': counts, 'orders': torch.as_tensor(orders), 'weights': model.weights}
        torch.save(checkpoint, out / 'final-checkpoint.pt')
        write(out / 'fit-settings.json', {'identity': identity, 'recipe': recipe, 'configuration': model.configuration()})
        receipt = {'status': 'complete', 'evaluation_calls': 0, 'counts': counts, 'recipe': recipe,
                   'identity': identity, 'final_weights_sha256': _tensor_hash(model.weights),
                   'files': {p.name: {'sha256': common.sha(p), 'bytes': p.stat().st_size} for p in out.iterdir()},
                   'wall_seconds': .01, 'work': {}, 'timings': {}}
        write(out / 'completed.json', receipt)
        state['fit_count'] += 1
        if state['mutate_after_fit'] is not None:
            state['mutate_after_fit'](out, model, state['fit_count'])
        return receipt

    def fake_load(entry, receipt, root):
        events.append(('restore', entry['mode']))
        assert state['fit_count'] == 18
        return {'mode': entry['mode']}

    def fake_evaluate(model, data, **kwargs):
        assert len(data['valid']) == 32 and state['fit_count'] == 18
        events.append(('evaluate', model['mode']))
        if state['mutate_on_evaluate'] is not None:
            state['mutate_on_evaluate']()
        return {'all': {'ce': .1}, 'per_episode': []}

    monkeypatch.setattr(runner, 'authenticate', lambda *args: (copy.deepcopy(recipe), copy.deepcopy(packet), {}))
    monkeypatch.setattr(runner, 'CardAssociativeMemory', FakeModel)
    monkeypatch.setattr(runner, 'copy_shared_initialization', copy_initial)
    monkeypatch.setattr(runner, 'train_fit', fake_fit)
    monkeypatch.setattr(runner, 'load_model', fake_load)
    monkeypatch.setattr(runner, 'evaluate', fake_evaluate)
    monkeypatch.setattr(torch, 'manual_seed', lambda seed: state.update(seed=seed))
    monkeypatch.setattr(torch, 'set_num_threads', lambda *args: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda *args: None)
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', lambda *args: None)
    arguments = {'root': root, 'out': root / 'run', **paths,
                 'data_sha256': common.sha(paths['data'] / 'completed.json'),
                 'preflight_sha256': common.sha(paths['preflight'] / 'completed.json')}
    return arguments, state, events, packet


def test_all_eighteen_fits_and_bindings_precede_development(campaign):
    arguments, _, events, _ = campaign
    runner.run(**arguments)
    assert [kind for kind, _ in events].count('fit') == 18
    assert [kind for kind, _ in events].count('evaluate') == 18
    first = next(i for i, (kind, _) in enumerate(events) if kind == 'evaluate')
    assert sum(kind == 'fit' for kind, _ in events[:first]) == 18
    completed = json.loads((arguments['out'] / 'completed.json').read_text())
    assert completed['fits'] == 18 and completed['updates'] == 2304
    assert completed['native_evaluation_calls'] == 0


def test_last_final_checkpoint_corruption_blocks_every_evaluation(campaign):
    arguments, state, events, _ = campaign

    def corrupt(out, model, count):
        if count == 18:
            path = out / 'final-checkpoint.pt'
            checkpoint = torch.load(path, weights_only=True)
            checkpoint['weights']['shared'] += 50
            torch.save(checkpoint, path)

    state['mutate_after_fit'] = corrupt
    with pytest.raises(ValueError, match='member changed'):
        runner.run(**arguments)
    assert not any(kind == 'evaluate' for kind, _ in events)
    assert (arguments['out'] / 'failed.json').exists()
    assert not (arguments['out'] / 'training-completed.json').exists()


@pytest.mark.parametrize('changed', ['mode', 'orders', 'data', 'initial_weights', 'final_weights'])
def test_resealed_wrong_checkpoint_payload_rejected(campaign, changed):
    arguments, state, events, _ = campaign

    def corrupt(out, model, count):
        if count != 18:
            return
        path = out / 'final-checkpoint.pt'
        checkpoint = torch.load(path, weights_only=True)
        if changed == 'mode':
            checkpoint['configuration']['mode'] = 'delta'
        elif changed == 'orders':
            checkpoint['orders'][0] = checkpoint['orders'][0].flip(0)
        elif changed in ('data', 'initial_weights'):
            checkpoint['identity'][changed + '_sha256'] = '0' * 64
        else:
            checkpoint['weights']['shared'] += 1
        torch.save(checkpoint, path)
        receipt_path = out / 'completed.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['files']['final-checkpoint.pt'] = {'sha256': common.sha(path), 'bytes': path.stat().st_size}
        write(receipt_path, receipt)

    state['mutate_after_fit'] = corrupt
    with pytest.raises(ValueError):
        runner.run(**arguments)
    assert not any(kind == 'evaluate' for kind, _ in events)


def test_resealed_pair_initialization_change_rejected(campaign):
    arguments, state, events, _ = campaign

    def corrupt(out, model, count):
        if count != 18:
            return
        path = arguments['out'] / (out.name + '-initial.pt')
        initial = torch.load(path, weights_only=True)
        initial['weights']['shared'] += 10
        torch.save(initial, path)
        # Runner binds this current initial file after train_fit returns, so the
        # independent checkpoint initial identity must catch the altered tensor.

    state['mutate_after_fit'] = corrupt
    with pytest.raises(ValueError, match='identity mismatch'):
        runner.run(**arguments)
    assert not any(kind == 'evaluate' for kind, _ in events)


@pytest.mark.parametrize('kind', ['started', 'packed', 'raw', 'case_order'])
def test_preparation_corruption_blocks_model_construction(campaign, kind):
    arguments, _, events, _ = campaign
    folder = arguments['data']
    if kind == 'started':
        (folder / 'started.json').write_text('{}')
    elif kind == 'packed':
        (folder / 'train.npz').write_bytes(b'corrupt')
    elif kind == 'raw':
        (folder / 'train/000.npz').write_bytes(b'corrupt')
    else:
        path = folder / 'episodes.json'
        ledger = json.loads(path.read_text())
        ledger[0], ledger[1] = ledger[1], ledger[0]
        write(path, ledger)
        receipt = json.loads((folder / 'completed.json').read_text())
        receipt['episodes_sha256'] = common.sha(path)
        write(folder / 'completed.json', receipt)
        arguments['data_sha256'] = common.sha(folder / 'completed.json')
    with pytest.raises(ValueError):
        runner.run(**arguments)
    assert not events


def test_bad_permutation_blocks_model_construction(campaign):
    arguments, _, events, packet = campaign
    packet['fits'][0]['orders'][0] = [0] * 128
    with pytest.raises(ValueError, match='permutation'):
        runner.run(**arguments)
    assert not events


def test_development_failure_preserves_complete_training(campaign):
    arguments, state, events, _ = campaign
    state['mutate_on_evaluate'] = lambda: (_ for _ in ()).throw(RuntimeError('fake evaluation failure'))
    with pytest.raises(RuntimeError, match='fake evaluation'):
        runner.run(**arguments)
    assert sum(kind == 'fit' for kind, _ in events) == 18
    assert (arguments['out'] / 'training-completed.json').exists()
    assert (arguments['out'] / 'failed.json').exists()
    assert not (arguments['out'] / 'completed.json').exists()


def test_late_completion_failure_preserved_without_retry(campaign, monkeypatch):
    arguments, _, _, _ = campaign
    original = runner.write_json

    def fail(path, value):
        original(path, value)
        if path.name == 'completed.json' and path.parent == arguments['out']:
            raise TimeoutError('fake late cap')

    monkeypatch.setattr(runner, 'write_json', fail)
    with pytest.raises(TimeoutError, match='fake late cap'):
        runner.run(**arguments)
    assert (arguments['out'] / 'invalid-completion.json').exists()
    assert (arguments['out'] / 'training-completed.json').exists()
    with pytest.raises(FileExistsError):
        runner.run(**arguments)


def test_common_preserves_error_without_add_note():
    class LegacyError(Exception):
        add_note = None

    called = []
    error = LegacyError('original')
    common.preserve_failure(error, [lambda: (_ for _ in ()).throw(OSError('secondary')),
                                    lambda: called.append(True)])
    assert called == [True]
