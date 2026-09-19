"""Run every prespecified fit before any final development evaluation."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from card_memory_common import authenticate, check, deadline_check, preserve_failure, sha, write_json
from evaluate_card_memory import authenticate_fits, load_model

from openjev.research.card_associative_memory import MODES, CardAssociativeMemory, copy_shared_initialization
from openjev.research.card_memory_training import _tensor_hash, evaluate, train_fit, validate_data


def load_part(directory, part, receipt):
    path = directory / (part + '.npz')
    check(sha(path) == receipt['parts'][part], f'Dataset changed: {part}')
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def validate_finished_fits(root, out, checkpoint_map, finished, fit_inputs, train_data_hash, deadline):
    """Authenticate ALL final/initial payloads before opening development data.

    The native evaluator supplies the same exact successful-fit member check.
    This additionally binds training data, orders, mode and actual paired initial
    tensors. It performs no model construction, prediction or optimization.
    """
    fits = authenticate_fits(checkpoint_map, root)
    receipts = {item['id']: item for item in finished}
    check(set(receipts) == set(fits) and len(finished) == 18, 'Exact finished fit receipts required')
    initials = {}
    for name, bound in fits.items():
        deadline_check(deadline)
        entry, receipt = bound['entry'], bound['receipt']
        spec = fit_inputs[entry['pair']]
        initial_path = out / (name + '-initial.pt')
        check(sha(initial_path) == receipts[name]['initial_sha256'], 'Initial artifact changed')
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        checkpoint = torch.load(root / entry['checkpoint_path'], map_location='cpu', weights_only=True)
        check(checkpoint['model_class'] == 'CardAssociativeMemory'
              and checkpoint['resume_authorized'] is False, 'Unexpected final checkpoint class')
        check(initial['seed'] == spec['seed'], 'Initial seed mismatch')
        check(initial['configuration'] == checkpoint['configuration']
              and checkpoint['configuration']['mode'] == entry['mode'], 'Initial/final mode or configuration mismatch')
        expected_identity = {'initial_weights_sha256': _tensor_hash(initial['weights']),
                             'data_sha256': train_data_hash,
                             'orders_sha256': _tensor_hash({'orders': torch.tensor(spec['orders'], dtype=torch.int64)})}
        check(checkpoint['identity'] == receipt['identity'] == expected_identity, 'Initial/data/order identity mismatch')
        for key in ('recipe', 'counts'):
            check(json.dumps(checkpoint[key], sort_keys=True) == json.dumps(receipt[key], sort_keys=True),
                  f'Checkpoint/receipt {key} mismatch')
        check(torch.equal(checkpoint['orders'], torch.tensor(spec['orders'], dtype=torch.int64)), 'Saved orders changed')
        check(receipt['counts']['successful_updates'] == 128 and receipt['counts']['completed_epochs'] == 16,
              'Final fixed schedule mismatch')
        settings = json.loads((root / entry['completed_path']).with_name('fit-settings.json').read_text())
        check(settings['identity'] == expected_identity and settings['configuration'] == checkpoint['configuration'],
              'Pre-update fit binding mismatch')
        check(json.dumps(settings['recipe'], sort_keys=True) == json.dumps(receipt['recipe'], sort_keys=True),
              'Pre-update recipe mismatch')
        weights = checkpoint['weights']
        check(set(weights) == set(initial['weights']) and all(
            isinstance(v, torch.Tensor) and v.dtype == torch.float32 and v.device.type == 'cpu'
            and torch.isfinite(v).all() and v.shape == initial['weights'][key].shape
            for key, v in weights.items()), 'Final weight schema mismatch')
        check(_tensor_hash(weights) == receipt['final_weights_sha256'], 'Final weight identity mismatch')
        initials[name] = initial
    for pair in range(3):
        source = initials[f'innovation_local-pair{pair}']['weights']
        for mode in MODES:
            initial = initials[f'{mode}-pair{pair}']
            shared = tuple(sorted(source.keys() & initial['weights'].keys()))
            check(tuple(initial['copied_tensors']) == shared
                  and all(torch.equal(source[key], initial['weights'][key]) for key in shared),
                  'Shared paired initial tensors differ')
    deadline_check(deadline)
    return fits


def run(root, out, protocol, inputs, bindings, data, data_sha256, preflight, preflight_sha256):
    root, out = root.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    finished, checkpoint_map, current = [], {}, None
    try:
        write_json(out / 'started.json', {'status': 'started', 'protocol_sha256': sha(protocol),
                   'inputs_sha256': sha(inputs), 'bindings_sha256': sha(bindings),
                   'data_receipt_sha256': data_sha256, 'preflight_receipt_sha256': preflight_sha256})
        recipe, packet, _ = authenticate(root, protocol, inputs, bindings)
        deadline = started + recipe['training']['wall_cap_seconds']
        deadline_check(deadline)
        check(sha(data / 'completed.json') == data_sha256, 'Unauthenticated preparation')
        check(sha(preflight / 'completed.json') == preflight_sha256, 'Unauthenticated preflight')
        for directory, expected in ((data, 160), (preflight, 4)):
            completed = json.loads((directory / 'completed.json').read_text())
            check(completed['started_sha256'] == sha(directory / 'started.json'), 'Preparation provenance changed')
            start = json.loads((directory / 'started.json').read_text())
            check(completed['status'] == 'complete' and completed['episodes'] == expected, 'Incomplete preparation')
            check(not any((directory / marker).exists() for marker in ('failed.json', 'invalid-completion.json')),
                  'Failed preparation is inadmissible')
            for name, path in (('inputs', inputs), ('protocol', protocol), ('bindings', bindings)):
                check(start[name + '_sha256'] == sha(path), 'Preparation provenance mismatch')
            check(completed['work']['native_steps_returned'] == completed['work']['replay_steps_returned'], 'Incomplete native replay')
            check(sha(directory / 'episodes.json') == completed['episodes_sha256'], 'Preparation episode ledger changed')
            for part, digest in completed['parts'].items():
                check(sha(directory / (part + '.npz')) == digest, 'Preparation packed data changed')
        data_receipt = json.loads((data / 'completed.json').read_text())
        check(sha(data / 'episodes.json') == data_receipt['episodes_sha256'], 'Episode ledger changed')
        ledger = json.loads((data / 'episodes.json').read_text())
        expected_cases = [(part, index, case['seed'], case['behavior_seed'])
                          for part, cases in packet['data'].items() for index, case in enumerate(cases)]
        actual_cases = [(case['part'], case['index'], case['seed'], case['behavior_seed']) for case in ledger]
        check(actual_cases == expected_cases, 'Collected case identity/order differs from inputs')
        for episode in ledger:
            check(sha(data / episode['npz_path']) == episode['npz_sha256'] and episode['replay'] == 'passed', 'Raw episode changed')
        train_data = load_part(data, 'train', data_receipt)
        check(len(train_data['valid']) == 128, 'Wrong training episode count')
        train_data_hash = _tensor_hash(validate_data(train_data))
        deadline_check(deadline)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        fits = packet['fits']
        check(len(fits) == 3 and [item['pair'] for item in fits] == [0, 1, 2], 'Wrong paired fit list')
        check(len({item['seed'] for item in fits}) == 3, 'Paired initializations must be distinct')
        for fit in fits:
            pair = fit['pair']
            deadline_check(deadline)
            check(type(fit['seed']) is int and 0 <= fit['seed'] < 2**32, 'Explicit uint32 initialization seed required')
            check(np.asarray(fit['orders']).dtype.kind in 'iu', 'Orders must be integer permutations')
            orders = np.asarray(fit['orders'], dtype=np.int64)
            check(orders.shape == (16, 128), 'Wrong training schedule')
            check(all(np.array_equal(np.sort(row), np.arange(128)) for row in orders), 'Wrong training permutation')
            check(set(fit['mode_order']) == set(MODES) and len(fit['mode_order']) == 6, 'Wrong arm order')
            torch.manual_seed(fit['seed'])
            shared = CardAssociativeMemory('innovation_local')
            for mode in fit['mode_order']:
                deadline_check(deadline)
                current = f'{mode}-pair{pair}'
                torch.manual_seed(fit['seed'])
                model = CardAssociativeMemory(mode)
                copied = copy_shared_initialization(shared, model)
                init_path = out / (current + '-initial.pt')
                with init_path.open('xb') as handle:
                    torch.save({'configuration': model.configuration(), 'weights': model.state_dict(),
                                'seed': fit['seed'], 'copied_tensors': copied}, handle)
                fit_dir = out / 'fits' / current
                completed = train_fit(model, train_data, orders, fit_dir, lr=.003, batch_size=16,
                                      wall_deadline=min(deadline, time.monotonic() + recipe['training']['per_fit_wall_cap_seconds']))
                check(completed['counts']['successful_updates'] == 128, 'Incomplete fit schedule')
                entry = {'mode': mode, 'pair': pair,
                         'checkpoint_path': str((fit_dir / 'final-checkpoint.pt').relative_to(root)),
                         'checkpoint_sha256': sha(fit_dir / 'final-checkpoint.pt'),
                         'completed_path': str((fit_dir / 'completed.json').relative_to(root)),
                         'completed_sha256': sha(fit_dir / 'completed.json')}
                checkpoint_map[current] = entry
                finished.append({'id': current, 'initial_sha256': sha(init_path), **model.parameter_counts(),
                                 'wall_seconds': completed['wall_seconds'], 'counts': completed['counts'],
                                 'work': completed['work'], 'timings': completed['timings'], **entry})
                write_json(out / (current + '-receipt.json'), finished[-1])
                print(json.dumps({'fit': current, 'status': 'complete', 'updates': 128,
                                  'wall_seconds': completed['wall_seconds']}), flush=True)
                del model
            del shared
        check(len(finished) == 18, 'All 18 fits required before evaluation')
        authenticate(root, protocol, inputs, bindings)
        deadline_check(deadline)
        validated_fits = validate_finished_fits(root, out, checkpoint_map, finished, fits, train_data_hash, deadline)
        write_json(out / 'checkpoint-map.json', checkpoint_map)
        write_json(out / 'training-completed.json', {'status': 'complete', 'fits': finished,
                   'checkpoint_map_sha256': sha(out / 'checkpoint-map.json'),
                   'wall_seconds': time.monotonic() - started, 'development_or_native_evaluations': 0})
        # Only now open the development corpus and restore final checkpoints.
        dev_started = time.monotonic()
        dev_deadline = min(deadline, dev_started + recipe['training']['development_wall_cap_seconds'])
        dev_data = load_part(data, 'dev', data_receipt)
        check(len(dev_data['valid']) == 32, 'Wrong development episode count')
        dev_results = {}
        for current, entry in checkpoint_map.items():
            deadline_check(dev_deadline)
            model = load_model(entry, validated_fits[current]['receipt'], root)
            metrics = evaluate(model, dev_data, wall_deadline=dev_deadline)
            dev_results[current] = metrics
            write_json(out / (current + '-development.json'), metrics)
            del model
        authenticate(root, protocol, inputs, bindings)
        authenticate_fits(checkpoint_map, root)
        check(sha(data / 'completed.json') == data_sha256 and sha(data / 'train.npz') == data_receipt['parts']['train']
              and sha(data / 'dev.npz') == data_receipt['parts']['dev'], 'Preparation changed during training/evaluation')
        deadline_check(deadline)
        write_json(out / 'completed.json', {'status': 'complete', 'fits': 18, 'updates': 2304,
                   'started_sha256': sha(out / 'started.json'),
                   'training_completed_sha256': sha(out / 'training-completed.json'),
                   'checkpoint_map_sha256': sha(out / 'checkpoint-map.json'),
                   'development': {key: {k: v for k, v in metrics.items() if k != 'per_episode'} for key, metrics in dev_results.items()},
                   'wall_seconds': time.monotonic() - started, 'development_seconds': time.monotonic() - dev_started,
                   'native_evaluation_calls': 0})
        deadline_check(deadline)
    except BaseException as error:
        actions = []
        if (out / 'completed.json').exists():
            actions.append(lambda: (out / 'completed.json').rename(out / 'invalid-completion.json'))
        actions.append(lambda: write_json(out / 'failed.json', {'status': 'failed', 'error': traceback.format_exc(),
                       'current': current, 'completed_fits': len(finished), 'wall_seconds': time.monotonic() - started,
                       'automatic_retry': False, 'resume_authorized': False}))
        preserve_failure(error, actions)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'out', 'protocol', 'inputs', 'bindings', 'data', 'preflight'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('data-sha256', 'preflight-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(**vars(args))
