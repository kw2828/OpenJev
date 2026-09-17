"""Single-factor zero-critic PPO follow-up using the immutable original trainer.

Completed original-head controls and initialization-probe receipts are required
before preparation. Evaluation reuses the same scored development seeds. The
only training intervention is zeroing the fresh value head, without RNG draws.
"""

import argparse
import copy
import importlib.util
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def imported_reference(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts/associative_ppo_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# One copy stays untouched for validating the completed controls. The other is
# scoped to the new protocol/factory while its original function bodies run.
reference = imported_reference('_zero_critic_reference')
base = imported_reference('_zero_critic_execution')
REFERENCE_FIT = base.fit
MODEL = reference.AssociativePolicy
PROTOCOL = {
    **copy.deepcopy(reference.PROTOCOL),
    'version': 'zero-critic-ppo-v1',
    'intervention': 'Zero fresh value.weight and value.bias in place after normal initialization, no RNG draws',
    'control': 'Reuse every completed original-head associative-ppo-v1 fit; no selective control reruns',
    'evaluation_scope': 'Same previously scored paired development seeds and every original intervention',
    'optimization_continuation': {'minimum_same_size_gain': .1, 'minimum_improved_arms': 3,
                                'maximum_same_size_degradation': .05},
    'claim': 'Development initialization comparison; not independent confirmation or algorithmic novelty',
}
NEW_SOURCES = ['scripts/zero_critic_ppo_study.py', 'tests/test_zero_critic_ppo_study.py']
sha, write_new = reference.sha, reference.write_new


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameter_digest(state):
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def zero_model(*args, **kwargs):
    """Return a fresh model and evidence that only its value head changed."""
    model = MODEL(*args, **kwargs)
    rng = torch.get_rng_state().clone()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    # Fixed deterministic tensors test episode masks, nonzero prior state and
    # continuing recurrence without drawing RNG or opening an environment.
    obs = torch.arange(3*reference.OBS_SIZE, dtype=torch.float32).reshape(3, -1).remainder(2)
    previous = torch.tensor([0, 3, 6])
    state = torch.arange(3*model.state_size, dtype=torch.float32).reshape(3, -1)/1000
    reset = torch.tensor([True, False, False])
    with torch.no_grad():
        logits, _, following = model.observe(obs, previous, state, reset)
        next_logits, _, next_state = model.observe(obs.flip(0), previous, following, ~reset)
        model.value.weight.zero_()
        model.value.bias.zero_()
        zero_logits, zero_values, zero_following = model.observe(obs, previous, state, reset)
        zero_next_logits, zero_next_values, zero_next_state = model.observe(
            obs.flip(0), previous, zero_following, ~reset)
    unchanged = all(torch.equal(value, model.state_dict()[name])
                    for name, value in before.items() if not name.startswith('value.'))
    outputs_equal = all(torch.equal(a, b) for a, b in [(logits, zero_logits), (following, zero_following),
                                                      (next_logits, zero_next_logits), (next_state, zero_next_state)])
    require(unchanged and outputs_equal, 'Value-head intervention changed policy parameters, logits or state')
    require(torch.equal(rng, torch.get_rng_state()), 'Value-head intervention consumed RNG draws')
    require(not torch.count_nonzero(zero_values) and not torch.count_nonzero(zero_next_values), 'Zero head not zero')
    require(model.value.weight.requires_grad and model.value.bias.requires_grad, 'Zero value head must remain trainable')
    require(not any(p.grad is not None for p in model.parameters()), 'Preflight accumulated gradients')
    receipt = {'only_value_parameters_changed': unchanged, 'policy_logits_and_states_identical': outputs_equal,
               'rng_unchanged': True, 'zero_value_predictions': True, 'zero_value_head_trainable': True,
               'initial_original_parameters_sha256': parameter_digest(before),
               'initial_zero_parameters_sha256': parameter_digest(model.state_dict()),
               'initial_non_value_parameters_sha256': parameter_digest(
                   {name: value for name, value in before.items() if not name.startswith('value.')})}
    return model, receipt


def factory(*args, **kwargs):
    return zero_model(*args, **kwargs)[0]


def checked_fit(arm, seed, out, *, updates=None):
    require(arm in PROTOCOL['arms'], 'Undeclared architecture')
    smoke = seed == 7 and updates == 2
    require(smoke or seed in PROTOCOL['seeds'] and updates is None, 'Only full declared fits or seed7 two-update smoke')
    receipts = []
    previous_factory = base.AssociativePolicy

    def capture(*args, **kwargs):
        model, receipt = zero_model(*args, **kwargs)
        receipts.append(receipt)
        return model

    base.AssociativePolicy = capture
    try:
        result = REFERENCE_FIT(arm, seed, out, updates=updates)
    finally:
        base.AssociativePolicy = previous_factory
    require(len(receipts) == 1, 'Trainer constructed an unexpected number of models')
    write_new(out/'initialization.json', {'arm': arm, 'seed': seed, 'smoke': smoke,
                                         'protocol': PROTOCOL['version'], **receipts[0]})
    return result


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def validate_inputs(paths):
    """Re-read completed evidence and freeze every reused control artifact."""
    required = {'control_plan', 'control_run', 'control_report', 'probe_plan', 'probe_run'}
    require(set(paths) == required, 'Missing prerequisite paths')
    p = {key: ROOT/value for key, value in paths.items()}
    require((p['control_run']/'completed.json').is_file() and
            (p['control_report']/'completed.json').is_file(), 'Original autonomous study and report must finish first')
    reference.verify(p['control_plan'])
    control_done = read(p['control_run']/'completed.json')
    control_report_done = read(p['control_report']/'completed.json')
    control = read(p['control_report']/'summary.json')
    require(control_done['status'] == control_report_done['status'] == 'completed', 'Incomplete original controls')
    require(control_done['plan_sha256'] == control_report_done['plan_sha256'] == control['plan_sha256'] ==
            sha(p['control_plan']), 'Original control plan mismatch')
    require(control_done['fits'] == 12 and control_done['evaluations'] == 55, 'Incomplete original control panel')
    require(control_report_done['summary_sha256'] == sha(p['control_report']/'summary.json'), 'Control report hash mismatch')
    require(control['protocol'] == reference.PROTOCOL and read(p['control_report']/'plan.json') == read(p['control_plan']),
            'Control report recipe changed')
    training = read(p['control_run']/'training-completed.json')
    require(training['status'] == 'completed' and training['fits'] == control['training'], 'Control training receipts differ')
    expected = {(arm, seed) for arm in PROTOCOL['arms'] for seed in PROTOCOL['seeds']}
    require(len(control['training']) == 12 and {(r['arm'], r['seed']) for r in control['training']} == expected,
            'Control fit identities differ')
    artifacts = [p['control_plan'], p['control_run']/'completed.json', p['control_run']/'started.json',
                 p['control_run']/'fit_order.json', p['control_run']/'training-completed.json',
                 p['control_report']/'summary.json', p['control_report']/'completed.json', p['control_report']/'plan.json']
    for row in control['training']:
        directory = p['control_run']/'fits'/f"{row['arm']}-{row['seed']}"
        require(read(directory/'completed.json') == row and row['status'] == 'completed', 'Control fit receipt mismatch')
        require(row['interactions'] == 1048576 and row['gradient_steps'] == 4096, 'Control fit budget changed')
        for filename, key in [('model.pt', 'checkpoint_sha256'), ('learning.jsonl', 'learning_sha256')]:
            require(sha(directory/filename) == row[key], f'Control {filename} changed')
        artifacts.extend(directory/name for name in ['started.json', 'completed.json', 'model.pt', 'learning.jsonl'])
    expected_evaluations = {(arm, seed, mode) for arm, seed in expected for mode in reference.modes(arm)}
    observed, records, random_count = set(), [], 0
    for path in sorted((p['control_run']/'evaluation').glob('*.json')):
        record = read(path)
        records.append(record)
        if record['arm'] == 'random':
            random_count += 1
            require(record['mode'] == 'intact', 'Control random mode changed')
        else:
            identity = record['arm'], record['seed'], record['mode']
            require(identity in expected_evaluations and identity not in observed, 'Control evaluation identity mismatch')
            observed.add(identity)
            require(record['checkpoint_sha256'] == sha(p['control_run']/'fits'/f'{identity[0]}-{identity[1]}'/'model.pt'),
                    'Control evaluation checkpoint mismatch')
        require(len(record['results']) == 3 and {r['size'] for r in record['results']} == {11, 17, 23},
                'Control evaluation sizes changed')
        for result in record['results']:
            reference.validate_episode_result(result)
        artifacts.append(path)
    require(observed == expected_evaluations and random_count == 1 and records == control['evaluation'],
            'Control evaluation records differ from completed report')
    require(reference.gates(control['averages'], control['per_fit']) == control['checks'], 'Control selective gate differs')
    probe_plan, probe_done, probe = read(p['probe_plan']), read(p['probe_run']/'completed.json'), read(p['probe_run']/'summary.json')
    require(probe_done['status'] == probe['status'] == 'completed' and probe_done['smoke'] is False and probe['smoke'] is False,
            'Probe must be the completed declared panel')
    require(probe_done['plan_sha256'] == probe['plan_sha256'] == sha(p['probe_plan']), 'Probe plan hash mismatch')
    require(probe_done['summary_sha256'] == sha(p['probe_run']/'summary.json'), 'Probe summary hash mismatch')
    require(probe_done['rollouts'] == probe['rollouts'] == len(probe['results']) == 12 and
            probe_done['optimizer_steps'] == probe['optimizer_steps'] == 0, 'Probe scope mismatch')
    require(probe['protocol']['version'] == 'ppo-initialization-probe-v1' and
            all(probe[key] == value for key, value in probe_plan.items()), 'Probe recipe mismatch')
    require({(r['arm'], r['seed']) for r in probe['results']} == expected, 'Probe identities differ')
    for source, digest in probe_plan['source_sha256'].items():
        require(sha(ROOT/source) == digest, f'Probe source changed: {source}')
    require(all(r['parameters_unchanged'] is True and r['rollout']['paired_logits_states_and_actions_identical'] is True
                for r in probe['results']), 'Probe did not establish paired initialization parity')
    artifacts.extend([p['probe_plan'], p['probe_run']/'completed.json', p['probe_run']/'summary.json'])
    return {'paths': paths, 'artifact_sha256': {relative(path): sha(path) for path in sorted(set(artifacts))}}


def signature(paths):
    original = reference.signature()
    allowed = {'version', 'claim', 'intervention', 'control', 'evaluation_scope', 'optimization_continuation'}
    require(all(PROTOCOL[key] == value for key, value in reference.PROTOCOL.items() if key not in allowed),
            'A setting other than the value-head intervention changed')
    return {**original, 'protocol': PROTOCOL,
            'source_hashes': {**original['source_hashes'], **{name: sha(ROOT/name) for name in NEW_SOURCES}},
            'bindings': validate_inputs(paths)}


def verify(plan):
    frozen = read(plan)
    require(frozen == signature(frozen['bindings']['paths']), 'Frozen zero-critic recipe, sources or evidence changed')
    return frozen


@contextmanager
def configured(paths=None):
    old = {name: getattr(base, name) for name in ['PROTOCOL', 'AssociativePolicy', 'fit', 'signature']}
    base.PROTOCOL = PROTOCOL
    base.AssociativePolicy = factory
    base.fit = checked_fit
    if paths is not None:
        base.signature = lambda: signature(paths)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(base, name, value)


def prepare(paths, out):
    write_new(out, signature({key: relative(path) for key, path in paths.items()}))


def run(plan, out):
    frozen = verify(plan)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'started.json', {'plan_sha256': sha(plan), 'started_unix': time.time(), 'protocol': PROTOCOL['version']})
    with configured(frozen['bindings']['paths']):
        base.run(plan, out/'core')
    verify(plan)
    initializations = {}
    for arm in PROTOCOL['arms']:
        for seed in PROTOCOL['seeds']:
            path = out/'core/fits'/f'{arm}-{seed}'/'initialization.json'
            row = read(path)
            require(row['arm'] == arm and row['seed'] == seed and row['smoke'] is False,
                    'Initialization identity mismatch')
            initializations[f'{arm}-{seed}'] = sha(path)
    write_new(out/'completed.json', {'status': 'completed', 'plan_sha256': sha(plan),
                                     'protocol': PROTOCOL['version'], 'fits': 12, 'evaluations': 55,
                                     'core_receipt_sha256': sha(out/'core/completed.json'),
                                     'initialization_sha256': initializations, 'finished_unix': time.time()})


def comparison(control, zero):
    """Pair exact architecture, training seed, mode, map size and episode seed."""
    def random_episodes(summary):
        records = [r for r in summary['evaluation'] if r['arm'] == 'random']
        require(len(records) == 1 and records[0]['mode'] == 'intact', 'Expected one random/intact reference')
        results = records[0]['results']
        require(len(results) == len(PROTOCOL['eval_sizes']) and
                {r['size'] for r in results} == set(PROTOCOL['eval_sizes']), 'Random reference size coverage differs')
        for result in results:
            reference.validate_episode_result(result)
        # Timings can differ across executions. Every episode field, including
        # action counts, return and timeout, must match the repeated reference.
        return {r['size']: r['episodes'] for r in results}

    require(random_episodes(control) == random_episodes(zero), 'Repeated random reference episodes differ')

    def records(summary):
        rows = [((r['arm'], r['seed'], r['mode'], size['size']), size)
                for r in summary['evaluation'] if r['arm'] != 'random' for size in r['results']]
        mapping = dict(rows)
        require(len(mapping) == len(rows), 'Duplicated paired evaluation condition')
        return mapping

    original, intervention = records(control), records(zero)
    expected = {(arm, seed, mode, size) for arm in PROTOCOL['arms'] for seed in PROTOCOL['seeds']
                for mode in reference.modes(arm) for size in PROTOCOL['eval_sizes']}
    require(set(original) == set(intervention) == expected, 'Paired evaluation condition coverage differs')
    paired = []
    for identity in sorted(expected):
        first, second = original[identity], intervention[identity]
        require([r['seed'] for r in first['episodes']] == [r['seed'] for r in second['episodes']],
                'Paired episode seeds differ')
        for record in [first, second]:
            reference.validate_episode_result(record)
        paired.append({'arm': identity[0], 'seed': identity[1], 'mode': identity[2], 'size': identity[3],
                       'original_success': first['success'], 'zero_success': second['success'],
                       'success_difference': second['success']-first['success'],
                       'assigned_pairs': len(first['episodes']),
                       'original_only_success': sum(a['success'] and not b['success']
                                                    for a, b in zip(first['episodes'], second['episodes'], strict=True)),
                       'zero_only_success': sum(b['success'] and not a['success']
                                                for a, b in zip(first['episodes'], second['episodes'], strict=True))})
    gains = {arm: {str(size): sum(r['success_difference'] for r in paired if r['arm'] == arm and
                                  r['size'] == size and r['mode'] == 'intact')/len(PROTOCOL['seeds'])
                   for size in PROTOCOL['eval_sizes']} for arm in PROTOCOL['arms']}
    rule, same = PROTOCOL['optimization_continuation'], str(PROTOCOL['train_size'])
    improved = {arm: gains[arm][same] >= rule['minimum_same_size_gain'] for arm in PROTOCOL['arms']}
    checks = {'minimum_improved_arms': sum(improved.values()) >= rule['minimum_improved_arms'],
              'no_excessive_degradation': all(gains[arm][same] >= -rule['maximum_same_size_degradation']
                                              for arm in PROTOCOL['arms'])}
    return {'protocol': PROTOCOL, 'paired_fits': paired, 'mean_success_gain': gains,
            'improved_arms': improved, 'optimization_checks': checks, 'optimization_continuation_passed': all(checks.values()),
            'original_selective_checks': reference.gates(control['averages'], control['per_fit']),
            'zero_selective_checks': reference.gates(zero['averages'], zero['per_fit']),
            'independent_confirmation': False, 'novelty_established': False,
            'uniform_random_reference': 'Repeated same seeded uniform policy and development episodes; not independent',
            'repeated_random_episode_identity_verified': True,
            'scope': 'Reused scored development seeds; original controls completed before zero-head treatment'}


def report(plan, execution, out):
    frozen = verify(plan)
    done = read(execution/'completed.json')
    require(done['status'] == 'completed' and done['plan_sha256'] == sha(plan) and done['fits'] == 12
            and done['evaluations'] == 55 and done['core_receipt_sha256'] == sha(execution/'core/completed.json'),
            'Zero-critic execution is incomplete')
    expected = {f'{arm}-{seed}' for arm in PROTOCOL['arms'] for seed in PROTOCOL['seeds']}
    require(set(done['initialization_sha256']) == expected, 'Initialization receipt coverage differs')
    for identity, digest in done['initialization_sha256'].items():
        require(sha(execution/'core/fits'/identity/'initialization.json') == digest, 'Initialization receipt changed')
    out.mkdir(parents=True, exist_ok=False)
    with configured(frozen['bindings']['paths']):
        base.report(plan, execution/'core', out/'core')
    control_path = ROOT/frozen['bindings']['paths']['control_report']/'summary.json'
    result = comparison(read(control_path), read(out/'core/summary.json'))
    write_new(out/'comparison.json', {'plan_sha256': sha(plan), 'control_summary_sha256': sha(control_path),
                                     'zero_summary_sha256': sha(out/'core/summary.json'), **result})
    write_new(out/'completed.json', {'status': 'completed', 'plan_sha256': sha(plan),
                                     'comparison_sha256': sha(out/'comparison.json'),
                                     'zero_summary_sha256': sha(out/'core/summary.json'),
                                     'repeated_random_episode_identity_verified':
                                         result['repeated_random_episode_identity_verified']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'smoke', 'run', 'report'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--run', type=Path)
    for name in ['control_plan', 'control_run', 'control_report', 'probe_plan', 'probe_run']:
        parser.add_argument('--'+name.replace('_', '-'), type=Path)
    args = parser.parse_args()
    torch.set_num_threads(PROTOCOL['torch_threads'])
    if args.command == 'prepare':
        paths = {key: getattr(args, key) for key in ['control_plan', 'control_run', 'control_report', 'probe_plan', 'probe_run']}
        require(all(value is not None for value in paths.values()), 'Preparation requires all control and probe paths')
        prepare(paths, args.out)
    elif args.command == 'smoke':
        args.out.mkdir(parents=True, exist_ok=False)
        with configured():
            for arm in PROTOCOL['arms']:
                checked_fit(arm, 7, args.out/arm, updates=2)
    else:
        require(args.plan is not None, 'A prepared frozen plan is required')
        if args.command == 'run':
            run(args.plan, args.out)
        else:
            require(args.run is not None, 'Report requires a completed execution')
            report(args.plan, args.run, args.out)


if __name__ == '__main__':
    main()
