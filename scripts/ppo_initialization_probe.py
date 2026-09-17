"""Matched first-rollout critic-initialization audit, with no optimizer updates.

Fresh policies collect the frozen PPO recipe's first actual rollout. The paired
model differs only in its zeroed value head. Actual rewards are preserved; a
separate all-zero-reward counterfactual isolates bootstrap-only advantages.
Gradient components are measured before clipping on the same full rollout,
not the trainer's shuffled minibatches. This is not training or efficacy evidence.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from minigrid.envs import MemoryEnv
from torch.distributions import Categorical
from torch.nn import functional as F

from openjev.research.associative_policy import AssociativePolicy
from openjev.research.cue_memory_env import CueMemoryBatch
from openjev.research.memory_env import OBS_SIZE
from openjev.research.predictive_memory import advantages

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_initialization_reference', ROOT/'scripts/associative_ppo_study.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
REFERENCE_FIELDS = ('arms', 'seeds', 'train_size', 'max_steps', 'environments', 'rollout',
                    'hidden', 'store_dimension', 'adapter_hidden', 'gamma', 'gae_lambda',
                    'clip', 'value_coefficient', 'entropy_coefficient', 'training_seed_base',
                    'policy_rng_offset', 'torch_threads')
PROTOCOL = {
    **{key: reference.PROTOCOL[key] for key in REFERENCE_FIELDS},
    'version': 'ppo-initialization-probe-v1', 'reference': 'associative-ppo-v1',
    'value_heads': ['original', 'zero'],
    'reward_conditions': ['actual', 'synthetic_all_zero'],
    'intervention': 'Deep-copy fresh policy, then zero only value.weight and value.bias without RNG draws',
    'collection': 'First actual rollout under original policy; verify paired logits, states and sampled actions exactly',
    'gradient_scope': 'Same full rollout, ratio at initial policy, before gradient clipping, no optimizer updates',
    'gradient_components': ['actor_normalized', 'actor_raw', 'actor_centered_raw', 'entropy_weighted', 'critic_weighted'],
    'raw_actor': 'Uncentered raw GAE; centered_raw also provided to isolate standard-deviation rescaling',
    'groups': ['all', 'shared_trunk', 'actor_head', 'value_head'],
    'undefined_cosine': 'null whenever either component norm is zero',
    'smoke': 'Seed7 only; full16x64 collection per architecture, explicitly not the scored panel',
    'optimizer_steps': 0, 'trained_checkpoints_loaded': False, 'efficacy_evaluation': False,
}
SOURCES = ['scripts/ppo_initialization_probe.py', 'tests/test_ppo_initialization_probe.py',
           'scripts/associative_ppo_study.py', 'src/openjev/research/associative_policy.py',
           'src/openjev/research/predictive_memory.py', 'src/openjev/research/cue_memory_env.py',
           'src/openjev/research/memory_env.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def signature():
    if any(PROTOCOL[key] != reference.PROTOCOL[key] for key in REFERENCE_FIELDS):
        raise ValueError('First-rollout settings differ from the reference PPO recipe')
    return {'protocol': PROTOCOL, 'source_sha256': {name: sha(ROOT/name) for name in SOURCES},
            'environment_source_sha256': sha(inspect.getfile(MemoryEnv)),
            'dependencies': {name: importlib.metadata.version(name)
                             for name in ('torch', 'numpy', 'gymnasium', 'minigrid')},
            'python': platform.python_version(), 'platform': platform.platform(),
            'lock_sha256': sha(ROOT/'uv.lock')}


def paired_models(arm, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    original = AssociativePolicy(arm, OBS_SIZE, PROTOCOL['hidden'],
                                 PROTOCOL['store_dimension'], PROTOCOL['adapter_hidden'])
    state = torch.get_rng_state().clone()
    zero = copy.deepcopy(original)
    with torch.no_grad():
        zero.value.weight.zero_()
        zero.value.bias.zero_()
    if not torch.equal(state, torch.get_rng_state()):
        raise ValueError('Value-head intervention consumed RNG draws')
    for name, value in original.state_dict().items():
        if not name.startswith('value.') and not torch.equal(value, zero.state_dict()[name]):
            raise ValueError('The value-head intervention changed policy parameters')
    return {'original': original, 'zero': zero}


@torch.no_grad()
def collect(models, seed, *, count=None, length=None):
    """Collect one shared rollout; optional shorter dimensions are for unit tests."""
    p = PROTOCOL
    n, length = p['environments'] if count is None else count, p['rollout'] if length is None else length
    if n < 1 or length < 1:
        raise ValueError('Positive collection dimensions required')
    # Exactly the reference trainer's post-initialization policy RNG reset.
    torch.manual_seed(p['policy_rng_offset']+seed)
    entries = {key: [] for key in ('obs', 'previous', 'reset', 'actions', 'logprob',
                                  'values_original', 'next_values_original', 'values_zero',
                                  'next_values_zero', 'rewards', 'terminated', 'ended')}
    episodes = []
    with CueMemoryBatch(n, p['train_size'], p['training_seed_base']+seed*100000, p['max_steps']) as envs:
        obs = torch.from_numpy(envs.reset())
        previous = torch.zeros(n, dtype=torch.long)
        states = {name: model.initial_state(n) for name, model in models.items()}
        reset = torch.ones(n, dtype=torch.bool)
        for _ in range(length):
            predictions = {name: model.observe(obs, previous, states[name], reset)
                           for name, model in models.items()}
            logits, _, state = predictions['original']
            zero_logits, _, zero_state = predictions['zero']
            if not torch.equal(logits, zero_logits) or not torch.equal(state, zero_state):
                raise ValueError('Value-head pair changed policy logits or recurrent state')
            before_sample = torch.get_rng_state().clone()
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            after_sample = torch.get_rng_state().clone()
            torch.set_rng_state(before_sample)
            zero_actions = Categorical(logits=zero_logits).sample()
            if not torch.equal(actions, zero_actions) or not torch.equal(after_sample, torch.get_rng_state()):
                raise ValueError('Paired action sampling differs')
            torch.set_rng_state(after_sample)
            new_obs, rewards, terminal, timeout, infos = envs.step(actions.numpy())
            true_next = new_obs.copy()
            for i, info in enumerate(infos):
                if 'episode' in info:
                    episodes.append(info['episode'])
                    true_next[i] = info['terminal_observation']
            next_obs = torch.from_numpy(true_next)
            values = {'obs': obs, 'previous': previous, 'reset': reset, 'actions': actions,
                      'logprob': distribution.log_prob(actions), 'rewards': torch.from_numpy(rewards),
                      'terminated': torch.from_numpy(terminal), 'ended': torch.from_numpy(terminal | timeout)}
            for name, model in models.items():
                _, value, states[name] = predictions[name]
                _, next_value, _ = model.observe(next_obs, actions, states[name], torch.zeros_like(reset))
                values[f'values_{name}'], values[f'next_values_{name}'] = value, next_value
            for key, history in entries.items():
                history.append(values[key])
            obs, previous, reset = torch.from_numpy(new_obs), actions, values['ended']
    data = {key: torch.stack(value) for key, value in entries.items()}
    if torch.count_nonzero(data['values_zero']) or torch.count_nonzero(data['next_values_zero']):
        raise ValueError('Zero critic produced nonzero values')
    return data, {'interactions': n*length, 'positive_reward_events': int((data['rewards'] > 0).sum()),
                  'reward_sum': float(data['rewards'].sum()), 'completed_episodes': episodes,
                  'terminated_events': int(data['terminated'].sum()),
                  'truncation_only_events': int((data['ended'] & ~data['terminated']).sum()),
                  'action_counts': torch.bincount(data['actions'].flatten(), minlength=7).tolist(),
                  'paired_logits_states_and_actions_identical': True,
                  'data_sha256': tensor_digest(data)}


def statistics(tensor):
    value = tensor.detach().double()
    return {'mean': float(value.mean()), 'std_population': float(value.std(unbiased=False)),
            'minimum': float(value.min()), 'maximum': float(value.max()),
            'l2': float(value.norm()), 'nonzero_count': int(torch.count_nonzero(value))}


def gradient_summary(model, losses):
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    grouped = {group: {} for group in PROTOCOL['groups']}
    for component, loss in losses.items():
        gradients = torch.autograd.grad(loss, [value for _, value in parameters],
                                        allow_unused=True, retain_graph=True)
        for group, components in grouped.items():
            vectors = []
            for (name, value), grad in zip(parameters, gradients, strict=True):
                include = (group == 'all' or group == 'actor_head' and name.startswith('actor.')
                           or group == 'value_head' and name.startswith('value.')
                           or group == 'shared_trunk' and not name.startswith(('actor.', 'value.')))
                if include:
                    vectors.append((torch.zeros_like(value) if grad is None else grad).detach().flatten().double())
            vector = torch.cat(vectors)
            if not torch.isfinite(vector).all():
                raise ValueError('Nonfinite gradient component')
            components[component] = vector
    output = {}
    for group, vectors in grouped.items():
        norms = {name: float(vector.norm()) for name, vector in vectors.items()}
        cosines = {}
        for left, right in itertools.combinations(vectors, 2):
            denominator = norms[left]*norms[right]
            cosines[f'{left}__{right}'] = (float(torch.dot(vectors[left], vectors[right])/denominator)
                                           if denominator else None)
        output[group] = {'l2_norms': norms, 'cosines': cosines,
                         'normalized_actor_to_weighted_entropy_norm_ratio':
                         norms['actor_normalized']/norms['entropy_weighted'] if norms['entropy_weighted'] else None}
    return output


def analyze(model, data, head, reward_condition):
    if head not in PROTOCOL['value_heads'] or reward_condition not in PROTOCOL['reward_conditions']:
        raise ValueError('Unknown probe condition')
    rewards = data['rewards'] if reward_condition == 'actual' else torch.zeros_like(data['rewards'])
    p = PROTOCOL
    with torch.no_grad():
        raw, returns = advantages(rewards, data[f'values_{head}'], data[f'next_values_{head}'],
                                  data['terminated'], data['ended'], p['gamma'], p['gae_lambda'])
        centered = raw-raw.mean()
        scale = raw.std(unbiased=False)+1e-8
        normalized = centered/scale
    logits, values, _ = model.sequence(data['obs'], data['previous'], model.initial_state(data['obs'].shape[1]),
                                       data['reset'])
    distribution = Categorical(logits=logits)
    logprob = distribution.log_prob(data['actions'])
    if not torch.equal(logprob.detach(), data['logprob']) or not torch.equal(values.detach(), data[f'values_{head}']):
        raise ValueError('Full-batch replay differs from initial collection')
    ratio = (logprob-data['logprob']).exp()

    def actor(advantage):
        return -torch.minimum(ratio*advantage, ratio.clamp(1-p['clip'], 1+p['clip'])*advantage).mean()

    losses = {'actor_normalized': actor(normalized), 'actor_raw': actor(raw),
              'actor_centered_raw': actor(centered),
              'entropy_weighted': -p['entropy_coefficient']*distribution.entropy().mean(),
              'critic_weighted': p['value_coefficient']*F.mse_loss(values, returns)}
    return {'value_head': head, 'reward_condition': reward_condition,
            'synthetic_rewards': reward_condition != 'actual', 'raw_advantage': statistics(raw),
            'normalized_advantage': statistics(normalized), 'normalization_divisor': float(scale),
            'values': statistics(data[f'values_{head}']), 'return_targets': statistics(returns),
            'losses': {name: float(value.detach()) for name, value in losses.items()},
            'policy_entropy_nats': float(distribution.entropy().mean().detach()),
            'gradient_components': gradient_summary(model, losses)}


def probe(arm, seed, out, *, smoke=False):
    if arm not in PROTOCOL['arms'] or seed not in ([7] if smoke else PROTOCOL['seeds']):
        raise ValueError('Only declared identities or seed7 smoke are permitted')
    out.mkdir(parents=True, exist_ok=False)
    models = paired_models(arm, seed)
    before = {name: tensor_digest(model.state_dict()) for name, model in models.items()}
    data, rollout = collect(models, seed)
    np.savez_compressed(out/'rollout.npz', **{name: value.numpy() for name, value in data.items()})
    analyses = [analyze(models[head], data, head, condition)
                for head in PROTOCOL['value_heads'] for condition in PROTOCOL['reward_conditions']]
    after = {name: tensor_digest(model.state_dict()) for name, model in models.items()}
    if before != after or any(value.grad is not None for model in models.values() for value in model.parameters()):
        raise ValueError('Gradient audit mutated model parameters or accumulated .grad buffers')
    result = {'arm': arm, 'seed': seed, 'smoke': smoke, 'rollout': rollout,
              'rollout_npz_sha256': sha(out/'rollout.npz'), 'analyses': analyses,
              'initial_parameter_sha256': before, 'final_parameter_sha256': after,
              'parameters_unchanged': True, 'optimizer_steps': 0}
    write_new(out/'result.json', result)
    return result


def run(plan, out, *, smoke=False):
    if not smoke and plan is None:
        raise ValueError('A frozen plan is required for the declared panel')
    frozen = signature()
    if not smoke and json.loads(plan.read_text()) != frozen:
        raise ValueError('Frozen source, protocol, runtime or reference changed')
    out.mkdir(parents=True, exist_ok=False)
    provenance = {'plan_sha256': None if smoke else sha(plan), 'smoke': smoke, **frozen}
    write_new(out/'started.json', {**provenance, 'started_unix': time.time()})
    begin = time.perf_counter()
    jobs = [(arm, seed) for arm in PROTOCOL['arms'] for seed in ([7] if smoke else PROTOCOL['seeds'])]
    results = []
    for arm, seed in jobs:
        result = probe(arm, seed, out/f'{arm}-{seed}', smoke=smoke)
        results.append(result)
        print(json.dumps({'arm': arm, 'seed': seed, 'actual_reward_events': result['rollout']['positive_reward_events'],
                          'optimizer_steps': 0}), flush=True)
    if {(r['arm'], r['seed']) for r in results} != set(jobs) or len(results) != len(jobs):
        raise ValueError('Incomplete initialization probe identities')
    if signature() != frozen:
        raise ValueError('Probe source, protocol or dependencies changed during collection')
    result = {**provenance, 'status': 'completed', 'results': results, 'rollouts': len(results),
              'environment_interactions': sum(row['rollout']['interactions'] for row in results),
              'elapsed_seconds': time.perf_counter()-begin, 'optimizer_steps': 0,
              'limitations': [
                  'Initial gradient audit only; no policy training or effectiveness evaluation',
                  'Synthetic all-zero rewards are a separate counterfactual, not the actual observed environment',
                  'Full-rollout unclipped gradient measurements differ from shuffled minibatch training updates',
                  'Critic initialization changes both bootstrap advantages and shared-trunk critic gradients',
                  'Gradient magnitude alone does not establish harmful optimization or explain later collapse',
              ]}
    write_new(out/'summary.json', result)
    write_new(out/'completed.json', {'status': 'completed', 'plan_sha256': provenance['plan_sha256'],
                                     'smoke': smoke, 'rollouts': len(results), 'optimizer_steps': 0,
                                     'summary_sha256': sha(out/'summary.json')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'smoke', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(PROTOCOL['torch_threads'])
    if args.command == 'freeze':
        write_new(args.out, signature())
    else:
        run(args.plan, args.out, smoke=args.command == 'smoke')


if __name__ == '__main__':
    main()
