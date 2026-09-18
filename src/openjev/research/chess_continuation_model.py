"""Matched recurrent actors with training-only candidate outcome supervision.

The critic shares the actor's candidate hidden activation, never its output
layer. Inference remains the inherited actor policy without critic evaluation.
Missing continuation labels are selected out before arithmetic, not multiplied
by zero. Auxiliary losses average distinct actions within each root first.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_spatial import ENCODING_VERSION, _check_candidates, _check_plan_hash

OBJECTIVES = ('policy', 'best_value', 'continuation')
CHECKPOINT_VERSION = 'openjev-chess-continuation-v1'
WIDTH, DEPTH = 32, 4
CRITIC_SEED_OFFSET = 1000003


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError('seed must be an integer in [0,2**63)')
    return value


def _critic_seed(seed):
    return (seed+CRITIC_SEED_OFFSET) % 2**63


def _objective(value):
    if type(value) is not str or value not in OBJECTIVES:
        raise ValueError('objective must be policy, best_value or continuation')
    return value


class ContinuationChess(AnchorChess):
    def __init__(self, objective='policy', seed=17, critic_seed=None):
        objective, seed = _objective(objective), _seed(seed)
        critic_seed = _seed(_critic_seed(seed) if critic_seed is None else critic_seed)
        # AnchorChess seeds through torch.manual_seed, which also touches the
        # accelerator generator. Preserve it as well as the CPU generator.
        device_type = 'mps' if torch.backends.mps.is_available() else 'cuda'
        devices = list(range(torch.get_device_module(device_type).device_count()))
        with torch.random.fork_rng(devices=devices, device_type=device_type):
            super().__init__('residual', seed=seed, width=WIDTH, depth=DEPTH)
        self.objective, self.critic_seed = objective, critic_seed
        # The private generator initializes the extra head without reseeding any
        # accelerator. Restore the CPU draws consumed by Linear construction.
        with torch.random.fork_rng(devices=[]):
            self.candidate_critic = nn.Linear(2*WIDTH, 1)
            generator = torch.Generator(device='cpu').manual_seed(critic_seed)
            nn.init.kaiming_uniform_(self.candidate_critic.weight, a=math.sqrt(5), generator=generator)
            bound = 1/math.sqrt(2*WIDTH)
            nn.init.uniform_(self.candidate_critic.bias, -bound, bound, generator=generator)

    def candidate_values(self, hidden, candidates):
        """Return bounded [B,L] training values; padded rows must not be supervised."""
        parameter = next(self.parameters())
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 4
                or hidden.shape[1:] != (WIDTH, 8, 8) or hidden.shape[0] == 0
                or not hidden.is_floating_point() or not torch.isfinite(hidden).all()
                or hidden.dtype != parameter.dtype or hidden.device != parameter.device):
            raise ValueError('Hidden state must be finite floating [batch,32,8,8]')
        if (not isinstance(candidates, torch.Tensor) or candidates.ndim != 3
                or candidates.shape[0] != hidden.shape[0] or candidates.shape[1] == 0
                or candidates.device != hidden.device):
            raise ValueError('Candidates must match hidden batch and device with a nonempty menu')
        _check_candidates(candidates)
        squares = hidden.flatten(2).transpose(1, 2)
        batch = torch.arange(len(hidden), device=hidden.device).unsqueeze(1)
        pooled = hidden.mean(dim=(2, 3))
        features = torch.cat((squares[batch, candidates[..., 0]], squares[batch, candidates[..., 1]],
                              pooled.unsqueeze(1).expand(-1, candidates.shape[1], -1),
                              self.promotion_embedding(candidates[..., 2]),
                              self.dx_embedding(candidates[..., 3]),
                              self.dy_embedding(candidates[..., 4])), dim=-1)
        shared = self.policy_head[1](self.policy_head[0](features))
        return torch.tanh(self.candidate_critic(shared)).squeeze(-1)

    def parameter_report(self):
        stored = sum(p.numel() for p in self.parameters())
        critic = sum(p.numel() for p in self.candidate_critic.parameters())
        unused_board_head = sum(p.numel() for p in self.aux_head.parameters())
        actor_active = stored-critic-unused_board_head
        active_critic = 0 if self.objective == 'policy' else critic
        return {'stored_parameters': stored, 'stored_actor_parameters': stored-critic,
                'stored_critic_parameters': critic, 'unused_board_head_parameters': unused_board_head,
                'training_active_actor_parameters': actor_active,
                'training_active_critic_parameters': active_critic,
                'training_active_parameters': actor_active+active_critic,
                'inference_active_actor_parameters': actor_active,
                'inference_active_critic_parameters': 0, 'inference_active_parameters': actor_active}

    def save(self, path, *, plan_sha256):
        """Exclusively save primitive metadata and finite float32 state tensors."""
        _check_plan_hash(plan_sha256)
        reference = type(self)(self.objective, _seed(self.seed), _seed(self.critic_seed))
        if (self.width, self.depth, self.recurrence, self.mode) != (WIDTH, DEPTH, 'residual', 'recurrent'):
            raise ValueError('Continuation architecture changed')
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        _check_state(state, reference.state_dict())
        payload = {'format_version': CHECKPOINT_VERSION, 'encoding': ENCODING_VERSION,
                   'objective': self.objective, 'recurrence': 'residual', 'seed': self.seed,
                   'critic_seed': self.critic_seed, 'width': WIDTH, 'depth': DEPTH,
                   'plan_sha256': plan_sha256, 'state_dict': state}
        with Path(path).open('xb') as handle:
            torch.save(payload, handle)

    @classmethod
    def load(cls, path, *, expected_plan_sha256, expected_seed, expected_objective,
             expected_critic_seed=None):
        """Require explicit plan/seed/objective identities and tensor-only loading."""
        _check_plan_hash(expected_plan_sha256)
        expected_seed, expected_objective = _seed(expected_seed), _objective(expected_objective)
        expected_critic_seed = _seed(_critic_seed(expected_seed) if expected_critic_seed is None
                                     else expected_critic_seed)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        metadata = {'format_version': CHECKPOINT_VERSION, 'encoding': ENCODING_VERSION,
                    'objective': expected_objective, 'recurrence': 'residual', 'seed': expected_seed,
                    'critic_seed': expected_critic_seed, 'width': WIDTH, 'depth': DEPTH,
                    'plan_sha256': expected_plan_sha256}
        if (not isinstance(payload, dict) or set(payload) != set(metadata) | {'state_dict'}
                or any(type(payload[k]) is not type(v) or payload[k] != v for k, v in metadata.items())):
            raise ValueError('Continuation checkpoint metadata or requested identity mismatch')
        model = cls(expected_objective, expected_seed, expected_critic_seed)
        _check_state(payload['state_dict'], model.state_dict())
        model.load_state_dict(payload['state_dict'], strict=True)
        return model.eval()


def _check_state(state, expected):
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Continuation checkpoint state membership mismatch')
    for name, value in state.items():
        if (type(value) is not torch.Tensor or value.layout != torch.strided
                or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid continuation checkpoint tensor: {name}')


def _floating(value, shape, device, name, *, bounded=False, dtype=None):
    if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != device
            or not value.is_floating_point() or (dtype is not None and value.dtype != dtype)
            or not torch.isfinite(value).all()
            or (bounded and (value.abs() > 1).any())):
        raise ValueError(f'{name} must be finite floating tensors with matching shape/device/dtype'
                         + (' and values in [-1,1]' if bounded else ''))


def loss_components(objective, logits, root_predictions, candidate_predictions, legal_mask,
                    targets, teacher_values, behavior_indices=None, continuation_values=None,
                    continuation_mask=None):
    """Compute CE + .5 root MSE + one per-root mean candidate MSE when enabled.

Missing continuation labels permit NaN values and index -1. Present behavior
indices must be legal. Available labels must be finite/bounded even when their
action duplicates the teacher; duplicates contribute no extra supervision.
Counts refer to critic targets actually used, not the separate CE/root targets.
    """
    _objective(objective)
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or min(logits.shape) < 1
            or not logits.is_floating_point()):
        raise ValueError('Logits must be floating [batch,moves] with nonempty dimensions')
    batch, moves = logits.shape
    device, shape = logits.device, torch.Size([batch])
    if (not isinstance(legal_mask, torch.Tensor) or legal_mask.dtype != torch.bool
            or legal_mask.shape != logits.shape or legal_mask.device != device
            or not legal_mask.any(-1).all() or not torch.isfinite(logits[legal_mask]).all()
            or not torch.isneginf(logits[~legal_mask]).all()):
        raise ValueError('Legal mask/logits must match and padding logits must be negative infinity')
    if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.long
            or targets.shape != shape or targets.device != device
            or ((targets < 0) | (targets >= moves)).any()):
        raise ValueError('Teacher targets must be int64 legal indices matching the batch')
    rows = torch.arange(batch, device=device)
    if not legal_mask[rows, targets].all():
        raise ValueError('Teacher targets include padded/illegal candidates')
    _floating(root_predictions, shape, device, 'Root predictions', bounded=True, dtype=logits.dtype)
    _floating(teacher_values, shape, device, 'Teacher values', bounded=True, dtype=logits.dtype)
    supplied = [v is not None for v in (behavior_indices, continuation_values, continuation_mask)]
    if any(supplied) and not all(supplied):
        raise ValueError('Supply all three continuation fields or none')
    if not any(supplied):
        behavior_indices = torch.full((batch,), -1, dtype=torch.long, device=device)
        continuation_values = torch.full_like(teacher_values, torch.nan)
        continuation_mask = torch.zeros(batch, dtype=torch.bool, device=device)
    if (not isinstance(continuation_mask, torch.Tensor) or continuation_mask.dtype != torch.bool
            or continuation_mask.shape != shape or continuation_mask.device != device
            or not isinstance(behavior_indices, torch.Tensor) or behavior_indices.dtype != torch.long
            or behavior_indices.shape != shape or behavior_indices.device != device
            or not isinstance(continuation_values, torch.Tensor) or not continuation_values.is_floating_point()
            or continuation_values.shape != shape or continuation_values.device != device
            or continuation_values.dtype != logits.dtype):
        raise ValueError('Continuation fields must have matching shapes/devices and correct dtypes')
    present = behavior_indices >= 0
    if ((behavior_indices < -1) | (behavior_indices >= moves)).any() or (continuation_mask & ~present).any():
        raise ValueError('Available continuations require valid behavior indices')
    if not legal_mask[rows[present], behavior_indices[present]].all():
        raise ValueError('Behavior indices include padded/illegal candidates')
    available = continuation_values[continuation_mask]
    if not torch.isfinite(available).all() or (available.abs() > 1).any():
        raise ValueError('Available continuation values must be finite and bounded in [-1,1]')
    ce = F.cross_entropy(logits, targets)
    root_mse = F.mse_loss(root_predictions, teacher_values)
    auxiliary = ce.new_zeros(())
    teacher_count, continuation_count = 0, 0
    if objective != 'policy':
        _floating(candidate_predictions, logits.shape, device, 'Candidate predictions', bounded=True, dtype=logits.dtype)
        errors = (candidate_predictions[rows, targets]-teacher_values).square()
        teacher_count = batch
        if objective == 'continuation':
            used = continuation_mask & (behavior_indices != targets)
            indices = rows[used]
            continuation_count = len(indices)
            if continuation_count:
                extra = (candidate_predictions[indices, behavior_indices[used]]-continuation_values[used]).square()
                errors = errors.index_add(0, indices, extra)/(1+used.to(errors.dtype))
        auxiliary = errors.mean()
    elif candidate_predictions is not None:
        raise ValueError('Policy objective must not compute a critic prediction')
    return {'total': ce+.5*root_mse+auxiliary, 'policy_ce': ce, 'root_value_mse': root_mse,
            'candidate_value_mse': auxiliary, 'teacher_targets_used': teacher_count,
            'continuation_targets_used': continuation_count,
            'supervised_targets_used': teacher_count+continuation_count}


def training_objective(model, observations, candidates, legal_mask, targets, teacher_values,
                       behavior_indices=None, continuation_values=None, continuation_mask=None):
    """Run the actor once; invoke the training critic only for auxiliary arms."""
    if not isinstance(model, ContinuationChess):
        raise TypeError('Expected a ContinuationChess model')
    parameter = next(model.parameters())
    if (not isinstance(observations, torch.Tensor) or not observations.is_floating_point()
            or not torch.isfinite(observations).all() or observations.dtype != parameter.dtype
            or observations.device != parameter.device):
        raise ValueError('Observations must be finite floating tensors')
    logits, roots, hidden = model(observations, candidates, legal_mask)
    values = None if model.objective == 'policy' else model.candidate_values(hidden, candidates)
    return loss_components(model.objective, logits, roots, values, legal_mask, targets, teacher_values,
                           behavior_indices, continuation_values, continuation_mask)
