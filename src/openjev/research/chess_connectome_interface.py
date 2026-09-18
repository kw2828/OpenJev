"""Hard spatial-port permutation prototype, not a trained or checkpointed study.

One 64-square bijection is shared by every channel and tied between gather and
scatter. The original adapter's trainable projections, graph and recurrence are
unchanged. No soft mixtures, dense interface, inference labels or save/load
contract are introduced. Search warmup, proposal budget and evaluation remain
the responsibility of a separately frozen experiment.
"""

from __future__ import annotations

import hashlib
import math
import time

import torch
from torch.nn import functional as F

from openjev.research.chess_candidate import _sync
from openjev.research.chess_connectome_adapter import ConnectomeChessAdapter

ARCHITECTURE = 'connectome-hard-square-interface-prototype-v1'
SQUARES = 64


def _mapping_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().astype('<i8').tobytes()).hexdigest()


def _permutation(value, device):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.long
            or value.ndim != 1 or value.numel() != SQUARES):
        raise ValueError('Permutation must be a torch.int64 vector of length64')
    # Small discrete mapping administration is explicitly CPU work. Neural
    # inference remains on its original device; this is not a model fallback.
    value = value.detach().cpu().clone()
    if not torch.equal(value.sort().values, torch.arange(SQUARES)):
        raise ValueError('Permutation must contain every square index exactly once')
    return value.to(device=device)


class HardSquareConnectomeAdapter(ConnectomeChessAdapter):
    """Absolute original-square-to-current-square assignment, identical initially."""

    def __init__(self, backbone, graph, **adapter_kwargs):
        super().__init__(backbone, graph, **adapter_kwargs)
        self.register_buffer('original_node_slots', self.node_slots.detach().clone())
        self.register_buffer('original_slot_occupancy', self.slot_occupancy.detach().clone())
        self.register_buffer('square_permutation', torch.arange(SQUARES, device=self.node_slots.device))
        self._original_mapping_sha256 = _mapping_hash(self.original_node_slots)
        self._assert_interface()

    def _original(self):
        original = self.original_node_slots.detach().cpu()
        if (original.dtype != torch.long or original.shape != (self.nodes,)
                or _mapping_hash(original) != self._original_mapping_sha256
                or not torch.equal(self.original_slot_occupancy.detach().cpu(),
                                   torch.bincount(original, minlength=self.slot_count)
                                   .to(self.slot_occupancy.dtype))):
            raise ValueError('Original slot assignment or occupancy changed')
        return original

    def _assert_interface(self):
        original = self._original()
        permutation = _permutation(self.square_permutation, torch.device('cpu'))
        expected = original.div(SQUARES, rounding_mode='floor') * SQUARES
        expected = expected + permutation[original % SQUARES]
        if (not torch.equal(expected, self.node_slots.detach().cpu())
                or not torch.equal(self.slot_occupancy.detach().cpu(),
                                   torch.bincount(expected, minlength=self.slot_count)
                                   .to(self.slot_occupancy.dtype))):
            raise ValueError('Gather/scatter mapping disagrees with the tied hard permutation')

    @torch.no_grad()
    def apply_permutation(self, permutation):
        """Set an absolute bijection; input tensors are copied, never retained by reference."""
        original = self._original()
        permutation = _permutation(permutation, torch.device('cpu'))
        slots = original.div(SQUARES, rounding_mode='floor') * SQUARES
        slots = slots + permutation[original % SQUARES]
        occupancy = torch.bincount(slots, minlength=self.slot_count).to(self.slot_occupancy.dtype)
        self.node_slots.copy_(slots)
        self.slot_occupancy.copy_(occupancy)
        self.square_permutation.copy_(permutation)
        return self

    def restore_permutation(self):
        """Restore the original fixed mapping exactly."""
        return self.apply_permutation(torch.arange(SQUARES, device=self.node_slots.device))

    @property
    def config(self):
        self._assert_interface()
        return {**super().config, 'architecture': ARCHITECTURE,
                'original_mapping_sha256': self._original_mapping_sha256,
                'mapping_sha256': _mapping_hash(self.node_slots),
                'square_permutation': self.square_permutation.detach().cpu().tolist(),
                'node_interface': 'One hard64-square permutation shared across channels; tied gather/scatter',
                'permutation_training': 'External bounded training-only swap proposals; no soft mixtures',
                'mapping_administration': 'Explicit CPU bijection/hash/occupancy checks and device transfers; '
                                          'neural computation stays on the configured device',
                'checkpoint_contract': 'Prototype only; no save/load or original-adapter checkpoint compatibility'}

    @property
    def provenance(self):
        return {**super().provenance, 'interface_architecture': ARCHITECTURE,
                'original_mapping_sha256': self._original_mapping_sha256,
                'interface_permutation_learnable': True,
                'interface_policy': 'Discrete square swaps only; original channel assignments and graph unchanged',
                'experiment_status': 'Untrained interface prototype, not a frozen scientific trial'}

    def forward(self, observations, candidates, legal_mask):
        self._assert_interface()
        return super().forward(observations, candidates, legal_mask)

    @torch.no_grad()
    def choose(self, board):
        """Keep the inherited decision and explicitly identify the new interface."""
        device = self.edge_log_gain.device
        started = time.perf_counter()
        _sync(device)
        result = super().choose(board)
        result.update({'interface_architecture': ARCHITECTURE,
                       'square_permutation_sha256': _mapping_hash(self.square_permutation),
                       'node_mapping_sha256': _mapping_hash(self.node_slots),
                       'original_mapping_sha256': self._original_mapping_sha256,
                       'model_kind': 'Hard-square-interface prototype on frozen direct graph adapter; '
                                     'no search, soft mixing or cross-move state',
                       'timing_scope': 'Entire choose wrapper, initial/final device synchronization, '
                                       'native preparation, neural inference and CPU interface metadata/checks'})
        _sync(device)
        result['latency_ms'] = (time.perf_counter() - started) * 1000
        return result


def _batch(model, batch):
    keys = {'observations', 'candidates', 'legal_mask', 'targets', 'values'}
    if not isinstance(batch, dict) or set(batch) != keys or any(
            not isinstance(value, torch.Tensor) for value in batch.values()):
        raise ValueError('Training batch must contain exactly five named tensors')
    batch = {key: value.detach().clone() for key, value in batch.items()}
    model._check_inputs(batch['observations'], batch['candidates'], batch['legal_mask'])
    targets, values, mask = batch['targets'], batch['values'], batch['legal_mask']
    count = len(mask)
    if (targets.dtype != torch.long or targets.shape != (count,)
            or targets.device != mask.device or values.device != mask.device
            or values.dtype != batch['observations'].dtype or values.shape != (count,)
            or not torch.isfinite(values).all() or (values.abs() > 1).any()
            or (targets < 0).any() or (targets >= mask.shape[1]).any()):
        raise ValueError('Invalid policy indices or bounded value targets')
    if not mask[torch.arange(count, device=mask.device), targets].all():
        raise ValueError('A supervised candidate is padded or illegal')
    return batch


def evaluate_square_swap(model, batch, first, second, *, policy='learned', value_weight=.5):
    """Evaluate exactly two fixed-weight objectives on one copied training batch.

    Swap the images of ``first`` and ``second`` original square slots. Learned
    mode retains the proposed mapping only for a strict finite loss decrease;
    fixed mode always retains its entry mapping despite doing identical work.
    Existing gradients must have been cleared by the caller. On an exception,
    the entry mapping, parameter tensors and training mode are restored. The
    helper performs no optimizer update and creates no gradient graph.
    """
    if not isinstance(model, HardSquareConnectomeAdapter):
        raise TypeError('Expected a HardSquareConnectomeAdapter')
    started = time.perf_counter()
    device = model.edge_log_gain.device
    _sync(device)
    if (type(first) is not int or type(second) is not int or first == second
            or not 0 <= first < SQUARES or not 0 <= second < SQUARES):
        raise ValueError('Swap requires two distinct integer square indices in [0,64)')
    if policy not in ('learned', 'fixed'):
        raise ValueError('Proposal policy must be learned or fixed')
    if (type(value_weight) not in (int, float) or not math.isfinite(value_weight) or value_weight < 0):
        raise ValueError('Value objective weight must be finite and nonnegative')
    model._assert_interface()
    batch = _batch(model, batch)
    frozen_batch = {key: value.clone() for key, value in batch.items()}
    parameters = dict(model.named_parameters())
    if any(value.grad is not None or not torch.isfinite(value).all() for value in parameters.values()):
        raise ValueError('Proposals require finite parameters and cleared gradients')
    frozen = {key: value.detach().clone() for key, value in parameters.items()}
    starting = model.square_permutation.detach().clone()
    proposal = starting.clone()
    proposal[first], proposal[second] = starting[second], starting[first]
    training = model.training
    complete = accepted = False

    def unchanged():
        if (set(dict(model.named_parameters())) != set(parameters)
                or any(not torch.equal(value, frozen[key]) or value.grad is not None
                       for key, value in model.named_parameters())):
            raise ValueError('A fixed-weight proposal altered parameters or gradients')
        if any(not torch.equal(value, frozen_batch[key]) for key, value in batch.items()):
            raise ValueError('A proposal modified its fixed training batch')

    def objective():
        logits, values, _ = model(batch['observations'], batch['candidates'], batch['legal_mask'])
        policy_ce = F.cross_entropy(logits, batch['targets'])
        value_mse = F.mse_loss(values, batch['values'])
        loss = policy_ce + value_weight * value_mse
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError('Proposal objective must be a finite scalar')
        unchanged()
        return {'loss': float(loss), 'policy_ce': float(policy_ce), 'value_mse': float(value_mse)}

    try:
        model.eval()
        with torch.no_grad():
            current = objective()
            model.apply_permutation(proposal)
            proposed = objective()
            improved = proposed['loss'] < current['loss']
            accepted = policy == 'learned' and improved
            if not accepted:
                model.apply_permutation(starting)
            result = {'status': 'completed_proposal', 'policy': policy, 'swap': [first, second],
                      'current': current, 'proposed': proposed, 'strictly_improved': improved,
                      'accepted': accepted, 'forward_calls': 2, 'optimizer_updates': 0,
                      'evaluated_examples': 2 * len(batch['targets']), 'value_weight': value_weight,
                      'starting_permutation': starting.cpu().tolist(),
                      'proposed_permutation': proposal.cpu().tolist(),
                      'final_permutation': model.square_permutation.cpu().tolist()}
            complete = True
    finally:
        if not complete:
            with torch.no_grad():
                for key, value in parameters.items():
                    value.copy_(frozen[key])
                    value.grad = None
                model.apply_permutation(starting)
        model.train(training)
        _sync(device)
    result['wall_seconds'] = time.perf_counter() - started
    result['timing_scope'] = ('Full helper, initial/final device synchronization, CPU mapping validation, '
                              'batch copies, parameter snapshots, both neural forwards, objective computation, '
                              'invariant checks, mapping restoration when rejected, and training-mode restoration')
    result['neural_device'] = str(device)
    result['mapping_administration'] = 'Explicit CPU work and transfers; no neural CPU fallback'
    return result
