"""Matched spatial chess recurrence with a residual or fixed encoder anchor.

The frozen SpatialChess initializer supplies exactly the same modules, parameters
and initial state in both arms. Only the recurrent addition differs. Given
``x = encoder(board)`` and ``h0 = x``, each iteration uses
``F(h) = conv2(ReLU(conv1(h)))`` and updates either ``ReLU(h + F(h))``
or ``ReLU(x + F(h))``. The encoder anchor is not detached.

These are internal refinements of one visible board. They perform no search and
carry no state between moves. Future-board prediction remains an optional head
with the same explicit action and target semantics as the frozen model.
"""

from pathlib import Path

import torch

from openjev.research.chess_spatial import (
    DEPTH,
    ENCODING_VERSION,
    INPUT_CHANNELS,
    WIDTH,
    SpatialChess,
    _check_candidates,
    _check_plan_hash,
)

RECURRENCES = ('residual', 'anchor')
# Deliberately differs in type/value from SpatialChess's integer format version.
# The frozen SpatialChess loader therefore rejects an anchor-format checkpoint.
CHECKPOINT_VERSION = 'openjev-chess-anchor-v1'


class AnchorChess(SpatialChess):
    def __init__(self, recurrence='residual', seed=17, width=WIDTH, depth=DEPTH):
        if type(recurrence) is not str or recurrence not in RECURRENCES:
            raise ValueError('recurrence must be residual or anchor')
        super().__init__('recurrent', seed=seed, width=width, depth=depth)
        self.recurrence = recurrence

    def forward(self, observations, candidates, legal_mask, depth=None):
        """Return legal logits, bounded side-to-move value and final spatial state.

Inputs and output shapes match SpatialChess. ``depth`` may be any positive
integer. The local encoder anchor is freshly computed on every call and remains
connected to the gradient graph. No module state is changed during recurrence.
        """
        depth = self._depth(depth)
        if observations.ndim != 4 or observations.shape[1:] != (INPUT_CHANNELS, 8, 8):
            raise ValueError('Observations must have shape [batch,19,8,8]')
        if candidates.ndim != 3 or candidates.shape[0] != observations.shape[0]:
            raise ValueError('Candidate features must have shape [batch,moves,5]')
        _check_candidates(candidates)
        if (legal_mask.dtype != torch.bool or legal_mask.shape != candidates.shape[:2]
                or not legal_mask.any(dim=-1).all()):
            raise ValueError('Every input requires a nonempty boolean legal-move mask')
        if observations.device != candidates.device or legal_mask.device != candidates.device:
            raise ValueError('Observations, candidates and mask must use the same device')
        anchor = self.encoder(observations)
        hidden = anchor
        for _ in range(depth):
            update = self.core.conv2(self.core.activation(self.core.conv1(hidden)))
            residual = hidden if self.recurrence == 'residual' else anchor
            hidden = self.core.activation(residual+update)
        squares = hidden.flatten(2).transpose(1, 2)
        batch = torch.arange(len(hidden), device=hidden.device).unsqueeze(1)
        pooled = hidden.mean(dim=(2, 3))
        features = torch.cat((squares[batch, candidates[..., 0]], squares[batch, candidates[..., 1]],
                              pooled.unsqueeze(1).expand(-1, candidates.shape[1], -1),
                              self.promotion_embedding(candidates[..., 2]),
                              self.dx_embedding(candidates[..., 3]),
                              self.dy_embedding(candidates[..., 4])), dim=-1)
        logits = self.policy_head(features).squeeze(-1).masked_fill(~legal_mask, -torch.inf)
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)
        return logits, value, hidden

    @torch.no_grad()
    def choose(self, board, depth=None):
        result = super().choose(board, depth=depth)
        return {**result, 'recurrence': self.recurrence,
                'recurrence_semantics': ('ReLU(h + F(h)); h0=encoder(board)' if self.recurrence == 'residual'
                                         else 'ReLU(x + F(h)); h0=x=encoder(board); fixed differentiable x'),
                'model_kind': 'spatial chess recurrence control; no search or cross-move state'}

    def save(self, path, *, plan_sha256):
        """Write a new checkpoint; never overwrite an existing artifact."""
        _check_plan_hash(plan_sha256)
        payload = {'format_version': CHECKPOINT_VERSION, 'encoding': ENCODING_VERSION,
                   'recurrence': self.recurrence, 'seed': self.seed, 'width': self.width,
                   'depth': self.depth, 'plan_sha256': plan_sha256, 'state_dict': self.state_dict()}
        with Path(path).open('xb') as handle:
            torch.save(payload, handle)

    @classmethod
    def load(cls, path, *, expected_plan_sha256=None, expected_recurrence=None, expected_seed=None):
        """Read tensor-only weights with strict format, metadata and state checks.

The caller can bind the requested recurrence, seed and frozen training plan.
Old SpatialChess checkpoints are rejected rather than implicitly reinterpreted.
        """
        payload = torch.load(path, map_location='cpu', weights_only=True)
        required = {'format_version', 'encoding', 'recurrence', 'seed', 'width', 'depth',
                    'plan_sha256', 'state_dict'}
        if (not isinstance(payload, dict) or set(payload) != required
                or payload['format_version'] != CHECKPOINT_VERSION or payload['encoding'] != ENCODING_VERSION):
            raise ValueError('Anchor checkpoint format or encoding mismatch')
        _check_plan_hash(payload['plan_sha256'])
        if expected_plan_sha256 is not None:
            _check_plan_hash(expected_plan_sha256)
            if payload['plan_sha256'] != expected_plan_sha256:
                raise ValueError('Anchor checkpoint plan mismatch')
        if expected_recurrence is not None:
            if type(expected_recurrence) is not str or expected_recurrence not in RECURRENCES:
                raise ValueError('Expected recurrence must be residual or anchor')
            if payload['recurrence'] != expected_recurrence:
                raise ValueError('Anchor checkpoint recurrence mismatch')
        if expected_seed is not None and (type(expected_seed) is not int or payload['seed'] != expected_seed):
            raise ValueError('Anchor checkpoint seed mismatch')
        model = cls(payload['recurrence'], payload['seed'], payload['width'], payload['depth'])
        model.load_state_dict(payload['state_dict'], strict=True)
        return model.eval()
