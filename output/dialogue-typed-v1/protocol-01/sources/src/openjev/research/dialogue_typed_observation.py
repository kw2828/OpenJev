"""Matched flat and typed normalization of public candidate evidence.

This is a conditional observation decoder, not a recurrent memory or a calibrated
Bayesian filter. Both modes compute identical candidate scores and branch gates.
The intervention is their normalization: typed separates NONE, DONTCARE and
concrete-value probability mass. Registered parameter equality does not imply
equal effective gradient paths. In particular, typed singleton within-branch
scores cancel, both scalar head biases are common shifts, and the entropy input
is always zero. Public candidate types are not the current target's type.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.dialogue_conditional_observation import (
    OUTPUT_TOLERANCE,
    DialogueConditionalObservation,
)

VERSION = "dialogue-typed-observation-v1"
MODES = ("flat", "typed")
CANDIDATE_TYPES = {"NONE": 0, "DONTCARE": 1, "TRUE": 2, "FALSE": 3, "OTHER": 4}
BRANCHES = ("NONE", "DONTCARE", "concrete")
PARENT_SHA256 = "2141d8f24f973bdad51f98f3963990ccf7f932b50b0777ac87628db434771982"


class DialogueTypedObservation(DialogueConditionalObservation):
    """Candidate-attention decoder with a matched normalization-only control.

    Inputs follow DialogueConditionalObservation's candidate mode, plus int64
    candidate_types[B,C], exactly -1 off support and 0..4 on support. Every row
    contains one NONE, one DONTCARE and at least one concrete candidate. The
    previous one-hot is the caller's explicitly privileged prior value. No
    current label, transition label or learned sequence state is accepted.
    """

    def __init__(self, mode: str, input_dim: int = 384,
                 projection_dim: int = 64, hidden_dim: int = 64):
        if mode not in MODES:
            raise ValueError("Unknown typed observation mode")
        # Keep the frozen parent's pooling dispatch as candidate attention.
        super().__init__("candidate", input_dim, projection_dim, hidden_dim)
        self.decision_mode = mode
        self.feature = nn.Sequential(nn.Linear(6*projection_dim+17, hidden_dim), nn.Tanh())
        self.gate = nn.Linear(hidden_dim, 1)

    def configuration(self):
        config = super().configuration()
        config.update({
            "version": VERSION, "mode": self.decision_mode,
            "attention_mode": "candidate", "parent_source_sha256": PARENT_SHA256,
            "feature_dim": 6*self.projection_dim+17,
            "feature_order": "u,q,c,u*q,u*c,q*c,lexical10,type_onehot5,previous_onehot,zero_entropy",
            "candidate_types": dict(CANDIDATE_TYPES), "padding_type": -1,
            "branches": list(BRANCHES),
            "branch_gate": "Shared Linear(hidden,1) on masked mean hidden per branch",
            "normalization": ("log_softmax(z_c+g_branch)" if self.decision_mode == "flat"
                              else "log_softmax(g)_branch + within_branch_log_softmax(z)_c"),
            "softmax_shift_parameters": 2,
            "inactive_parameters": "Entropy-column weights have zero input; writer and gate scalar biases are common shifts. Typed singleton within-branch writer scores have zero output effect",
            "parameter_count_scope": "Registered counts only, not equal effective capacity or gradient paths; same tensors and all shared score computation in both modes",
            "initialization": "Same constructor draws in both modes; copy full new-model state before fitting. No old-checkpoint identity claim",
            "scope": "Public candidate ontology types and privileged previous value; no current target or transition labels, no recurrence, no calibration claim",
        })
        return config

    def _types(self, candidate_types, candidate_mask):
        if (not isinstance(candidate_types, torch.Tensor)
                or candidate_types.shape != candidate_mask.shape
                or candidate_types.dtype != torch.int64
                or candidate_types.device != candidate_mask.device
                or (candidate_types[~candidate_mask] != -1).any()
                or ((candidate_types[candidate_mask] < 0)
                    | (candidate_types[candidate_mask] > 4)).any()):
            raise ValueError("Candidate types require int64 0..4 on support and -1 on padding")
        branch_ids = candidate_types.clamp(min=0, max=2)
        branch_mask = (branch_ids[:, None] == torch.arange(3, device=branch_ids.device)[None, :, None])
        branch_mask = branch_mask & candidate_mask[:, None]
        counts = branch_mask.sum(-1)
        if ((counts[:, :2] != 1).any() or (counts[:, 2] < 1).any()):
            raise ValueError("Each row requires one NONE, one DONTCARE and at least one concrete candidate")
        return branch_ids, branch_mask

    def _scores(self, observation, query, candidates, candidate_mask, lexical,
                previous_onehot, candidate_types, token_mask, token_prior):
        b, c = self._schema(query, candidates, candidate_mask)
        self._float(lexical, (b, c, 10), "lexical features")
        self._float(previous_onehot, (b, c), "previous one-hot")
        if (not ((previous_onehot == 0) | (previous_onehot == 1)).all()
                or not (previous_onehot.sum(-1) == 1).all()
                or (previous_onehot[~candidate_mask] != 0).any()):
            raise ValueError("Previous value must be exact supported one-hot")
        branch_ids, branch_mask = self._types(candidate_types, candidate_mask)
        evidence = self._pool(observation, query, candidates, candidate_mask, token_mask, token_prior)
        u = self.turn_projection(evidence).tanh()
        q = self.query_projection(query).tanh()[:, None].expand(b, c, self.projection_dim)
        cp = self.candidate_projection(torch.where(candidate_mask[..., None], candidates, 0.)).tanh()
        lex = torch.where(candidate_mask[..., None], lexical, 0.)
        flags = F.one_hot(candidate_types.clamp(min=0), 5).to(u.dtype)
        flags = torch.where(candidate_mask[..., None], flags, 0.)
        features = torch.cat((u, q, cp, u*q, u*cp, q*cp, lex, flags,
                              previous_onehot[..., None], torch.zeros_like(previous_onehot[..., None])), -1)
        hidden = self.feature(features)
        z = self.head(hidden).squeeze(-1)
        membership = branch_mask.to(hidden.dtype)
        means = torch.einsum("btc,bch->bth", membership, hidden)/membership.sum(-1, keepdim=True)
        g = self.gate(means).squeeze(-1)
        if not torch.isfinite(z).all() or not torch.isfinite(g).all():
            raise ValueError("Nonfinite typed candidate or branch scores")
        return z, g, branch_ids, branch_mask

    def _log_probabilities(self, z, g, branch_ids, branch_mask, candidate_mask):
        if self.decision_mode == "flat":
            log_probs = (z+g.gather(1, branch_ids)).masked_fill(~candidate_mask, -torch.inf).log_softmax(-1)
        else:
            within = z[:, None].expand_as(branch_mask).masked_fill(~branch_mask, -torch.inf).log_softmax(-1)
            log_probs = g.log_softmax(-1).gather(1, branch_ids)+within.gather(1, branch_ids[:, None]).squeeze(1)
            log_probs = log_probs.masked_fill(~candidate_mask, -torch.inf)
        if (not torch.isfinite(log_probs[candidate_mask]).all()
                or not torch.isneginf(log_probs[~candidate_mask]).all()
                or ((log_probs.exp().to(torch.float64).sum(-1)-1).abs() > OUTPUT_TOLERANCE).any()):
            raise ValueError("Invalid typed output: finite supported log probabilities and normalized mass required")
        return log_probs

    def forward(self, observation, query, candidates, candidate_mask, lexical,
                previous_onehot, candidate_types, *, token_mask=None, token_prior=None):
        factors = self._scores(observation, query, candidates, candidate_mask, lexical,
                               previous_onehot, candidate_types, token_mask, token_prior)
        return self._log_probabilities(*factors, candidate_mask)


def copy_initialization(source: DialogueTypedObservation, target: DialogueTypedObservation):
    """Validate then copy the complete paired initialization, without aliasing."""
    if not isinstance(source, DialogueTypedObservation) or not isinstance(target, DialogueTypedObservation):
        raise TypeError("Paired initialization requires typed observation decoders")
    src, dst = dict(source.named_parameters()), dict(target.named_parameters())
    if src.keys() != dst.keys():
        raise ValueError("Incompatible initialization tensor names")
    for name, value in src.items():
        if (value.shape != dst[name].shape or value.dtype != dst[name].dtype
                or value.device != dst[name].device or not torch.isfinite(value).all()):
            raise ValueError(f"Incompatible initialization tensor: {name}")
    with torch.no_grad():
        for name, value in src.items():
            dst[name].copy_(value)
    return list(src)
