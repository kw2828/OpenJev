"""Matched token comparison with opposite means or bidirectional soft alignment.

This adapts the established align/compare/aggregate pattern to a conditional
observation scorer. It is neither a recurrent memory nor a calibrated filter.
Both modes use the same tensors and flat candidate-plus-branch normalization.
The mean control has the same token access and comparison capacity; alignment
adds pairwise computation, not parameters. Training cost remains unmeasured.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.dialogue_conditional_observation import PRIOR_TOLERANCE
from openjev.research.dialogue_copy_memory import LEXICAL_FIELDS
from openjev.research.dialogue_typed_observation import (
    BRANCHES,
    CANDIDATE_TYPES,
    DialogueTypedObservation,
)

VERSION = "dialogue-token-alignment-v1"
MODES = ("mean", "aligned")
PARENT_SHA256 = "cd3c38ffea6d177431e73c8a4e2d6b10644d5e1d90e4b949451d44d50f35e493"
COMMON_TENSORS = (
    "query_projection.weight", "query_projection.bias",
    "candidate_projection.weight", "candidate_projection.bias",
    "feature.0.weight", "feature.0.bias", "head.weight", "head.bias",
    "gate.weight", "gate.bias",
)


class DialogueTokenAlignment(DialogueTypedObservation):
    """One conditional query per row, same dtype/device as model (float32/64).

    forward has the typed parent's seven positional inputs. observation[B,L,D]
    and schema_tokens[B,C,S,D] are raw frozen token vectors. The required keyword
    masks/priors select public token support, with positive priors summing to one.
    Padded candidates have empty schema support and exactly zero schema priors.
    No current target, transition label, time axis or persistent state is accepted.
    """

    def __init__(self, mode: str, input_dim: int = 384,
                 projection_dim: int = 64, hidden_dim: int = 64):
        # Construct only used modules. The frozen parent's old attention and turn
        # projection are deliberately absent, including their initialization draws.
        nn.Module.__init__(self)
        if mode not in MODES:
            raise ValueError("Unknown token alignment mode")
        if any(type(n) is not int or n < 1 for n in (input_dim, projection_dim, hidden_dim)):
            raise ValueError("Dimensions must be positive integers")
        self.mode, self.decision_mode = mode, "flat"
        self.input_dim, self.projection_dim, self.hidden_dim = input_dim, projection_dim, hidden_dim
        self.query_projection = nn.Linear(input_dim, projection_dim)
        self.candidate_projection = nn.Linear(input_dim, projection_dim)
        self.feature = nn.Sequential(nn.Linear(6*projection_dim+17, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(hidden_dim, 1)
        self.token_projection = nn.Linear(input_dim, projection_dim)
        self.compare = nn.Sequential(nn.Linear(4*projection_dim, projection_dim), nn.Tanh())
        self.combine = nn.Sequential(nn.Linear(2*projection_dim, projection_dim), nn.Tanh())

    def configuration(self):
        total = sum(p.numel() for p in self.parameters())
        common = sum(p.numel() for name, p in self.named_parameters() if name in COMMON_TENSORS)
        return {
            "class": type(self).__name__, "version": VERSION, "mode": self.mode,
            "input_dim": self.input_dim, "projection_dim": self.projection_dim, "hidden_dim": self.hidden_dim,
            "parameters": total, "common_scorer_parameters": common, "token_comparison_parameters": total-common,
            "parent_source_sha256": PARENT_SHA256, "candidate_chunk_size": 1,
            "feature_dim": 6*self.projection_dim+17,
            "feature_order": "u,q,c,u*q,u*c,q*c,lexical10,type_onehot5,previous_onehot,zero_entropy",
            "lexical_fields": list(LEXICAL_FIELDS), "candidate_types": dict(CANDIDATE_TYPES), "padding_type": -1,
            "branches": list(BRANCHES), "normalization": "log_softmax(z_c+g_branch)",
            "projection": "Shared Linear(D,P) with bias, no pre-alignment activation or L2 normalization",
            "comparison": "Shared tanh(Linear([x,a,x-a,x*a])); supplied-prior pooling on each side; tanh(Linear([context,schema]))",
            "alignment": ("Opposite sequence supplied-prior weighted mean" if self.mode == "mean" else
                          "Bidirectional softmax(dot/sqrt(P)+log(opposite_prior)), masked on opposite tokens"),
            "work": "Context projection once; schema projection and comparisons once per candidate, including padded slots. Aligned score matrices are B*L*S per candidate, never B*C*L*S*P",
            "memory": "Sequential candidate processing limits individual temporaries; autograd retains candidate graphs through backward. Not a peak-memory or runtime bound",
            "parameter_count_scope": "Registered counts, identical across new modes. No unused attention or turn-projection modules. Entropy-column weights receive zero input; head/gate scalar biases are common shifts",
            "initialization": "Copy complete tensors between mean/aligned; copy only COMMON_TENSORS to/from the separate flat baseline. No old fitted checkpoint reuse",
            "scope": "Public candidate types and privileged previous one-hot; no current target, recurrent state, calibration or novelty claim",
        }

    def work_counts(self, batch_size, candidate_count, context_tokens, schema_tokens):
        """Shape-based padded work, not traffic, allocated memory or timing.

        Every candidate slot is computed, even a padded one. The alignment matrix
        count includes its two directional normalizations; mean mode has no matrix.
        """
        if any(type(n) is not int or n < 1 for n in (batch_size, candidate_count, context_tokens, schema_tokens)):
            raise ValueError("Work dimensions must be positive integers")
        b, c, l, s = batch_size, candidate_count, context_tokens, schema_tokens
        matrix = b*c*l*s if self.mode == "aligned" else 0
        return {
            "candidate_chunks": c, "context_projection_rows": b*l, "schema_projection_rows": b*c*s,
            "comparison_rows": b*c*(l+s), "combined_evidence_rows": b*c,
            "pairwise_score_scalars": matrix, "directional_alignment_scalars": 2*matrix,
            "largest_score_matrix_scalars": b*l*s if self.mode == "aligned" else 0,
        }

    def _float(self, value, shape, name):
        ref = self.token_projection.weight
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype not in (torch.float32, torch.float64) or value.dtype != ref.dtype
                or value.device != ref.device or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid {name}: shape, dtype, device or finite values")

    def _mask(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != torch.bool or value.device != self.token_projection.weight.device):
            raise ValueError(f"Invalid {name}")

    def _tokens(self, observation, candidate_mask, token_mask, token_prior,
                schema_tokens, schema_mask, schema_prior):
        b, c = candidate_mask.shape
        if (not isinstance(observation, torch.Tensor) or observation.ndim != 3 or observation.shape[1] < 1
                or not isinstance(schema_tokens, torch.Tensor) or schema_tokens.ndim != 4 or schema_tokens.shape[2] < 1):
            raise ValueError("Nonempty observation[B,L,D] and schema_tokens[B,C,S,D] required")
        l, s = observation.shape[1], schema_tokens.shape[2]
        self._float(observation, (b, l, self.input_dim), "observation")
        self._float(schema_tokens, (b, c, s, self.input_dim), "schema tokens")
        self._mask(token_mask, (b, l), "token mask")
        self._mask(schema_mask, (b, c, s), "schema mask")
        self._float(token_prior, (b, l), "token prior")
        self._float(schema_prior, (b, c, s), "schema prior")
        if (not token_mask.any(-1).all() or (token_prior[~token_mask] != 0).any()
                or (token_prior[token_mask] <= 0).any()
                or ((token_prior.to(torch.float64).sum(-1)-1).abs() > PRIOR_TOLERANCE).any()):
            raise ValueError("Context priors require nonempty positive support, zero padding and unit mass")
        if (not torch.equal(schema_mask.any(-1), candidate_mask)
                or (schema_prior[~schema_mask] != 0).any() or (schema_prior[schema_mask] <= 0).any()
                or ((schema_prior.to(torch.float64).sum(-1)-candidate_mask.to(torch.float64)).abs() > PRIOR_TOLERANCE).any()):
            raise ValueError("Schema priors require candidate-matched positive support, zero padding and unit mass")

    def _evidence(self, observation, candidate_mask, token_mask, token_prior,
                  schema_tokens, schema_mask, schema_prior):
        self._tokens(observation, candidate_mask, token_mask, token_prior, schema_tokens, schema_mask, schema_prior)
        clean = torch.where(token_mask[..., None], observation, 0.)
        x = self.token_projection(clean)
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite projected context")
        prior = torch.where(token_mask, token_prior, 0.)
        context_mean = torch.einsum("bl,blp->bp", prior, x) if self.mode == "mean" else None
        context_log_prior = token_prior.masked_fill(~token_mask, 1.).log()
        evidence = []
        for index in range(candidate_mask.shape[1]):
            active = candidate_mask[:, index]
            sm = schema_mask[:, index]
            # Padded candidate has one numerical sentinel token, then is zeroed
            # before the shared scorer. No all--inf softmax or fabricated evidence
            # enters an actual supported candidate. Off-support gradients are zero.
            sentinel = torch.zeros_like(sm)
            sentinel[:, 0] = True
            safe_mask = sm | (~active[:, None] & sentinel)
            supported_prior = torch.where(sm, schema_prior[:, index], 0.)
            sp = torch.where(active[:, None], supported_prior, sentinel.to(prior.dtype))
            y = self.token_projection(torch.where(sm[..., None], schema_tokens[:, index], 0.))
            if not torch.isfinite(y).all():
                raise ValueError("Nonfinite projected schema")
            if self.mode == "aligned":
                scores = torch.bmm(x, y.transpose(1, 2))/math.sqrt(self.projection_dim)
                if not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite alignment scores")
                xy = (scores+sp.masked_fill(~safe_mask, 1.).log()[:, None]).masked_fill(~safe_mask[:, None], -torch.inf).softmax(-1)
                yx = (scores.transpose(1, 2)+context_log_prior[:, None]).masked_fill(~token_mask[:, None], -torch.inf).softmax(-1)
                aligned_y, aligned_x = torch.bmm(xy, y), torch.bmm(yx, x)
            else:
                aligned_y = torch.einsum("bs,bsp->bp", sp, y)[:, None].expand_as(x)
                aligned_x = context_mean[:, None].expand_as(y)
            left = torch.cat((x, aligned_y, x-aligned_y, x*aligned_y), -1)
            right = torch.cat((y, aligned_x, y-aligned_x, y*aligned_x), -1)
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError("Nonfinite token comparison features")
            cx, cy = self.compare(left), self.compare(right)
            pooled = torch.cat((torch.einsum("bl,blp->bp", prior, cx),
                                torch.einsum("bs,bsp->bp", sp, cy)), -1)
            u = self.combine(pooled)
            if not torch.isfinite(u).all():
                raise ValueError("Nonfinite comparison evidence")
            evidence.append(torch.where(active[:, None], u, 0.))
        return torch.stack(evidence, 1)

    def pool(self, observation, query, candidates, candidate_mask, *, token_mask, token_prior,
             schema_tokens, schema_mask, schema_prior):
        """Return [B,C,P] comparison evidence, already transformed for the scorer."""
        self._schema(query, candidates, candidate_mask)
        return self._evidence(observation, candidate_mask, token_mask, token_prior, schema_tokens, schema_mask, schema_prior)

    def _scores(self, observation, query, candidates, candidate_mask, lexical, previous_onehot,
                candidate_types, token_mask, token_prior, schema_tokens, schema_mask, schema_prior):
        b, c = self._schema(query, candidates, candidate_mask)
        self._float(lexical, (b, c, 10), "lexical features")
        self._float(previous_onehot, (b, c), "previous one-hot")
        if (not ((previous_onehot == 0) | (previous_onehot == 1)).all()
                or not (previous_onehot.sum(-1) == 1).all() or (previous_onehot[~candidate_mask] != 0).any()):
            raise ValueError("Previous value must be exact supported one-hot")
        branch_ids, branch_mask = self._types(candidate_types, candidate_mask)
        u = self._evidence(observation, candidate_mask, token_mask, token_prior, schema_tokens, schema_mask, schema_prior)
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
            raise ValueError("Nonfinite candidate or branch scores")
        return z, g, branch_ids, branch_mask

    def forward(self, observation, query, candidates, candidate_mask, lexical, previous_onehot,
                candidate_types, *, token_mask, token_prior, schema_tokens, schema_mask, schema_prior):
        factors = self._scores(observation, query, candidates, candidate_mask, lexical, previous_onehot,
                               candidate_types, token_mask, token_prior, schema_tokens, schema_mask, schema_prior)
        return self._log_probabilities(*factors, candidate_mask)


def _copy(source, target, names):
    src, dst = dict(source.named_parameters()), dict(target.named_parameters())
    for name in names:
        if (name not in src or name not in dst or src[name].shape != dst[name].shape
                or src[name].dtype != dst[name].dtype or src[name].device != dst[name].device
                or not torch.isfinite(src[name]).all()):
            raise ValueError(f"Incompatible initialization tensor: {name}")
    with torch.no_grad():
        for name in names:
            dst[name].copy_(src[name])
    return list(names)


def copy_initialization(source: DialogueTokenAlignment, target: DialogueTokenAlignment):
    """Validate then copy all paired tensors without aliasing or new draws."""
    if not isinstance(source, DialogueTokenAlignment) or not isinstance(target, DialogueTokenAlignment):
        raise TypeError("Full initialization copy requires token alignment models")
    names = tuple(dict(source.named_parameters()))
    if set(names) != set(dict(target.named_parameters())):
        raise ValueError("Incompatible initialization tensor names")
    return _copy(source, target, names)


def copy_common_initialization(source: DialogueTypedObservation, target: DialogueTypedObservation):
    """Copy compatible q/c, feature, writer and gate tensors, excluding pooling.

    Caller must pair fresh initial models; this helper does not authenticate or
    load old checkpoints. Full mean/aligned pairing uses copy_initialization.
    """
    if not isinstance(source, DialogueTypedObservation) or not isinstance(target, DialogueTypedObservation):
        raise TypeError("Common initialization copy requires typed or alignment scorers")
    return _copy(source, target, COMMON_TENSORS)
