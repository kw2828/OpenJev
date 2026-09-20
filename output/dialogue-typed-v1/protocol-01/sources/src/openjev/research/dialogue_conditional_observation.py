"""Conditional observation decoding with an explicitly privileged previous value.

Each batch row is one admitted query at one current USER turn. The caller supplies
the correct previous categorical value, not the current target. There is no
learned state, transition, sequence axis, departure head or retained activation.
The ten lexical inputs may already contain causal literal-register history.

The feature layout follows DialogueCopyMemory's writer, but this standalone
decoder does not call its readout (which resets the prior) or scalar transition.
Token attention follows DialogueTokenMemory's shared-token/chunk-prior formula.
This is a representation diagnostic, not a deployable dialogue-memory policy.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.dialogue_copy_memory import LEXICAL_FIELDS

VERSION = "dialogue-conditional-observation-v1"
MODES = ("mean", "slot", "candidate")
ATTENTION_DIM = 64
POOL_EPS = 1e-12
PRIOR_TOLERANCE = 2e-6
OUTPUT_TOLERANCE = 2e-6
COMMON_TENSORS = (
    "turn_projection.weight", "turn_projection.bias",
    "query_projection.weight", "query_projection.bias",
    "candidate_projection.weight", "candidate_projection.bias",
    "feature.0.weight", "feature.0.bias", "head.weight", "head.bias",
)
ATTENTION_TENSORS = ("token_key.weight", "evidence_query.weight")


class DialogueConditionalObservation(nn.Module):
    """One independent query per row; float32/64 inputs match model dtype/device.

    forward(observation, query, candidates, candidate_mask, lexical,
            previous_onehot, *, token_mask=None, token_prior=None) -> [B,C]

    The return value is masked log probability, with exactly -inf off support.
    observation is [B,D] in mean mode and [B,L,D] otherwise; schema tensors are
    [B,D], [B,C,D], [B,C], lexical is [B,C,10], previous_onehot is [B,C].
    All rows are real observations; no label-selected update mask is accepted.
    """

    def __init__(self, mode: str, input_dim: int = 384,
                 projection_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown conditional observation mode")
        if any(type(n) is not int or n < 1 for n in (input_dim, projection_dim, hidden_dim)):
            raise ValueError("Dimensions must be positive integers")
        self.mode, self.input_dim = mode, input_dim
        self.projection_dim, self.hidden_dim = projection_dim, hidden_dim
        self.turn_projection = nn.Linear(input_dim, projection_dim)
        self.query_projection = nn.Linear(input_dim, projection_dim)
        self.candidate_projection = nn.Linear(input_dim, projection_dim)
        self.feature = nn.Sequential(nn.Linear(6*projection_dim+12, hidden_dim), nn.Tanh())
        # Only the writer row exists. No unused departure parameters or draws.
        self.head = nn.Linear(hidden_dim, 1)
        if mode != "mean":
            self.token_key = nn.Linear(input_dim, ATTENTION_DIM, bias=False)
            self.evidence_query = nn.Linear(2*input_dim, ATTENTION_DIM, bias=False)

    def configuration(self):
        total = sum(p.numel() for p in self.parameters())
        adapter = sum(p.numel() for name, p in self.named_parameters() if name in ATTENTION_TENSORS)
        return {
            "class": type(self).__name__, "version": VERSION, "mode": self.mode,
            "input_dim": self.input_dim, "projection_dim": self.projection_dim,
            "hidden_dim": self.hidden_dim, "feature_dim": 6*self.projection_dim+12,
            "lexical_fields": list(LEXICAL_FIELDS), "parameters": total,
            "shared_scorer_parameters": total-adapter, "attention_parameters": adapter,
            "zero_input_parameters": self.hidden_dim,
            "softmax_shift_parameters": 1,
            "inactive_parameters": "feature.0.weight final entropy column has zero input; scalar head.bias is a common softmax shift",
            "parameter_count_scope": "Registered counts only, not an effective dimension estimate. Slot uses duplicate query inputs; roundoff can produce a tiny common-bias gradient",
            "writer": "head Linear(hidden_dim,1); V2 writer feature order, no departure row",
            "previous_value": "Caller-supplied correct previous one-hot, supported and exact 0/1; privileged in training and evaluation",
            "entropy": "Exactly zero; previous categorical value is certain",
            "attention_width": None if self.mode == "mean" else ATTENTION_DIM,
            "attention": None if self.mode == "mean" else "softmax(log(token_prior)+dot(Wq[pair],Wk[token])/sqrt(64)); prior positive on support and sum within2e-6",
            "schema_pair": None if self.mode == "mean" else ("[query;query]" if self.mode == "slot" else "[query;candidate]"),
            "pooling": "Mean uses supplied cached normalized vector without renormalizing; token modes L2-normalize weighted raw token sum, eps1e-12",
            "output_validation": "Finite supported log probabilities; -inf off support; raw exponentiated output sums within2e-6, without floor or renormalization",
            "initialization": "Copy common new decoder tensors across all arms; copy attention tensors across slot/candidate. No claim of V2 checkpoint initialization identity",
            "scope": "Independent conditional decoder; no current gold, operation label, recurrent state or sequence rollout",
        }

    def _float(self, value, shape, name):
        ref = self.turn_projection.weight
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype not in (torch.float32, torch.float64) or value.dtype != ref.dtype
                or value.device != ref.device or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid {name}: shape, dtype, device or finite values")

    def _mask(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != torch.bool or value.device != self.turn_projection.weight.device):
            raise ValueError(f"Invalid {name}")

    def _schema(self, query, candidates, candidate_mask):
        if not isinstance(query, torch.Tensor) or query.ndim != 2:
            raise ValueError("query must be [B,D]")
        if not isinstance(candidates, torch.Tensor) or candidates.ndim != 3:
            raise ValueError("candidates must be [B,C,D]")
        b, c = query.shape[0], candidates.shape[1]
        if min(b, c) < 1:
            raise ValueError("Empty batch or candidate axis")
        self._float(query, (b, self.input_dim), "query")
        self._float(candidates, (b, c, self.input_dim), "candidates")
        self._mask(candidate_mask, (b, c), "candidate mask")
        if not candidate_mask.any(-1).all():
            raise ValueError("Each observation requires a candidate")
        return b, c

    def _pool(self, observation, query, candidates, candidate_mask, token_mask, token_prior):
        b, c = candidate_mask.shape
        if self.mode == "mean":
            if token_mask is not None or token_prior is not None:
                raise ValueError("Mean mode does not accept unused token inputs")
            self._float(observation, (b, self.input_dim), "mean observation")
            return torch.where(candidate_mask[..., None], observation[:, None].expand(b, c, self.input_dim), 0.)
        if not isinstance(observation, torch.Tensor) or observation.ndim != 3 or observation.shape[1] < 1:
            raise ValueError("Token observation must be nonempty [B,L,D]")
        length = observation.shape[1]
        self._float(observation, (b, length, self.input_dim), "token observation")
        self._mask(token_mask, (b, length), "token mask")
        self._float(token_prior, (b, length), "token prior")
        if (not token_mask.any(-1).all() or (token_prior[~token_mask] != 0).any()
                or (token_prior[token_mask] <= 0).any()
                or ((token_prior.to(torch.float64).sum(-1)-1).abs() > PRIOR_TOLERANCE).any()):
            raise ValueError("Token priors require positive nonempty support, zero padding and unit row mass")
        clean = torch.where(token_mask[..., None], observation, 0.)
        q = query[:, None].expand_as(candidates)
        pair = torch.cat((q, q if self.mode == "slot" else candidates), -1)
        schema = self.evidence_query(torch.where(candidate_mask[..., None], pair, 0.))
        keys = self.token_key(clean)
        logits = torch.einsum("bck,blk->bcl", schema, keys)/math.sqrt(ATTENTION_DIM)
        logits = logits+token_prior.masked_fill(~token_mask, 1.).log()[:, None]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite attention logits")
        weights = logits.masked_fill(~token_mask[:, None], -torch.inf).softmax(-1)
        evidence = F.normalize(torch.einsum("bcl,bld->bcd", weights, clean), dim=-1, eps=POOL_EPS)
        if not torch.isfinite(evidence).all():
            raise ValueError("Nonfinite pooled evidence")
        return torch.where(candidate_mask[..., None], evidence, 0.)

    def pool(self, observation, query, candidates, candidate_mask, *, token_mask=None, token_prior=None):
        """Return [B,C,D] evidence only, with the same checks as forward."""
        self._schema(query, candidates, candidate_mask)
        return self._pool(observation, query, candidates, candidate_mask, token_mask, token_prior)

    def forward(self, observation, query, candidates, candidate_mask, lexical,
                previous_onehot, *, token_mask=None, token_prior=None):
        b, c = self._schema(query, candidates, candidate_mask)
        self._float(lexical, (b, c, 10), "lexical features")
        self._float(previous_onehot, (b, c), "previous one-hot")
        if (not ((previous_onehot == 0) | (previous_onehot == 1)).all()
                or not (previous_onehot.sum(-1) == 1).all()
                or (previous_onehot[~candidate_mask] != 0).any()):
            raise ValueError("Previous value must be exact supported one-hot")
        evidence = self._pool(observation, query, candidates, candidate_mask, token_mask, token_prior)
        u = self.turn_projection(evidence).tanh()
        q = self.query_projection(query).tanh()[:, None].expand(b, c, self.projection_dim)
        cp = self.candidate_projection(torch.where(candidate_mask[..., None], candidates, 0.)).tanh()
        lex = torch.where(candidate_mask[..., None], lexical, 0.)
        features = torch.cat((u, q, cp, u*q, u*cp, q*cp, lex,
                              previous_onehot[..., None], torch.zeros_like(previous_onehot[..., None])), -1)
        logits = self.head(self.feature(features)).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite conditional writer logits")
        log_probs = logits.masked_fill(~candidate_mask, -torch.inf).log_softmax(-1)
        if (not torch.isfinite(log_probs[candidate_mask]).all()
                or not torch.isneginf(log_probs[~candidate_mask]).all()
                or ((log_probs.exp().to(torch.float64).sum(-1)-1).abs() > OUTPUT_TOLERANCE).any()):
            raise ValueError("Invalid conditional output: finite supported log probabilities and normalized mass required")
        return log_probs


def copy_initialization(source: DialogueConditionalObservation,
                        target: DialogueConditionalObservation, *, include_attention: bool = False):
    """Copy paired tensors without aliasing storage, constructing modules or RNG.

    Use include_attention=True for slot/candidate. Validate every selected tensor
    before any mutation. This is a caller-controlled initialization helper, not
    an instruction to overwrite fitted parameters or load old trained weights.
    """
    if not isinstance(source, DialogueConditionalObservation) or not isinstance(target, DialogueConditionalObservation):
        raise TypeError("Paired initialization requires conditional decoders")
    if type(include_attention) is not bool:
        raise TypeError("include_attention must be boolean")
    names = COMMON_TENSORS+(ATTENTION_TENSORS if include_attention else ())
    src, dst = dict(source.named_parameters()), dict(target.named_parameters())
    for name in names:
        if (name not in src or name not in dst or src[name].shape != dst[name].shape
                or src[name].dtype != dst[name].dtype or src[name].device != dst[name].device):
            raise ValueError(f"Incompatible initialization tensor: {name}")
    with torch.no_grad():
        for name in names:
            dst[name].copy_(src[name])
    return list(names)
