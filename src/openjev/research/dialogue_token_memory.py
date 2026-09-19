"""Shared-token evidence adapter for the unchanged normalized dialogue head.

The encoder is outside this module. Inputs are finite raw contextual token
states, frozen schema vectors, and an explicit chunk-pooling prior. The two
attention modes change only whether the second schema vector is the candidate
or another copy of the query. Attention weights are not semantic confidence.

The existing monitor remains outside every actual head update::

    class MonitoredToken(MonitoredCopyMemoryV2, DialogueTokenMemory):
        pass

Use begin_batch with the eight-argument actor tuple and public layouts. The
first six positions retain the old meaning, with tokens replacing turn vectors.
This adapter does not override _advance, so incoming, feature-prior, outgoing
and released-mass checks still wrap the inherited joint/V2 transition.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.dialogue_joint_memory import VERSION as JOINT_VERSION
from openjev.research.dialogue_joint_memory import DialogueJointMemory

VERSION = "dialogue-token-memory-v1"
ATTENTION_DIM = 64
PRIOR_TOLERANCE = 2e-6
POOL_EPS = 1e-12
ADAPTER_PARAMETERS = ("token_key.weight", "evidence_query.weight")


class DialogueTokenMemory(DialogueJointMemory):
    """Query-independent encoder cache; schema-conditioned token pooling.

    Float32/64 inputs match parameter dtype/device. Token tensors are never
    repeated along question/candidate axes. Logits [B,Q,C,L] and pooled evidence
    [B,Q,C,D] are materialized, so candidate-dependent work is still paid.
    Parent parameters are initialized first with exactly the original draws.
    """

    def __init__(self, method: str, *, attention_mode: str = "candidate", input_dim: int = 384,
                 projection_dim: int = 64, hidden_dim: int = 64, gru_width: int = 16):
        if attention_mode not in ("slot", "candidate"):
            raise ValueError("attention_mode must be slot or candidate")
        super().__init__(method, input_dim, projection_dim, hidden_dim, gru_width)
        self.attention_mode = attention_mode
        self.token_key = nn.Linear(input_dim, ATTENTION_DIM, bias=False)
        self.evidence_query = nn.Linear(2*input_dim, ATTENTION_DIM, bias=False)

    def configuration(self):
        result = super().configuration()
        adapter = sum(p.numel() for name, p in self.named_parameters() if name in ADAPTER_PARAMETERS)
        result.update({
            "implementation_version": VERSION, "normalized_head_parent_version": JOINT_VERSION,
            "attention_mode": self.attention_mode, "attention_width": ATTENTION_DIM,
            "adapter_parameter_names": list(ADAPTER_PARAMETERS), "adapter_parameters": adapter,
            "parent_parameters": result["parameters"]-adapter,
            "attention": "softmax(log(token_prior) + dot(Wq[schema_pair], Wk[token])/sqrt(64)) over tokens only",
            "schema_pair": "[query;candidate]" if self.attention_mode == "candidate" else "[query;query]",
            "pooling": "L2-normalize weighted raw token sum, eps=1e-12; zero vector remains zero",
            "token_prior": "Positive on token_mask, zero elsewhere; real-turn sum within 2e-6 of one",
            "observation_paths": {"tokens": "[B,T,L,D]; shared raw tokens, candidate-specific pooled evidence"},
            "observation_mode_binding": "Caller authenticates raw contextual tokens and chunk-weighted priors",
            "representation_control": "Same normalized head; two added bias-free projections; no encoder calls",
            "attention_work": "Forward: schema query B*Q*C once, token keys B*T*L, token logits B*T*Q*C*L; step repeats schema projection",
        })
        return result

    def _check_tokens(self, tokens, valid, token_mask, token_prior, batch, *, sequence):
        rank = 4 if sequence else 3
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != rank or min(tokens.shape[1:-1]) < 1:
            raise ValueError("tokens require nonempty [B,T,L,D] or one-step [B,L,D]")
        shape = (batch, *tokens.shape[1:-1])
        self._float(tokens, (*shape, self.input_dim), "token states")
        self._mask(valid, shape[:-1], "turn validity")
        self._mask(token_mask, shape, "token mask")
        self._float(token_prior, shape, "token prior")
        if ((token_prior < 0).any() or (token_prior[~token_mask] != 0).any()
                or (token_prior[token_mask] <= 0).any()):
            raise ValueError("Token priors must be positive on support and zero off support")
        if not token_mask.any(-1)[valid].all():
            raise ValueError("Real turns require nonempty token support")
        total = token_prior.to(torch.float64).sum(-1)[valid]
        if ((total-1).abs() > PRIOR_TOLERANCE).any():
            raise ValueError("Real-turn token prior must sum to one")

    def _attention_query(self, query, candidates, candidate_mask):
        q = query[:, :, None].expand_as(candidates)
        other = candidates if self.attention_mode == "candidate" else q
        pair = torch.cat((q, other), -1)
        return self.evidence_query(torch.where(candidate_mask[..., None], pair, 0.))

    def _pool(self, tokens, valid, candidate_mask, token_mask, token_prior, schema_query):
        active = valid[:, None] & token_mask
        clean = torch.where(active[..., None], tokens, 0.)
        keys = self.token_key(clean)
        logits = torch.einsum("bqck,blk->bqcl", schema_query, keys) / math.sqrt(ATTENTION_DIM)
        # Avoid log(0) even in masked graph branches; all-masked padded turns use
        # harmless uniform weights on zeroed tokens and yield exact zero evidence.
        prior_log = token_prior.masked_fill(~active, 1.).log()
        logits = logits + prior_log[:, None, None]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite token attention logits")
        logits = logits.masked_fill(~active[:, None, None], -torch.inf)
        logits = torch.where(valid[:, None, None, None], logits, 0.)
        weights = logits.softmax(-1)
        evidence = torch.einsum("bqcl,bld->bqcd", weights, clean)
        evidence = F.normalize(evidence, dim=-1, eps=POOL_EPS)
        if not torch.isfinite(evidence).all():
            raise ValueError("Nonfinite pooled evidence")
        return torch.where(candidate_mask[..., None] & valid[:, None, None, None], evidence, 0.)

    def pool(self, tokens, valid, query, candidates, candidate_mask, token_mask, token_prior):
        """One-step evidence [B,Q,C,D], with no memory update or retained tensors."""
        b, _, _ = self._schema(query, candidates, candidate_mask)
        self._check_tokens(tokens, valid, token_mask, token_prior, b, sequence=False)
        return self._pool(tokens, valid, candidate_mask, token_mask, token_prior,
                          self._attention_query(query, candidates, candidate_mask))

    def step(self, state, tokens, valid, query, candidates, candidate_mask, lexical, token_mask, token_prior):
        """One causal functional update, returning the unchanged state/diagnostic schema."""
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._check_tokens(tokens, valid, token_mask, token_prior, b, sequence=False)
        self._float(lexical, (b, q, c, 10), "lexical")
        self._state(state, candidate_mask)
        evidence = self._pool(tokens, valid, candidate_mask, token_mask, token_prior,
                              self._attention_query(query, candidates, candidate_mask))
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        return self._advance(state, evidence, valid, qp, cp, candidate_mask, lexical)

    def forward(self, tokens, valid, query, candidates, candidate_mask, lexical, token_mask, token_prior, none_index=None):
        """Return [B,T,Q,C] log probabilities; schema pooling queries are computed once."""
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._check_tokens(tokens, valid, token_mask, token_prior, b, sequence=True)
        t = tokens.shape[1]
        self._float(lexical, (b, t, q, c, 10), "lexical sequence")
        state = self.initial(candidate_mask, none_index)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        schema_query = self._attention_query(query, candidates, candidate_mask)
        rows = []
        for i in range(t):
            evidence = self._pool(tokens[:, i], valid[:, i], candidate_mask, token_mask[:, i], token_prior[:, i], schema_query)
            state, _ = self._advance(state, evidence, valid[:, i], qp, cp, candidate_mask, lexical[:, i])
            rows.append(state["log_b"])
        return torch.stack(rows, 1)
