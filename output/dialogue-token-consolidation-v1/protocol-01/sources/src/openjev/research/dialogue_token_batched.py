"""Time-batched token pooling with the frozen sequential normalized head.

Only independent observation pooling is batched. Softmax still runs over tokens
within each turn; recurrent transitions and their monitor hooks remain ordered.
No parameters, initialization draws, state fields, or one-step APIs are added.
This implementation makes no measured speed or memory claim.

The existing monitor composes as follows::

    class MonitoredTokenBatched(MonitoredCopyMemoryV2, DialogueTokenBatchedMemory):
        pass

Call begin_batch before forward as for the original token model. All original
incoming, feature, outgoing and released-mass checks wrap each _advance call.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_memory import ATTENTION_DIM, POOL_EPS, DialogueTokenMemory
from openjev.research.dialogue_token_memory import VERSION as PARENT_VERSION

VERSION = "dialogue-token-batched-v1"
PARENT_SOURCE = "src/openjev/research/dialogue_token_memory.py"
PARENT_SHA256 = "07a9b5d11ba4569417ede208bf1b6498cfac3488db8c8daec1062053a8bdc5eb"


class DialogueTokenBatchedMemory(DialogueTokenMemory):
    """Same constructor/state_dict and inherited step/pool as DialogueTokenMemory.

    Forward materializes [B,T,Q,C,L] scores and [B,T,Q,C,D] evidence. Raw tokens
    remain [B,T,L,D], never repeated over schema axes. Matrix batching can change
    floating-point accumulation; exact training-trajectory identity is not claimed.
    """

    def configuration(self):
        result = super().configuration()
        result.update({
            "implementation_version": VERSION,
            "pooling_parent_version": PARENT_VERSION,
            "pooling_parent_source": {"path": PARENT_SOURCE, "sha256": PARENT_SHA256},
            "pooling_execution": "All T independent observations batched before unchanged sequential _advance",
            "attention_work": "Forward: one schema projection and one B*T*L token-key projection; B*T*Q*C*L scores unchanged; T head updates; inherited step unchanged",
        })
        return result

    def _pool_sequence(self, tokens, valid, candidate_mask, token_mask, token_prior, schema_query):
        active = valid[..., None] & token_mask
        clean = torch.where(active[..., None], tokens, 0.)
        keys = self.token_key(clean)
        logits = torch.einsum("bqck,btlk->btqcl", schema_query, keys) / math.sqrt(ATTENTION_DIM)
        prior_log = token_prior.masked_fill(~active, 1.).log()
        logits = logits + prior_log[:, :, None, None]
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite token attention logits")
        logits = logits.masked_fill(~active[:, :, None, None], -torch.inf)
        logits = torch.where(valid[:, :, None, None, None], logits, 0.)
        weights = logits.softmax(-1)
        evidence = torch.einsum("btqcl,btld->btqcd", weights, clean)
        evidence = F.normalize(evidence, dim=-1, eps=POOL_EPS)
        if not torch.isfinite(evidence).all():
            raise ValueError("Nonfinite pooled evidence")
        return torch.where(candidate_mask[:, None, ..., None] & valid[:, :, None, None, None], evidence, 0.)

    def forward(self, tokens, valid, query, candidates, candidate_mask, lexical, token_mask, token_prior, none_index=None):
        """Original [B,T,Q,C] result, validation and head order; no state retained."""
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._check_tokens(tokens, valid, token_mask, token_prior, b, sequence=True)
        t = tokens.shape[1]
        self._float(lexical, (b, t, q, c, 10), "lexical sequence")
        state = self.initial(candidate_mask, none_index)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        schema_query = self._attention_query(query, candidates, candidate_mask)
        evidence = self._pool_sequence(tokens, valid, candidate_mask, token_mask, token_prior, schema_query)
        rows = []
        for i in range(t):
            state, _ = self._advance(state, evidence[:, i], valid[:, i], qp, cp, candidate_mask, lexical[:, i])
            rows.append(state["log_b"])
        return torch.stack(rows, 1)
