"""Identity-initialized attention placement on a trained scalar copy memory.

The backbone is supplied frozen. This is head continuation, not a resumed
optimizer or a new world model. The original shared turn projection and B=1
query layout are retained so zero attention reproduces the original arithmetic.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from openjev.research.dialogue_belief_pooling import DialogueBeliefPooling
from openjev.research.dialogue_copy_memory import transition
from openjev.research.dialogue_copy_memory_v2 import normalize_log_b

VERSION = "dialogue-warm-pooling-v1"


class DialogueWarmPooling(DialogueBeliefPooling):
    """Apply an identity-preserving directional residual before scalar updates.

    The residual is normalize(pooled + correction) - normalize(pooled). Adding
    it to the stored pooled vector preserves that vector exactly at zero,
    including its float32 norm rounding. Away from zero the result need not
    have exactly unit norm. All four arms retain the original belief features.
    """

    def configuration(self):
        result = super().configuration()
        result.update({"version": VERSION, "initialization": "trained scalar weights plus zero output adapter",
            "observation": "pooled + normalize(pooled+correction) - normalize(pooled)",
            "projection": "original shared turn projection plus bias-free projected observation delta",
            "scalar_layout": "one dialogue batch; all supplied queries",
            "scope": "attention placement during head continuation with a frozen trained encoder"})
        return result

    def observe(self, tokens, pooled, query, candidates, candidate_mask, log_b):
        if self.method == "pooled":
            return pooled[None].expand(query.shape[0], -1), None
        normalized, weights = super().observe(tokens, pooled, query, candidates, candidate_mask, log_b)
        baseline = F.normalize(pooled[None].expand(query.shape[0], -1), dim=-1)
        return pooled[None] + (normalized - baseline), weights

    def _advance_observed(self, state, pooled, observed, qp, cp, mask, lexical):
        """Original scalar transition with only the observation input changed."""
        b, q, c = mask.shape
        # Compute the shared term with the exact old [1,D] GEMM shape. Moving
        # queries into the batch or renormalizing the saved vector breaks this
        # identity even when the attention output matrix is exactly zero.
        base = self.memory.turn_projection(pooled[None])
        delta = F.linear(observed - pooled[None], self.memory.turn_projection.weight)
        u = (base + delta).tanh()[None, :, None].expand(b, q, c, -1)
        query = qp[:, :, None].expand_as(cp)
        lex = torch.where(mask[..., None], lexical, 0.)
        log_b = normalize_log_b(state["log_b"])
        belief = log_b.exp()
        safe_log = log_b.masked_fill(torch.isneginf(log_b), 0.)
        entropy = -(belief * safe_log).sum(-1, keepdim=True)
        entropy = entropy / mask.sum(-1, keepdim=True).to(belief.dtype).clamp_min(2).log()
        features = torch.cat((u, query, cp, u * query, u * cp, query * cp,
            lex, belief[..., None], entropy[..., None].expand(b, q, c, 1)), -1)
        write, departure = self.memory.head(self.memory.feature(features)).unbind(-1)
        log_next, _, _, _ = transition(log_b, write, departure, mask, selective=False)
        return {"none_index": state["none_index"].clone(), "log_b": normalize_log_b(log_next)}

    def forward(self, token_states, pooled_turns, query, candidates, candidate_mask, lexical, none_index):
        self.last_audit = {"forward_attempts": 1, "forward_returns": 0, "observation_attempts": 0,
            "observation_returns": 0, "step_attempts": 0, "step_returns": 0,
            "state_checks": 0, "real_question_updates": 0, "attention_positions": 0}
        self._validate(token_states, pooled_turns, query, candidates, candidate_mask, lexical, none_index)
        mask = candidate_mask[None]
        state = self.memory.initial(mask, none_index[None])
        qp, cp = self.memory._project_schema(query[None], candidates[None], mask)
        rows = []
        for time, tokens in enumerate(token_states):
            self.last_audit["observation_attempts"] += 1
            observed, _ = self.observe(tokens, pooled_turns[time], query, candidates,
                                       candidate_mask, state["log_b"][0])
            self.last_audit["observation_returns"] += 1
            self.last_audit["attention_positions"] += (query.shape[0] * (len(tokens) + 1)
                                                       if self.method != "pooled" else 0)
            self.last_audit["step_attempts"] += 1
            state = self._advance_observed(state, pooled_turns[time], observed, qp, cp, mask, lexical[time][None])
            self.last_audit["step_returns"] += 1
            self.memory._state(state, mask)
            self.last_audit["state_checks"] += 1
            self.last_audit["real_question_updates"] += query.shape[0]
            rows.append(state["log_b"])
        self.last_audit["forward_returns"] += 1
        return torch.stack(rows, dim=1)
