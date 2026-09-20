"""Observation-only extension of the normalized candidate-copy implementation.

The original [B,T,D] observation path delegates to V2 unchanged. The new
[B,T,Q,C,D] path supplies one public evidence embedding per query/candidate;
heads, parameter construction, lexical features and scalar transition stay the
same. This module neither constructs embeddings nor verifies their provenance.

The existing read-only monitor can be composed without replacing its checks::

    class MonitoredJoint(MonitoredCopyMemoryV2, DialogueJointMemory):
        pass

That MRO routes forward/_advance through the monitor, then this adapter, then
V2. Call the inherited begin_batch with the same actor tuple and public layouts.
The actual feature tensor still ends with own belief and entropy, and scalar
diagnostics retain V2's departure_mass. Do not reverse those base classes:
the candidate-specific _advance must remain inside the monitor's wrapper.
"""
from __future__ import annotations

import torch

from openjev.research.dialogue_copy_memory import transition
from openjev.research.dialogue_copy_memory_v2 import VERSION as PARENT_VERSION
from openjev.research.dialogue_copy_memory_v2 import DialogueCopyMemoryV2, normalize_log_b

VERSION = "dialogue-joint-memory-v1"
METHODS = ("readout", "scalar")


class DialogueJointMemory(DialogueCopyMemoryV2):
    """Same parameter names, counts and initialization draws as corresponding V2.

    Float32/float64 inputs must match parameter dtype/device. All public input
    entries must be finite, including padding. Padding is sanitized for compute
    and leaves recurrent state unchanged. No tensors are retained externally.
    """

    def __init__(self, method: str, input_dim: int = 384, projection_dim: int = 64,
                 hidden_dim: int = 64, gru_width: int = 16):
        if method not in METHODS:
            raise ValueError("Joint observation control supports readout or scalar only")
        super().__init__(method, input_dim, projection_dim, hidden_dim, gru_width)

    def configuration(self):
        result = super().configuration()
        result.update({
            "implementation_version": VERSION,
            "normalized_parent_version": PARENT_VERSION,
            "observation_paths": {"independent": "[B,T,D]; exact V2 super path",
                                  "joint": "[B,T,Q,C,D]; candidate-specific turn projection"},
            "observation_mode_binding": "Caller must bind independent/joint mode and embedding provenance",
            "supported_methods": list(METHODS),
            "representation_control": "Same heads/parameters/normalization; only supplied turn embedding indexing differs",
        })
        return result

    def _advance(self, state, turn, valid, query, candidates, mask, lexical):
        if turn.ndim == 2:
            return super()._advance(state, turn, valid, query, candidates, mask, lexical)
        # V2 normalizes before the actual own-belief/entropy features and again
        # after the proposal. Keep this same sequence for the joint path.
        log_b = normalize_log_b(state["log_b"])
        b, q, c = mask.shape
        clean = torch.where(valid[:, None, None, None] & mask[..., None], turn, 0.)
        u = self.turn_projection(clean).tanh()
        query = query[:, :, None].expand_as(candidates)
        lex = torch.where(mask[..., None] & valid[:, None, None, None], lexical, 0.)
        if self.method == "readout":
            log_b = torch.full_like(log_b, -torch.inf).scatter(-1, state["none_index"][..., None], 0.)
        belief = log_b.exp()
        safe_log = log_b.masked_fill(torch.isneginf(log_b), 0.)
        entropy = -(belief * safe_log).sum(-1, keepdim=True)
        entropy = entropy / mask.sum(-1, keepdim=True).to(belief.dtype).clamp_min(2).log()
        features = torch.cat((u, query, candidates, u * query, u * candidates, query * candidates,
                              lex, belief[..., None], entropy[..., None].expand(b, q, c, 1)), -1)
        write, departure = self.head(self.feature(features)).unbind(-1)
        diagnostic = {}
        if self.method == "readout":
            log_next = write.masked_fill(~mask, -torch.inf).log_softmax(-1)
        else:
            log_next, log_w, r, mass = transition(log_b, write, departure, mask, selective=False)
            diagnostic = {"write_log_probs": log_w, "departure_probabilities": r.masked_fill(~mask, 0.),
                          "departure_mass": mass, "valid": valid.clone()}
        log_next = torch.where(valid[:, None, None], log_next, state["log_b"])
        if torch.isnan(log_next).any() or torch.isposinf(log_next).any():
            raise ValueError("Nonfinite candidate prediction")
        log_next = normalize_log_b(log_next)
        proposed = {"none_index": state["none_index"].clone(),
                    "log_b": torch.where(valid[:, None, None], log_next, state["log_b"])}
        return proposed, diagnostic

    def step(self, state, turn, valid, query, candidates, candidate_mask, lexical):
        """Accept independent [B,D] or joint [B,Q,C,D] evidence for one step."""
        if isinstance(turn, torch.Tensor) and turn.ndim == 2:
            return super().step(state, turn, valid, query, candidates, candidate_mask, lexical)
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._float(turn, (b, q, c, self.input_dim), "joint turn")
        self._mask(valid, (b,), "turn validity")
        self._float(lexical, (b, q, c, 10), "lexical")
        self._state(state, candidate_mask)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        return self._advance(state, turn, valid, qp, cp, candidate_mask, lexical)

    def forward(self, turns, valid, query, candidates, candidate_mask, lexical, none_index=None):
        """Return [B,T,Q,C] log probabilities, with V2's unchanged argument order."""
        if isinstance(turns, torch.Tensor) and turns.ndim == 3:
            return super().forward(turns, valid, query, candidates, candidate_mask, lexical, none_index)
        b, q, c = self._schema(query, candidates, candidate_mask)
        if not isinstance(turns, torch.Tensor) or turns.ndim != 5 or turns.shape[1] < 1:
            raise ValueError("turns must be nonempty independent [B,T,D] or joint [B,T,Q,C,D]")
        t = turns.shape[1]
        self._float(turns, (b, t, q, c, self.input_dim), "joint turn sequence")
        self._mask(valid, (b, t), "sequence validity")
        self._float(lexical, (b, t, q, c, 10), "lexical sequence")
        state = self.initial(candidate_mask, none_index)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        rows = []
        for index in range(t):
            state, _ = self._advance(state, turns[:, index], valid[:, index], qp, cp,
                                     candidate_mask, lexical[:, index])
            rows.append(state["log_b"])
        return torch.stack(rows, dim=1)
