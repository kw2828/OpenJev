"""Autonomous belief-conditioned pooling of supplied, frozen token features.

The three attention arms move the same differentiable candidate-belief summary
between a read query and one appended state token. They have identical tensor
shapes and parameters. This is a placement ablation, not a world model or a
lossless representation of the full categorical distribution. No assets load
on import; labels and annotation masks are not accepted by the actor.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.dialogue_copy_memory_v2 import DialogueCopyMemoryV2

METHODS = ("pooled", "schema_attention", "belief_query", "state_token")
VERSION = "dialogue-belief-pooling-v1"


class DialogueBeliefPooling(nn.Module):
    """Query-specific token reads followed by the unchanged scalar transition.

    Each query is an independent batch member of DialogueCopyMemoryV2. Its
    categorical state starts at the supplied NONE candidate and is never reset
    inside a public stream. All arms retain that head's belief/entropy inputs.
    The encoder features are inputs, so this module cannot change the backbone.
    """

    def __init__(self, method, input_dim=384, attention_dim=64,
                 projection_dim=64, hidden_dim=64):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown belief-pooling method")
        if any(type(n) is not int or n < 1 for n in
               (input_dim, attention_dim, projection_dim, hidden_dim)):
            raise ValueError("Positive integer dimensions required")
        self.method, self.input_dim, self.attention_dim = method, input_dim, attention_dim
        self.memory = DialogueCopyMemoryV2("scalar", input_dim=input_dim,
            projection_dim=projection_dim, hidden_dim=hidden_dim)
        self.read_query = nn.Linear(2 * input_dim, attention_dim, bias=False)
        self.read_key = nn.Linear(input_dim, attention_dim, bias=False)
        self.read_value = nn.Linear(input_dim, attention_dim, bias=False)
        self.read_output = nn.Linear(attention_dim, input_dim, bias=False)
        self.last_audit = {}
        # Paired fits start from exactly the pooled observation. Attention
        # parameters get learning signal after the output map's first update.
        nn.init.zeros_(self.read_output.weight)

    def configuration(self):
        total = sum(p.numel() for p in self.parameters())
        memory = sum(p.numel() for p in self.memory.parameters())
        return {"version": VERSION, "method": self.method,
            "input_dim": self.input_dim, "attention_dim": self.attention_dim,
            "parameters": total, "active_parameters": memory if self.method == "pooled" else total,
            "memory": self.memory.configuration(), "encoder_trainable": False,
            "summary": "candidate embedding expectation over all supported probabilities",
            "summary_is_lossless": False, "gold_feedback": False,
            "state_gradient": "through autonomous probabilities; no detach",
            "attention_output_initialization": "zero; initial observations equal pooled control"}

    def _float(self, value, shape, name):
        reference = self.read_query.weight
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != reference.dtype or value.device != reference.device
                or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid {name}: shape, dtype, device or finite values")

    def _validate(self, token_states, pooled_turns, query, candidates, candidate_mask, lexical, none_index):
        if not isinstance(query, torch.Tensor) or query.ndim != 2 or query.shape[0] < 1:
            raise ValueError("Nonempty [Q,D] schema queries required")
        if not isinstance(candidates, torch.Tensor) or candidates.ndim != 3 or candidates.shape[1] < 1:
            raise ValueError("Nonempty [Q,C,D] schema candidates required")
        q, c = query.shape[0], candidates.shape[1]
        if not isinstance(token_states, (list, tuple)) or not token_states:
            raise ValueError("One nonempty token matrix per public turn required")
        t = len(token_states)
        self._float(query, (q, self.input_dim), "query")
        self._float(candidates, (q, c, self.input_dim), "candidates")
        self._float(pooled_turns, (t, self.input_dim), "pooled turns")
        self._float(lexical, (t, q, c, 10), "lexical")
        if ((torch.linalg.vector_norm(pooled_turns, dim=-1) <= 1e-12).any()
                or not torch.allclose(torch.linalg.vector_norm(pooled_turns, dim=-1),
                                      pooled_turns.new_ones(t), atol=1e-5, rtol=0)):
            raise ValueError("Pooled turns must be unit vectors")
        if (not isinstance(candidate_mask, torch.Tensor) or candidate_mask.shape != (q, c)
                or candidate_mask.dtype != torch.bool or candidate_mask.device != query.device
                or not candidate_mask.any(-1).all()):
            raise ValueError("Nonempty supported candidate masks required")
        if not ((lexical == 0) | (lexical == 1)).all():
            raise ValueError("Binary public lexical features required")
        if (lexical.masked_select(~candidate_mask[None, ..., None]) != 0).any():
            raise ValueError("Lexical padding must be zero")
        for value in token_states:
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 1:
                raise ValueError("Nonempty [L,D] valid token features required")
            self._float(value, (value.shape[0], self.input_dim), "token features")
        if (not isinstance(none_index, torch.Tensor) or none_index.shape != (q,)
                or none_index.dtype != torch.int64 or none_index.device != query.device):
            raise ValueError("One integer NONE index per query required")
        # Reuse the scalar's exact supplied-NONE validation.
        self.memory.initial(candidate_mask[:, None], none_index[:, None])

    def observe(self, tokens, pooled, query, candidates, candidate_mask, log_b):
        """Return [Q,D] observations and [Q,L+1] weights (None for pooled).

        Inputs are validated by forward. Keeping this operation explicit makes
        the state-to-observation path inspectable without changing the head.
        Padding contributes to neither schema nor belief summaries.
        """
        q = query.shape[0]
        if self.method == "pooled":
            return F.normalize(pooled[None].expand(q, -1), dim=-1), None
        clean = candidates.masked_fill(~candidate_mask[..., None], 0.)
        schema = clean.sum(1) / candidate_mask.sum(1, keepdim=True)
        belief = (log_b.exp()[..., None] * clean).sum(1)
        conditioner = belief if self.method == "belief_query" else schema
        state_token = belief if self.method == "state_token" else schema
        # Every attention arm computes both summaries and the same Q/K/V work.
        rq = self.read_query(torch.cat((query, conditioner), -1))
        keys = torch.cat((self.read_key(tokens)[None].expand(q, -1, -1),
                          self.read_key(state_token)[:, None]), 1)
        values = torch.cat((self.read_value(tokens)[None].expand(q, -1, -1),
                            self.read_value(state_token)[:, None]), 1)
        weights = (torch.einsum("qd,qld->ql", rq, keys) / self.attention_dim**.5).softmax(-1)
        correction = self.read_output(torch.einsum("ql,qld->qd", weights, values))
        observation = pooled[None] + correction
        if (not torch.isfinite(observation).all()
                or (torch.linalg.vector_norm(observation, dim=-1) <= 1e-12).any()):
            raise ValueError("Invalid conditioned observation")
        return F.normalize(observation, dim=-1), weights

    def forward(self, token_states, pooled_turns, query, candidates, candidate_mask, lexical, none_index):
        self.last_audit = {"forward_attempts": 1, "forward_returns": 0, "observation_attempts": 0,
            "observation_returns": 0, "step_attempts": 0, "step_returns": 0,
            "state_checks": 0, "real_question_updates": 0, "attention_positions": 0}
        self._validate(token_states, pooled_turns, query, candidates, candidate_mask, lexical, none_index)
        q = query.shape[0]
        state = self.memory.initial(candidate_mask[:, None], none_index[:, None])
        valid = torch.ones(q, dtype=torch.bool, device=query.device)
        rows = []
        for time, tokens in enumerate(token_states):
            self.last_audit["observation_attempts"] += 1
            observed, _ = self.observe(tokens, pooled_turns[time], query, candidates,
                                       candidate_mask, state["log_b"][:, 0])
            self.last_audit["observation_returns"] += 1
            self.last_audit["attention_positions"] += (q * (len(tokens) + 1) if self.method != "pooled" else 0)
            self.last_audit["step_attempts"] += 1
            state, _ = self.memory.step(state, observed, valid, query[:, None], candidates[:, None],
                                       candidate_mask[:, None], lexical[time, :, None])
            self.last_audit["step_returns"] += 1
            self.memory._state(state, candidate_mask[:, None])
            self.last_audit["state_checks"] += 1
            self.last_audit["real_question_updates"] += q
            rows.append(state["log_b"][:, 0])
        self.last_audit["forward_returns"] += 1
        return torch.stack(rows)[None]
