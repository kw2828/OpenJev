"""Candidate-indexed, query-conditioned memories with predicted-state feedback.

The selective transition is T_ij=(1-r_i)*1[i=j]+r_i*w_j. Its matched scalar
control replaces every r_i by m=sum(b_i*r_i) only in the retained-state term.
This is a discriminative stochastic transition, not exact Bayesian inference
or a claim of a new copying algorithm. There are no trainable candidate IDs.

Lexical inputs are supplied public-text features, in this exact order:
user_match, system_match, unique_longest_user, unique_longest_system,
literal_current, literal_previous, is_none, is_dontcare,
affirmative_cue_for_true, negative_cue_for_false. The last two are noisy cues,
not annotated polarity. All arms receive the same public inputs. No-lexical
zeros the first six and final two features; it preserves reserved-state flags.

At request time the caller may replay the public prefix for a supplied schema.
Grouping independent requested-query streams in a batch is permitted, including
queries first requested later. Query-set membership or annotation eligibility
must not enter an update mask, feature, or another query's computation. All
streams update on each public valid turn. No state or data is retained externally.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("readout", "scalar", "selective", "selective_no_lexical", "candidate_gru")
LEXICAL_FIELDS = ("user_match", "system_match", "unique_longest_user", "unique_longest_system",
                  "literal_current", "literal_previous", "is_none", "is_dontcare",
                  "affirmative_cue_for_true", "negative_cue_for_false")
State = dict[str, torch.Tensor]


def transition(log_b, write_logits, departure_logits, mask, *, selective):
    """Stable matched transitions; caller validates shapes and finite logits.

    Masked candidates never contribute to sums. Sanitizing the two invalid
    logaddexp operands prevents NaN derivatives from logaddexp(-inf,-inf).
    """
    log_w = write_logits.masked_fill(~mask, -torch.inf).log_softmax(-1)
    log_r, log_keep = F.logsigmoid(departure_logits), F.logsigmoid(-departure_logits)
    log_m = torch.logsumexp(log_b + log_r, -1, keepdim=True)
    retained = log_b + (log_keep if selective else
                        torch.logsumexp(log_b + log_keep, -1, keepdim=True))
    written = log_m + log_w
    result = torch.logaddexp(retained.masked_fill(~mask, 0), written.masked_fill(~mask, 0))
    return result.masked_fill(~mask, -torch.inf), log_w, log_r.exp(), log_m.exp()


class DialogueCopyMemory(nn.Module):
    """Shared candidate scorers and explicit categorical or candidate-GRU state.

    The three transport arms have identical parameter names/shapes. Copy their
    state_dict explicitly for paired initialization. Departure starts uniformly
    at sigmoid(-3), so scalar/selective are initially algebraically equivalent.
    The readout arm uses initial NONE, not its previous predictions, as its
    belief feature; its only historical inputs are the public lexical features.
    """

    def __init__(self, method: str, input_dim: int = 384, projection_dim: int = 64,
                 hidden_dim: int = 64, gru_width: int = 16):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown copy-memory method")
        if any(type(n) is not int or n < 1 for n in (input_dim, projection_dim, hidden_dim, gru_width)):
            raise ValueError("Dimensions must be positive integers")
        self.method, self.input_dim = method, input_dim
        self.projection_dim, self.hidden_dim, self.gru_width = projection_dim, hidden_dim, gru_width
        self.turn_projection = nn.Linear(input_dim, projection_dim)
        self.query_projection = nn.Linear(input_dim, projection_dim)
        self.candidate_projection = nn.Linear(input_dim, projection_dim)
        self.feature = nn.Sequential(nn.Linear(6 * projection_dim + 12, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, 2)
        with torch.no_grad():
            self.head.weight[1].zero_()
            self.head.bias[1].fill_(-3.)
        if method == "candidate_gru":
            self.gru = nn.GRUCell(hidden_dim, gru_width)
            self.gru_head = nn.Linear(gru_width, 1)

    def configuration(self):
        return {"class": type(self).__name__, "method": self.method, "input_dim": self.input_dim,
                "projection_dim": self.projection_dim, "hidden_dim": self.hidden_dim,
                "gru_width": self.gru_width, "lexical_fields": list(LEXICAL_FIELDS),
                "feature_dim": 6 * self.projection_dim + 12,
                "departure_initial_logit": -3., "initial_belief": "one-hot supplied NONE index",
                "belief_feature": "own probability and entropy divided by log(valid candidate count)",
                "updates": {"scalar": "(1-m)*b+m*w", "selective": "(1-r)*b+m*w",
                            "m": "sum(b*r)", "readout": "softmax(write logits); no learned state feedback",
                            "candidate_gru": "shared per-candidate GRU then candidate softmax"},
                "no_lexical_keeps": [6, 7], "query_conditioned_updates": True,
                "gold_feedback": False, "parameters": sum(p.numel() for p in self.parameters()),
                "scope": "conditional stochastic transition; not calibrated Bayes or architecture novelty"}

    def _float(self, value, shape, name, *, finite=True):
        ref = self.turn_projection.weight
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype not in (torch.float32, torch.float64) or value.dtype != ref.dtype
                or value.device != ref.device or (finite and not torch.isfinite(value).all())):
            raise ValueError(f"Invalid {name}: shape, dtype, device or finite values")

    def _mask(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != torch.bool or value.device != self.turn_projection.weight.device):
            raise ValueError(f"Invalid {name}")

    def _schema(self, query, candidates, mask):
        if not isinstance(query, torch.Tensor) or query.ndim != 3:
            raise ValueError("query must be [B,Q,D]")
        b, q, _ = query.shape
        if not isinstance(candidates, torch.Tensor) or candidates.ndim != 4:
            raise ValueError("candidates must be [B,Q,C,D]")
        c = candidates.shape[2]
        if min(b, q, c) < 1:
            raise ValueError("Empty batch, query or candidate axis")
        self._float(query, (b, q, self.input_dim), "query")
        self._float(candidates, (b, q, c, self.input_dim), "candidates")
        self._mask(mask, (b, q, c), "candidate mask")
        if not mask.any(-1).all():
            raise ValueError("Each query requires a candidate")
        return b, q, c

    def initial(self, candidate_mask, none_index=None) -> State:
        if (not isinstance(candidate_mask, torch.Tensor) or candidate_mask.ndim != 3
                or min(candidate_mask.shape) < 1):
            raise ValueError("candidate_mask must be nonempty [B,Q,C]")
        self._mask(candidate_mask, candidate_mask.shape, "candidate mask")
        b, q, c = candidate_mask.shape
        ref = self.turn_projection.weight
        if none_index is None:
            none_index = torch.zeros((b, q), dtype=torch.int64, device=ref.device)
        if (not isinstance(none_index, torch.Tensor) or none_index.shape != (b, q)
                or none_index.dtype != torch.int64 or none_index.device != ref.device
                or (none_index < 0).any() or (none_index >= c).any()
                or not candidate_mask.gather(-1, none_index[..., None]).all()):
            raise ValueError("NONE must identify a valid candidate")
        state = {"log_b": ref.new_full((b, q, c), -torch.inf).scatter(-1, none_index[..., None], 0.),
                 "none_index": none_index.clone()}
        if self.method == "candidate_gru":
            state["hidden"] = ref.new_zeros(b, q, c, self.gru_width)
        return state

    def _state(self, state, mask):
        expected = {"log_b", "none_index"} | ({"hidden"} if self.method == "candidate_gru" else set())
        if type(state) is not dict or set(state) != expected:
            raise ValueError("Wrong state keys")
        b, q, c = mask.shape
        self._float(state["log_b"], (b, q, c), "log belief", finite=False)
        log_b = state["log_b"]
        if (torch.isnan(log_b).any() or torch.isposinf(log_b).any()
                or not torch.isneginf(log_b[~mask]).all()
                or not torch.allclose(log_b.logsumexp(-1), torch.zeros_like(log_b[..., 0]), atol=1e-5, rtol=0)):
            raise ValueError("Belief must be normalized and masked")
        index = state["none_index"]
        if (not isinstance(index, torch.Tensor) or index.shape != (b, q) or index.dtype != torch.int64
                or index.device != mask.device or (index < 0).any() or (index >= c).any()
                or not mask.gather(-1, index[..., None]).all()):
            raise ValueError("Invalid state NONE index")
        if "hidden" in state:
            self._float(state["hidden"], (b, q, c, self.gru_width), "candidate hidden state")

    def _project_schema(self, query, candidates, mask):
        return (self.query_projection(query).tanh(),
                self.candidate_projection(torch.where(mask[..., None], candidates, 0.)).tanh())

    def _advance(self, state, turn, valid, query, candidates, mask, lexical):
        b, q, c = mask.shape
        clean = torch.where(valid[:, None], turn, 0.)
        u = self.turn_projection(clean).tanh()[:, None, None].expand(b, q, c, -1)
        query = query[:, :, None].expand_as(candidates)
        lex = torch.where(mask[..., None] & valid[:, None, None, None], lexical, 0.)
        if self.method == "selective_no_lexical":
            keep = torch.zeros(10, dtype=lex.dtype, device=lex.device)
            keep[6:8] = 1
            lex = lex * keep
        log_b = state["log_b"]
        if self.method == "readout":
            log_b = torch.full_like(log_b, -torch.inf).scatter(-1, state["none_index"][..., None], 0.)
        belief = log_b.exp()
        # 0*log(0) is defined as zero without an undefined gradient path.
        safe_log = log_b.masked_fill(torch.isneginf(log_b), 0.)
        entropy = -(belief * safe_log).sum(-1, keepdim=True)
        entropy = entropy / mask.sum(-1, keepdim=True).to(belief.dtype).clamp_min(2).log()
        features = torch.cat((u, query, candidates, u * query, u * candidates, query * candidates,
                              lex, belief[..., None], entropy[..., None].expand(b, q, c, 1)), -1)
        hidden_features = self.feature(features)
        write, departure = self.head(hidden_features).unbind(-1)
        proposed = {"none_index": state["none_index"].clone()}
        diagnostics = {}
        if self.method == "candidate_gru":
            hidden = self.gru(hidden_features.reshape(b * q * c, -1),
                              state["hidden"].reshape(b * q * c, -1)).reshape(b, q, c, -1)
            log_next = self.gru_head(hidden).squeeze(-1).masked_fill(~mask, -torch.inf).log_softmax(-1)
            proposed["hidden"] = torch.where((valid[:, None, None] & mask)[..., None], hidden, state["hidden"])
        elif self.method == "readout":
            log_next = write.masked_fill(~mask, -torch.inf).log_softmax(-1)
        else:
            log_next, log_w, r, mass = transition(log_b, write, departure, mask,
                                                selective=self.method != "scalar")
            diagnostics = {"write_log_probs": log_w, "departure_probabilities": r.masked_fill(~mask, 0.),
                           "departure_mass": mass, "valid": valid.clone()}
        proposed["log_b"] = torch.where(valid[:, None, None], log_next, state["log_b"])
        if torch.isnan(proposed["log_b"]).any() or torch.isposinf(proposed["log_b"]).any():
            raise ValueError("Nonfinite candidate prediction")
        return proposed, diagnostics

    def step(self, state, turn, valid, query, candidates, candidate_mask, lexical):
        """One functional update for supplied queries; returns state, diagnostics."""
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._float(turn, (b, self.input_dim), "turn")
        self._mask(valid, (b,), "turn validity")
        self._float(lexical, (b, q, c, 10), "lexical")
        self._state(state, candidate_mask)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        return self._advance(state, turn, valid, qp, cp, candidate_mask, lexical)

    def forward(self, turns, valid, query, candidates, candidate_mask, lexical, none_index=None):
        """Return [B,T,Q,C] log probabilities after every real or padded turn."""
        b, q, c = self._schema(query, candidates, candidate_mask)
        if not isinstance(turns, torch.Tensor) or turns.ndim != 3 or turns.shape[1] < 1:
            raise ValueError("turns must be nonempty [B,T,D]")
        t = turns.shape[1]
        self._float(turns, (b, t, self.input_dim), "turn sequence")
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
