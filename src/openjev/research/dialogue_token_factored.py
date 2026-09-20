"""Split the existing head's exogenous and live recurrent-state linear terms.

Group-local token pooling/normalization and turn projection are unchanged. The
first feature layer's observation/schema/lexical term is computed before the
ordered recurrence, then combined with actual belief/entropy using its last two
weight columns. Its existing tanh runs only after addition. No new parameters.

The unchanged monitor remains outside every actual head update::

    class MonitoredFactored(MonitoredCopyMemoryV2, DialogueTokenFactoredMemory):
        pass

The feature pre-hook sees the actual two state inputs consumed by the split
operation. A private marker preserves the ordinary inherited step/pool path.
No actor tensors, activations or outcomes are cached between calls. Logical work
diagnostics are integers, not timing or physical memory-traffic measurements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.dialogue_copy_memory import transition
from openjev.research.dialogue_copy_memory_v2 import normalize_log_b
from openjev.research.dialogue_token_memory import ATTENTION_DIM, POOL_EPS, DialogueTokenMemory
from openjev.research.dialogue_token_memory import VERSION as PARENT_VERSION
from openjev.research.dialogue_token_packed import BUCKET_WIDTH, _packing_layout
from openjev.research.dialogue_token_projected import _projected_work

VERSION = "dialogue-token-factored-v1"
PARENT_SOURCE = "src/openjev/research/dialogue_token_memory.py"
PARENT_SHA256 = "07a9b5d11ba4569417ede208bf1b6498cfac3488db8c8daec1062053a8bdc5eb"
LAYOUT_SOURCE = "src/openjev/research/dialogue_token_packed.py"
LAYOUT_SHA256 = "1297e68c89b98cbb65fbc7fc5a871980b61cb6e13471aa1b5b52df7a41122aa0"
HEAD_SOURCE = "src/openjev/research/dialogue_joint_memory.py"
HEAD_SHA256 = "30a3e9180916537b261dfb7b0450730ff97542bf07b259fe4a88cd5d2ba0ba5e"
COUNTER_SOURCE = "src/openjev/research/dialogue_token_projected.py"
COUNTER_SHA256 = "25de6cb1656b4cce3904f1a708c5bc32a65f765cb0d37861f33bed5543fcf895"
REFERENCE_ONLY_FIELDS = [
    "scatter_evidence_scalars", "dense_evidence_scalars", "dense_turn_projection_positions",
    "projected_scatter_scalars", "dense_projected_scalars", "projected_bias_fill_positions",
    "projected_skipped_positions", "projected_scatter_bytes_float32", "dense_projected_bytes_float32",
    "dense_feature_macs",
]


@dataclass(frozen=True)
class _ExogenousTurn:
    """Internal dispatch marker, not a claim of tensor immutability."""

    value: torch.Tensor


class _FactoredFeature(nn.Sequential):
    """Same Linear/Tanh objects and state_dict names, with an explicit split call."""

    def forward(self, features, exogenous=None):
        if exogenous is None:
            return super().forward(features)
        # The monitor's pre-hook reads features[..., -2], the live belief used
        # by this exact linear operation, rather than a substitute audit tensor.
        return self[1](exogenous + F.linear(features, self[0].weight[:, -2:]))


def _factored_work(base_work, candidate_mask, time_steps, projection_width, hidden_width):
    if type(hidden_width) is not int or hidden_width < 1:
        raise ValueError("Hidden width must be a positive integer")
    work = _projected_work(base_work, projection_width)
    s, n, g = (work[k] for k in ("packed_evidence_positions", "dense_evidence_positions", "pooling_groups"))
    k, e, h = candidate_mask.numel(), 6*projection_width+10, hidden_width
    return {**work, "exogenous_width": e,
            "group_feature_positions": s, "fill_feature_positions": k, "state_feature_positions": n,
            "group_feature_calls": g, "fill_feature_calls": 1, "state_feature_calls": time_steps,
            "group_exogenous_input_scalars": s*e, "fill_exogenous_input_scalars": k*e,
            "state_feature_input_scalars": 2*n, "grouped_head_schema_scalars": 2*s*projection_width,
            "grouped_lexical_scalars": 10*s,
            "scatter_hidden_scalars": s*h, "dense_hidden_scalars": n*h, "fill_hidden_scalars": k*h,
            "group_feature_macs": s*e*h, "fill_feature_macs": k*e*h, "state_feature_macs": 2*n*h,
            "factored_feature_macs": (s+k)*e*h+2*n*h, "dense_feature_macs": n*(e+2)*h,
            "scatter_hidden_bytes_float32": 4*s*h, "dense_hidden_bytes_float32": 4*n*h}


class DialogueTokenFactoredMemory(DialogueTokenMemory):
    """Same constructor, initialization, state_dict, and inherited step/pool.

    Grouping and the split linear reduction change floating arithmetic, including
    possible extreme-input overflow behavior. No bitwise trajectory, speed or
    all-finite-domain equivalence claim. Failures may precede all head updates.
    """

    last_packing_work = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve the actual objects and initialization draws; the monitor's
        # constructor registers its feature hook after this constructor returns.
        self.feature = _FactoredFeature(*self.feature.children())

    def configuration(self):
        result = super().configuration()
        result.update({
            "implementation_version": VERSION,
            "pooling_parent_version": PARENT_VERSION,
            "pooling_parent_source": {"path": PARENT_SOURCE, "sha256": PARENT_SHA256},
            "packing_layout_source": {"path": LAYOUT_SOURCE, "sha256": LAYOUT_SHA256},
            "preprojected_head_source": {"path": HEAD_SOURCE, "sha256": HEAD_SHA256},
            "projected_counter_source": {"path": COUNTER_SOURCE, "sha256": COUNTER_SHA256},
            "packing_bucket_width": BUCKET_WIDTH,
            "packing": "Real turns/token support and candidate support including dummy NONE; public (bucket,pair-count) groups",
            "attention_work": "Valid-token key projection; grouped raw pooling/normalization and biased turn projection/tanh; exogenous first-layer term; hidden-width scatter; ordered live-state term",
            "skipped_evidence": "Full-schema exogenous fill from tanh(turn bias), projected query/candidate interactions and zero lexical; no cross-update cache",
            "packing_work_scope": "Integer logical positions/payloads, not memory traffic or timing; float32 byte fields are dtype-specific estimates",
            "packing_reference_only_fields": list(REFERENCE_ONLY_FIELDS),
            "factored_head": "Existing first weight columns+bias on exogenous features; final two columns without bias on actual belief/entropy; add then existing tanh/head",
            "feature_monitor": "Split call argument0 is actual two-feature state input; ordinary full-feature calls delegate unchanged",
        })
        return result

    @staticmethod
    def packing_work(valid, candidate_mask, token_mask, token_width=384, projection_width=64, hidden_width=64):
        """Mask-only work, with explicitly labeled prior dense/scatter references."""
        return _factored_work(_packing_layout(valid, candidate_mask, token_mask, token_width)[1],
                              candidate_mask, valid.shape[1], projection_width, hidden_width)

    def _exogenous(self, u, query, candidates, lexical):
        features = torch.cat((u, query, candidates, u*query, u*candidates, query*candidates, lexical), -1)
        return F.linear(features, self.feature[0].weight[:, :-2], self.feature[0].bias)

    def _pool_factored(self, tokens, valid, candidate_mask, token_mask, token_prior, schema_query, qp, cp, lexical):
        b, t, _, width = tokens.shape
        q, c = candidate_mask.shape[1:]
        groups, base_work = _packing_layout(valid, candidate_mask, token_mask, width)
        active = valid[..., None] & token_mask
        token_indices = active.reshape(-1).nonzero(as_tuple=False).flatten()
        raw = tokens.reshape(-1, width).index_select(0, token_indices)
        keys = self.token_key(raw)
        priors = token_prior.reshape(-1).index_select(0, token_indices)
        if not torch.isfinite(schema_query).all():
            raise ValueError("Nonfinite token attention logits")
        # The old skipped feature includes projected schema and interactions,
        # not just turn bias. Compute all schema fill rows once, without a hook.
        qfull = qp[:, :, None].expand_as(cp)
        u0 = self.turn_projection.bias.tanh().expand_as(cp)
        fill = self._exogenous(u0, qfull, cp, tokens.new_zeros((b, q, c, 10)))
        dense = fill[:, None].expand(b, t, q, c, self.hidden_dim).reshape(b*t*q*c, self.hidden_dim)
        if not groups:
            # Preserve zero (not absent) weight/input gradients on an empty batch.
            empty = self.turn_projection(raw).tanh()
            dense = (dense + empty.sum()*0 + keys.sum()*0 + priors.sum()*0
                     + (schema_query*0).sum() + (lexical*0).sum())
        else:
            lookup = torch.full(active.shape, -1, dtype=torch.long, device=tokens.device)
            lookup[active] = torch.arange(len(raw), device=tokens.device)
            pairs = [candidate_mask[bi].reshape(-1).nonzero(as_tuple=False).flatten() for bi in range(b)]
            values, destinations = [], []
            for (bucket, pair_count), turns in sorted(groups.items()):
                count = len(turns)
                indices = torch.zeros((count, bucket), dtype=torch.long, device=tokens.device)
                supported = torch.zeros((count, bucket), dtype=torch.bool, device=tokens.device)
                schema_indices, output_indices = [], []
                for i, (bi, ti) in enumerate(turns):
                    ids = lookup[bi, ti][token_mask[bi, ti]]
                    indices[i, :len(ids)] = ids
                    supported[i, :len(ids)] = True
                    schema_indices.append(pairs[bi]+bi*q*c)
                    output_indices.append(pairs[bi]+(bi*t+ti)*q*c)
                gather = indices.reshape(-1)
                group_raw = raw.index_select(0, gather).reshape(count, bucket, width)
                group_raw = torch.where(supported[..., None], group_raw, 0.)
                group_keys = keys.index_select(0, gather).reshape(count, bucket, ATTENTION_DIM)
                group_keys = torch.where(supported[..., None], group_keys, 0.)
                group_prior = priors.index_select(0, gather).reshape(count, bucket).masked_fill(~supported, 1.)
                schema_ids = torch.stack(schema_indices).reshape(-1)
                output_ids = torch.stack(output_indices).reshape(-1)
                schema = schema_query.reshape(-1, ATTENTION_DIM).index_select(0, schema_ids)
                schema = schema.reshape(count, pair_count, ATTENTION_DIM)
                logits = torch.bmm(schema, group_keys.transpose(1, 2))/math.sqrt(ATTENTION_DIM)
                logits = logits + group_prior.log()[:, None]
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite token attention logits")
                weights = logits.masked_fill(~supported[:, None], -torch.inf).softmax(-1)
                evidence = F.normalize(torch.bmm(weights, group_raw), dim=-1, eps=POOL_EPS)
                if not torch.isfinite(evidence).all():
                    raise ValueError("Nonfinite pooled evidence")
                projected = self.turn_projection(evidence).tanh()
                group_q = qp.reshape(-1, self.projection_dim).index_select(0, schema_ids//c).reshape_as(projected)
                group_c = cp.reshape(-1, self.projection_dim).index_select(0, schema_ids).reshape_as(projected)
                group_lex = lexical.reshape(-1, 10).index_select(0, output_ids).reshape(count, pair_count, 10)
                values.append(self._exogenous(projected, group_q, group_c, group_lex).reshape(-1, self.hidden_dim))
                destinations.append(output_ids)
            dense = dense.index_copy(0, torch.cat(destinations), torch.cat(values))
        if torch.isnan(dense).any():
            raise ValueError("NaN exogenous preactivation")
        self.last_packing_work = _factored_work(base_work, candidate_mask, t, self.projection_dim, self.hidden_dim)
        return dense.reshape(b, t, q, c, self.hidden_dim)

    def _advance(self, state, turn, valid, query, candidates, mask, lexical):
        if not isinstance(turn, _ExogenousTurn):
            return super()._advance(state, turn, valid, query, candidates, mask, lexical)
        # Same joint/V2 state and transition equations. The feature hook sees
        # the actual live state inputs, and the existing tanh follows addition.
        log_b = normalize_log_b(state["log_b"])
        b, q, c = mask.shape
        if self.method == "readout":
            log_b = torch.full_like(log_b, -torch.inf).scatter(-1, state["none_index"][..., None], 0.)
        belief = log_b.exp()
        safe_log = log_b.masked_fill(torch.isneginf(log_b), 0.)
        entropy = -(belief * safe_log).sum(-1, keepdim=True)
        entropy = entropy / mask.sum(-1, keepdim=True).to(belief.dtype).clamp_min(2).log()
        features = torch.cat((belief[..., None], entropy[..., None].expand(b, q, c, 1)), -1)
        write, departure = self.head(self.feature(features, turn.value)).unbind(-1)
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

    def forward(self, tokens, valid, query, candidates, candidate_mask, lexical, token_mask, token_prior, none_index=None):
        """Original validation and ordered monitored state updates; no persistent cache."""
        self.last_packing_work = None
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._check_tokens(tokens, valid, token_mask, token_prior, b, sequence=True)
        t = tokens.shape[1]
        self._float(lexical, (b, t, q, c, 10), "lexical sequence")
        state = self.initial(candidate_mask, none_index)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        schema_query = self._attention_query(query, candidates, candidate_mask)
        exogenous = self._pool_factored(tokens, valid, candidate_mask, token_mask, token_prior, schema_query, qp, cp, lexical)
        rows = []
        for i in range(t):
            state, _ = self._advance(state, _ExogenousTurn(exogenous[:, i]), valid[:, i], qp, cp,
                                     candidate_mask, lexical[:, i])
            rows.append(state["log_b"])
        return torch.stack(rows, 1)
