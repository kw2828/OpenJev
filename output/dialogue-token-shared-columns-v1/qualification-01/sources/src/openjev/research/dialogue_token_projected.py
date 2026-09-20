"""Group-local evidence projection with the frozen ordered categorical head.

Attention still pools and L2-normalizes in the original token dimension. Only
then does the existing biased turn projection and tanh run, before scattering
projection-width vectors. Skipped cells contain tanh(bias), as in the original
head applied to zero evidence. This changes execution, not parameters or state.

The unchanged monitor remains outside every actual head update::

    class MonitoredProjected(MonitoredCopyMemoryV2, DialogueTokenProjectedMemory):
        pass

An internal marker distinguishes projected forward inputs from ordinary token
step inputs; inherited step/pool therefore retain their original APIs. No actor
tensors are cached between calls. Work diagnostics contain integer counts only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from openjev.research.dialogue_copy_memory import transition
from openjev.research.dialogue_copy_memory_v2 import normalize_log_b
from openjev.research.dialogue_token_memory import ATTENTION_DIM, POOL_EPS, DialogueTokenMemory
from openjev.research.dialogue_token_memory import VERSION as PARENT_VERSION
from openjev.research.dialogue_token_packed import BUCKET_WIDTH, _packing_layout

VERSION = "dialogue-token-projected-v1"
PARENT_SOURCE = "src/openjev/research/dialogue_token_memory.py"
PARENT_SHA256 = "07a9b5d11ba4569417ede208bf1b6498cfac3488db8c8daec1062053a8bdc5eb"
LAYOUT_SOURCE = "src/openjev/research/dialogue_token_packed.py"
LAYOUT_SHA256 = "1297e68c89b98cbb65fbc7fc5a871980b61cb6e13471aa1b5b52df7a41122aa0"
HEAD_SOURCE = "src/openjev/research/dialogue_joint_memory.py"
HEAD_SHA256 = "30a3e9180916537b261dfb7b0450730ff97542bf07b259fe4a88cd5d2ba0ba5e"


@dataclass(frozen=True)
class _ProjectedTurn:
    """Internal dispatch marker, not a claim of tensor immutability."""

    value: torch.Tensor


def _projected_work(work, projection_width):
    if type(projection_width) is not int or projection_width < 1:
        raise ValueError("Projection width must be a positive integer")
    packed, dense, groups = (work[k] for k in
                             ("packed_evidence_positions", "dense_evidence_positions", "pooling_groups"))
    return {**work,
            "packed_turn_projection_positions": packed,
            "dense_turn_projection_positions": dense,
            "turn_projection_calls": max(1, groups),
            "empty_turn_projection_calls": int(groups == 0),
            "projected_scatter_scalars": packed*projection_width,
            "dense_projected_scalars": dense*projection_width,
            "projected_bias_fill_positions": dense,
            "projected_skipped_positions": dense-packed,
            "projected_scatter_bytes_float32": 4*packed*projection_width,
            "dense_projected_bytes_float32": 4*dense*projection_width}


class DialogueTokenProjectedMemory(DialogueTokenMemory):
    """Same constructor, initialization, state_dict, and inherited step/pool.

    Floating reduction order changes with grouped matrices. Successful output
    and gradient parity requires numerical checks; no speed or trajectory claim
    follows from the counts. Pooling failures may precede all head updates.
    """

    last_packing_work = None

    def configuration(self):
        result = super().configuration()
        result.update({
            "implementation_version": VERSION,
            "pooling_parent_version": PARENT_VERSION,
            "pooling_parent_source": {"path": PARENT_SOURCE, "sha256": PARENT_SHA256},
            "packing_layout_source": {"path": LAYOUT_SOURCE, "sha256": LAYOUT_SHA256},
            "preprojected_head_source": {"path": HEAD_SOURCE, "sha256": HEAD_SHA256},
            "packing_bucket_width": BUCKET_WIDTH,
            "packing": "Real turns/token support and candidate support including dummy NONE; public (bucket,pair-count) groups",
            "attention_work": "Valid-token key projection; grouped attention and raw-token pooling; normalize before biased turn projection/tanh; projection-width scatter; ordered head",
            "skipped_evidence": "tanh(turn_projection.bias); zero-row projection preserves all-padding gradient membership",
            "packing_work_scope": "Integer logical positions/payloads, not memory traffic or timing; float32 byte fields are dtype-specific estimates",
            "packing_reference_only_fields": ["scatter_evidence_scalars", "dense_evidence_scalars",
                                               "dense_turn_projection_positions"],
            "projected_head": "Internal marker only; inherited step accepts raw evidence and delegates unchanged",
        })
        return result

    @staticmethod
    def packing_work(valid, candidate_mask, token_mask, token_width=384, projection_width=64):
        """Public-mask counts; original 384-wide scatter/dense fields are references."""
        return _projected_work(_packing_layout(valid, candidate_mask, token_mask, token_width)[1], projection_width)

    def _pool_projected(self, tokens, valid, candidate_mask, token_mask, token_prior, schema_query):
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
        # Zero evidence in the old head produces tanh(bias), including padding.
        dense = self.turn_projection.bias.tanh().expand(b*t*q*c, self.projection_dim)
        if not groups:
            # Preserve zero (not absent) weight/input gradients on an empty batch.
            empty = self.turn_projection(raw).tanh()
            dense = dense + empty.sum()*0 + keys.sum()*0 + priors.sum()*0 + (schema_query*0).sum()
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
                schema = schema_query.reshape(-1, ATTENTION_DIM).index_select(0, torch.stack(schema_indices).reshape(-1))
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
                values.append(projected.reshape(-1, self.projection_dim))
                destinations.append(torch.stack(output_indices).reshape(-1))
            dense = dense.index_copy(0, torch.cat(destinations), torch.cat(values))
        if not torch.isfinite(dense).all():
            raise ValueError("Nonfinite projected evidence")
        self.last_packing_work = _projected_work(base_work, self.projection_dim)
        return dense.reshape(b, t, q, c, self.projection_dim)

    def _advance(self, state, turn, valid, query, candidates, mask, lexical):
        if not isinstance(turn, _ProjectedTurn):
            return super()._advance(state, turn, valid, query, candidates, mask, lexical)
        # Same joint/V2 head equations, with its projection already performed.
        log_b = normalize_log_b(state["log_b"])
        b, q, c = mask.shape
        u = turn.value
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

    def forward(self, tokens, valid, query, candidates, candidate_mask, lexical, token_mask, token_prior, none_index=None):
        """Original actor validation and monitored head order; group-local projection."""
        self.last_packing_work = None
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._check_tokens(tokens, valid, token_mask, token_prior, b, sequence=True)
        t = tokens.shape[1]
        self._float(lexical, (b, t, q, c, 10), "lexical sequence")
        state = self.initial(candidate_mask, none_index)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        schema_query = self._attention_query(query, candidates, candidate_mask)
        projected = self._pool_projected(tokens, valid, candidate_mask, token_mask, token_prior, schema_query)
        rows = []
        for i in range(t):
            state, _ = self._advance(state, _ProjectedTurn(projected[:, i]), valid[:, i], qp, cp,
                                     candidate_mask, lexical[:, i])
            rows.append(state["log_b"])
        return torch.stack(rows, 1)
