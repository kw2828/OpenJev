"""Consolidate supported pooled rows before the existing two projections.

Public-mask attention groups and raw-token L2 normalization are unchanged.
Normalized supported rows are concatenated once, then the existing turn and
exogenous feature projections run once each. No raw tokens are repeated over
schema axes. This trades a supported token-width concatenation for fewer calls;
work counters do not establish runtime, traffic or peak-memory improvements.

The factored parent's parameter construction, forward validation, private head
marker, ordered normalized recurrence and inherited step/pool APIs are reused::

    class MonitoredConsolidated(MonitoredCopyMemoryV2, DialogueTokenConsolidatedMemory):
        pass

No model values or tensors are cached across calls or optimizer updates.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_factored import VERSION as PARENT_VERSION
from openjev.research.dialogue_token_factored import DialogueTokenFactoredMemory, _factored_work
from openjev.research.dialogue_token_memory import ATTENTION_DIM, POOL_EPS
from openjev.research.dialogue_token_packed import _packing_layout

VERSION = "dialogue-token-consolidated-v1"
FACTORED_SOURCE = "src/openjev/research/dialogue_token_factored.py"
FACTORED_SHA256 = "d817ae4646da4bc4cb4380cf64e678c927237f71f7a2ee27bfd198e5008b6f42"


def _consolidated_work(work, token_width):
    rows = work["packed_evidence_positions"]
    return {**work, "turn_projection_calls": 1, "empty_turn_projection_calls": int(rows == 0),
            "group_feature_calls": int(rows > 0), "pooled_concat_scalars": rows*token_width,
            "pooled_concat_bytes_float32": 4*rows*token_width}


class DialogueTokenConsolidatedMemory(DialogueTokenFactoredMemory):
    """Same constructor, tensors and mathematical head; grouped GEMM order differs.

    Numerical parity requires checks, including every input/parameter gradient.
    No bitwise trajectory or extreme-finite-input overflow equivalence claim.
    """

    def configuration(self):
        result = super().configuration()
        result.update({
            "implementation_version": VERSION,
            "factored_parent_version": PARENT_VERSION,
            "factored_parent_source": {"path": FACTORED_SOURCE, "sha256": FACTORED_SHA256},
            "attention_work": "Unchanged grouped attention/normalization; concatenate supported token-width rows; one turn projection and one supported exogenous projection; same schema fill and ordered head",
            "projection_call_counts": "turn_projection_calls includes one zero-row empty call; group_feature_calls is 1 iff supported rows exist, not the attention group count",
            "pooled_concatenation": "Supported rows only, ordered by sorted public groups then turn and candidate indices; schema/destination gathers use that same order",
            "pooling_group_count": "pooling_groups remains the actual attention-group count",
        })
        return result

    @staticmethod
    def packing_work(valid, candidate_mask, token_mask, token_width=384, projection_width=64, hidden_width=64):
        """Mask-derived actual calls and logical payloads, with parent references."""
        work = DialogueTokenFactoredMemory.packing_work(valid, candidate_mask, token_mask,
                                                       token_width, projection_width, hidden_width)
        return _consolidated_work(work, token_width)

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
        qfull = qp[:, :, None].expand_as(cp)
        u0 = self.turn_projection.bias.tanh().expand_as(cp)
        fill = self._exogenous(u0, qfull, cp, tokens.new_zeros((b, q, c, 10)))
        dense = fill[:, None].expand(b, t, q, c, self.hidden_dim).reshape(b*t*q*c, self.hidden_dim)
        if not groups:
            # Exactly the parent's empty graph, including lexical zero gradients.
            empty = self.turn_projection(raw).tanh()
            dense = (dense + empty.sum()*0 + keys.sum()*0 + priors.sum()*0
                     + (schema_query*0).sum() + (lexical*0).sum())
        else:
            lookup = torch.full(active.shape, -1, dtype=torch.long, device=tokens.device)
            lookup[active] = torch.arange(len(raw), device=tokens.device)
            pairs = [candidate_mask[bi].reshape(-1).nonzero(as_tuple=False).flatten() for bi in range(b)]
            pooled_rows, schemas, destinations = [], [], []
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
                pooled_rows.append(evidence.reshape(-1, width))
                schemas.append(schema_ids)
                destinations.append(torch.stack(output_indices).reshape(-1))
            # Keep these three concatenations aligned; no candidate-dependent
            # filtering or reordering is permitted after public layout assembly.
            pooled = torch.cat(pooled_rows)
            schema_ids = torch.cat(schemas)
            output_ids = torch.cat(destinations)
            projected = self.turn_projection(pooled).tanh()
            supported_q = qp.reshape(-1, self.projection_dim).index_select(0, schema_ids//c)
            supported_c = cp.reshape(-1, self.projection_dim).index_select(0, schema_ids)
            supported_lex = lexical.reshape(-1, 10).index_select(0, output_ids)
            values = self._exogenous(projected, supported_q, supported_c, supported_lex)
            dense = dense.index_copy(0, output_ids, values)
        if torch.isnan(dense).any():
            raise ValueError("NaN exogenous preactivation")
        work = _factored_work(base_work, candidate_mask, t, self.projection_dim, self.hidden_dim)
        self.last_packing_work = _consolidated_work(work, width)
        return dense.reshape(b, t, q, c, self.hidden_dim)
