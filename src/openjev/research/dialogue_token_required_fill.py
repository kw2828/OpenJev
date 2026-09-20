"""Omit only schema-fill rows overwritten at every public time position.

This additive implementation changes neither parameters nor the ordered head.
The public predicate is (~all_t(valid)) OR (~candidate_mask). Partial fill uses
only those schema rows; omitted storage is initialized and fully overwritten.
All-padding retains the exact inherited empty graph, including zero gradients.
No labels, predictions, timing thresholds or persistent activation caches enter
layout decisions. Reduction shape changes require numerical parity checks;
there is no universal extreme-finite-input or bitwise-trajectory claim.

Use the unchanged monitor-first MRO::

    class MonitoredRequiredFill(MonitoredCopyMemoryV2, DialogueTokenRequiredFillMemory):
        pass
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_consolidated import VERSION as PARENT_VERSION
from openjev.research.dialogue_token_consolidated import DialogueTokenConsolidatedMemory, _consolidated_work
from openjev.research.dialogue_token_factored import _factored_work
from openjev.research.dialogue_token_memory import ATTENTION_DIM, POOL_EPS
from openjev.research.dialogue_token_packed import _packing_layout

VERSION = "dialogue-token-required-fill-v1"
CONSOLIDATED_SOURCE = "src/openjev/research/dialogue_token_consolidated.py"
CONSOLIDATED_SHA256 = "03fa31be4ac35ff302fe34b367ef8a632416b7ff42c4aa11c331d7b25bf26ec8"
FULL_FILL_REFERENCE_FIELDS = [
    "full_fill_reference_positions", "full_fill_reference_calls", "full_fill_reference_exogenous_scalars",
    "full_fill_reference_hidden_scalars", "full_fill_reference_macs",
]


def _required_fill_work(work, valid, candidate_mask, fill_count, projection_width, hidden_width):
    """Scalar payload accounting, not traffic, peak allocation or elapsed work.

    Two boolean reductions visit B*T and K entries. Boolean intermediates are
    all_t(valid), its negation, not(candidate_mask), and their broadcast OR.
    Sum returns one int64 scalar. Partial fill creates F schema and F query
    indices, gathers two F*P schema arrays and index-copies F*H output scalars.
    Index-copy's returned schema tensor shares the K*H geometry already recorded
    as zero storage; this payload accounting is not a lifetime allocation sum.
    """
    b, t = valid.shape
    k, f, p, h = candidate_mask.numel(), fill_count, projection_width, hidden_width
    e, s, n = work["exogenous_width"], work["packed_evidence_positions"], work["dense_evidence_positions"]
    partial = 0 < f < k
    gathered, lexical, storage = (2*f*p if partial else 0), 10*f, (k*h if f < k else 0)
    return {**work,
            "fill_feature_positions": f, "fill_feature_calls": int(f > 0),
            "fill_exogenous_input_scalars": f*e, "fill_hidden_scalars": f*h,
            "fill_feature_macs": f*e*h, "factored_feature_macs": (s+f)*e*h+2*n*h,
            "full_fill_reference_positions": k, "full_fill_reference_calls": 1,
            "full_fill_reference_exogenous_scalars": k*e, "full_fill_reference_hidden_scalars": k*h,
            "full_fill_reference_macs": k*e*h, "omitted_fill_positions": k-f,
            "fill_mask_reduction_positions": b*t+k, "fill_mask_boolean_scalars": 2*b+2*k,
            "fill_mask_boolean_bytes": 2*b+2*k, "fill_count_int64_scalars": 1, "fill_count_int64_bytes": 8,
            "fill_index_int64_scalars": 2*f if partial else 0, "fill_index_int64_bytes": 16*f if partial else 0,
            "fill_gathered_schema_scalars": gathered, "fill_zero_lexical_scalars": lexical,
            "fill_zero_storage_scalars": storage, "fill_schema_scatter_positions": f if partial else 0,
            "fill_schema_scatter_scalars": f*h if partial else 0,
            "fill_auxiliary_float32_bytes": 4*(gathered+lexical+storage), "fill_mask_predicate_calls": 1}


class DialogueTokenRequiredFillMemory(DialogueTokenConsolidatedMemory):
    """Same constructor, initialization, state_dict, step and monitored forward."""

    def configuration(self):
        result = super().configuration()
        result.update({
            "implementation_version": VERSION,
            "consolidated_parent_version": PARENT_VERSION,
            "consolidated_parent_source": {"path": CONSOLIDATED_SOURCE, "sha256": CONSOLIDATED_SHA256},
            "skipped_evidence": "Only public schema rows with some non-overwritten time position get exact schema fill; all other initialized rows are overwritten before head use",
            "required_fill_predicate": "(~valid.all(time))[:,None,None] | (~candidate_mask)",
            "required_fill_routes": "F=0 skips fill projection; F=K retains full-schema fill; otherwise gather F and index_copy into initialized zeros",
            "attention_work": "Inherited grouped attention, raw normalization, supported concatenation and two projections; required-only schema fill; unchanged ordered monitored head",
            "packing_reference_only_fields": result["packing_reference_only_fields"]+FULL_FILL_REFERENCE_FIELDS,
            "required_fill_work_scope": "Actual F-based fill counters; mask reductions and bool/int payloads separate from float32 auxiliary payload estimates. No traffic, peak allocation or speed claim; aliases must not be summed as allocations",
        })
        return result

    @staticmethod
    def packing_work(valid, candidate_mask, token_mask, token_width=384, projection_width=64, hidden_width=64):
        """Pure public-mask reference using the same actual fill predicate."""
        work = DialogueTokenConsolidatedMemory.packing_work(
            valid, candidate_mask, token_mask, token_width, projection_width, hidden_width)
        needs_fill = (~valid.all(dim=1))[:, None, None] | (~candidate_mask)
        return _required_fill_work(work, valid, candidate_mask, int(needs_fill.sum()), projection_width, hidden_width)

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
        # Only rows with some non-overwritten time position need schema fill.
        needs_fill = (~valid.all(dim=1))[:, None, None] | (~candidate_mask)
        fill_count = int(needs_fill.sum())
        schema_count = candidate_mask.numel()
        if fill_count == schema_count:
            qfull = qp[:, :, None].expand_as(cp)
            u0 = self.turn_projection.bias.tanh().expand_as(cp)
            fill = self._exogenous(u0, qfull, cp, tokens.new_zeros((b, q, c, 10)))
        else:
            # Initialized zeros are safe because omitted rows are supported on
            # every time step; the unique supported scatter overwrites them all.
            fill = tokens.new_zeros((schema_count, self.hidden_dim))
            if fill_count:
                fill_ids = needs_fill.reshape(-1).nonzero(as_tuple=False).flatten()
                fill_query = qp.reshape(-1, self.projection_dim).index_select(0, fill_ids//c)
                fill_candidates = cp.reshape(-1, self.projection_dim).index_select(0, fill_ids)
                u0 = self.turn_projection.bias.tanh().expand(fill_count, self.projection_dim)
                values = self._exogenous(u0, fill_query, fill_candidates, tokens.new_zeros((fill_count, 10)))
                fill = fill.index_copy(0, fill_ids, values)
            fill = fill.reshape(b, q, c, self.hidden_dim)
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
        self.last_packing_work = _required_fill_work(
            _consolidated_work(work, width), valid, candidate_mask, fill_count, self.projection_dim, self.hidden_dim)
        return dense.reshape(b, t, q, c, self.hidden_dim)
