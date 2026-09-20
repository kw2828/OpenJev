"""Public-mask packing for the frozen token evidence function and ordered head.

Supported dummy NONE candidates remain computed. Grouping uses only masks, never
labels, attention values or recurrent state. Raw tokens are shared over schema
axes; no tensor is retained between calls. last_packing_work contains only integer
payload/work counts after successful pooling, not a memory or speed measurement.

The existing monitor composes unchanged::

    class MonitoredPacked(MonitoredCopyMemoryV2, DialogueTokenPackedMemory):
        pass
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_memory import ATTENTION_DIM, POOL_EPS, DialogueTokenMemory
from openjev.research.dialogue_token_memory import VERSION as PARENT_VERSION

VERSION = "dialogue-token-packed-v1"
PARENT_SOURCE = "src/openjev/research/dialogue_token_memory.py"
PARENT_SHA256 = "07a9b5d11ba4569417ede208bf1b6498cfac3488db8c8daec1062053a8bdc5eb"
BUCKET_WIDTH = 16


def _packing_layout(valid, candidate_mask, token_mask, token_width):
    if (type(token_width) is not int or token_width < 1
            or not all(isinstance(x, torch.Tensor) and x.dtype == torch.bool for x in (valid, candidate_mask, token_mask))
            or valid.ndim != 2 or candidate_mask.ndim != 3 or token_mask.ndim != 3):
        raise ValueError("Packing requires boolean masks and a positive token width")
    b, t = valid.shape
    if (min(b, t, *candidate_mask.shape[1:], token_mask.shape[-1]) < 1
            or candidate_mask.shape[0] != b or token_mask.shape[:2] != (b, t)
            or candidate_mask.device != valid.device or token_mask.device != valid.device
            or not candidate_mask.any(-1).all() or not token_mask.any(-1)[valid].all()):
        raise ValueError("Invalid packing mask geometry or empty real support")
    q, c = candidate_mask.shape[1:]
    length = token_mask.shape[-1]
    support = candidate_mask.sum((1, 2)).tolist()
    lengths = token_mask.sum(-1).tolist()
    groups = {}
    key_rows = score_rows = evidence_rows = bucket_rows = 0
    real_turns = valid.nonzero(as_tuple=False).tolist()
    for bi, ti in real_turns:
        n, m = lengths[bi][ti], support[bi]
        bucket = min(length, BUCKET_WIDTH*((n+BUCKET_WIDTH-1)//BUCKET_WIDTH))
        groups.setdefault((bucket, m), []).append((bi, ti))
        key_rows += n
        score_rows += bucket*m
        evidence_rows += m
        bucket_rows += bucket
    work = {"real_turns": len(real_turns), "supported_schema_pairs": sum(support),
        "packed_token_key_positions": key_rows, "dense_token_key_positions": b*t*length,
        "packed_score_positions": score_rows, "dense_score_positions": b*t*q*c*length,
        "packed_evidence_positions": evidence_rows, "dense_evidence_positions": b*t*q*c,
        "pooling_groups": len(groups), "bucket_token_positions": bucket_rows,
        "packed_raw_token_scalars": key_rows*token_width, "grouped_raw_token_scalars": bucket_rows*token_width,
        "grouped_key_scalars": bucket_rows*ATTENTION_DIM, "grouped_schema_scalars": evidence_rows*ATTENTION_DIM,
        "scatter_evidence_scalars": evidence_rows*token_width, "dense_evidence_scalars": b*t*q*c*token_width}
    return groups, work


class DialogueTokenPackedMemory(DialogueTokenMemory):
    """Same constructor, initialization, state_dict and inherited step/pool APIs.

    Each group is (min(input_L, 16*ceil(valid_tokens/16)), supported_schema_pairs).
    Original head/monitor work stays dense and sequential. Packing alters floating
    reduction order; no bitwise optimizer-trajectory or performance claim is made.
    """

    last_packing_work = None

    def configuration(self):
        result = super().configuration()
        result.update({
            "implementation_version": VERSION,
            "pooling_parent_version": PARENT_VERSION,
            "pooling_parent_source": {"path": PARENT_SOURCE, "sha256": PARENT_SHA256},
            "packing_bucket_width": BUCKET_WIDTH,
            "packing": "Real turns and token_mask support; all candidate_mask support including dummy NONE; deterministic (bucket,pair-count) groups",
            "attention_work": "One valid-token key projection; grouped masked token attention; one dense evidence scatter; unchanged T head updates; step inherited",
            "packing_work_scope": "Integer positions and gathered/scattered scalar payloads; not measured traffic, peak allocation or speed",
        })
        return result

    @staticmethod
    def packing_work(valid, candidate_mask, token_mask, token_width=384):
        """Pure mask-derived counts, independent of model values, labels and state."""
        return _packing_layout(valid, candidate_mask, token_mask, token_width)[1]

    def _pool_packed(self, tokens, valid, candidate_mask, token_mask, token_prior, schema_query):
        b, t, _, width = tokens.shape
        q, c = candidate_mask.shape[1:]
        groups, work = _packing_layout(valid, candidate_mask, token_mask, width)
        active = valid[..., None] & token_mask
        token_indices = active.reshape(-1).nonzero(as_tuple=False).flatten()
        raw = tokens.reshape(-1, width).index_select(0, token_indices)
        keys = self.token_key(raw)
        priors = token_prior.reshape(-1).index_select(0, token_indices)
        if not torch.isfinite(schema_query).all():
            raise ValueError("Nonfinite token attention logits")
        dense = tokens.new_zeros((b*t*q*c, width))
        if not groups:
            # The original masked graph still returns zero gradients, not None,
            # for these paths on an all-padded batch. Keep their zero connections.
            dense = dense + keys.sum()*0 + priors.sum()*0 + (schema_query*0).sum()
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
                values.append(evidence.reshape(-1, width))
                destinations.append(torch.stack(output_indices).reshape(-1))
            dense = dense.index_copy(0, torch.cat(destinations), torch.cat(values))
        if not torch.isfinite(dense).all():
            raise ValueError("Nonfinite pooled evidence")
        self.last_packing_work = dict(work)
        return dense.reshape(b, t, q, c, width)

    def forward(self, tokens, valid, query, candidates, candidate_mask, lexical, token_mask, token_prior, none_index=None):
        """Original validation and ordered updates, with public-mask packed pooling."""
        self.last_packing_work = None
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._check_tokens(tokens, valid, token_mask, token_prior, b, sequence=True)
        t = tokens.shape[1]
        self._float(lexical, (b, t, q, c, 10), "lexical sequence")
        state = self.initial(candidate_mask, none_index)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        schema_query = self._attention_query(query, candidates, candidate_mask)
        evidence = self._pool_packed(tokens, valid, candidate_mask, token_mask, token_prior, schema_query)
        rows = []
        for i in range(t):
            state, _ = self._advance(state, evidence[:, i], valid[:, i], qp, cp, candidate_mask, lexical[:, i])
            rows.append(state["log_b"])
        return torch.stack(rows, 1)
