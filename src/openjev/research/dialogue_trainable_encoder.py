"""Differentiable pooled MiniLM-style observations for autonomous scalar memory.

The caller supplies an encoder and integer content tokens, never a tokenizer,
text file, target or previous gold state. No pretrained weights load on import.
All encoding is in eval mode, including the trainable-encoder comparison.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from openjev.research.dialogue_copy_memory_v2 import DialogueCopyMemoryV2

NONE = "reserved:NOT_MENTIONED"
DONTCARE = "reserved:DONTCARE"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def encode_token_lists(encoder, content_tokens, *, cls_id, sep_id, pad_id,
                       trainable, chunk_tokens=254, chunk_batch_size=32):
    """Return [U,D] unit vectors and actual successful forward work counts.

    Each chunk mean includes CLS/SEP and excludes padding. Chunk means are
    weighted by their content lengths; only the combined text vector is L2
    normalized. No truncation, NumPy conversion, detach, or cached activations.
    The caller owns parameter requires_grad flags; ``trainable=False`` also
    suppresses encoder autograd using no_grad, not inference_mode.
    """
    require(isinstance(encoder, nn.Module) and type(trainable) is bool, "Supplied encoder and trainability flag")
    require(all(_integer(v) for v in (cls_id, sep_id, pad_id)) and _integer(chunk_tokens, minimum=1)
            and _integer(chunk_batch_size, minimum=1), "Special-token IDs and positive chunk dimensions")
    require(type(content_tokens) is list and content_tokens
            and all(type(v) is list and v and all(_integer(t) for t in v) for v in content_tokens),
            "Nonempty lists of nonnegative integer content tokens")
    reference = next(encoder.parameters(), None)
    require(reference is not None and reference.dtype in (torch.float32, torch.float64), "Float32/64 encoder parameters")
    encoder.eval()
    chunks = [(i, [cls_id, *ids[start:start+chunk_tokens], sep_id], len(ids[start:start+chunk_tokens]))
              for i, ids in enumerate(content_tokens) for start in range(0, len(ids), chunk_tokens)]
    work = {"input_texts": len(content_tokens), "content_tokens": sum(map(len, content_tokens)),
        "encoder_sequences": len(chunks), "encoder_calls": 0, "special_token_positions": 2*len(chunks),
        "valid_token_positions": sum(len(seq) for _, seq, _ in chunks), "padded_token_positions": 0,
        "padded_attention_positions": 0, "padding_token_positions": 0,
        "overlength_texts_chunked": sum(len(v) > chunk_tokens for v in content_tokens), "truncated_tokens": 0}
    combined = None
    with nullcontext() if trainable else torch.no_grad():
        for start in range(0, len(chunks), chunk_batch_size):
            batch = chunks[start:start+chunk_batch_size]
            width = max(len(seq) for _, seq, _ in batch)
            ids = torch.tensor([seq+[pad_id]*(width-len(seq)) for _, seq, _ in batch],
                               device=reference.device, dtype=torch.int64)
            mask = torch.tensor([[1]*len(seq)+[0]*(width-len(seq)) for _, seq, _ in batch],
                                device=reference.device, dtype=torch.int64)
            hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            require(isinstance(hidden, torch.Tensor) and hidden.ndim == 3 and hidden.shape[:2] == ids.shape
                    and hidden.shape[2] > 0 and hidden.dtype == reference.dtype and hidden.device == reference.device
                    and torch.isfinite(hidden).all().item(), "Finite encoder last_hidden_state with matching geometry")
            pooled = hidden.masked_fill(~mask.bool()[..., None], 0.).sum(1)/mask.sum(1)[:, None]
            owners = torch.tensor([i for i, _, _ in batch], device=hidden.device, dtype=torch.int64)
            weights = hidden.new_tensor([n for _, _, n in batch])
            if combined is None:
                combined = hidden.new_zeros(len(content_tokens), hidden.shape[-1])
            require(hidden.shape[-1] == combined.shape[-1], "Stable encoder width")
            combined = combined.index_add(0, owners, pooled*weights[:, None])
            work["encoder_calls"] += 1
            work["padded_token_positions"] += ids.numel()
            work["padded_attention_positions"] += len(batch)*width*width
        combined = combined/combined.new_tensor([len(v) for v in content_tokens])[:, None]
        norms = torch.linalg.vector_norm(combined, dim=-1, keepdim=True)
        require(torch.isfinite(combined).all().item() and torch.isfinite(norms).all().item()
                and (norms > 1e-12).all().item(), "Finite nonzero pooled text vectors")
        vectors = combined/norms
    work["padding_token_positions"] = work["padded_token_positions"]-work["valid_token_positions"]
    return vectors, work


def build_actor(vectors, *, turn_text_ids, query_text_ids, candidate_text_ids,
                candidate_ids, lexical, memory_device, memory_dtype=None):
    """Build one complete dialogue's six actor tensors and explicit NONE indices.

    ``lexical`` may be any torch.as_tensor-compatible public numeric array.
    Validity uses complete public chronology only; no scored-row mask is accepted.
    Text indices may repeat: index_select propagates all uses to their shared
    encoder graph. Schema transfers to memory_device remain differentiable.
    """
    require(isinstance(vectors, torch.Tensor) and vectors.ndim == 2 and min(vectors.shape) > 0
            and vectors.dtype in (torch.float32, torch.float64) and torch.isfinite(vectors).all().item(),
            "Finite text-vector matrix")
    def refs(values):
        return type(values) is list and values and all(_integer(v) and v < len(vectors) for v in values)
    require(refs(turn_text_ids) and refs(query_text_ids), "Complete public turn/query indices")
    q, t = len(query_text_ids), len(turn_text_ids)
    require(type(candidate_text_ids) is list and len(candidate_text_ids) == q
            and all(refs(v) for v in candidate_text_ids)
            and type(candidate_ids) is list and len(candidate_ids) == q, "Complete candidate text mapping")
    for text_ids, ids in zip(candidate_text_ids, candidate_ids, strict=True):
        require(type(ids) is list and len(ids) == len(text_ids) and all(type(cid) is str for cid in ids)
                and len(set(ids)) == len(ids) and NONE in ids and DONTCARE in ids, "Unique supported candidates and reserved states")
    used = set(turn_text_ids+query_text_ids+[v for group in candidate_text_ids for v in group])
    require(used == set(range(len(vectors))), "Every supplied unique text must be used by the actor")
    target = vectors.to(device=memory_device, dtype=memory_dtype or vectors.dtype)
    require(target.dtype in (torch.float32, torch.float64), "Float32/64 memory inputs")
    c = max(map(len, candidate_ids))
    mask = torch.tensor([[True]*len(v)+[False]*(c-len(v)) for v in candidate_ids], device=target.device)
    indices = torch.tensor([v+[0]*(c-len(v)) for v in candidate_text_ids], dtype=torch.int64, device=target.device)
    candidates = target.index_select(0, indices.flatten()).reshape(q, c, -1).masked_fill(~mask[..., None], 0.)
    query = target.index_select(0, torch.tensor(query_text_ids, dtype=torch.int64, device=target.device))
    turns = target.index_select(0, torch.tensor(turn_text_ids, dtype=torch.int64, device=target.device))
    lex = torch.as_tensor(lexical, dtype=target.dtype, device=target.device)
    require(tuple(lex.shape) == (t, q, c, 10) and torch.isfinite(lex).all().item()
            and ((lex == 0) | (lex == 1)).all().item()
            and (lex.masked_select(~mask[None, ..., None]) == 0).all().item(), "Binary public lexical fields with zero padding")
    actor = (turns[None], torch.ones((1, t), dtype=torch.bool, device=target.device),
             query[None], candidates[None], mask[None], lex[None])
    none = torch.tensor([[v.index(NONE) for v in candidate_ids]], dtype=torch.int64, device=target.device)
    return actor, none


class DialogueTrainableEncoder(nn.Module):
    """Supplied encoder plus unchanged normalized autonomous scalar memory.

    For a separately monitored memory, call encode_actor, its begin_batch on the
    returned actor/public layout, then memory(*actor, none_index=none). The ordinary
    forward path accepts no labels and never resets between public turns.
    """
    def __init__(self, encoder, memory, *, train_encoder, cls_id, sep_id, pad_id,
                 chunk_tokens=254, chunk_batch_size=32):
        super().__init__()
        require(isinstance(encoder, nn.Module) and isinstance(memory, DialogueCopyMemoryV2)
                and getattr(memory, "method", None) == "scalar" and type(train_encoder) is bool,
                "Supplied encoder, scalar memory and trainability flag")
        self.encoder, self.memory, self.train_encoder = encoder, memory, train_encoder
        self.encoder.requires_grad_(train_encoder)
        self.encoder.eval()
        self.encoding = {"cls_id": cls_id, "sep_id": sep_id, "pad_id": pad_id,
                         "chunk_tokens": chunk_tokens, "chunk_batch_size": chunk_batch_size}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_actor(self, tokens, turn_text_ids, query_text_ids, candidate_text_ids, candidate_ids, lexical):
        vectors, work = encode_token_lists(self.encoder, tokens, trainable=self.train_encoder, **self.encoding)
        reference = self.memory.turn_projection.weight
        actor, none = build_actor(vectors, turn_text_ids=turn_text_ids, query_text_ids=query_text_ids,
            candidate_text_ids=candidate_text_ids, candidate_ids=candidate_ids, lexical=lexical,
            memory_device=reference.device, memory_dtype=reference.dtype)
        t, q, c = len(turn_text_ids), len(query_text_ids), max(map(len, candidate_ids))
        work.update(public_turns=t, supplied_queries=q, candidate_occurrences=sum(map(len, candidate_ids)),
                    real_question_updates=t*q, real_candidate_updates=t*sum(map(len, candidate_ids)),
                    padded_candidate_updates=t*q*c, turn_embedding_occurrences=t, query_embedding_occurrences=q,
                    candidate_embedding_occurrences=sum(map(len, candidate_ids)),
                    encoder_output_scalars=vectors.numel())
        return actor, none, work

    def forward(self, tokens, turn_text_ids, query_text_ids, candidate_text_ids, candidate_ids, lexical):
        actor, none, work = self.encode_actor(tokens, turn_text_ids, query_text_ids, candidate_text_ids, candidate_ids, lexical)
        return self.memory(*actor, none_index=none), work
