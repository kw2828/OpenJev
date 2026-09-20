"""Tiny CPU encoders and artificial integer tokens; no pretrained assets."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openjev.research.dialogue_copy_memory_v2 import DialogueCopyMemoryV2
from openjev.research.dialogue_trainable_encoder import (
    DONTCARE,
    NONE,
    DialogueTrainableEncoder,
    build_actor,
    encode_token_lists,
)

torch.set_num_threads(1)


class TinyEncoder(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.embeddings = nn.Embedding(40, width)
        self.attention = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(.8)
        self.calls = []

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone(), self.training))
        x = self.embeddings(input_ids)
        scores = (self.attention(x)@x.transpose(-1, -2))/x.shape[-1]**.5
        weights = scores.masked_fill(~attention_mask.bool()[:, None, :], -torch.inf).softmax(-1)
        return SimpleNamespace(last_hidden_state=self.dropout(x+weights@x))


class LookupEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(40, 3, dtype=torch.float64)
        with torch.no_grad():
            ids = torch.arange(40, dtype=torch.float64)
            self.embeddings.weight.copy_(torch.stack((ids+1, ids.square()+3, ids%3+1), -1))
        self.calls = []

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


def payload():
    return {"tokens": [[1, 2, 3], [4, 5], [6, 7], [8], [9], [10], [11], [12], [13]],
        "turn_text_ids": [0, 1], "query_text_ids": [2, 3], "candidate_text_ids": [[4, 5, 6], [7, 4, 5, 8]],
        "candidate_ids": [[NONE, DONTCARE, "value:X"], ["value:Y", NONE, DONTCARE, "value:Z"]],
        "lexical": torch.zeros(2, 2, 4, 10)}


def wrapper(trainable=True):
    torch.manual_seed(718)
    encoder = TinyEncoder()
    memory = DialogueCopyMemoryV2("scalar", input_dim=8, projection_dim=4, hidden_dim=6)
    return DialogueTrainableEncoder(encoder, memory, train_encoder=trainable,
                                   cls_id=30, sep_id=31, pad_id=0, chunk_batch_size=4)


def loss(logs):
    return -(logs[0, -1, 0, 2]+logs[0, -1, 1, 0])/2


@pytest.mark.parametrize("trainable", [True, False])
def test_end_to_end_gradients_and_frozen_encoder_exclusion(trainable):
    model = wrapper(trainable).train()
    logs, work = model(**payload())
    loss(logs).backward()
    assert logs.shape == (1, 2, 2, 4)
    assert torch.isneginf(logs[0, :, 0, 3]).all()
    assert torch.allclose(logs.logsumexp(-1), torch.zeros(1, 2, 2), atol=1e-6)
    assert work["public_turns"] == 2 and work["real_question_updates"] == 4
    assert work["real_candidate_updates"] == 14 and work["encoder_calls"] == 3
    for name, parameter in model.encoder.named_parameters():
        if trainable:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert not parameter.requires_grad and parameter.grad is None, name
    if trainable:
        # Disjoint source tokens for turn, query, candidate paths all receive credit.
        for token in (1, 6, 11):
            assert model.encoder.embeddings.weight.grad[token].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.memory.parameters())
    assert model.memory.head.weight.grad.abs().sum() > 0


def test_chunk_pooling_exact_recipe_over_254_and_masked_padding():
    encoder = LookupEncoder()
    tokens = [[i%20+1 for i in range(259)], [5, 6]]
    vectors, work = encode_token_lists(encoder, tokens, cls_id=30, sep_id=31, pad_id=0,
                                      trainable=True, chunk_batch_size=2)
    expected = []
    for ids in tokens:
        chunks = [ids[start:start+254] for start in range(0, len(ids), 254)]
        pooled = sum(encoder.embeddings.weight[[30, *chunk, 31]].mean(0)*len(chunk) for chunk in chunks)/len(ids)
        expected.append(pooled/pooled.norm())
    assert torch.allclose(vectors, torch.stack(expected), atol=1e-14, rtol=1e-14)
    assert work == {"input_texts": 2, "content_tokens": 261, "encoder_sequences": 3, "encoder_calls": 2,
        "special_token_positions": 6, "valid_token_positions": 267, "padded_token_positions": 516,
        "padded_attention_positions": 2*256**2+4**2, "padding_token_positions": 249,
        "overlength_texts_chunked": 1, "truncated_tokens": 0}
    vectors[:, 0].sum().backward()
    assert encoder.embeddings.weight.grad[30].abs().sum() > 0
    assert encoder.embeddings.weight.grad[0].abs().sum() == 0


def test_chunk_batch_partition_preserves_outputs_and_gradients():
    a, b = LookupEncoder(), LookupEncoder()
    tokens = [[1, 2, 3, 4, 5, 6, 7], [8], [9, 10]]
    va, wa = encode_token_lists(a, tokens, cls_id=30, sep_id=31, pad_id=0, trainable=True, chunk_tokens=3, chunk_batch_size=1)
    vb, wb = encode_token_lists(b, tokens, cls_id=30, sep_id=31, pad_id=0, trainable=True, chunk_tokens=3, chunk_batch_size=4)
    assert torch.allclose(va, vb, atol=1e-14, rtol=1e-14)
    va.square()[:, 1].sum().backward(); vb.square()[:, 1].sum().backward()
    assert torch.allclose(a.embeddings.weight.grad, b.embeddings.weight.grad, atol=1e-13, rtol=1e-13)
    assert wa["encoder_sequences"] == wb["encoder_sequences"] == 5
    assert wa["encoder_calls"] == 5 and wb["encoder_calls"] == 2


def test_eval_dropout_stays_disabled_while_training_gradients_remain_enabled():
    model = wrapper().train()
    assert model.training and model.memory.training and not model.encoder.training
    first, _ = model(**payload())
    second, _ = model(**payload())
    assert torch.equal(first, second) and first.requires_grad
    assert all(not training for _, _, training in model.encoder.calls)
    with torch.no_grad():
        evaluated, _ = model(**payload())
    assert not evaluated.requires_grad


def test_candidate_permutation_and_nonzero_none_index():
    model, p = wrapper(), payload()
    original, _ = model(**p)
    altered = copy.deepcopy(p)
    orders = [[2, 0, 1], [3, 2, 1, 0]]
    for q, order in enumerate(orders):
        altered["candidate_text_ids"][q] = [p["candidate_text_ids"][q][c] for c in order]
        altered["candidate_ids"][q] = [p["candidate_ids"][q][c] for c in order]
        altered["lexical"][:, q, :len(order)] = p["lexical"][:, q, order]
    moved, _ = model(**altered)
    for q, order in enumerate(orders):
        assert torch.allclose(moved[0, :, q, :len(order)], original[0, :, q, order], atol=1e-6)
    actor, none, _ = model.encode_actor(**altered)
    assert none.tolist() == [[1, 2]] and actor[1].all()


def test_query_independence_and_public_turn_causality():
    model, p = wrapper(), payload()
    before, _ = model(**p)
    other = copy.deepcopy(p)
    other["tokens"][3] = [18, 19, 20]  # Only the second query's schema.
    after, _ = model(**other)
    assert torch.equal(before[0, :, 0], after[0, :, 0])
    future = copy.deepcopy(p)
    future["tokens"][1] = [21, 22, 23]
    future_output, _ = model(**future)
    assert torch.equal(before[:, 0], future_output[:, 0])
    assert not torch.equal(before[:, 1], future_output[:, 1])


def test_complete_dialogue_equals_unreset_manual_stream_and_final_loss_reaches_earlier_turn():
    model, p = wrapper(), payload()
    actor, none, _ = model.encode_actor(**p)
    actor[0].retain_grad()
    full = model.memory(*actor, none_index=none)
    state = model.memory.initial(actor[4], none)
    stream = []
    for t in range(actor[0].shape[1]):
        state, _ = model.memory.step(state, actor[0][:, t], actor[1][:, t], actor[2], actor[3], actor[4], actor[5][:, t])
        stream.append(state["log_b"])
    assert torch.allclose(full, torch.stack(stream, 1), atol=1e-6)
    loss(full).backward()
    assert actor[0].grad[0, 0].abs().sum() > 0
    assert actor[0].grad[0, 1].abs().sum() > 0


def test_differentiable_dtype_transfer_and_lexical_list_acceptance():
    model, p = wrapper(), payload()
    vectors, _ = encode_token_lists(model.encoder, p["tokens"], cls_id=30, sep_id=31, pad_id=0, trainable=True)
    vectors.retain_grad()
    actor, none = build_actor(vectors, **{k: v for k, v in p.items() if k not in ("tokens", "lexical")},
        lexical=p["lexical"].tolist(), memory_device="cpu", memory_dtype=torch.float64)
    model.memory.double()
    loss(model.memory(*actor, none_index=none)).backward()
    assert vectors.grad is not None and torch.isfinite(vectors.grad).all()
    assert model.encoder.embeddings.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("tokens", [[], [[]], [[True]], [[-1]], [[1.5]]])
def test_invalid_content_tokens_rejected_before_encoder_call(tokens):
    encoder = TinyEncoder()
    with pytest.raises(ValueError):
        encode_token_lists(encoder, tokens, cls_id=30, sep_id=31, pad_id=0, trainable=True)
    assert encoder.calls == []


@pytest.mark.parametrize("damage", ["missing_none", "duplicate_candidate", "bad_mapping", "lexical_padding", "nonfinite_lexical"])
def test_actor_schema_and_padding_guards(damage):
    p = payload()
    vectors = torch.randn(len(p["tokens"]), 8)
    if damage == "missing_none":
        p["candidate_ids"][0][0] = "value:unsupported"
    elif damage == "duplicate_candidate":
        p["candidate_ids"][0][1] = NONE
    elif damage == "bad_mapping":
        p["turn_text_ids"][0] = True
    elif damage == "lexical_padding":
        p["lexical"][0, 0, 3, 0] = 1
    else:
        p["lexical"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        build_actor(vectors, **{k: v for k, v in p.items() if k != "tokens"}, memory_device="cpu")


def test_encoder_nonfinite_or_zero_pooling_rejected():
    for bad in (float("nan"), 0.):
        encoder = LookupEncoder()
        with torch.no_grad():
            encoder.embeddings.weight.fill_(bad)
        with pytest.raises(ValueError):
            encode_token_lists(encoder, [[1]], cls_id=30, sep_id=31, pad_id=0, trainable=True)
