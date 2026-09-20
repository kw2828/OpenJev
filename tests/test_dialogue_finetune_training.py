"""Tiny artificial Torch arithmetic and fake encoder only; no assets or fits."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openjev.research.dialogue_copy_memory_v2 import DialogueCopyMemoryV2
from openjev.research.dialogue_finetune_training import supervised_loss
from openjev.research.dialogue_trainable_encoder import DialogueTrainableEncoder

torch.set_num_threads(1)


def row(time=0, query=0, label=0, stratum=0):
    return {"time": time, "query_position": query, "label_index": label, "stratum_index": stratum}


@pytest.mark.parametrize("weights", [[1., 1., 1.], [.3, 2.5, 1.2]])
def test_unequal_microbatch_loss_and_parameter_gradients_equal_combined_endpoint_mean(weights):
    with torch.random.fork_rng():
        torch.manual_seed(1901)
        micro_model = nn.Linear(3, 4, dtype=torch.float64)
        combined_model = copy.deepcopy(micro_model)
        # Four full public steps but only three selected endpoints; second dialogue has one.
        x1 = torch.tensor([[.1, .3, -.2], [1., -.3, .7], [.4, .9, .2], [-.1, .2, .8]], dtype=torch.float64)
        x2 = torch.tensor([[.2, .8, .4], [.9, -.3, .2]], dtype=torch.float64)
        rows1 = [row(3, label=2, stratum=2), row(0, label=0), row(1, label=1, stratum=1)]
        rows2 = [row(1, label=3, stratum=2)]
        log1 = micro_model(x1).log_softmax(-1)[None, :, None, :]
        log2 = micro_model(x2).log_softmax(-1)[None, :, None, :]
        loss1 = supervised_loss(log1, rows1, weights, 4)
        loss2 = supervised_loss(log2, rows2, weights, 4)
        loss1.backward()
        loss2.backward()
        # Independent reference is one concatenated batch of selected endpoint logits.
        selected_x = torch.cat((x1[[3, 0, 1]], x2[[1]]))
        targets = torch.tensor([2, 0, 1, 3])
        per_endpoint = torch.nn.functional.cross_entropy(combined_model(selected_x), targets, reduction="none")
        ws = torch.tensor([weights[2], weights[0], weights[1], weights[2]], dtype=torch.float64)
        expected = (per_endpoint * ws).sum() / 4
        expected.backward()
        torch.testing.assert_close(loss1 + loss2, expected, atol=1e-14, rtol=1e-14)
        for a, b in zip(micro_model.parameters(), combined_model.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-14, rtol=1e-14)


def test_endpoint_count_not_dialogue_mean_or_weight_sum():
    logs_a = torch.tensor([[-1., -.45867514538708193]] * 3, dtype=torch.float64)[None, :, None, :]
    logs_b = torch.tensor([[-9., -.00012341741970310867]], dtype=torch.float64)[None, :, None, :]
    rows_a, rows_b = [row(t) for t in range(3)], [row()]
    loss = supervised_loss(logs_a, rows_a, [2., 1., 1.], 4)
    loss += supervised_loss(logs_b, rows_b, [2., 1., 1.], 4)
    assert loss.item() == 6.  # (3*2*1 + 1*2*9) / 4, not a mean over the two dialogues.
    assert loss.item() != (2. + 18.) / 2
    assert loss.item() != (3*2. + 18.) / (4*2.)


def test_tail_one_and_unscored_positions_have_correct_input_gradients():
    logits = torch.tensor([[[[.2, -.1], [.7, .4]], [[-.2, .3], [.1, .9]]]],
                          dtype=torch.float64, requires_grad=True)
    logs = logits.log_softmax(-1)
    loss = supervised_loss(logs, [row(1, query=1, label=1, stratum=2)], [1., 1., 3.], 1)
    expected = -3 * logs[0, 1, 1, 1]
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1, 0]) == 0
    assert logits.grad[0, 1, 1].abs().sum() > 0


def test_masked_padding_allowed_but_selected_zero_probability_rejected():
    logp = torch.tensor([[[[-.2, -1.70777180097052, -torch.inf]]]], requires_grad=True)
    loss = supervised_loss(logp, [row(label=1)], [1., 1., 1.], 1)
    loss.backward()
    assert logp.grad.tolist() == [[[[0., -1., 0.]]]]
    with pytest.raises(ValueError, match="Selected target"):
        supervised_loss(logp, [row(label=2)], [1., 1., 1.], 1)


@pytest.mark.parametrize("mutation", ["time_bool", "query_oob", "label_negative", "stratum_oob", "missing", "duplicate"])
def test_invalid_endpoint_metadata(mutation):
    rows = [row()]
    if mutation == "time_bool":
        rows[0]["time"] = False
    elif mutation == "query_oob":
        rows[0]["query_position"] = 1
    elif mutation == "label_negative":
        rows[0]["label_index"] = -1
    elif mutation == "stratum_oob":
        rows[0]["stratum_index"] = 3
    elif mutation == "missing":
        del rows[0]["label_index"]
    else:
        rows.append(row(label=1))
    with pytest.raises(ValueError):
        supervised_loss(torch.zeros(1, 2, 1, 2).log_softmax(-1), rows, [1., 1., 1.], 2)


@pytest.mark.parametrize("weights", [[1., 1.], [1., 0., 1.], [1., -1., 1.], [1., float("nan"), 1.],
                                    [1., float("inf"), 1.]])
def test_invalid_fixed_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        supervised_loss(torch.zeros(1, 1, 1, 2).log_softmax(-1), [row()], weights, 1)


@pytest.mark.parametrize("denominator", [0, -1, True, 1.5])
def test_invalid_batch_denominator(denominator):
    with pytest.raises(ValueError, match="endpoint count"):
        supervised_loss(torch.zeros(1, 1, 1, 2).log_softmax(-1), [row()], [1., 1., 1.], denominator)


def test_empty_rows_or_denominator_smaller_than_dialogue_rejected():
    logp = torch.zeros(1, 2, 1, 2).log_softmax(-1)
    with pytest.raises(ValueError, match="Nonempty"):
        supervised_loss(logp, [], [1., 1., 1.], 1)
    with pytest.raises(ValueError, match="endpoint count"):
        supervised_loss(logp, [row(), row(1)], [1., 1., 1.], 1)


def test_nonmapping_endpoint_rejected():
    with pytest.raises(TypeError, match="mapping"):
        supervised_loss(torch.zeros(1, 1, 1, 2).log_softmax(-1), [0], [1., 1., 1.], 1)


@pytest.mark.parametrize("logp", [torch.zeros(2, 1, 1, 2), torch.zeros(1, 0, 1, 2),
                                 torch.zeros(1, 2, 2), torch.zeros(1, 1, 1, 2, dtype=torch.long),
                                 torch.tensor([[[[float("nan"), -.5]]]]),
                                 torch.tensor([[[[float("inf"), -.5]]]])])
def test_invalid_log_probability_shapes_or_nonfinite_values(logp):
    with pytest.raises(ValueError):
        supervised_loss(logp, [row(label=1)], [1., 1., 1.], 1)


def test_arithmetic_overflow_rejected_without_floor_or_rescale():
    with pytest.raises(ValueError, match="Nonfinite accumulated"):
        supervised_loss(torch.tensor([[[[-1e30, 0.]]]]), [row()], [1e20, 1., 1.], 1)


def test_final_scored_loss_reaches_earlier_unscored_encoder_token_through_scalar_state():
    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(30, 8)
            self.context = nn.Linear(8, 8)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.context(self.embeddings(input_ids)))

    with torch.random.fork_rng():
        torch.manual_seed(1942)
        model = DialogueTrainableEncoder(FakeEncoder(),
            DialogueCopyMemoryV2("scalar", input_dim=8, projection_dim=4, hidden_dim=6),
            train_encoder=True, cls_id=28, sep_id=29, pad_id=0)
        # Content token 1 appears only in the earlier unscored public turn.
        payload = {"tokens": [[1], [2], [3], [4], [5], [6]], "turn_text_ids": [0, 1],
                   "query_text_ids": [2], "candidate_text_ids": [[3, 4, 5]],
                   "candidate_ids": [["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:blue"]],
                   "lexical": torch.zeros(2, 1, 3, 10)}
        actor, none, _ = model.encode_actor(**payload)
        actor[0].retain_grad()
        logs = model.memory(*actor, none_index=none)
        supervised_loss(logs, [row(1, label=2, stratum=2)], [1., 1., 1.], 1).backward()
        assert torch.isfinite(actor[0].grad).all() and actor[0].grad[0, 0].abs().sum() > 0
        gradient = model.encoder.embeddings.weight.grad
        assert torch.isfinite(gradient).all() and gradient[1].abs().sum() > 0
        assert model.memory.head.weight.grad.abs().sum() > 0
