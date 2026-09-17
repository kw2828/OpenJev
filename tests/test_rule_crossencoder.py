import runpy
from pathlib import Path

import pytest
import torch

MODULE = runpy.run_path(str(Path(__file__).parents[1]/'scripts/rule_crossencoder_study.py'))


class RecordingTokenizer:
    def __init__(self, length=8):
        self.length = length
        self.calls = []

    def __call__(self, contexts, assertions, **kwargs):
        self.calls.append((contexts, assertions, kwargs))
        mask = torch.ones(len(contexts), self.length, dtype=torch.int64)
        return {'input_ids': mask.clone(), 'attention_mask': mask}


def world():
    return [{'context': 'A is green. Green things are blue.', 'id': 'private', 'proof': 'secret',
             'questions': [{'text': 'A is blue.', 'gold': 'true', 'depth': 1, 'proof': 'secret'}]}]


def test_gold_and_annotations_stay_out_of_joint_tokenizer():
    tokenizer, w = RecordingTokenizer(), world()
    before = MODULE['tokenize_worlds'](w, tokenizer)
    w[0]['questions'][0].update(gold='false', depth=4, proof='changed')
    after = MODULE['tokenize_worlds'](w, tokenizer)
    assert tokenizer.calls[0] == tokenizer.calls[1]
    assert not torch.equal(before['gold'], after['gold'])
    assert tokenizer.calls[0][:2] == (['A is green. Green things are blue.'], ['A is blue.'])


def test_long_context_rejected_instead_of_silent_truncation():
    with pytest.raises(ValueError, match='no truncation'):
        MODULE['tokenize_worlds'](world(), RecordingTokenizer(length=513))


def test_batch_only_removes_padding():
    data = {'inputs': {'input_ids': torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]]),
                       'attention_mask': torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])}}
    actual = MODULE['batch'](data, torch.tensor([0]), 'cpu')
    assert actual['input_ids'].tolist() == [[1, 2, 3]]
