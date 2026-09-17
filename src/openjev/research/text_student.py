"""Small shared cross-encoder for caller-defined candidate decisions.

Research-only student. No generation; score all context/question/candidate pairs.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ENCODER_ID = 'sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MAX_LENGTH = 512


def ordered_candidates(record):
    return sorted(record['candidates'], key=lambda c: c['description'])


def pair_text(record):
    return f"Question: {record['question']}\nContext: {record['context']}"


class CandidateStudent(nn.Module):
    def __init__(self, seed=17, device='mps'):
        super().__init__()
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION,
                                                       local_files_only=True)
        self.encoder = AutoModel.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION,
                                                local_files_only=True, attn_implementation='eager')
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.device_name = device
        self.to(device)

    def tokenize(self, record):
        candidates = ordered_candidates(record)
        if not 2 <= len(candidates) <= 12 or len({c['id'] for c in candidates}) != len(candidates):
            raise ValueError('Expected 2-12 distinct candidate IDs')
        if len({c['description'].strip().casefold() for c in candidates}) != len(candidates):
            raise ValueError('Candidate descriptions must be distinct')
        inputs = self.tokenizer([pair_text(record)]*len(candidates),
                                [c['description'] for c in candidates], padding=True,
                                truncation=False, return_tensors='pt')
        if inputs['input_ids'].shape[1] > MAX_LENGTH:
            raise ValueError('Student context exceeds frozen token limit; no truncation')
        return {k: v.to(self.device_name) for k, v in inputs.items()}, candidates

    def forward(self, inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def decide_record(self, record):
        self.eval()
        start = time.perf_counter()
        inputs, candidates = self.tokenize(record)
        logits = self(inputs)
        probabilities = logits.softmax(0).cpu().numpy()
        elapsed = time.perf_counter()-start
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite student scores')
        return {'choice': candidates[int(probabilities.argmax())]['id'],
                'probabilities': {c['id']: float(v) for c, v in zip(candidates, probabilities, strict=True)},
                'latency_ms': elapsed*1000, 'candidate_count': len(candidates),
                'pair_tokens': int(inputs['attention_mask'].sum().item()),
                'probability_semantics': 'uncalibrated softmax over candidate scores, not token probabilities'}

    def score(self, request):
        """Use the existing decision request; do not fabricate vocabulary token mass."""
        answers = []
        for q in request.questions:
            r = self.decide_record({'context': request.context, 'question': q.question,
                                   'candidates': [c.model_dump() for c in q.candidates]})
            answers.append({'id': q.id, **r})
        return answers

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save({k: v.detach().cpu() for k, v in self.state_dict().items()}, path / 'weights.pt')
        (path / 'config.json').write_text(json.dumps({'encoder': ENCODER_ID,
            'revision': ENCODER_REVISION, 'max_pair_tokens': MAX_LENGTH, 'calibration': 'none'}))

    @classmethod
    def load(cls, path, device='cpu'):
        path = Path(path)
        config = json.loads((path / 'config.json').read_text())
        if (config['encoder'], config['revision'], config['max_pair_tokens']) != (
                ENCODER_ID, ENCODER_REVISION, MAX_LENGTH):
            raise ValueError('Checkpoint architecture mismatch')
        model = cls(device=device)
        model.load_state_dict(torch.load(path / 'weights.pt', map_location='cpu', weights_only=True), strict=True)
        return model.eval()


def categorical_metrics(rows):
    if not rows:
        raise ValueError('Cannot score an empty evaluation')
    accuracy, nll, brier, classes = [], [], [], {}
    for row in rows:
        p, gold = row['probabilities'], row['gold']
        if gold not in p or not np.isfinite(list(p.values())).all() or any(v < 0 for v in p.values()):
            raise ValueError('Invalid probabilities or missing gold candidate')
        if not np.isclose(sum(p.values()), 1., atol=1e-5):
            raise ValueError('Probabilities must sum to one')
        if row['choice'] not in p or p[row['choice']] != max(p.values()):
            raise ValueError('Choice must be a highest-probability candidate')
        accuracy.append(row['choice'] == gold)
        classes.setdefault(gold, []).append(row['choice'] == gold)
        nll.append(-np.log(max(p[gold], 1e-12)))
        brier.append(sum((v-float(k == gold))**2 for k, v in p.items()))
    return {'examples': len(rows), 'accuracy': float(np.mean(accuracy)),
            'macro_class_accuracy': float(np.mean([np.mean(v) for v in classes.values()])),
            'class_accuracy': {k: float(np.mean(v)) for k, v in classes.items()}, 'nll': float(np.mean(nll)),
            'multiclass_brier': float(np.mean(brier)),
            'median_latency_ms': float(np.median([r['latency_ms'] for r in rows])),
            'p95_latency_ms': float(np.percentile([r['latency_ms'] for r in rows], 95))}
