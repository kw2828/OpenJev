"""Prototype-gated associative recall; research classifier, not a new Hopfield theorem."""
import torch
from torch.nn import functional as F


class AssociativeRouter:
    def __init__(self, memory, labels, classes, top_k=32, beta=30., anchor=0.5, steps=2):
        if memory.ndim != 2 or len(memory) != len(labels) or len(memory) < top_k:
            raise ValueError('Invalid memory shape or top-k')
        if not torch.isfinite(memory).all() or (memory.norm(dim=1) == 0).any():
            raise ValueError('Memory must be finite and nonzero')
        if not 0 <= anchor <= 1 or beta <= 0 or steps < 1:
            raise ValueError('Invalid recurrence settings')
        self.memory = F.normalize(memory.float(), dim=1)
        self.labels = labels.long()
        if sorted(set(labels.tolist())) != list(range(classes)):
            raise ValueError('Every contiguous class must have memory examples')
        self.classes, self.top_k, self.beta, self.anchor, self.steps = classes, top_k, beta, anchor, steps
        self.prototypes = F.normalize(torch.stack([self.memory[labels == k].mean(0) for k in range(classes)]), dim=1)

    def prototype(self, query):
        query = F.normalize(query, dim=1)
        scores = query @ self.prototypes.T
        best = scores.topk(2, dim=1).values
        return scores, best[:, 0], best[:, 0] - best[:, 1]

    def class_scores(self, similarities):
        scores = torch.full((len(similarities), self.classes), -torch.inf, dtype=similarities.dtype)
        return scores.scatter_reduce_(1, self.labels.expand(len(similarities), -1), similarities,
                                      reduce='amax', include_self=True)

    def recall(self, query, recurrent=False):
        """Refine against a fixed sparse neighborhood, then reread the whole memory.

        Confidence retains the original query's support: attraction to a stored
        exemplar must not silently turn an unfamiliar query into a familiar one.
        """
        query = F.normalize(query, dim=1)
        similarities = query @ self.memory.T
        original_support = similarities.max(1).values
        if not recurrent:
            return self.class_scores(similarities), original_support
        indices = similarities.topk(self.top_k, dim=1).indices
        neighbors = self.memory[indices]
        state = query
        for _ in range(self.steps):
            weights = torch.softmax(self.beta * torch.einsum('bd,bkd->bk', state, neighbors), dim=1)
            retrieved = torch.einsum('bk,bkd->bd', weights, neighbors)
            state = F.normalize(self.anchor * query + (1-self.anchor) * retrieved, dim=1)
        similarities = state @ self.memory.T
        return self.class_scores(similarities), torch.minimum(original_support, similarities.max(1).values)

    def predict(self, query, mode, gate_threshold=None):
        if mode not in ('prototype', 'nearest', 'recurrent', 'gated_nearest', 'gated_recurrent'):
            raise ValueError('Unknown architecture')
        scores, support, margin = self.prototype(query)
        if mode == 'prototype':
            active = torch.zeros(len(query), dtype=torch.bool)
        elif mode.startswith('gated_'):
            if gate_threshold is None:
                raise ValueError('Frozen gate threshold required')
            active = margin <= gate_threshold
        else:
            active = torch.ones(len(query), dtype=torch.bool)
        if active.any():
            scores[active], support[active] = self.recall(query[active], recurrent='recurrent' in mode)
        passes = int(active.sum()) * (2 if 'recurrent' in mode else 1)
        return scores, support, active, passes


def conformal_threshold(probabilities, gold, alpha=.1):
    """ID-only marginal prediction sets; does not guarantee detection of unknown intents."""
    import math
    if not 0 < alpha < 1 or len(probabilities) != len(gold) or not len(gold):
        raise ValueError('Invalid calibration sample')
    if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
            or not torch.allclose(probabilities.sum(1), torch.ones(len(gold)), atol=1e-5)):
        raise ValueError('Invalid probability distribution')
    scores = 1-probabilities[torch.arange(len(gold)), gold]
    rank = math.ceil((len(gold)+1)*(1-alpha))
    return 1. if rank > len(gold) else float(scores.sort().values[rank-1])
