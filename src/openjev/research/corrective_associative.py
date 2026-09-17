"""Bounded corrective-recurrence ablation with no additional trained parameters."""
import torch
from torch.nn import functional as F

from .learned_associative import LearnedAssociativeHead

VARIANTS = ('metric', 'feedforward', 'attractive_2', 'inhibitory_2', 'residual_2', 'residual_3')


class CorrectiveAssociativeHead(LearnedAssociativeHead):
    def __init__(self, mode, *args, **kwargs):
        if mode not in VARIANTS:
            raise ValueError('Unknown corrective variant')
        super().__init__('metric' if mode == 'metric' else 'feedforward', *args, **kwargs)
        self.variant = mode
        self.iterations = int(mode.rsplit('_', 1)[1]) if '_' in mode else 0
        self.correction_rate = .25

    def forward(self, x):
        q0 = self.initial_state(x)
        state = q0
        p = F.normalize(self.prototypes, dim=1)
        for _ in range(self.iterations):
            recall = torch.softmax(self.beta*(state @ p.T), dim=1) @ p
            if self.variant.startswith('attractive'):
                state = F.normalize(self.anchor*q0 + (1-self.anchor)*recall, dim=1)
            elif self.variant.startswith('inhibitory'):
                state = F.normalize(q0-self.correction_rate*recall, dim=1)
            else:
                state = F.normalize(state+self.correction_rate*(q0-recall), dim=1)
        return self.log_scale.exp().clamp(max=100.)*(state @ p.T)

    def cost(self):
        result = super().cost()
        result['matrix_multiply_accumulates_per_query'] += (
            self.iterations*2*self.project.out_features*len(self.prototypes))
        return result
