"""Parameter-matched learned metric, feedforward and tied associative heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

MODES = ('metric', 'feedforward', 'recurrent', 'sparse_recurrent')


class LearnedAssociativeHead(nn.Module):
    def __init__(self, mode, input_dim=384, latent_dim=128, classes=150, seed=17,
                 steps=2, anchor=.5, beta=10., top_k=8):
        super().__init__()
        if mode not in MODES or steps < 0 or not 0 <= anchor <= 1 or not 1 <= top_k <= classes:
            raise ValueError('Invalid head configuration')
        torch.manual_seed(seed)
        self.project = nn.Linear(input_dim, latent_dim)
        self.prototypes = nn.Parameter(torch.randn(classes, latent_dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.)))
        self.mode, self.steps, self.anchor, self.beta, self.top_k = mode, steps, anchor, beta, top_k

    def initial_state(self, x):
        h = self.project(x)
        return F.normalize(h if self.mode == 'metric' else F.gelu(h), dim=-1)

    @torch.no_grad()
    def initialize_prototypes(self, x, y):
        if sorted(set(y.tolist())) != list(range(len(self.prototypes))):
            raise ValueError('Initialization needs training examples from every class')
        h = self.initial_state(x)
        means = torch.stack([h[y == k].mean(0) for k in range(len(self.prototypes))])
        self.prototypes.copy_(F.normalize(means, dim=1))

    def forward(self, x):
        q0 = self.initial_state(x)
        state = q0
        p = F.normalize(self.prototypes, dim=1)
        if self.mode in ('recurrent', 'sparse_recurrent'):
            for _ in range(self.steps):
                similarity = state @ p.T
                if self.mode == 'sparse_recurrent':
                    values, indices = similarity.topk(self.top_k, dim=1)
                    weights = torch.softmax(self.beta*values, dim=1)
                    recall = torch.einsum('bk,bkd->bd', weights, p[indices])
                else:
                    recall = torch.softmax(self.beta*similarity, dim=1) @ p
                state = F.normalize(self.anchor*q0 + (1-self.anchor)*recall, dim=1)
        return self.log_scale.exp().clamp(max=100.)*(state @ p.T)

    def cost(self):
        inputs, latent = self.project.in_features, self.project.out_features
        classes = len(self.prototypes)
        macs = inputs*latent + latent*classes
        if self.mode == 'recurrent':
            macs += self.steps*2*latent*classes
        elif self.mode == 'sparse_recurrent':
            macs += self.steps*latent*(classes+self.top_k)
        return {'trainable_parameters': sum(p.numel() for p in self.parameters()),
                'matrix_multiply_accumulates_per_query': macs,
                'excludes': 'encoder, rejection gate, bias, normalization, activation, softmax, top-k and data movement'}
