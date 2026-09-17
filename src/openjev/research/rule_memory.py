"""English sentence memory heads, with tied reads and optional coverage inhibition.

These are research baselines built on frozen transformer features, not a new
language-model architecture. No labels, depths or proof graphs enter forward().
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

MODES = ('question_only', 'mean_context', 'single_read', 'recurrent_3', 'recurrent_6', 'coverage_3')
HOPS = {'question_only': 0, 'mean_context': 1, 'single_read': 1,
        'recurrent_3': 3, 'recurrent_6': 6, 'coverage_3': 3}


def english_inputs(world):
    """Allowlist English only; retain duplicate clauses and their original order."""
    clauses = [s.strip()+'.' for s in world['context'].split('.') if s.strip()]
    if not clauses or not world['questions']:
        raise ValueError('Empty English world or questions')
    return clauses, [q['text'] for q in world['questions']]


class RuleMemoryHead(nn.Module):
    def __init__(self, mode, input_dim=384, latent_dim=128, seed=17):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown memory head')
        torch.manual_seed(seed)
        self.mode, self.hops = mode, HOPS[mode]
        self.query = nn.Linear(input_dim, latent_dim)
        self.key = nn.Linear(input_dim, latent_dim)
        self.value = nn.Linear(input_dim, latent_dim)
        self.update = nn.Linear(latent_dim, latent_dim)
        self.query_norm = nn.LayerNorm(latent_dim)
        self.key_norm = nn.LayerNorm(latent_dim)
        self.state_norm = nn.LayerNorm(latent_dim)
        self.classifier = nn.Sequential(nn.Linear(3*latent_dim, latent_dim), nn.GELU(),
                                        nn.Linear(latent_dim, 2))

    def forward(self, question, memory, mask):
        q0 = self.query_norm(self.query(question))
        state = q0
        if self.mode != 'question_only':
            if mask.dtype != torch.bool or mask.shape != memory.shape[:2] or not mask.any(1).all():
                raise ValueError('Each world needs at least one unmasked sentence')
            values = self.value(memory)
            if self.mode == 'mean_context':
                recall = (values*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True)
                state = self.state_norm(state+F.gelu(self.update(recall)))
            else:
                keys = self.key_norm(self.key(memory))
                coverage = torch.zeros_like(mask, dtype=question.dtype)
                for _ in range(self.hops):
                    scores = torch.einsum('bd,bnd->bn', state, keys)/math.sqrt(keys.shape[-1])
                    if self.mode == 'coverage_3':
                        scores = scores-coverage
                    weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
                    recall = torch.einsum('bn,bnd->bd', weights, values)
                    state = self.state_norm(state+F.gelu(self.update(recall)))
                    coverage = coverage+weights
        return self.classifier(torch.cat((q0, state, q0*state), dim=-1))

    def cost(self, slots):
        """Dense MAC estimate per query including repeated world projections."""
        d, h = self.query.in_features, self.query.out_features
        inactive = ()
        macs = d*h + 3*h*h + 2*h
        if self.mode == 'question_only':
            inactive = ('key.', 'value.', 'update.', 'key_norm.', 'state_norm.')
        elif self.mode == 'mean_context':
            inactive = ('key.', 'key_norm.')
            macs += slots*d*h+h*h
        else:
            macs += 2*slots*d*h+self.hops*(2*slots*h+h*h)
        return {'allocated_parameters': sum(p.numel() for p in self.parameters()),
                'active_parameters': sum(p.numel() for n, p in self.named_parameters()
                                         if not n.startswith(inactive)),
                'matrix_macs_per_query': macs, 'context_slots': slots, 'memory_reads': self.hops,
                'excludes': 'encoder, bias, normalization, activations, softmax, elementwise operations, data movement'}
