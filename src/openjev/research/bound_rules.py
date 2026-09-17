"""Explicit entity binding for RuleTaker's controlled English, not general NLP.

The parser reads only context/question strings. It never accepts dataset proofs
or supplied logical forms. A fixed rule operator is an essential strong control.
"""
import re
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, order=True)
class Literal:
    predicate: str
    arguments: tuple[str, ...]
    negative: bool = False

    @property
    def atom(self):
        return self.predicate, self.arguments

    def bind(self, entity):
        return Literal(self.predicate, tuple(entity if a == '$x' else a for a in self.arguments), self.negative)


@dataclass(frozen=True)
class Rule:
    body: tuple[Literal, ...]
    head: Literal


def entity(text):
    text = text.strip().casefold()
    text = text.removeprefix('the ')
    if text in ('someone', 'something', 'they', 'it'):
        return '$x'
    if not re.fullmatch(r'[a-z]+(?: [a-z]+)*', text):
        raise ValueError('Unsupported entity: '+text)
    return text


def literal(text, implicit_subject=None):
    text = ' '.join(text.strip().rstrip('.').casefold().split())
    m = re.fullmatch(r'(.+?) (?:is|are) (not )?([a-z]+)', text)
    if m:
        return Literal('is_'+m[3], (entity(m[1]),), bool(m[2]))
    m = re.fullmatch(r'(.+?) (does not |do not )?(like[s]?|chase[s]?|eat[s]?|see[s]?|visit[s]?|need[s]?) (.+)', text)
    if m:
        verb = m[3] if m[3].endswith('s') else m[3]+'s'
        return Literal(verb, (entity(m[1]), entity(m[4])), bool(m[2]))
    if implicit_subject is not None and re.fullmatch(r'(?:not )?[a-z]+', text):
        return Literal('is_'+text.removeprefix('not '), (implicit_subject,), text.startswith('not '))
    raise ValueError('Unsupported English literal: '+text)


def parse(context, questions):
    facts, rules = [], []
    for sentence in context.split('.'):
        text = ' '.join(sentence.strip().casefold().split())
        if not text:
            continue
        if text.startswith('if '):
            before, after = text[3:].split(' then ')
            body = []
            for clause in before.split(' and '):
                body.append(literal(clause, body[0].arguments[0] if body else None))
            rules.append(Rule(tuple(body), literal(after)))
        elif ' are ' in text and re.search(r'\b(?:people|things)\b', text):
            before, after = text.split(' are ')
            before = re.sub(r'^(?:all )?', '', before)
            before = re.sub(r' (?:people|things)$', '', before)
            body = tuple(literal(x.strip(), '$x') for x in re.split(r', | and ', before))
            rules.append(Rule(body, literal('something is '+after)))
        else:
            facts.append(literal(text))
    queries = [literal(q) for q in questions]
    if any('$x' in f.arguments for f in facts+queries):
        raise ValueError('Facts and questions must name concrete entities')
    if not facts and not rules:
        raise ValueError('Empty rulebase')
    return facts, rules, queries


def ground(context, questions, collapse_entities=False):
    facts, rules, queries = parse(context, questions)
    all_literals = facts+queries+[a for r in rules for a in (*r.body, r.head)]
    entities = sorted({a for lit in all_literals for a in lit.arguments if a != '$x'})
    grounded = []
    for rule in rules:
        variables = any('$x' in a.arguments for a in (*rule.body, rule.head))
        for e in entities if variables else [None]:
            grounded.append(Rule(tuple(a.bind(e) for a in rule.body), rule.head.bind(e)))
    if collapse_entities:
        def collapse(lit):
            return Literal(lit.predicate, tuple('$entity' for _ in lit.arguments), lit.negative)
        facts, queries = [collapse(a) for a in facts], [collapse(a) for a in queries]
        grounded = [Rule(tuple(collapse(a) for a in r.body), collapse(r.head)) for r in grounded]
    atoms = sorted({a.atom for a in facts+queries+[a for r in grounded for a in (*r.body, r.head)]})
    lookup = {a: i for i, a in enumerate(atoms)}
    positive, negative = torch.zeros(len(atoms)), torch.zeros(len(atoms))
    for f in facts:
        (negative if f.negative else positive)[lookup[f.atom]] = 1
    return {'atoms': atoms, 'facts': positive, 'inhibitors': negative,
        'bodies': [[lookup[a.atom] for a in r.body] for r in grounded],
        'body_negative': [[a.negative for a in r.body] for r in grounded],
        'heads': [lookup[r.head.atom] for r in grounded], 'head_negative': [r.head.negative for r in grounded],
        'query': [lookup[a.atom] for a in queries], 'query_negative': [a.negative for a in queries]}


def collate(graphs, max_body=4):
    """Disjoint union of rule graphs; no padded cross-world message edges."""
    facts, inhibitors, bodies, signs, masks, heads, head_signs, query, query_signs = [], [], [], [], [], [], [], [], []
    offset = 0
    for g in graphs:
        facts.append(g['facts'])
        inhibitors.append(g['inhibitors'])
        for body, neg in zip(g['bodies'], g['body_negative'], strict=True):
            if not 1 <= len(body) <= max_body:
                raise ValueError('Unsupported rule arity')
            bodies.append([a+offset for a in body]+[0]*(max_body-len(body)))
            signs.append(neg+[False]*(max_body-len(body)))
            masks.append([True]*len(body)+[False]*(max_body-len(body)))
        heads.extend(a+offset for a in g['heads'])
        head_signs.extend(g['head_negative'])
        query.extend(a+offset for a in g['query'])
        query_signs.extend(g['query_negative'])
        offset += len(g['atoms'])
    return {'facts': torch.cat(facts), 'inhibitors': torch.cat(inhibitors),
        'body': torch.tensor(bodies, dtype=torch.int64).reshape(-1, max_body),
        'body_negative': torch.tensor(signs, dtype=torch.bool).reshape(-1, max_body),
        'body_mask': torch.tensor(masks, dtype=torch.bool).reshape(-1, max_body),
        'head': torch.tensor(heads, dtype=torch.int64), 'head_negative': torch.tensor(head_signs, dtype=torch.bool),
        'query': torch.tensor(query, dtype=torch.int64), 'query_negative': torch.tensor(query_signs, dtype=torch.bool)}


class BoundRuleNet(nn.Module):
    """Shared learned conjunction, explicit signed edges and inhibitory aggregation.

OR (max), negation (1-p), entity unification and inhibitory gating are supplied
inductive biases. Only the conjunction operator is learned, so this cannot be
described as learning a theorem prover from scratch.
"""
    def __init__(self, mode='learned', steps=16, seed=17):
        super().__init__()
        if mode not in ('learned', 'fixed') or steps < 0:
            raise ValueError('Invalid rule operator')
        torch.manual_seed(seed)
        self.operator = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 1))
        self.mode, self.steps = mode, steps

    def forward(self, graph):
        state = graph['facts']*(1-graph['inhibitors'])
        previous = state
        for _ in range(self.steps):
            if not len(graph['head']):
                break
            values = state[graph['body']]
            values = torch.where(graph['body_negative'], 1-values, values)
            values = values.masked_fill(~graph['body_mask'], 1.)
            if self.mode == 'fixed':
                activation = values.min(1).values
            else:
                # Sort makes body order irrelevant; count distinguishes padding.
                features = torch.cat((values.sort(1).values, graph['body_mask'].sum(1, keepdim=True)/4.), 1)
                activation = self.operator(features).sigmoid().squeeze(1)
            positive = graph['facts'].scatter_reduce(0, graph['head'],
                activation*(~graph['head_negative']), reduce='amax', include_self=True)
            negative = graph['inhibitors'].scatter_reduce(0, graph['head'],
                activation*graph['head_negative'], reduce='amax', include_self=True)
            previous, state = state, positive*(1-negative)
        p = state[graph['query']]
        p = torch.where(graph['query_negative'], 1-p, p)
        change = (state-previous).abs()[graph['query']]
        return p, change
