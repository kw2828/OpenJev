"""Learn a shared rule operator on explicit English-derived entity graphs."""
import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from openjev.research.bound_rules import BoundRuleNet, collate, ground
from openjev.research.text_distillation import digest, file_hash, write_new

PROTOCOL = {'version': 'bound-rules-v1', 'seeds': [17, 29, 43], 'epochs': 40,
    'world_batch': 64, 'learning_rate': .01, 'weight_decay': .0001, 'gradient_clip': 1., 'cpu_threads': 4,
    'variants': {'learned_1': [1, False], 'learned_6': [6, False], 'learned_16': [16, False],
                 'collapsed_16': [16, True]},
    'controls': {'facts_only': 0, 'fixed_1': 1, 'fixed_32': 32},
    'supervision': 'Question truth labels only; no supplied proof or logical representation',
    'architecture_prior': 'Handwritten controlled-English parser, exact binding, negation, max-OR and inhibition',
    'learned_component': '225-parameter shared conjunction MLP; random initialization; no truth-table pretraining',
    'training': 'Identical 2000 worlds and 19809 gold questions; same world shuffles per seed; final epoch',
    'primary_metric': 'Equal weight to four gold-label/question-negation groups on fresh depth-3-5 development worlds',
    'challenge_seed': 48597, 'challenge_pairs': 240, 'challenge_depths': [3, 4, 5, 6, 7, 8],
    'continuation': {'min_macro_accuracy': .95, 'max_macro_gap_to_fixed_solver': .01,
                     'min_gain_over_collapsed': .10, 'min_counterfactual_pair_success': .95,
                     'interpretation': 'Functional candidate only; matching a classical solver is not novelty'},
}


def signature():
    import openjev.research.bound_rules as architecture
    return {'driver': file_hash(__file__), 'architecture': file_hash(architecture.__file__)}


def load(packet):
    manifest = json.loads((packet/'manifest.json').read_text())
    worlds, seen = {}, set()
    if manifest['version'] != 'bound-rules-data-v1':
        raise ValueError('Need fresh bound-rule development packet')
    for part in ('train', 'dev_in', 'dev_shift'):
        if file_hash(packet/f'{part}.json') != manifest['parts'][part]['file_sha256']:
            raise ValueError('Packet changed')
        w = json.loads((packet/f'{part}.json').read_text())
        keys = {x['world_sha256'] for x in w}
        if seen & keys or len(keys) != len(w):
            raise ValueError('Overlapping worlds')
        seen.update(keys)
        worlds[part] = w
    return manifest, worlds


def challenge():
    rng = random.Random(PROTOCOL['challenge_seed'])
    predicates = ['red', 'blue', 'green', 'warm', 'quiet', 'smart', 'nice', 'kind', 'tall', 'young', 'rough', 'big']
    rows = []
    for i in range(PROTOCOL['challenge_pairs']):
        depth = PROTOCOL['challenge_depths'][i % 6]
        chain = rng.sample(predicates, depth+1)
        target, other, distractor = rng.sample(['Mira', 'Nemi', 'Zora', 'Fora', 'Luma', 'Sora'], 3)
        relation = rng.choice(['visits', 'likes', 'needs'])
        kind = 'inhibition' if (i//6) % 2 else 'support'
        if kind == 'support':
            rules = [f'If someone is {chain[0]} and they {relation[:-1]} {other} then they are {chain[1]}.']
        else:
            rules = [f'All {chain[0]} people are {chain[1]}.']
        rules += [f'All {a} people are {b}.' for a, b in itertools.pairwise(chain[1:])]
        if kind == 'inhibition':
            rules.append(f'If someone {relation} {other} then they are not {chain[-1]}.')
        rng.shuffle(rules)
        contexts = [f'{target} is {chain[0]}. {target} {relation} {other}. '+ ' '.join(rules),
                    f'{target} is {chain[0]}. {other} {relation} {target}. '+ ' '.join(rules)]
        for changed, context in enumerate(contexts):
            truth = (changed == 0) == (kind == 'support')
            questions = [{'text': f'{target} is {chain[-1]}.', 'gold': 'true' if truth else 'false'},
                         {'text': f'{target} is not {chain[-1]}.', 'gold': 'false' if truth else 'true'}]
            rows.append({'context': context, 'questions': questions, 'pair': i, 'changed': changed,
                         'kind': kind, 'depth': depth,
                         'irrelevant_context': context+f' {distractor} is {chain[0]}. {distractor} {relation} {other}.'})
    return rows


def graphs(worlds, collapsed=False, field='context'):
    return [ground(w[field], [q['text'] for q in w['questions']], collapsed) for w in worlds]


def targets(worlds):
    return torch.tensor([int(q['gold'] == 'true') for w in worlds for q in w['questions']], dtype=torch.float32)


def metrics(p, worlds):
    gold = targets(worlds)
    neg = torch.tensor(['not' in q['text'].casefold().split() for w in worlds for q in w['questions']])
    good = (p >= .5) == (gold >= .5)
    groups = {}
    for label in (0, 1):
        for n in (False, True):
            mask = (gold == label) & (neg == n)
            if not mask.any():
                raise ValueError('All four groups required for primary metric')
            groups[f'{label}:{n}'] = {'n': int(mask.sum()), 'accuracy': float(good[mask].float().mean())}
    return {'accuracy': float(good.float().mean()), 'macro_accuracy': float(np.mean([g['accuracy'] for g in groups.values()])),
            'nll': float(F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), gold)),
            'brier': float(((p-gold)**2).mean()), 'groups': groups}


@torch.no_grad()
def predict(model, g):
    output, change = [], []
    for i in range(0, len(g), PROTOCOL['world_batch']):
        p, delta = model(collate(g[i:i+PROTOCOL['world_batch']]))
        output.append(p)
        change.append(delta)
    return torch.cat(output), torch.cat(change)


@torch.no_grad()
def evaluation(model, parts, graph_parts, stress, stress_graphs):
    result, arrays = {}, {}
    for part in ('dev_in', 'dev_shift'):
        p, change = predict(model, graph_parts[part])
        result[part] = metrics(p, parts[part]) | {'queries_with_last_step_change_over_001': int((change > .01).sum())}
        arrays[part] = p.numpy()
    p, _ = predict(model, stress_graphs[0])
    extra, _ = predict(model, stress_graphs[1])
    correct = (p >= .5) == (targets(stress) >= .5)
    result['challenge'] = metrics(p, stress) | {
        'counterfactual_pair_success': float(correct.reshape(-1, 4).all(1).float().mean()),
        'irrelevant_prediction_agreement': float(((p >= .5) == (extra >= .5)).float().mean()),
        'max_irrelevant_probability_change': float((p-extra).abs().max())}
    arrays['challenge'], arrays['challenge_irrelevant'] = p.numpy(), extra.numpy()
    return result, arrays


def freeze(args):
    manifest, _ = load(args.packet)
    write_new(args.out, {'protocol': PROTOCOL, 'code': signature(), 'packet_sha256': digest(manifest),
                         'challenge_sha256': digest(challenge()),
                         'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})


def checked(args):
    plan = json.loads(args.plan.read_text())
    manifest, parts = load(args.packet)
    if (plan['protocol'] != PROTOCOL or plan['code'] != signature() or plan['packet_sha256'] != digest(manifest) or
            plan['challenge_sha256'] != digest(challenge())):
        raise ValueError('Frozen source, protocol, data or challenge changed')
    torch.set_num_threads(PROTOCOL['cpu_threads'])
    return plan, parts


def run(args):
    plan, parts = checked(args)
    if args.out.exists():
        raise ValueError('Fresh run directory required')
    args.out.mkdir(parents=True)
    write_new(args.out/'started.json', {'plan_sha256': digest(plan)})
    t = time.perf_counter()
    full = {part: graphs(w) for part, w in parts.items()}
    collapsed = {part: graphs(w, True) for part, w in parts.items()}
    stress = challenge()
    stress_graphs = {False: (graphs(stress), graphs(stress, field='irrelevant_context')),
                    True: (graphs(stress, True), graphs(stress, True, 'irrelevant_context'))}
    preparation_seconds = time.perf_counter()-t
    write_new(args.out/'prepared.json', {'plan_sha256': digest(plan), 'english_parse_and_ground_seconds': preparation_seconds,
        'coverage': {p: {'worlds': len(v), 'parsed': len(full[p]), 'questions': len(targets(v))} for p, v in parts.items()},
        'training_graph_atoms': sum(len(g['atoms']) for g in full['train']),
        'training_ground_rules': sum(len(g['heads']) for g in full['train'])})
    # Only now, after the plan is frozen, evaluate fixed controls on fresh worlds.
    for name, steps in PROTOCOL['controls'].items():
        model = BoundRuleNet('fixed', steps)
        m, predictions = evaluation(model, parts, full, stress, stress_graphs[False])
        path = args.out/f'{name}-probabilities.npz'
        np.savez(path, **predictions)
        write_new(args.out/f'{name}.json', {'name': name, 'metrics': m, 'plan_sha256': digest(plan),
            'probabilities_sha256': file_hash(path), 'learned_parameters': 0})
    write_new(args.out/'challenge.json', stress)
    for seed in PROTOCOL['seeds']:
        for name, (steps, collapse) in PROTOCOL['variants'].items():
            gp = collapsed if collapse else full
            model = BoundRuleNet('learned', steps, seed)
            opt = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['learning_rate'], weight_decay=PROTOCOL['weight_decay'])
            rng = np.random.default_rng(seed)
            history, start = [], time.perf_counter()
            for epoch in range(PROTOCOL['epochs']):
                order = rng.permutation(len(parts['train']))
                losses, counts = 0., 0
                for start_index in range(0, len(order), PROTOCOL['world_batch']):
                    indices = order[start_index:start_index+PROTOCOL['world_batch']]
                    g = collate([gp['train'][i] for i in indices])
                    gold = targets([parts['train'][i] for i in indices])
                    opt.zero_grad(set_to_none=True)
                    p, _ = model(g)
                    loss = F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), gold)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL['gradient_clip'], error_if_nonfinite=True)
                    opt.step()
                    losses += float(loss.detach())*len(gold)
                    counts += len(gold)
                history.append({'epoch': epoch+1, 'loss': losses/counts, 'updates': int(np.ceil(len(order)/PROTOCOL['world_batch']))})
                if (epoch+1) % 10 == 0:
                    print(json.dumps({'seed': seed, 'variant': name, **history[-1]}), flush=True)
            seconds = time.perf_counter()-start
            model.eval()
            m, predictions = evaluation(model, parts, gp, stress, stress_graphs[collapse])
            path = args.out/f'{name}-{seed}.pt'
            torch.save(model.state_dict(), path)
            np.savez(args.out/f'{name}-{seed}-probabilities.npz', **predictions)
            result = {'name': name, 'seed': seed, 'plan_sha256': digest(plan), 'metrics': m, 'history': history,
                'training_seconds': seconds, 'learned_parameters': sum(p.numel() for p in model.parameters()),
                'checkpoint_sha256': file_hash(path),
                'probabilities_sha256': file_hash(args.out/f'{name}-{seed}-probabilities.npz')}
            write_new(args.out/f'{name}-{seed}.json', result)
            print(json.dumps({'seed': seed, 'variant': name, 'training_seconds': seconds,
                              'macro_accuracy': m['dev_shift']['macro_accuracy'], 'challenge': m['challenge']}), flush=True)
    write_new(args.out/'completed.json', {'plan_sha256': digest(plan), 'fits': 12, 'controls': 3, 'status': 'completed'})


def report(args):
    plan, parts = checked(args)
    if json.loads((args.runs/'completed.json').read_text()) != {
            'plan_sha256': digest(plan), 'fits': 12, 'controls': 3, 'status': 'completed'}:
        raise ValueError('Incomplete fit family')
    stress = challenge()
    methods = {}
    for name in list(PROTOCOL['controls'])+list(PROTOCOL['variants']):
        fits = []
        for seed in [None] if name in PROTOCOL['controls'] else PROTOCOL['seeds']:
            prefix = name if seed is None else f'{name}-{seed}'
            fit = json.loads((args.runs/f'{prefix}.json').read_text())
            if (fit['plan_sha256'] != digest(plan) or fit.get('seed') != seed or fit.get('name') != name or
                    fit['probabilities_sha256'] != file_hash(args.runs/f'{prefix}-probabilities.npz')):
                raise ValueError('Fit identity mismatch')
            if seed is not None and fit['checkpoint_sha256'] != file_hash(args.runs/f'{prefix}.pt'):
                raise ValueError('Checkpoint hash mismatch')
            with np.load(args.runs/f'{prefix}-probabilities.npz', allow_pickle=False) as data:
                for part in ('dev_in', 'dev_shift', 'challenge'):
                    replay = metrics(torch.from_numpy(data[part]), stress if part == 'challenge' else parts[part])
                    if any(replay[k] != fit['metrics'][part][k] for k in replay):
                        raise ValueError('Metric replay mismatch')
                p, other = data['challenge'], data['challenge_irrelevant']
                correct = (p >= .5) == (targets(stress).numpy() >= .5)
                # Torch rounding is part of the original serialized metric.
                pair_success = float(torch.from_numpy(correct.reshape(-1, 4).all(1)).float().mean())
                if pair_success != fit['metrics']['challenge']['counterfactual_pair_success']:
                    raise ValueError('Challenge pair replay mismatch')
                if float(np.abs(p-other).max()) != fit['metrics']['challenge']['max_irrelevant_probability_change']:
                    raise ValueError('Invariance replay mismatch')
            fits.append(fit)
        methods[name] = {'fits': fits, 'means': {part: {metric: float(np.mean([f['metrics'][part][metric] for f in fits]))
            for metric in ('accuracy', 'macro_accuracy', 'nll', 'brier')} for part in ('dev_in', 'dev_shift', 'challenge')}}
    selected = max(('learned_6', 'learned_16'), key=lambda m: methods[m]['means']['dev_shift']['macro_accuracy'])
    score = methods[selected]['means']['dev_shift']['macro_accuracy']
    gate = PROTOCOL['continuation']
    pair = float(np.mean([f['metrics']['challenge']['counterfactual_pair_success'] for f in methods[selected]['fits']]))
    checks = {'macro_accuracy': score >= gate['min_macro_accuracy'],
        'fixed_solver_gap': methods['fixed_32']['means']['dev_shift']['macro_accuracy']-score <= gate['max_macro_gap_to_fixed_solver'],
        'binding_gain': score-methods['collapsed_16']['means']['dev_shift']['macro_accuracy'] >= gate['min_gain_over_collapsed'],
        'counterfactual_pairs': pair >= gate['min_counterfactual_pair_success']}
    write_new(args.out, {'status': 'fresh_development_and_constructed_challenge', 'plan': plan, 'methods': methods,
        'selected': selected, 'continuation_checks': checks, 'functional_gate_passed': all(checks.values()),
        'novelty_established': False, 'confirmation_executed': False,
        'limits': ['Handwritten English grammar and explicit logical operations provide strong prior knowledge.',
                   'Fixed symbolic solver is a comparator, not learned-model efficacy.',
                   'Fresh development worlds support candidate selection, not independent confirmation.',
                   'Constructed challenges share the controlled grammar and supplied semantics.',
                   'No claim of general-language understanding, novel reasoning algorithm or Astra supervision.']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'run', 'report'])
    for name in ('packet', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('plan', 'runs'):
        p.add_argument('--'+name, type=Path)
    args = p.parse_args()
    {'freeze': freeze, 'run': run, 'report': report}[args.stage](args)
