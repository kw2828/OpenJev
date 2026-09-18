# SPDX-License-Identifier: GPL-3.0-only
"""Native frozen-feature probes and bounded artificial-update replay for pin heads."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import chess
import torch
from torch.nn import functional as F

import chess_pin_factor_screen as inputs
import chess_wldn_preflight as parent
import chess_wldn_interventions as diagnostic
from openjev.research.chess_pin_factors import candidate_factors
from openjev.research.chess_pin_factor_head import ARMS, VERSION, ChessPinFactorHead, pack_factors

ROOT = parent.ROOT
source = parent.source
INPUT_PLAN = 'evidence/chess-pin-factors-v1/protocol/plan.json'
INPUT_AUDIT = 'evidence/chess-pin-factors-v1/audit/receipt.json'
CODE = ['src/openjev/research/chess_pin_factor_head.py', 'tests/test_chess_pin_factor_head.py',
        'scripts/chess_pin_factor_preflight.py', 'tests/test_chess_pin_factor_preflight.py']
PROTOCOL = {
    'version': VERSION, 'scope': 'Bounded engineering, no teacher-quality experiment.',
    'arms': list(ARMS), 'parameters': 16658, 'backbone_seed': 97,
    'roots': 128, 'selection': 'First128 original training roots, all legal candidates, no label-dependent selection.',
    'probe_seeds': [211, 223, 227], 'root_seed_arm_checks': 768,
    'projection': 'Fresh output zero must preserve backbone exactly; probes replace it with Normal(0,.1), generator900000+head_seed.',
    'probe_dtype': 'float32 actual frozen backbone features', 'score_tolerance': 1e-5,
    'probes': 'Compare one batched call with a full individual call for every legal candidate, for all3 seeds and2 arms. Audit uses independently extracted factors and factor-by-factor pooling, and compares every saved batched score.',
    'head_seed': 1197, 'updates_per_arm': 3, 'total_updates': 6,
    'training_dtype': 'float64 head on cast frozen float32 features; this is not float64 backbone inference.',
    'target': 'First legal move for each root. Historical tensor packs contain labels but targets/values are never used as supervision or quality outcomes.',
    'optimizer': 'Adam lr.001 betas(.9,.999) eps1e-8 weight_decay0;norm clip1.',
    'update_replay_tolerance': 1e-9,
    'replay': 'Initial tensors exact; reference-replayed final tensors, loss and gradient norms within absolute1e-9. Different batched/scalar affine arithmetic can round differently; no bitwise training-equivalence claim. All four factor parameter blocks must change after3 updates in both arms.',
    'input_checks': 'Reconstruct all128 native root/child attack graphs; verify cached masks, menus and FENs. Independently reconstruct every pin factor. Candidate packing order must match action/graph order.',
    'runtime': 'CPU2 deterministic algorithms, shared host with existing union training.',
    'time_cap_seconds': 900,
    'limits': 'Factor stages share neural layers with the production head. Reference independently implements factor detection, coordinate mapping, status assignment and pooling. Engineering checks neither reproduce a prior FGNN benchmark nor establish chess quality, novelty, generic interactions, float32 optimization parity or inference-speed advantage.'}


def read(path):
    return json.loads(Path(path).read_text())


def signature():
    pin = read(ROOT/INPUT_PLAN)
    if pin != inputs.signature():
        raise ValueError('Frozen pin input protocol changed')
    audit = read(ROOT/INPUT_AUDIT)
    execution = ROOT/'runs/chess-pin-factors-v1/execution'
    if (audit['status'] != 'passed' or not audit['all_root_candidate_witnesses_exact']
            or audit['summary']['roots'] != 1024 or audit['summary']['candidates'] != 30930
            or audit['plan_sha256'] != source.file_hash(ROOT/INPUT_PLAN)
            or audit['completed_sha256'] != source.file_hash(execution/'completed.json')):
        raise ValueError('Pin input replay identity changed')
    diagnostic.manifest(execution, source.file_hash(ROOT/INPUT_PLAN))
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'native_parent_signature': parent.signature(), 'pin_input_signature': pin,
            'input_plan_sha256': source.file_hash(ROOT/INPUT_PLAN),
            'input_audit_sha256': source.file_hash(ROOT/INPUT_AUDIT)}


def deadline(begin):
    if time.monotonic()-begin > PROTOCOL['time_cap_seconds']:
        raise TimeoutError('Frozen fifteen-minute pin-head engineering ceiling exceeded')


def single_args(args, root, slot, compact):
    return [args[0][root:root+1], args[1][root:root+1, slot:slot+1],
            args[2][root:root+1, slot:slot+1], torch.ones(1, 1, dtype=torch.bool),
            args[4][root:root+1, slot:slot+1], args[5][root:root+1], args[6][compact:compact+1]]


def single_factors(factors, compact):
    result = factors[factors[:, 0] == compact].clone()
    result[:, 0] = 0
    return result


def native_inputs(graphs, rows, reference):
    batch = graphs.batch(torch.arange(128))
    boards = [chess.Board(row['fen']) for row in rows]
    native = source.candidate_graphs(boards)
    if (batch['fens'] != [row['fen'] for row in rows]
            or batch['menus'] != list(map(tuple, native['menus']))
            or not torch.equal(batch['root'], native['root'])
            or not torch.equal(batch['children'], native['children'][native['mask']])):
        raise AssertionError('Native graphs differ from frozen cache')
    records = [candidate_factors(board, reference=reference) for board in boards]
    if [tuple(c['uci'] for c in r['candidates']) for r in records] != batch['menus']:
        raise AssertionError('Factor candidate order differs')
    factors = pack_factors(records)
    summary = inputs.summarize(records)
    if len(factors) == 0:
        raise AssertionError('No factor pathway exercised')
    summary['factor_rows'] = len(factors)
    summary['factor_sha256'] = hashlib.sha256(factors.numpy().tobytes()).hexdigest()
    return batch, factors, summary


@torch.no_grad()
def probes(args, graph, factors, rows, begin, reference):
    records, scores = [], []
    for seed in PROTOCOL['probe_seeds']:
        for arm in ARMS:
            model = ChessPinFactorHead(arm, seed=seed)
            if not torch.equal(model(*args, factors, reference=reference), args[4]):
                raise AssertionError('Fresh backbone preservation failed')
            generator = torch.Generator().manual_seed(900000+seed)
            model.output.weight.copy_(torch.randn(model.output.weight.shape, generator=generator)*.1)
            batched = model(*args, factors, reference=reference)
            scores.append(batched[args[3]].clone())
            offset = 0
            for index, menu in enumerate(graph['menus']):
                error = 0.
                for slot in range(len(menu)):
                    value = model(*single_args(args, index, slot, offset+slot),
                                  single_factors(factors, offset+slot), reference=reference)[0, 0]
                    if not torch.isfinite(value):
                        raise AssertionError('Nonfinite probe score')
                    error = max(error, float((value-batched[index, slot]).abs()))
                offset += len(menu); deadline(begin)
                records.append({'seed': seed, 'arm': arm, 'index': index, 'id': rows[index]['id'],
                                'candidates': len(menu), 'single_batch_error': error})
                if error > PROTOCOL['score_tolerance']:
                    raise AssertionError('Frozen probe score tolerance exceeded')
    return records, torch.stack(scores)


def updates(args, factors, begin, reference):
    args = [t.double() if t.is_floating_point() else t for t in args]
    states, records = {}, {}
    common = None
    for arm in ARMS:
        head = ChessPinFactorHead(arm, seed=1197).double()
        initial = {k: v.clone() for k, v in head.state_dict().items()}
        if sum(p.numel() for p in head.parameters()) != 16658:
            raise AssertionError('Parameter count changed')
        if common is None:
            common = initial
        if not all(torch.equal(v, common[k]) for k, v in initial.items()):
            raise AssertionError('Arms did not receive identical tensors')
        optimizer = torch.optim.Adam(head.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
        logs = []
        for update in range(3):
            deadline(begin); optimizer.zero_grad(set_to_none=True)
            logits = head(*args, factors, reference=reference)
            loss = F.cross_entropy(logits, torch.zeros(128, dtype=torch.long))
            if not torch.isfinite(loss):
                raise AssertionError('Nonfinite artificial loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise AssertionError('Missing or nonfinite gradient')
            optimizer.step()
            logs.append({'update': update+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
        final = head.state_dict()
        if any(torch.equal(final[name], initial[name]) for name in final if name.startswith('factor.')):
            raise AssertionError('A factor parameter block did not train')
        states[arm] = initial, final
        records[arm] = logs
    return states, records


def compare_states(actual, expected, tolerance):
    if actual.keys() != expected.keys():
        raise AssertionError('State membership differs')
    if any(actual[k].shape != expected[k].shape or actual[k].dtype != expected[k].dtype for k in actual):
        raise AssertionError('State shape or dtype differs')
    errors = [float((actual[k]-expected[k]).abs().max()) for k in actual]
    if any(not math.isfinite(x) for x in errors):
        raise AssertionError('Nonfinite state discrepancy')
    error = max(errors)
    if error > tolerance:
        raise AssertionError('Frozen checkpoint replay tolerance exceeded')
    return error


def execute(plan_path, out, execution=None):
    plan = read(plan_path)
    if plan != signature():
        raise ValueError('Frozen engineering plan changed')
    plan_hash = source.file_hash(plan_path); reference = execution is not None
    if reference:
        diagnostic.manifest(execution, plan_hash)
    out.mkdir(parents=True, exist_ok=False); begin = time.monotonic()
    source.prior.write(out/'started.json', {'plan_sha256': plan_hash, 'reference': reference, 'pid': os.getpid()})
    try:
        model, data, features, graphs, rows, setup = parent.setup()
        graph, factors, input_summary = native_inputs(graphs, rows, reference)
        args = source.arguments(model, data, features, torch.arange(128), graph)
        probes_meta, score_values = probes(args, graph, factors, rows, begin, reference)
        states, training = updates(args, factors, begin, reference)
        if len(probes_meta) != 768 or score_values.shape != (6, input_summary['candidates']):
            raise AssertionError('Probe membership differs')
        if plan != signature():
            raise ValueError('Sources changed during execution')
        deadline(begin)
        if not reference:
            for arm, (initial, final) in states.items():
                source.prior.save(out/f'{arm}-initial.pt', initial)
                source.prior.save(out/f'{arm}-weights.pt', final)
            source.prior.save(out/'scores.pt', score_values)
            summary = {'status': 'completed', 'plan_sha256': plan_hash, 'inputs': input_summary,
                       'probes': probes_meta, 'score_checks': score_values.numel(),
                       'max_single_batch_error': max(r['single_batch_error'] for r in probes_meta),
                       'parameters': 16658, 'artificial_updates': 6, 'training': training,
                       'setup': setup, 'wall_seconds': time.monotonic()-begin, 'limits': PROTOCOL['limits']}
            source.prior.write(out/'summary.json', summary)
            deadline(begin)
            source.prior.write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                'files': {p.name: source.file_hash(p) for p in out.iterdir() if p.is_file()}})
            print(json.dumps({k: v for k, v in summary.items() if k not in ('probes', 'training')}), flush=True)
        else:
            old = read(execution/'summary.json')
            if (old['status'] != 'completed' or old['plan_sha256'] != plan_hash
                    or old['inputs'] != input_summary or old['parameters'] != 16658 or old['artificial_updates'] != 6
                    or old['limits'] != PROTOCOL['limits'] or not 0 < old['wall_seconds'] <= 900
                    or old['score_checks'] != score_values.numel() or len(old['probes']) != len(probes_meta)):
                raise AssertionError('Primary summary differs')
            if old['max_single_batch_error'] != max(r['single_batch_error'] for r in old['probes']):
                raise AssertionError('Primary error aggregation differs')
            if old['setup']['cache_max_error'] != setup['cache_max_error']:
                raise AssertionError('Frozen-feature reconstruction differs')
            for a, b in zip(probes_meta, old['probes']):
                if ({k: v for k, v in a.items() if k != 'single_batch_error'} !=
                        {k: v for k, v in b.items() if k != 'single_batch_error'} or
                        not math.isfinite(b['single_batch_error']) or not 0 <= b['single_batch_error'] <= 1e-5):
                    raise AssertionError('Primary probe record differs')
            saved = torch.load(execution/'scores.pt', weights_only=True)
            if saved.shape != score_values.shape:
                raise AssertionError('Score shape differs')
            score_error = float((saved-score_values).abs().max())
            if not math.isfinite(score_error) or score_error > 1e-5:
                raise AssertionError('Frozen independent score tolerance exceeded')
            final_error, log_error = 0., 0.
            for arm, (initial, final) in states.items():
                compare_states(initial, torch.load(execution/f'{arm}-initial.pt', weights_only=True), 0.)
                final_error = max(final_error, compare_states(final, torch.load(execution/f'{arm}-weights.pt', weights_only=True), 1e-9))
                if len(old['training'][arm]) != 3:
                    raise AssertionError('Missing artificial updates')
                for a, b in zip(training[arm], old['training'][arm]):
                    if a['update'] != b['update']:
                        raise AssertionError('Update membership differs')
                    for field in ('loss', 'gradient_norm'):
                        discrepancy = abs(a[field]-b[field]); log_error = max(log_error, discrepancy)
                        if not math.isfinite(discrepancy) or discrepancy > 1e-9:
                            raise AssertionError('Frozen update replay tolerance exceeded')
            deadline(begin)
            receipt = {'status': 'passed', 'plan_sha256': plan_hash, 'summary_sha256': source.file_hash(execution/'summary.json'),
                       'completed_sha256': source.file_hash(execution/'completed.json'), 'inputs': input_summary,
                       'root_seed_arm_checks': 768, 'score_checks': score_values.numel(), 'score_replay_max_error': score_error,
                       'reference_single_batch_max_error': max(r['single_batch_error'] for r in probes_meta),
                       'artificial_updates_replayed': 6, 'final_checkpoint_max_error': final_error,
                       'training_log_max_error': log_error, 'wall_seconds': time.monotonic()-begin,
                       'limits': PROTOCOL['limits']}
            source.prior.write(out/'receipt.json', receipt); print(json.dumps(receipt), flush=True)
    except Exception as error:
        source.prior.write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic()-begin})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'run', 'audit'))
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--plan', type=Path)
    parser.add_argument('--execution', type=Path); args = parser.parse_args()
    if args.command == 'prepare':
        plan = signature(); args.out.mkdir(parents=True, exist_ok=False); source.prior.write(args.out/'plan.json', plan)
        print(json.dumps({'plan_sha256': source.file_hash(args.out/'plan.json'), 'root_seed_arm_checks': 768, 'artificial_updates': 6}))
    else:
        if args.plan is None or (args.command == 'audit' and args.execution is None):
            parser.error('Expected --plan and, for audit, --execution')
        execute(args.plan, args.out, args.execution if args.command == 'audit' else None)
