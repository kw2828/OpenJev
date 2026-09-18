# SPDX-License-Identifier: GPL-3.0-only
"""Frozen native engineering replay for pin information/capacity controls."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

import chess_pin_factor_preflight as parent
from openjev.research.chess_pin_factor_controls import ARMS, PARAMETERS, VERSION, ChessPinFactorControl

ROOT = parent.ROOT
source = parent.source
PARENT_PLAN = 'evidence/chess-pin-factor-preflight-v1/protocol/plan.json'
PARENT_AUDIT = 'evidence/chess-pin-factor-preflight-v1/audit/receipt.json'
PARENT_RUN = 'runs/chess-pin-factor-preflight-v1/execution'
CODE = ['src/openjev/research/chess_pin_factor_controls.py',
        'tests/test_chess_pin_factor_controls.py', 'scripts/chess_pin_control_preflight.py']
PROTOCOL = {
    'version': VERSION, 'arms': list(ARMS), 'parameters': PARAMETERS,
    'scope': 'Engineering only. No teacher-quality fitting, held-out predictions or cost claim.',
    'roots': 128, 'selection': 'First128 old training roots, all legal candidates.',
    'backbone_seed': 97, 'probe_seeds': [211, 223, 227], 'root_seed_arm_checks': 1152,
    'probe_dtype': 'float32', 'score_tolerance': 1e-5,
    'output_probe': 'Normal(0,.1), generator900000+seed after exact zero-output backbone preservation.',
    'head_seed': 1197, 'updates_per_arm': 3, 'total_updates': 9,
    'target': 'First legal candidate; teacher targets and values unused.',
    'update_dtype': 'float64 head on cast frozen float32 backbone features',
    'update_tolerance': 1e-9,
    'optimizer': 'Adam lr.001 betas(.9,.999) eps1e-8 weight_decay0; clip norm1.',
    'native_input_checks': 'Reuse frozen native-input checker; independent pin detector and status assignment in replay.',
    'references': 'Root-only filters root membership in a separate Python loop and uses scalar factor pooling. Counts use explicit Python tallies; counts/graph MLPs use individual candidate affine calls. Encoders and neural layers are shared.',
    'capacity_limits': 'Root-only has1568 first-layer weights on zero channels; counts has2 fewer parameters; graph_mlp matches stored count. No effective-capacity or FLOP parity claim.',
    'runtime': 'CPU2 deterministic, shared host with ongoing union study.',
    'time_cap_seconds': 900,
}


def read(path):
    return json.loads(Path(path).read_text())


def signature():
    frozen = read(ROOT/PARENT_PLAN)
    if frozen != parent.signature():
        raise ValueError('Parent pin-factor signature changed')
    complete = parent.diagnostic.manifest(ROOT/PARENT_RUN, source.file_hash(ROOT/PARENT_PLAN))
    audit = read(ROOT/PARENT_AUDIT)
    if (audit['status'] != 'passed' or audit['plan_sha256'] != source.file_hash(ROOT/PARENT_PLAN)
            or audit['completed_sha256'] != source.file_hash(ROOT/PARENT_RUN/'completed.json')
            or audit['summary_sha256'] != complete['files']['summary.json']
            or audit['score_checks'] != 23520 or audit['artificial_updates_replayed'] != 6):
        raise ValueError('Parent native preflight identity changed')
    return {'protocol': PROTOCOL, 'sources': {p: source.file_hash(ROOT/p) for p in CODE},
            'parent_signature': frozen, 'parent_audit_sha256': source.file_hash(ROOT/PARENT_AUDIT)}


def deadline(begin):
    if time.monotonic()-begin > PROTOCOL['time_cap_seconds']:
        raise TimeoutError('Frozen fifteen-minute control engineering ceiling exceeded')


def write(path, value):
    source.prior.write(path, value)


def execute(plan, out, *, reference=False):
    begin = time.monotonic()
    if read(plan) != signature():
        raise ValueError('Frozen control signature changed')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'started.json', {'plan_sha256': source.file_hash(plan), 'reference': reference, 'unix': time.time()})
    try:
        backbone, data, features, graphs, rows, setup = parent.parent.setup()
        graph, factors, inputs = parent.native_inputs(graphs, rows, reference)
        args = source.arguments(backbone, data, features, torch.arange(128), graph)
        scores, probes = [], []
        with torch.no_grad():
            for seed in PROTOCOL['probe_seeds']:
                for arm in ARMS:
                    model = ChessPinFactorControl(arm, seed=seed)
                    if sum(p.numel() for p in model.parameters()) != PARAMETERS[arm]:
                        raise AssertionError('Parameter budget changed')
                    if not torch.equal(model(*args, factors, reference=reference), args[4]):
                        raise AssertionError('Backbone preservation failed')
                    generator = torch.Generator().manual_seed(900000+seed)
                    model.output.weight.copy_(torch.randn(model.output.weight.shape, generator=generator)*.1)
                    batched = model(*args, factors, reference=reference)
                    values = batched[args[3]]
                    if not torch.isfinite(values).all():
                        raise AssertionError('Nonfinite probe scores')
                    scores.append(values.clone()); offset = 0
                    for index, menu in enumerate(graph['menus']):
                        errors = []
                        for slot in range(len(menu)):
                            single = model(*parent.single_args(args, index, slot, offset+slot),
                                           parent.single_factors(factors, offset+slot), reference=reference)[0, 0]
                            if not torch.isfinite(single):
                                raise AssertionError('Nonfinite single-candidate score')
                            errors.append(float((single-batched[index, slot]).abs()))
                        error = max(errors)
                        if error > PROTOCOL['score_tolerance']:
                            raise AssertionError('Frozen score tolerance exceeded')
                        probes.append({'seed': seed, 'arm': arm, 'index': index, 'id': rows[index]['id'],
                                       'candidates': len(menu), 'single_batch_error': error})
                        offset += len(menu); deadline(begin)
                    print(json.dumps({'probe': arm, 'seed': seed, 'reference': reference}), flush=True)
        source.prior.save(out/'scores.pt', torch.stack(scores))
        write(out/'probes.json', probes)
        double_args = [v.double() if v.is_floating_point() else v for v in args]
        logs = {}; common = None
        for arm in ARMS:
            model = ChessPinFactorControl(arm, seed=1197).double()
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            encoder = {k: v for k, v in initial.items() if k.startswith(('encoder.', 'difference.'))}
            if common is None:
                common = encoder
            parent.compare_states(common, encoder, 0)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
            logs[arm] = []
            for update in range(3):
                deadline(begin); optimizer.zero_grad(set_to_none=True)
                logits = model(*double_args, factors, reference=reference)
                loss = F.cross_entropy(logits, torch.zeros(128, dtype=torch.long))
                if not torch.isfinite(loss):
                    raise AssertionError('Nonfinite artificial loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise AssertionError('Missing or nonfinite gradient')
                optimizer.step()
                logs[arm].append({'update': update+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
            final = model.state_dict()
            changed = [name for name in initial if name.startswith('factor.') and not torch.equal(initial[name], final[name])]
            if len(changed) != 4:
                raise AssertionError('A factor parameter block did not update')
            source.prior.save(out/f'{arm}-initial.pt', initial)
            source.prior.save(out/f'{arm}-final.pt', final)
            print(json.dumps({'artificial_updates': arm, 'reference': reference}), flush=True)
        deadline(begin)
        summary = {'inputs': inputs, 'setup': setup, 'root_seed_arm_checks': len(probes),
                   'score_checks': sum(p['candidates'] for p in probes), 'artificial_updates': 9,
                   'max_single_batch_error': max(p['single_batch_error'] for p in probes),
                   'parameters': PARAMETERS, 'logs': logs, 'wall_seconds': time.monotonic()-begin}
        write(out/'summary.json', summary)
        files = {str(p.relative_to(out)): source.file_hash(p) for p in out.rglob('*') if p.is_file()}
        write(out/'completed.json', {'status': 'completed', 'plan_sha256': source.file_hash(plan), 'files': files})
        return summary
    except Exception as error:
        write(out/'failed.json', {'error': repr(error), 'wall_seconds': time.monotonic()-begin})
        raise


def audit(plan, execution, out):
    begin = time.monotonic()
    if read(plan) != signature():
        raise ValueError('Frozen control signature changed')
    parent.diagnostic.manifest(execution, source.file_hash(plan))
    out.mkdir(parents=True, exist_ok=False)
    try:
        actual = read(execution/'summary.json')
        expected = execute(plan, out/'replay', reference=True)
        if (actual['inputs'] != expected['inputs'] or actual['root_seed_arm_checks'] != 1152
                or actual['score_checks'] != 9*expected['inputs']['candidates']
                or actual['artificial_updates'] != 9 or actual['parameters'] != PARAMETERS):
            raise AssertionError('Native counts or budgets differ')
        a_probes, b_probes = read(execution/'probes.json'), read(out/'replay/probes.json')
        a_members = [{k: v for k, v in r.items() if k != 'single_batch_error'} for r in a_probes]
        b_members = [{k: v for k, v in r.items() if k != 'single_batch_error'} for r in b_probes]
        if a_members != b_members or len(a_members) != 1152:
            raise AssertionError('Probe membership differs')
        errors = [r['single_batch_error'] for r in a_probes]
        if any(not (0 <= e <= 1e-5) for e in errors) or actual['max_single_batch_error'] != max(errors):
            raise AssertionError('Primary probe error aggregation failed')
        def load(path):
            return torch.load(path, weights_only=True, map_location='cpu')
        score_error = parent.compare_states({'scores': load(execution/'scores.pt')},
                                            {'scores': load(out/'replay/scores.pt')}, 1e-5)
        state_error = log_error = 0.
        for arm in ARMS:
            parent.compare_states(load(execution/f'{arm}-initial.pt'), load(out/f'replay/{arm}-initial.pt'), 0)
            state_error = max(state_error, parent.compare_states(load(execution/f'{arm}-final.pt'),
                               load(out/f'replay/{arm}-final.pt'), 1e-9))
            a_logs, b_logs = actual['logs'][arm], expected['logs'][arm]
            if len(a_logs) != 3 or len(b_logs) != 3:
                raise AssertionError('Artificial update count differs')
            for a, b in zip(a_logs, b_logs):
                if a['update'] != b['update'] or set(a) != set(b):
                    raise AssertionError('Update membership differs')
                for key in ('loss', 'gradient_norm'):
                    error = abs(a[key]-b[key])
                    if not 0 <= error <= 1e-9:
                        raise AssertionError('Update replay tolerance exceeded')
                    log_error = max(log_error, error)
        deadline(begin)
        receipt = {'status': 'passed', 'plan_sha256': source.file_hash(plan),
                   'completed_sha256': source.file_hash(execution/'completed.json'),
                   'summary_sha256': source.file_hash(execution/'summary.json'),
                   'reference_completed_sha256': source.file_hash(out/'replay/completed.json'),
                   'score_checks': actual['score_checks'], 'artificial_updates_replayed': 9,
                   'score_replay_max_error': score_error, 'checkpoint_replay_max_error': state_error,
                   'training_log_max_error': log_error, 'wall_seconds': time.monotonic()-begin}
        write(out/'receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        write(out/'failed.json', {'error': repr(error), 'wall_seconds': time.monotonic()-begin})
        raise


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare'); prepare.add_argument('--out', type=Path, required=True)
    for command in ('run', 'audit'):
        p = sub.add_parser(command); p.add_argument('--plan', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        if command == 'audit': p.add_argument('--execution', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        frozen = signature(); args.out.mkdir(parents=True, exist_ok=False)
        write(args.out/'plan.json', frozen)
        print(json.dumps({'plan_sha256': source.file_hash(args.out/'plan.json')}), flush=True)
    elif args.command == 'run': execute(args.plan, args.out)
    else: audit(args.plan, args.execution, args.out)


if __name__ == '__main__':
    main()
