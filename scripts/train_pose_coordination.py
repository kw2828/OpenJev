"""Frozen-expert coordination screen with matched, train-only gate fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from run_pose_support import forecast_support
from train_pose_adaptation import load_data, loss_function, measure, predict

from openjev.research.pose_coordination import ConstantGate, RecurrentGate, SummaryGate, blend, context_tokens
from openjev.research.pose_support import PoseSupport
from openjev.research.pose_transport import PoseTransport

SEEDS = (1101, 1202, 1303)
TRAIN_VARIANTS = ('constant', 'summary', 'recurrent')
VARIANTS = ('fast', 'slow', 'position_fast', 'position_slow', 'half', *TRAIN_VARIANTS)
PANELS = ('test_sin', 'test_zigzag')
EPOCHS, BATCH = 30, 32
GATE = ('recurrent RMSE <=0.90*each positive control on both physical endpoints/panels; '
        'all3 paired MSE nonworse; at least8/10 parents nonworse; all10 leave-one-parent-out '
        'MSE strictly lower; median full-window latency<=1.5*new slow. All17 groups required.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sources():
    return [Path(p) for p in (
        'scripts/train_pose_coordination.py', 'scripts/audit_pose_coordination.py',
        'src/openjev/research/pose_coordination.py', 'tests/test_pose_coordination.py',
        'tests/test_pose_coordination_training.py', 'tests/test_audit_pose_coordination.py',
        'research/pose-coordination-protocol.md', 'scripts/run_pose_support.py',
        'scripts/audit_pose_support.py', 'scripts/audit_pose_adaptation.py',
        'scripts/audit_pose_transport.py', 'scripts/train_pose_adaptation.py',
        'src/openjev/research/pose_support.py', 'src/openjev/research/pose_adaptation.py',
        'src/openjev/research/pose_transport.py', 'src/openjev/research/pose_references.py',
        'src/openjev/research/rigid_motion.py')]


def make_gate(variant):
    return {'constant': ConstantGate, 'summary': SummaryGate, 'recurrent': RecurrentGate}[variant]()


def experts(protocol, seed):
    models = []
    for name, cls in (('meta', PoseSupport), ('gru', lambda s: PoseTransport('body', s))):
        binding = protocol['checkpoints'][f'{name}-{seed}']
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('expert changed')
        state = torch.load(binding['path'], map_location='cpu', weights_only=True)
        model = cls(state['scales'])
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        models.append(model)
    if not torch.equal(models[0].scales, models[1].scales):
        raise ValueError('expert scales differ')
    return tuple(models)


def private_rollouts(fast, slow, p, r, a):
    # Both private trajectories remain unchanged by the final output combination.
    fp, fr = forecast_support(fast, p[:, :32], r[:, :32], a[:, :31], a[:, 31:], 'decay_huber3')
    sp, sr = predict(slow, p, r, a, 'gru')
    return fp, fr, sp, sr


def forecast(fast, slow, gate, variant, context_p, context_r, past_actions, future_actions):
    if context_p.shape[1:] != (32, 3) or context_r.shape[1:] != (32, 3, 3):
        raise ValueError('context must contain exactly32 poses')
    if past_actions.shape != (len(context_p), 31, 40) or future_actions.shape != (len(context_p), 25, 40):
        raise ValueError('31 past and25 future actions required')
    if variant not in VARIANTS:
        raise ValueError('unknown coordination variant')
    actions = torch.cat((past_actions, future_actions), 1)
    if variant == 'fast':
        p, r = forecast_support(fast, context_p, context_r, past_actions, future_actions, 'decay_huber3')
        return p, r, p.new_ones(len(p), 2)
    if variant == 'slow':
        p, r = predict(slow, context_p, context_r, actions, 'gru')
        return p, r, p.new_zeros(len(p), 2)
    fp, fr, sp, sr = private_rollouts(fast, slow, context_p, context_r, actions)
    if variant in TRAIN_VARIANTS:
        alpha = gate(context_tokens(context_p, context_r, past_actions, fast.scales))
    else:
        values = {'position_fast': [1., 0.], 'position_slow': [0., 1.], 'half': [.5, .5]}[variant]
        alpha = fp.new_tensor(values).expand(len(fp), -1)
    p, r = blend(fp, fr, sp, sr, alpha)
    return p, r, alpha


def freeze(data, prior, report, out):
    old = json.loads((prior / 'protocol.json').read_text())
    done = json.loads((prior / 'run-01/completed.json').read_text())
    audited = json.loads((report / 'summary.json').read_text())
    receipt = json.loads((report / 'receipt.json').read_text())
    if old['study'] != 'pose-support-v1' or done['status'] != 'completed':
        raise ValueError('completed support study required')
    if done['protocol_sha256'] != sha(prior / 'protocol.json'):
        raise ValueError('support protocol binding')
    if audited['execution_completed_sha256'] != sha(prior / 'run-01/completed.json'):
        raise ValueError('support audit binding')
    if receipt['files']['summary.json']['sha256'] != sha(report / 'summary.json'):
        raise ValueError('support report seal')
    for name, binding in done['files'].items():
        if sha(prior / 'run-01' / name) != binding['sha256']:
            raise ValueError('support payload changed: ' + name)
    for name, digest in old['sources'].items():
        if sha(name) != digest:
            raise ValueError('support source changed: ' + name)
    data_hashes = {p.name: sha(p) for p in sorted(data.iterdir()) if p.is_file()}
    if old['data'] != str(data.resolve()) or data_hashes != old['data_hashes']:
        raise ValueError('data differ')
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        'study': 'pose-coordination-v1', 'scope': 'exposed-data frozen-expert coordination screen',
        'data': str(data.resolve()), 'data_hashes': data_hashes,
        'support_experiment': str(prior.resolve()), 'support_protocol_sha256': sha(prior / 'protocol.json'),
        'support_completed_sha256': sha(prior / 'run-01/completed.json'),
        'support_report': str(report.resolve()), 'support_summary_sha256': sha(report / 'summary.json'),
        'support_receipt_sha256': sha(report / 'receipt.json'),
        'checkpoints': {f'{v}-{s}': old['checkpoints'][f'{v}-{s}'] for s in SEEDS for v in ('meta', 'gru')},
        'sources': {str(p): sha(p) for p in sources()},
        'seeds': SEEDS, 'train_variants': TRAIN_VARIANTS, 'variants': VARIANTS, 'panels': PANELS,
        'primary': 'recurrent', 'context': 32, 'tokens': 31, 'token_dim': 49, 'horizon': 25,
        'epochs': EPOCHS, 'batch': BATCH, 'optimizer': 'Adam', 'learning_rate': .001, 'grad_norm_cap': 1.,
        'expected_fits': 9, 'updates_per_fit': 690, 'evaluation_rows': 48,
        'position_loss_scale_m': .1, 'rotation_loss_scale_rad': .1,
        'warmups_per_row': 3, 'timed_windows_per_row': 20,
        'blend': 'alpha is fast fraction; linear position, Exp(alpha_R*Log(R_fast*R_slow.T))*R_slow; exact endpoints',
        'training': 'Original720 training windows only; frozen expert forecasts cached once per seed; train final gate only. Same30 batch permutations across3 arms. Experts saw these windows, not out-of-fold stacking.',
        'gate': GATE, 'gate_groups': 17, 'gate_comparisons': 1501,
        'blend_tolerance': {'rtol': 2e-6, 'atol': 2e-6, 'hard_endpoints': 'exact'},
        'expert_replay_tolerance': {'rtol': 1e-6, 'atol': 2e-7},
        'execution_files': 147,
        'token_tolerance': {'rtol': 1e-4, 'atol': 1e-4},
        'no_selection': 'Fixed arms, final checkpoints and seeds; no retries, sweeps, exclusions or gate changes.',
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                    'threads': 1, 'platform': platform.platform()},
    }
    write_json(out / 'protocol.json', protocol)
    for p in sources():
        dest = out / 'source-snapshot' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    print(json.dumps({'frozen': str(out), 'sha256': sha(out / 'protocol.json')}))


def validate_protocol(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    for path, digest in protocol['sources'].items():
        if sha(path) != digest:
            raise ValueError('frozen source changed: ' + path)
    for name, digest in protocol['data_hashes'].items():
        if sha(Path(protocol['data']) / name) != digest:
            raise ValueError('data changed: ' + name)
    for field, suffix in (('protocol', 'protocol.json'), ('completed', 'run-01/completed.json')):
        if sha(Path(protocol['support_experiment']) / suffix) != protocol[f'support_{field}_sha256']:
            raise ValueError('support study changed')
    for field in ('summary', 'receipt'):
        if sha(Path(protocol['support_report']) / (field + '.json')) != protocol[f'support_{field}_sha256']:
            raise ValueError('support audit changed')
    for binding in protocol['checkpoints'].values():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('expert checkpoint changed')
    return protocol


def run(out):
    protocol = validate_protocol(out)
    dest = out / 'run-01'
    dest.mkdir(exist_ok=False)
    start = time.perf_counter()
    write_json(dest / 'started.json', {'protocol_sha256': sha(out / 'protocol.json')})
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        data = Path(protocol['data'])
        p, r, a, _ = load_data(data, 'train')
        if len(p) != 720:
            raise ValueError('720 training windows required')
        fits, caches = [], []
        for seed in SEEDS:
            tick = time.perf_counter()
            fast, slow = experts(protocol, seed)
            with torch.no_grad():
                tokens = context_tokens(p[:, :32], r[:, :32], a[:, :31], fast.scales)
                fp, fr, sp, sr = private_rollouts(fast, slow, p, r, a)
            if seed == SEEDS[0]:
                np.save(dest / 'train-tokens.npy', tokens.numpy(), allow_pickle=False)
            else:
                if not np.array_equal(tokens.numpy(), np.load(dest / 'train-tokens.npy', allow_pickle=False)):
                    raise ValueError('seed-dependent context tokens')
            for name, pp, rr in (('fast', fp, fr), ('slow', sp, sr)):
                np.savez_compressed(dest / f'train-{name}-{seed}.npz', p=pp.numpy(), R=rr.numpy())
            cache = {'seed': seed, 'seconds': time.perf_counter() - tick}
            caches.append(cache)
            generator = torch.Generator().manual_seed(seed + 1000)
            orders = [torch.randperm(len(p), generator=generator) for _ in range(EPOCHS)]
            np.save(dest / f'orders-{seed}.npy', torch.stack(orders).numpy(), allow_pickle=False)
            for variant in TRAIN_VARIANTS:
                tick = time.perf_counter()
                torch.manual_seed(seed)
                model = make_gate(variant)
                stem = f'{variant}-{seed}'
                torch.save(model.state_dict(), dest / (stem + '-initial.pt'))
                opt = torch.optim.Adam(model.parameters(), lr=.001)
                losses = []
                for epoch, order in enumerate(orders):
                    total = 0.
                    for j in range(0, len(p), BATCH):
                        idx = order[j:j+BATCH]
                        alpha = model(tokens[idx])
                        pp, rr = blend(fp[idx], fr[idx], sp[idx], sr[idx], alpha)
                        loss = loss_function(pp, rr, p[idx, 32:], r[idx, 32:])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite gate loss')
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        opt.step()
                        losses.append(float(loss.detach()))
                        total += losses[-1] * len(idx)
                    print(json.dumps({'fit': stem, 'epoch': epoch+1, 'train_loss': total / len(p)}), flush=True)
                torch.save(model.state_dict(), dest / (stem + '.pt'))
                np.save(dest / (stem + '-losses.npy'), np.asarray(losses), allow_pickle=False)
                fit = {'variant': variant, 'seed': seed, 'updates': len(losses),
                       'parameters': sum(x.numel() for x in model.parameters()),
                       'training_seconds': time.perf_counter()-tick,
                       'initial_sha256': sha(dest / (stem + '-initial.pt')),
                       'checkpoint_sha256': sha(dest / (stem + '.pt')),
                       'orders_sha256': sha(dest / f'orders-{seed}.npy')}
                write_json(dest / (stem + '-fit.json'), fit)
                fits.append(fit)
        write_json(dest / 'training-completed.json', {'fits': fits, 'caches': caches, 'evaluation_started': False})
        rows = []
        for panel in PANELS:
            tp, tr, ta, ids = load_data(data, panel)
            np.savez_compressed(dest / (panel + '-targets.npz'), p=tp[:, 32:].numpy(), R=tr[:, 32:].numpy(), ids=ids)
            for variant in VARIANTS:
                for seed in SEEDS:
                    fast, slow = experts(protocol, seed)
                    gate, digest = None, None
                    if variant in TRAIN_VARIANTS:
                        gate = make_gate(variant)
                        file = dest / f'{variant}-{seed}.pt'
                        fit = next(x for x in fits if x['variant'] == variant and x['seed'] == seed)
                        digest = fit['checkpoint_sha256']
                        if sha(file) != digest:
                            raise ValueError('trained gate checkpoint changed')
                        gate.load_state_dict(torch.load(file, map_location='cpu', weights_only=True), strict=True)
                        gate.eval()
                    def call(cp, cr, pa, fa, f=fast, s=slow, g=gate, v=variant):
                        return forecast(f, s, g, v, cp, cr, pa, fa)
                    with torch.no_grad():
                        pp, rr, alpha = call(tp[:, :32], tr[:, :32], ta[:, :31], ta[:, 31:])
                        metrics = measure(pp, rr, tp[:, 32:], tr[:, 32:])
                        timings = []
                        for i in range(23):
                            tick = time.perf_counter_ns()
                            call(tp[i:i+1, :32], tr[i:i+1, :32], ta[i:i+1, :31], ta[i:i+1, 31:])
                            ms = (time.perf_counter_ns()-tick)/1e6
                            if i >= 3:
                                timings.append(ms)
                    row = {'panel': panel, 'variant': variant, 'seed': seed, 'checkpoint_sha256': digest,
                           'expert_sha256': {n: protocol['checkpoints'][f'{n}-{seed}']['sha256'] for n in ('meta', 'gru')},
                           'metrics': metrics, 'latency_ms': timings}
                    prefix = f'{panel}-{variant}-{seed}'
                    np.savez_compressed(dest / (prefix + '-predictions.npz'), p=pp.numpy(), R=rr.numpy(), alpha=alpha.numpy())
                    write_json(dest / (prefix + '-evaluation.json'), row)
                    rows.append(row)
                    print(json.dumps({'row': prefix, 'metrics': metrics}), flush=True)
        validate_protocol(out)
        write_json(dest / 'completed.json', {'status': 'completed', 'fits': fits, 'caches': caches, 'rows': rows,
            'new_training_fits': 9, 'new_optimizer_updates': 6210, 'wall_seconds': time.perf_counter()-start,
            'protocol_sha256': sha(out / 'protocol.json'),
            'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                      for p in sorted(dest.iterdir()) if p.is_file()}})
        print(json.dumps({'completed': str(dest), 'wall_seconds': time.perf_counter()-start}), flush=True)
    except BaseException as error:
        write_json(dest / 'failed.json', {'error': repr(error), 'wall_seconds': time.perf_counter()-start})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        if any(x is None for x in (args.data, args.prior, args.report)):
            parser.error('freeze requires data, prior and report')
        freeze(args.data, args.prior, args.report, args.out)
    else:
        run(args.out)
