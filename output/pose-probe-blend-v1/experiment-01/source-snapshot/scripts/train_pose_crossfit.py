"""Parent-excluded versus included expert predictions for pose-selector training."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import train_pose_coordination as previous
from train_pose_adaptation import load_data, loss_function, make_model, measure, predict

from openjev.research.pose_constant import fit_constant
from openjev.research.pose_coordination import blend, context_tokens
from openjev.research.pose_support import PoseSupport
from openjev.research.pose_transport import training_scales

sha, write_json = previous.sha, previous.write_json
SEEDS = previous.SEEDS
PANELS = previous.PANELS
REGIMES = ('is', 'oof')
TRAIN_VARIANTS = ('is_summary', 'is_recurrent', 'oof_summary', 'oof_recurrent')
CONSTANT_VARIANTS = ('is_constant', 'oof_constant', 'full_constant')
VARIANTS = ('fast', 'slow', 'is_constant', 'is_summary', 'is_recurrent',
            'oof_constant', 'oof_summary', 'oof_recurrent', 'full_constant')
GATE = ('oof_recurrent RMSE <=0.90*each positive control on both physical endpoints/panels; '
        'all3 paired MSE nonworse; at least8/10 parents nonworse; all10 leave-one-parent-out '
        'MSE strictly lower; median full-window latency<=1.5*new slow. All17 groups required; '
        'all9 constant optimizations must also certify their projected-rotation objective gap.')
BLEND_TOLERANCE = {'rtol': 2e-6, 'atol': 2e-6, 'hard_endpoints': 'exact'}
MOTION_TOLERANCE = {'rtol': 1e-4, 'atol': 1e-7}


def sources():
    return previous.sources() + [Path(p) for p in (
        'scripts/train_pose_crossfit.py', 'scripts/audit_pose_crossfit.py',
        'src/openjev/research/pose_constant.py', 'tests/test_pose_constant.py',
        'tests/test_pose_crossfit_training.py', 'tests/test_audit_pose_crossfit.py',
        'research/pose-crossfit-protocol.md')]


def partition(ids):
    ids = np.asarray(ids)
    if ids.shape != (720, 2) or ids.dtype != np.int64:
        raise ValueError('720 source/start IDs required')
    if not np.array_equal(np.unique(ids[:, 0]), np.arange(30)):
        raise ValueError('original30 training parents required')
    if not all(np.sum(ids[:, 0] == p) == 24 for p in range(30)):
        raise ValueError('24 windows per parent required')
    return ids[:, 0] % 3


def fold_normalization(p, r, actions, groups, fold):
    if fold not in (0, 1, 2):
        raise ValueError('three fixed folds required')
    indices = np.flatnonzero(groups != fold)
    if len(indices) != 480:
        raise ValueError('480 eligible training windows required')
    raw = actions[indices].numpy().astype(np.float64)
    mean = raw.mean(axis=(0, 1))
    scale = np.sqrt(np.square(raw - mean).mean(axis=(0, 1))).clip(min=1e-5)
    return {'train_indices': indices.astype(np.int64),
            'action_mean': mean.astype(np.float32), 'action_scale': scale.astype(np.float32),
            'motion_scales': training_scales(p[indices], r[indices]).numpy()}


def mix_cache(fold_caches, groups, regime):
    if regime not in REGIMES:
        raise ValueError('unknown cache regime')
    folds = groups if regime == 'oof' else (groups + 1) % 3
    if set(fold_caches) != {0, 1, 2} or folds.shape != (720,):
        raise ValueError('complete fold caches required')
    combined = [torch.empty_like(x) for x in fold_caches[0]]
    for fold in range(3):
        mask = torch.from_numpy(folds == fold)
        for dst, src in zip(combined, fold_caches[fold], strict=True):
            dst[mask] = src[mask]
    return tuple(combined), folds.astype(np.int64)


def make_gate(variant):
    if variant not in TRAIN_VARIANTS:
        raise ValueError('unknown trained selector')
    return previous.make_gate(variant.split('_', 1)[1])


def forecast(fast, slow, gate, alpha, variant, p, r, past, future):
    if variant not in VARIANTS:
        raise ValueError('unknown crossfit variant')
    if variant in ('fast', 'slow'):
        return previous.forecast(fast, slow, None, variant, p, r, past, future)
    if variant in TRAIN_VARIANTS:
        return previous.forecast(fast, slow, gate, variant.split('_', 1)[1], p, r, past, future)
    if p.shape[1:] != (32, 3) or r.shape[1:] != (32, 3, 3):
        raise ValueError('exact32 observed poses required')
    if past.shape != (len(p), 31, 40) or future.shape != (len(p), 25, 40):
        raise ValueError('31 completed and25 future actions required')
    fp, fr, sp, sr = previous.private_rollouts(fast, slow, p, r, torch.cat((past, future), 1))
    weights = p.new_tensor(alpha).expand(len(p), -1)
    pp, rr = blend(fp, fr, sp, sr, weights)
    return pp, rr, weights


def freeze(data, prior, report, out):
    old = json.loads((prior / 'protocol.json').read_text())
    done = json.loads((prior / 'run-01/completed.json').read_text())
    summary = json.loads((report / 'summary.json').read_text())
    receipt = json.loads((report / 'receipt.json').read_text())
    if old['study'] != 'pose-coordination-v1' or done['status'] != 'completed':
        raise ValueError('completed coordination study required')
    if done['protocol_sha256'] != sha(prior / 'protocol.json') or summary['execution_completed_sha256'] != sha(prior / 'run-01/completed.json'):
        raise ValueError('parent protocol/completion identity')
    if receipt['files']['summary.json']['sha256'] != sha(report / 'summary.json'):
        raise ValueError('parent report identity')
    for name, binding in done['files'].items():
        if sha(prior / 'run-01' / name) != binding['sha256']:
            raise ValueError('parent payload changed')
    for name, digest in old['sources'].items():
        if sha(name) != digest:
            raise ValueError('parent source changed')
    data_hashes = {p.name: sha(p) for p in sorted(data.iterdir()) if p.is_file()}
    if old['data'] != str(data.resolve()) or old['data_hashes'] != data_hashes:
        raise ValueError('same prepared data required')
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        'study': 'pose-crossfit-v1', 'scope': 'exposed-data parent-exclusion training mechanism screen',
        'data': str(data.resolve()), 'data_hashes': data_hashes,
        'coordination_experiment': str(prior.resolve()), 'coordination_protocol_sha256': sha(prior / 'protocol.json'),
        'coordination_completed_sha256': sha(prior / 'run-01/completed.json'),
        'coordination_report': str(report.resolve()), 'coordination_summary_sha256': sha(report / 'summary.json'),
        'coordination_receipt_sha256': sha(report / 'receipt.json'), 'checkpoints': old['checkpoints'],
        'sources': {str(p): sha(p) for p in sources()}, 'seeds': SEEDS,
        'train_variants': TRAIN_VARIANTS, 'constant_variants': CONSTANT_VARIANTS, 'variants': VARIANTS, 'panels': PANELS,
        'primary': 'oof_recurrent', 'folds': 3, 'group_rule': 'source_id modulo3',
        'fold_training': 'exclude groupk; OOF uses expertk=g; IS uses expertk=(g+1)%3',
        'fold_windows': 480, 'fold_epochs': 46, 'gate_windows': 720, 'gate_epochs': 30,
        'batch': 32, 'updates_per_fit': 690, 'optimizer': 'Adam', 'learning_rate': .001, 'grad_norm_cap': 1.,
        'expert_fits': 18, 'gate_fits': 12, 'constant_fits': 9, 'optimizer_updates': 20700,
        'constant_tolerance_rad_squared': 1e-7, 'constant_max_evaluations': 4095,
        'constant_criterion': 'Analytic position LS; global interval search on float64 SO3-projected rotation objective. Production float32 blend unchanged.',
        'normalization': 'Fold-only action mean/populationSD per40coords, float64 thenfloat32, floor1e-5; fold-only four motion RMS scales. Gate tokens retain original full-data scales.',
        'action_normalization_tolerance': 'exact float32 arrays from NumPy float64 reductions',
        'motion_scale_tolerance': MOTION_TOLERANCE,
        'context': 32, 'tokens': 31, 'token_dim': 49, 'horizon': 25,
        'position_loss_scale_m': .1, 'rotation_loss_scale_rad': .1,
        'evaluation_rows': 54, 'execution_files': 288, 'warmups_per_row': 3, 'timed_windows_per_row': 20,
        'blend_tolerance': BLEND_TOLERANCE, 'token_tolerance': {'rtol': 1e-4, 'atol': 1e-4},
        'expert_replay_tolerance': {'rtol': 1e-6, 'atol': 2e-7},
        'gate': GATE, 'gate_groups': 17, 'gate_comparisons': 1921,
        'no_selection': 'Fixed parent groups, final checkpoints and budgets; no retries, sweeps, replacements, exclusions or criterion changes.',
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
    for name, digest in protocol['sources'].items():
        if sha(name) != digest:
            raise ValueError('frozen source changed: ' + name)
    for name, digest in protocol['data_hashes'].items():
        if sha(Path(protocol['data']) / name) != digest:
            raise ValueError('data changed: ' + name)
    prior = Path(protocol['coordination_experiment'])
    for field, name in (('protocol', 'protocol.json'), ('completed', 'run-01/completed.json')):
        if sha(prior / name) != protocol[f'coordination_{field}_sha256']:
            raise ValueError('parent experiment changed')
    parent_done = json.loads((prior / 'run-01/completed.json').read_text())
    for name, binding in parent_done['files'].items():
        if sha(prior / 'run-01' / name) != binding['sha256']:
            raise ValueError('parent payload changed')
    for field in ('summary', 'receipt'):
        if sha(Path(protocol['coordination_report']) / (field + '.json')) != protocol[f'coordination_{field}_sha256']:
            raise ValueError('parent report changed')
    for binding in protocol['checkpoints'].values():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('deployment expert changed')
    return protocol


def fit_neural(model, orders, objective, dest, stem, identity, orders_file):
    tick = time.perf_counter()
    torch.save(model.state_dict(), dest / (stem + '-initial.pt'))
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    losses = []
    for epoch, order in enumerate(orders):
        total = 0.
        for j in range(0, len(order), 32):
            idx = order[j:j+32]
            loss = objective(model, idx)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
            total += losses[-1] * len(idx)
        print(json.dumps({'fit': stem, 'epoch': epoch+1, 'train_loss': total/len(order)}), flush=True)
    if len(losses) != 690:
        raise ValueError('exact690 updates required')
    torch.save(model.state_dict(), dest / (stem + '.pt'))
    np.save(dest / (stem + '-losses.npy'), np.asarray(losses), allow_pickle=False)
    fit = {**identity, 'updates': len(losses), 'parameters': sum(x.numel() for x in model.parameters()),
           'training_seconds': time.perf_counter()-tick, 'initial_sha256': sha(dest / (stem + '-initial.pt')),
           'checkpoint_sha256': sha(dest / (stem + '.pt')), 'orders_sha256': sha(orders_file)}
    write_json(dest / (stem + '-fit.json'), fit)
    return fit


def run(out):
    protocol = validate_protocol(out)
    dest = out / 'run-01'
    dest.mkdir(exist_ok=False)
    start = time.perf_counter()
    write_json(dest / 'started.json', {'protocol_sha256': sha(out / 'protocol.json')})
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        p, r, a, ids = load_data(Path(protocol['data']), 'train')
        groups = partition(ids)
        expert_fits, gate_fits, constant_fits, cache_costs = [], [], [], []
        initial_parameters = {}
        full_prior = Path(protocol['coordination_experiment']) / 'run-01'
        for seed in SEEDS:
            full_fast, _full_slow = previous.experts(protocol, seed)
            with torch.no_grad():
                tokens = context_tokens(p[:, :32], r[:, :32], a[:, :31], full_fast.scales)
            if seed == SEEDS[0]:
                np.save(dest / 'train-tokens.npy', tokens.numpy(), allow_pickle=False)
            elif not np.array_equal(tokens.numpy(), np.load(dest / 'train-tokens.npy', allow_pickle=False)):
                raise ValueError('seed-dependent gate tokens')
            fold_caches = {}
            for fold in range(3):
                norm = fold_normalization(p, r, a, groups, fold)
                norm_file = dest / f'fold-{fold}-{seed}-normalization.npz'
                np.savez_compressed(norm_file, **norm)
                eligible = torch.from_numpy(norm['train_indices'])
                fa = (a - torch.from_numpy(norm['action_mean'])) / torch.from_numpy(norm['action_scale'])
                scales = torch.from_numpy(norm['motion_scales'])
                generator = torch.Generator().manual_seed(seed + 1000)
                orders = [eligible[torch.randperm(480, generator=generator)] for _ in range(46)]
                order_file = dest / f'fold-{fold}-{seed}-orders.npy'
                np.save(order_file, torch.stack(orders).numpy(), allow_pickle=False)
                models = []
                for variant in ('meta', 'gru'):
                    torch.manual_seed(seed)
                    model = make_model(variant, scales)
                    params = {k: v.detach().clone() for k, v in model.named_parameters()}
                    key = (variant, seed)
                    if key in initial_parameters:
                        if not all(torch.equal(params[k], v) for k, v in initial_parameters[key].items()):
                            raise ValueError('fold initial parameters differ')
                    else:
                        initial_parameters[key] = params
                    def objective(m, idx, v=variant, actions=fa):
                        pp, rr = predict(m, p[idx], r[idx], actions[idx], v)
                        return loss_function(pp, rr, p[idx, 32:], r[idx, 32:])
                    stem = f'fold-{fold}-{variant}-{seed}'
                    fit = fit_neural(model, orders, objective, dest, stem,
                        {'kind': 'expert', 'variant': variant, 'seed': seed, 'fold': fold,
                         'normalization_sha256': sha(norm_file)}, order_file)
                    expert_fits.append(fit)
                    # Reload the authenticated final weights; no initial or best checkpoint selection.
                    if sha(dest / (stem + '.pt')) != fit['checkpoint_sha256']:
                        raise ValueError('fold checkpoint changed')
                    weights = torch.load(dest / (stem + '.pt'), map_location='cpu', weights_only=True)
                    inferred = PoseSupport(scales) if variant == 'meta' else make_model('gru', scales)
                    inferred.load_state_dict(weights, strict=True)
                    inferred.eval().requires_grad_(False)
                    models.append(inferred)
                tick = time.perf_counter()
                with torch.no_grad():
                    fp, fr, sp, sr = previous.private_rollouts(*models, p, r, fa)
                for variant, pp, rr in (('fast', fp, fr), ('slow', sp, sr)):
                    np.savez_compressed(dest / f'fold-{fold}-{variant}-{seed}-cache.npz', p=pp.numpy(), R=rr.numpy())
                fold_caches[fold] = fp, fr, sp, sr
                cache_costs.append({'kind': 'fold', 'seed': seed, 'fold': fold, 'seconds': time.perf_counter()-tick})
            generator = torch.Generator().manual_seed(seed + 1000)
            gate_orders = [torch.randperm(720, generator=generator) for _ in range(30)]
            gate_order_file = dest / f'gate-{seed}-orders.npy'
            np.save(gate_order_file, torch.stack(gate_orders).numpy(), allow_pickle=False)
            combined_caches = {}
            for regime in REGIMES:
                tick = time.perf_counter()
                cache, assignments = mix_cache(fold_caches, groups, regime)
                fp, fr, sp, sr = cache
                np.savez_compressed(dest / f'cache-{regime}-{seed}.npz', fp=fp.numpy(), fR=fr.numpy(), sp=sp.numpy(),
                                    sR=sr.numpy(), ids=ids, folds=assignments)
                combined_caches[regime] = cache
                cache_costs.append({'kind': 'assembly', 'seed': seed, 'regime': regime, 'seconds': time.perf_counter()-tick})
                for kind in ('summary', 'recurrent'):
                    variant = regime + '_' + kind
                    torch.manual_seed(seed)
                    gate = make_gate(variant)
                    def objective(m, idx, cached=cache, token_data=tokens):
                        pp, rr = blend(*(x[idx] for x in cached), m(token_data[idx]))
                        return loss_function(pp, rr, p[idx, 32:], r[idx, 32:])
                    stem = f'{variant}-{seed}'
                    gate_fits.append(fit_neural(gate, gate_orders, objective, dest, stem,
                        {'kind': 'selector', 'variant': variant, 'seed': seed,
                         'cache_sha256': sha(dest / f'cache-{regime}-{seed}.npz')}, gate_order_file))
            full_cache = []
            for expert in ('fast', 'slow'):
                with np.load(full_prior / f'train-{expert}-{seed}.npz', allow_pickle=False) as arrays:
                    full_cache.extend((torch.from_numpy(arrays['p'].copy()), torch.from_numpy(arrays['R'].copy())))
            combined_caches['full'] = tuple(full_cache)
            for regime in ('is', 'oof', 'full'):
                tick = time.perf_counter()
                cache = combined_caches[regime]
                record = fit_constant(*(x.numpy() for x in cache), p[:, 32:].numpy(), r[:, 32:].numpy(),
                                      tolerance=1e-7, max_evaluations=4095)
                variant = regime + '_constant'
                record = {'variant': variant, 'seed': seed, 'regime': regime, 'optimization': record,
                          'seconds': time.perf_counter()-tick}
                file = dest / f'{variant}-{seed}-fit.json'
                write_json(file, record)
                constant_fits.append({'variant': variant, 'seed': seed, 'fit_sha256': sha(file), 'seconds': record['seconds']})
                print(json.dumps({'constant_fit': f'{variant}-{seed}', 'seconds': record['seconds']}), flush=True)
        write_json(dest / 'training-completed.json', {'expert_fits': expert_fits, 'gate_fits': gate_fits,
            'constant_fits': constant_fits, 'cache_costs': cache_costs, 'evaluation_started': False})
        rows = []
        for panel in PANELS:
            tp, tr, ta, test_ids = load_data(Path(protocol['data']), panel)
            np.savez_compressed(dest / (panel + '-targets.npz'), p=tp[:, 32:].numpy(), R=tr[:, 32:].numpy(), ids=test_ids)
            for variant in VARIANTS:
                for seed in SEEDS:
                    fast, slow = previous.experts(protocol, seed)
                    gate, alpha, digest = None, None, None
                    if variant in TRAIN_VARIANTS:
                        gate = make_gate(variant)
                        file = dest / f'{variant}-{seed}.pt'
                        fit = next(x for x in gate_fits if x['variant'] == variant and x['seed'] == seed)
                        digest = fit['checkpoint_sha256']
                        if sha(file) != digest:
                            raise ValueError('selector checkpoint changed')
                        gate.load_state_dict(torch.load(file, map_location='cpu', weights_only=True), strict=True)
                        gate.eval()
                    elif variant in CONSTANT_VARIANTS:
                        file = dest / f'{variant}-{seed}-fit.json'
                        fit = next(x for x in constant_fits if x['variant'] == variant and x['seed'] == seed)
                        digest = fit['fit_sha256']
                        if sha(file) != digest:
                            raise ValueError('constant record changed')
                        alpha = json.loads(file.read_text())['optimization']['alpha']
                    def call(cp, cr, pa, future, f=fast, s=slow, g=gate, c=alpha, v=variant):
                        return forecast(f, s, g, c, v, cp, cr, pa, future)
                    with torch.no_grad():
                        pp, rr, weights = call(tp[:, :32], tr[:, :32], ta[:, :31], ta[:, 31:])
                        metrics = measure(pp, rr, tp[:, 32:], tr[:, 32:])
                        timings = []
                        for i in range(23):
                            tick = time.perf_counter_ns()
                            call(tp[i:i+1, :32], tr[i:i+1, :32], ta[i:i+1, :31], ta[i:i+1, 31:])
                            elapsed = (time.perf_counter_ns()-tick)/1e6
                            if i >= 3:
                                timings.append(elapsed)
                    row = {'panel': panel, 'variant': variant, 'seed': seed, 'checkpoint_sha256': digest,
                           'expert_sha256': {n: protocol['checkpoints'][f'{n}-{seed}']['sha256'] for n in ('meta', 'gru')},
                           'metrics': metrics, 'latency_ms': timings}
                    prefix = f'{panel}-{variant}-{seed}'
                    np.savez_compressed(dest / (prefix + '-predictions.npz'), p=pp.numpy(), R=rr.numpy(), alpha=weights.numpy())
                    write_json(dest / (prefix + '-evaluation.json'), row)
                    rows.append(row)
                    print(json.dumps({'row': prefix, 'metrics': metrics}), flush=True)
        validate_protocol(out)
        write_json(dest / 'completed.json', {'status': 'completed', 'expert_fits': expert_fits, 'gate_fits': gate_fits,
            'constant_fits': constant_fits, 'cache_costs': cache_costs, 'rows': rows, 'new_optimizer_updates': 20700,
            'wall_seconds': time.perf_counter()-start, 'protocol_sha256': sha(out / 'protocol.json'),
            'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(dest.iterdir()) if p.is_file()}})
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
