"""Frozen training-parent screen of explicit, ordered prediction-error memory."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from train_pose_adaptation import load_data, loss_function, make_model, measure, sha, write_json

from openjev.research.pose_innovation import (
    RecurrentInnovation,
    SummaryInnovation,
    apply_correction,
    innovation_tokens,
    residual_targets,
)

PARENT = Path('output/pose-crossfit-v1/experiment-01')
DATA = Path('output/residual-dynamics-v1/data-01')
PARENT_PROTOCOL = 'b9e0d4c9a43527cd14d1931242cd130fd224eb948cf6f7d8662e43cb9d2b5143'
PARENT_COMPLETED = '3f57c8727e574a01e078ffea2e3cce5179619a72c8753e1b928d4f3115759be2'
SEEDS = (1101, 1202, 1303)
NEURAL = ('summary', 'recurrent', 'shuffled', 'noerror', 'error_shuffled')
VARIANTS = ('base', 'bias', 'ridge_summary', 'ridge_ordered', *NEURAL)
EPOCHS, BATCH = 40, 32
TEST_RANKS = (1, 4, 7, 9)
SOURCES = (
    'scripts/run_pose_innovation.py', 'scripts/audit_pose_innovation.py',
    'src/openjev/research/pose_innovation.py', 'tests/test_pose_innovation.py',
    'tests/test_pose_innovation_runner.py', 'tests/test_audit_pose_innovation.py',
    'src/openjev/research/pose_transport.py', 'src/openjev/research/rigid_motion.py',
    'src/openjev/research/pose_coordination.py', 'src/openjev/research/pose_adaptation.py',
    'src/openjev/research/pose_references.py', 'scripts/train_pose_adaptation.py',
    'research/pose-innovation-protocol.md',
)


def split_indices(ids, fold):
    ids = np.asarray(ids)
    if ids.shape != (720, 2) or ids.dtype != np.int64 or fold not in (0, 1, 2):
        raise ValueError('fixed original 720-window identity required')
    expected = np.column_stack((np.repeat(np.arange(30), 24), np.tile(np.arange(24)*50, 30)))
    if not np.array_equal(ids, expected):
        raise ValueError('original parent/start order required')
    selected = np.flatnonzero(ids[:, 0] % 3 == fold)
    rank = ids[selected, 0] // 3
    test = np.isin(rank, TEST_RANKS)
    return selected, ~test, test


def interior_permutations(count, seed):
    rng = np.random.default_rng(seed)
    return np.asarray([np.r_[0, rng.permutation(np.arange(1, 30)), 30]
                       for _ in range(count)], dtype=np.int64)


def variant_tokens(tokens, variant, permutations):
    if variant == 'noerror':
        result = tokens.clone()
        result[..., 49:] = 0
        return result
    if variant in ('shuffled', 'error_shuffled'):
        shuffled = tokens.gather(1, torch.from_numpy(permutations).unsqueeze(-1).expand_as(tokens))
        if variant == 'shuffled':
            return shuffled
        result = tokens.clone()
        result[..., 49:] = shuffled[..., 49:]
        return result
    return tokens


def summary_features(tokens):
    return torch.cat((tokens[:, -1], tokens.mean(1), tokens[:, -1] - tokens[:, 0]), -1)


def ridge_predict(features, targets, train_mask):
    """Centered, row-normalized dual ridge; lambda=1 and unpenalized intercept."""
    x = features.double()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    y = targets.double().flatten(1)
    mask = torch.from_numpy(train_mask)
    xm, ym = x[mask].mean(0), y[mask].mean(0)
    xc, yc = x[mask] - xm, y[mask] - ym
    dual = torch.linalg.solve(xc @ xc.T + torch.eye(len(xc), dtype=x.dtype), yc)
    weights = xc.T @ dual
    correction = ((x-xm) @ weights + ym).reshape_as(targets).float()
    return correction, {'x_mean': xm.numpy(), 'y_mean': ym.numpy(), 'weights': weights.numpy()}


@torch.no_grad()
def causal_cache(model, p, r, past, future):
    """Only context32 is accepted; predict before assimilating each successor."""
    if p.shape[1:] != (32, 3) or r.shape != (len(p), 32, 3, 3):
        raise ValueError('exact32 context poses')
    if past.shape != (len(p), 31, 40) or future.shape != (len(p), 25, 40):
        raise ValueError('exact completed and forecast actions')
    state = model.assimilate(model.initial(p[:, 0], r[:, 0]), p[:, 0], r[:, 0])
    one_p, one_r = [], []
    for t in range(31):
        state = model.advance(state, past[:, t])
        one_p.append(state['p'])
        one_r.append(state['R'])
        state = model.assimilate(state, p[:, t+1], r[:, t+1])
    forecast_p, forecast_r = [], []
    for t in range(25):
        state = model.advance(state, future[:, t])
        forecast_p.append(state['p'])
        forecast_r.append(state['R'])
    return tuple(torch.stack(values, 1) for values in (one_p, one_r, forecast_p, forecast_r))


def bindings():
    if sha(PARENT/'protocol.json') != PARENT_PROTOCOL or sha(PARENT/'run-01/completed.json') != PARENT_COMPLETED:
        raise ValueError('parent identity changed')
    protocol = json.loads((PARENT/'protocol.json').read_text())
    completed = json.loads((PARENT/'run-01/completed.json').read_text())
    result = {}
    for name in ('train.npz', 'normalization.npz', 'manifest.json', 'completed.json'):
        path = DATA/name
        if sha(path) != protocol['data_hashes'][name]:
            raise ValueError('training data changed')
        result[str(path)] = sha(path)
    for name, digest in protocol['sources'].items():
        if sha(name) != digest:
            raise ValueError('inherited source changed: '+name)
    for seed in SEEDS:
        for fold in range(3):
            stem = f'fold-{fold}-gru-{seed}'
            names = (stem+'.pt', stem+'-fit.json', f'fold-{fold}-{seed}-normalization.npz',
                     f'fold-{fold}-slow-{seed}-cache.npz')
            for name in names:
                path = PARENT/'run-01'/name
                if sha(path) != completed['files'][name]['sha256']:
                    raise ValueError('inherited artifact changed: '+name)
                result[str(path)] = sha(path)
    result[str(PARENT/'protocol.json')] = PARENT_PROTOCOL
    result[str(PARENT/'run-01/completed.json')] = PARENT_COMPLETED
    return result


def freeze(out):
    authenticated = bindings()
    protocol = {
        'study': 'pose-innovation-v1', 'scope': 'training-parent held-out mechanism screen, not confirmation',
        'seeds': SEEDS, 'folds': [0, 1, 2], 'variants': VARIANTS, 'neural_variants': NEURAL,
        'primary': 'recurrent', 'test_ranks': TEST_RANKS,
        'split': 'backbone excludes parent%3=fold; within excluded10, parent//3 in[1,4,7,9] is adapter test',
        'windows_per_parent': 24, 'train_windows_per_fold': 144, 'test_windows_per_fold': 96,
        'context': 32, 'tokens': 31, 'token_dim': 55, 'horizon': 25,
        'epochs': EPOCHS, 'batch': BATCH, 'updates_per_fit': 200, 'fits': 45, 'updates': 9000,
        'optimizer': 'Adam', 'learning_rate': .001, 'clip_norm': 1.,
        'correction_scales': [.1, .1], 'input_error_scales': 'frozen fold motion_scales[2:4]',
        'shuffle': 'one fixed independent permutation per case, first/last fixed; NumPy seed=seed+10000+fold; shuffled permutes all55 channels, error_shuffled only49:55',
        'bias': 'sum of31 completed root-frame one-step endpoint residuals /32, multiplied by lead1..25',
        'ridge': 'L2-normalize each feature row, center on adapter train; dual ridge lambda1, unpenalized intercept; tangent target squared loss',
        'neural_loss': 'physical position MSE/.1^2 + geodesic rotation MSE/.1^2, average25 leads',
        'screen': 'primary pooled RMSE <=.9*each control on both endpoints; all9 fold/seed paired MSE nonworse; >=10/12 parents nonworse; all12 leave-one-parent-out MSE strictly lower',
        'screen_scope': 'signal prerequisite only; not previous exposed-panel continuation gate or novel architecture claim',
        'no_selection': 'final checkpoint only; no retry, tuning, selection, dropped parents or external panels',
        'latency': 'cached correction head only;3 warmups and12 timed single windows per fit/variant; not full deployment cost',
        'bindings': authenticated, 'sources': {name: sha(name) for name in SOURCES},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                    'platform': platform.platform(), 'threads': 1},
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protocol.json', protocol)
    for name in SOURCES:
        target = out/'source-snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    print(json.dumps({'frozen': str(out), 'sha256': sha(out/'protocol.json')}), flush=True)


def validate(out, digest):
    if sha(out/'protocol.json') != digest:
        raise ValueError('external protocol pin mismatch')
    protocol = json.loads((out/'protocol.json').read_text())
    runtime = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
               'platform': platform.platform(), 'threads': 1}
    if protocol['runtime'] != runtime:
        raise ValueError('frozen runtime changed')
    for name, expected in {**protocol['bindings'], **protocol['sources']}.items():
        if sha(name) != expected:
            raise ValueError('frozen input changed: '+name)
    for name, expected in protocol['sources'].items():
        if sha(out/'source-snapshot'/name) != expected:
            raise ValueError('source snapshot changed')
    return protocol


def run(out, digest):
    validate(out, digest)
    dest = out/'run-01'
    dest.mkdir(exist_ok=False)
    begin = time.perf_counter()
    fits, rows, caches = [], [], []
    updates = 0
    try:
        write_json(dest/'started.json', {'protocol_sha256': digest})
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        p, r, actions, ids = load_data(DATA, 'train')
        output_scale = torch.tensor([.1, .1])
        for seed in SEEDS:
            for fold in range(3):
                stem = f'fold-{fold}-{seed}'
                indices, train_mask, test_mask = split_indices(ids, fold)
                with np.load(PARENT/'run-01'/f'{stem}-normalization.npz', allow_pickle=False) as norm:
                    scales = torch.from_numpy(norm['motion_scales'].copy())
                    normalized = (actions[indices]-torch.from_numpy(norm['action_mean']))/torch.from_numpy(norm['action_scale'])
                    if not np.array_equal(norm['train_indices'], np.flatnonzero(ids[:, 0] % 3 != fold)):
                        raise ValueError('backbone training exclusion mismatch')
                model = make_model('gru', scales)
                model.load_state_dict(torch.load(PARENT/'run-01'/f'fold-{fold}-gru-{seed}.pt', weights_only=True))
                model.eval().requires_grad_(False)
                cp, cr = p[indices, :32], r[indices, :32]
                tick = time.perf_counter()
                one_p, one_r, bp, br = causal_cache(model, cp, cr, normalized[:, :31], normalized[:, 31:])
                cache_seconds = time.perf_counter()-tick
                with np.load(PARENT/'run-01'/f'fold-{fold}-slow-{seed}-cache.npz', allow_pickle=False) as inherited:
                    replay = {key: float(np.max(np.abs(x.numpy()-inherited[key][indices])))
                              for key, x in (('p', bp), ('R', br))}
                    for key, x in (('p', bp), ('R', br)):
                        np.testing.assert_allclose(x.numpy(), inherited[key][indices], rtol=2e-6, atol=2e-7)
                root_r, tp, tr = cr[:, -1], p[indices, 32:], r[indices, 32:]
                tokens = innovation_tokens(cp, cr, normalized[:, :31], one_p, one_r, scales, scales[2:4])
                target = residual_targets(bp, br, tp, tr, root_r, output_scale)
                one_errors = residual_targets(one_p, one_r, cp[:, 1:], cr[:, 1:], root_r, output_scale)
                permutations = interior_permutations(len(indices), seed+10000+fold)
                np.savez_compressed(dest/(stem+'-cache.npz'), ids=ids[indices], indices=indices,
                    train_mask=train_mask, test_mask=test_mask, tokens=tokens.numpy(), target=target.numpy(),
                    base_p=bp.numpy(), base_R=br.numpy(), target_p=tp.numpy(), target_R=tr.numpy(),
                    root_R=root_r.numpy(), one_errors=one_errors.numpy(), permutations=permutations,
                    context_p=cp.numpy(), context_R=cr.numpy(), past_actions=normalized[:, :31].numpy(),
                    one_p=one_p.numpy(), one_R=one_r.numpy(), motion_scales=scales.numpy())
                cache_record = {'seed': seed, 'fold': fold, 'seconds': cache_seconds, 'replay_max': replay,
                                'cache_sha256': sha(dest/(stem+'-cache.npz'))}
                caches.append(cache_record)
                train_idx = torch.from_numpy(np.flatnonzero(train_mask))
                gen = torch.Generator().manual_seed(seed+1000+fold)
                orders = torch.stack([train_idx[torch.randperm(len(train_idx), generator=gen)] for _ in range(EPOCHS)])
                np.save(dest/(stem+'-orders.npy'), orders.numpy(), allow_pickle=False)
                corrections = {'base': torch.zeros_like(target),
                    'bias': one_errors.sum(1)[:, None]/32*torch.arange(1,26)[None,:,None]}
                for kind, features in (('ridge_summary', summary_features(tokens)), ('ridge_ordered', tokens.flatten(1))):
                    corrections[kind], ridge = ridge_predict(features, target, train_mask)
                    np.savez_compressed(dest/(stem+'-'+kind+'-weights.npz'), **ridge)
                for variant in NEURAL:
                    tick = time.perf_counter()
                    torch.manual_seed(seed)
                    head = SummaryInnovation() if variant == 'summary' else RecurrentInnovation()
                    head.train()
                    fitstem = stem+'-'+variant
                    torch.save(head.state_dict(), dest/(fitstem+'-initial.pt'))
                    inputs = variant_tokens(tokens, variant, permutations)
                    optimizer = torch.optim.Adam(head.parameters(), lr=.001)
                    losses = []
                    for order in orders:
                        for idx in order.split(BATCH):
                            correction = head(inputs[idx])
                            pp, rr = apply_correction(bp[idx], br[idx], root_r[idx], correction, output_scale)
                            loss = loss_function(pp, rr, tp[idx], tr[idx])
                            if not torch.isfinite(loss):
                                raise FloatingPointError('nonfinite training loss')
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                            optimizer.step()
                            updates += 1
                            losses.append(float(loss.detach()))
                    head.eval()
                    torch.save(head.state_dict(), dest/(fitstem+'.pt'))
                    np.save(dest/(fitstem+'-losses.npy'), np.asarray(losses), allow_pickle=False)
                    with torch.no_grad():
                        corrections[variant] = head(inputs)
                        timings = []
                        for j in np.flatnonzero(test_mask)[:15]:
                            tic = time.perf_counter_ns()
                            _ = apply_correction(bp[j:j+1], br[j:j+1], root_r[j:j+1], head(inputs[j:j+1]), output_scale)
                            timings.append((time.perf_counter_ns()-tic)/1e6)
                    record = {'seed': seed, 'fold': fold, 'variant': variant, 'updates': len(losses),
                        'parameters': sum(v.numel() for v in head.parameters()), 'seconds': time.perf_counter()-tick,
                        'initial_sha256': sha(dest/(fitstem+'-initial.pt')), 'final_sha256': sha(dest/(fitstem+'.pt')),
                        'head_latency_ms': timings[3:], 'warmup_ms': timings[:3],
                        'first_loss': losses[0], 'last_loss': losses[-1]}
                    fits.append(record)
                    write_json(dest/(fitstem+'-fit.json'), record)
                    print(json.dumps(record), flush=True)
                for variant in VARIANTS:
                    correction = corrections[variant]
                    with torch.no_grad():
                        pp, rr = apply_correction(bp, br, root_r, correction, output_scale)
                    np.savez_compressed(dest/(stem+'-'+variant+'-predictions.npz'),
                                        p=pp.numpy(), R=rr.numpy(), correction=correction.numpy())
                    record = {'seed': seed, 'fold': fold, 'variant': variant,
                        'train': measure(pp[train_mask], rr[train_mask], tp[train_mask], tr[train_mask]),
                        'test': measure(pp[test_mask], rr[test_mask], tp[test_mask], tr[test_mask])}
                    rows.append(record)
                    write_json(dest/(stem+'-'+variant+'-evaluation.json'), record)
        validate(out, digest)
        if len(fits) != 45 or updates != 9000 or len(rows) != 81:
            raise ValueError('fixed execution budget mismatch')
        files = {str(path.relative_to(dest)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(dest.iterdir()) if path.is_file()}
        write_json(dest/'completed.json', {'status': 'completed', 'protocol_sha256': digest,
            'fits': fits, 'rows': rows, 'caches': caches, 'optimizer_updates': updates,
            'external_panel_calls': 0, 'frozen_backbone_batch_calls': 9,
            'wall_seconds': time.perf_counter()-begin, 'files': files})
        print(json.dumps({'completed': str(dest), 'sha256': sha(dest/'completed.json')}), flush=True)
    except BaseException as error:
        try:
            if (dest/'completed.json').exists():
                (dest/'completed.json').rename(dest/'completion-before-error.json')
            write_json(dest/'failed.json', {'status': 'failed', 'error': repr(error), 'fits_completed': len(fits),
                'rows_completed': len(rows), 'optimizer_updates': updates, 'wall_seconds': time.perf_counter()-begin})
        except BaseException as receipt_error:  # noqa: BLE001 - preserve the original execution failure
            error.add_note('Failure receipt could not be written: '+repr(receipt_error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.out)
    else:
        if not args.protocol_sha256:
            parser.error('run requires an external protocol SHA256')
        run(args.out, args.protocol_sha256)
