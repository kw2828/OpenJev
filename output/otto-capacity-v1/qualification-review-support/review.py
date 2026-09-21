"""Closed TRAIN-only qualification readback; no optimizer or simulator calls.

The pinned runner supplies only lineage/process authentication. Separate pinned
audit helpers supply numerical arithmetic. Recorded gradients, Torch parity,
initial RNG, optimizer execution and physical timing remain producer evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLOCK = ROOT / 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
KINDS = ('mlp8', 'mlp128', 'deep128')
LIMITS = {'seconds': 120, 'rss_bytes': 4*1024**3, 'output_bytes': 16*1024**2}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def expired(*_):
    raise TimeoutError('qualification readback wall cap')


def compute(args, runner, audit, plan, receipt, check):
    import numpy as np

    fit_ids = [f'{kind}@10101' for kind in KINDS]
    require(plan['mode'] == receipt['mode'] == 'qualify' and receipt['completed_fits'] == 3
            and receipt['parity_passed'] and receipt['pending'] == [], 'complete disposable qualification')
    require(set(receipt['files']) == runner.payload_names('qualify'), 'all20 qualification payloads')
    require(receipt['plan_sha256'] == args.plan_sha256 and receipt['sources'] == plan['sources']
            and receipt['inputs'] == plan['inputs'], 'qualification provenance')
    run = args.run
    prep, summary = read(run/'preparation.json'), read(run/'summary.json')
    expected = np.linspace(0, 5588, 256, dtype=np.int64)
    with np.load(plan['inputs']['train_data']['path'], allow_pickle=False) as archive:
        full_x, full_y = archive['features'], archive['target']
    require(full_x.dtype == full_y.dtype == np.float32 and full_x.shape == (5589, 11028)
            and full_y.shape == (5589,) and np.isfinite(full_x).all() and np.isfinite(full_y).all(), 'original TRAIN geometry')
    c0 = float(np.float32(np.mean(full_y, dtype=np.float64)))
    x, y = full_x[expected], full_y[expected]
    del full_x, full_y
    require(prep['row_indices'] == {'train': expected.tolist()} and prep['c0_float32'] == c0
            and prep['features'] == {'train': {'rows': 256, 'sha256': hashlib.sha256(x.tobytes()).hexdigest()}},
            'fixed selection, feature bytes and whole-TRAIN baseline')
    audit.close(prep['alias_floor'], audit.duplicate_floor(x, y, np, check), 'qualification alias floor')
    fits, init, orders, updates, curves, parity = [rows(run/name) for name in
        ('fits.jsonl', 'initializations.jsonl', 'epoch-orders.jsonl', 'updates.jsonl', 'fit-curves.jsonl', 'parity.jsonl')]
    require([r['fit_id'] for r in fits] == [r['fit_id'] for r in init] == [r['fit_id'] for r in parity] == fit_ids,
            'all three original-order fits and parity witnesses')
    require(len(orders) == len(curves) == 18 and len(updates) == 36, 'all fixed epochs and updates')
    first_layers, metrics = {}, {}
    for index, kind in enumerate(KINDS):
        check()
        fit, initialization, p = fits[index], init[index], parity[index]
        checkpoints = []
        for phase in ('initial', 'final'):
            path = run/f'{phase}-{kind}-10101.npz'
            with np.load(path, allow_pickle=False) as archive:
                checkpoints.append(audit.checkpoint(dict(archive), kind, c0, np))
            require(sha(path) == fit['initial_sha256' if phase == 'initial' else 'checkpoint_sha256'], 'saved fit checkpoint pins')
        initial, final = checkpoints
        first_layers[kind] = initial[0][0]
        require(all(not np.any(bias) for _, bias in initial), 'zero initial biases')
        count = audit.count_parameters(final)
        require(initialization == {'fit_id': fit_ids[index], 'kind': kind, 'seed': 10101,
                'initial_sha256': fit['initial_sha256'], 'parameter_count': count, 'c0_float32': c0}, 'initialization record')
        require(fit['kind'] == kind and fit['seed'] == 10101 and fit['rows'] == 256 and fit['epochs'] == 6
                and fit['parameter_count'] == count and math.isfinite(fit['fit_seconds']) and fit['fit_seconds'] > 0,
                'fixed fitted architecture and paid time')
        parameter_names = {f'{name}_{i}' for i in range(len(final)) for name in ('weight', 'bias')}
        require(set(fit['optimizer_steps']) == parameter_names and set(fit['optimizer_steps'].values()) == {12},
                'every parameter has twelve recorded Adam steps')
        rng = np.random.default_rng(30101)
        for epoch in range(1, 7):
            order = rng.permutation(256)
            order_sha = hashlib.sha256(order.tobytes()).hexdigest()
            record = orders[index*6+epoch-1]
            require(record == {'fit_id': fit_ids[index], 'epoch': epoch, 'order': order.tolist(), 'sha256': order_sha},
                    'paired seed permutation and order hash')
            pair = updates[index*12+(epoch-1)*2:index*12+epoch*2]
            for batch, update in enumerate(pair):
                require(update['fit_id'] == fit_ids[index] and update['epoch'] == epoch and update['batch'] == batch
                        and update['rows'] == 128 and update['update_index'] == 2*(epoch-1)+batch+1
                        and math.isfinite(update['loss']) and update['loss'] >= 0
                        and math.isfinite(update['gradient_norm']) and update['gradient_norm'] >= 0, 'complete finite update witness')
            audit.close(curves[index*6+epoch-1], {'fit_id': fit_ids[index], 'epoch': epoch, 'rows': 256, 'updates': 2,
                'training_mse_normalized': math.fsum(u['loss']*128 for u in pair)/256, 'order_sha256': order_sha}, 'online curve witness')
        probe = np.concatenate((x[:8].astype(np.float64), np.zeros((1, 11028)), x[:1].astype(np.float64)*.5))
        actual = audit.dense_predict(probe, final, c0, np)
        require(p['fit_id'] == fit_ids[index] and p['rows'] == 10 and p['passed'] is True
                and p['features_sha256'] == hashlib.sha256(probe.tobytes()).hexdigest(), 'ten fixed parity inputs')
        for name in ('numpy', 'torch'):
            other = np.asarray(p[name], np.float64)
            require(other.shape == (10,) and np.isfinite(other).all()
                    and np.all(np.abs(actual-other) <= 1e-10+1e-10*np.abs(other)), 'independent final scalar parity')
        saved_numpy, saved_torch = np.asarray(p['numpy']), np.asarray(p['torch'])
        require(np.all(np.abs(saved_numpy-saved_torch) <= 1e-10+1e-10*np.abs(saved_torch)), 'original parity gate unchanged')
        with np.load(run/f'predictions-{kind}-10101.npz', allow_pickle=False) as archive:
            require(set(archive) == {'train'}, 'TRAIN-only final predictions')
            saved = archive['train']
        predicted = audit.dense_predict(x.astype(np.float64), final, c0, np)
        require(saved.dtype == np.float64 and saved.shape == (256,) and np.isfinite(saved).all()
                and np.all(np.abs(saved-predicted) <= 1e-10+1e-10*np.abs(predicted)), 'all final TRAIN predictions')
        metrics[fit_ids[index]] = {'train': audit.scalar_metrics(predicted, y, np)}
        audit.close(fit['metrics'], metrics[fit_ids[index]], 'fit scalar metrics')
    require(all(np.array_equal(first_layers[k][:8], first_layers['mlp8']) for k in KINDS)
            and np.array_equal(first_layers['mlp128'], first_layers['deep128']), 'paired first-layer initialization')
    work = rows(run/'work.jsonl')
    require(len(work) == 120, 'sixty attempted and returned operations')
    counts, seconds = {}, {}
    for i in range(0, len(work), 2):
        attempt, returned = work[i:i+2]
        require(attempt['id'] == returned['id'] == i//2+1 and attempt['event'] == 'attempt'
                and returned['event'] == 'return' and attempt['channel'] == returned['channel']
                and attempt['context'] == returned['context'] and math.isfinite(returned['seconds'])
                and returned['seconds'] >= 0, 'atomic operation accounting')
        channel = attempt['channel']
        counts[channel] = counts.get(channel, 0)+1
        seconds[channel] = seconds.get(channel, 0.)+returned['seconds']
    require(counts == {'model_initialization': 3, 'checkpoint_export': 6, 'optimizer_initialization': 3,
        'optimizer_update': 36, 'parity_restore': 3, 'parity_forward': 6, 'saved_prediction': 3}, 'exact operation counts')
    audit.close(receipt['calls'], {k: {'attempted': n, 'returned': n, 'seconds': seconds[k]} for k, n in counts.items()}, 'receipt counters')
    require(math.fsum(seconds.values()) <= receipt['wall_seconds'] and math.fsum(f['fit_seconds'] for f in fits) <= receipt['wall_seconds'], 'disjoint paid work enclosure')
    projected = 3*math.fsum(f['fit_seconds'] for f in fits)*(3520/12)
    audit.close(summary['metrics'], metrics, 'complete saved metrics')
    audit.close(summary['alias_floor'], prep['alias_floor'], 'saved qualification floor')
    audit.close(summary['training_costs'], {f['fit_id']: f['fit_seconds'] for f in fits}, 'all paid fit times')
    audit.close(summary['qualification_projection_seconds'], projected, 'prospective projection')
    require(summary['mode'] == 'qualify' and summary['rows'] == {'train': 256}
            and summary['checks'] == [] and summary['family_admission'] == {} and summary['capacity_screen_passed'] is None
            and summary['learned_architecture_advantage_established'] is False,
            'qualification does not select a family or establish efficacy')
    return {'agreement': True, 'fits': 3, 'updates': 36, 'paired_epoch_orders': 18, 'parity_rows': 30,
            'prediction_rows': 768, 'local_numpy_readout_calls': 6, 'qualification_projection_seconds': projected,
            'cost_admission_passed': projected <= 1200, 'optimizer_steps_each_parameter': 12,
            'scope': __doc__, 'native_steps': 0, 'training_calls': 0, 'external_model_calls': 0}


def execute(args):
    require(args.output.is_absolute() and not args.output.exists() and not any(p.is_symlink() for p in args.output.parents), 'exclusive regular output')
    args.output.mkdir(parents=True)
    receipt = {'status': 'started', 'source_sha256': sha(__file__), 'limits': LIMITS, 'request': {k: str(v) for k, v in vars(args).items()}}
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
    try:
        require(sha(CLOCK) == CLOCK_PIN, 'qualified native clock source')
        clock = load(CLOCK, '_capacity_qualification_review_clock').SuspendClock()
        start = clock.now_ns()
        def check():
            require(clock.now_ns()-start < LIMITS['seconds']*10**9, 'native review cap')
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
            require(rss <= LIMITS['rss_bytes'] and sum(p.stat().st_size for p in args.output.iterdir()) <= LIMITS['output_bytes'], 'review resource caps')
        for path, pin in ((args.plan, args.plan_sha256), (args.run/'receipt.json', args.receipt_sha256),
                          (args.terminal, args.terminal_sha256), (ROOT/'scripts/study_otto_capacity.py', args.runner_sha256),
                          (ROOT/'scripts/audit_otto_capacity.py', args.auditor_sha256)):
            check()
            require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)) and sha(path) == pin, 'external input pin')
        runner = load(ROOT/'scripts/study_otto_capacity.py', '_capacity_qualification_auth')
        plan = runner.authenticate(args)
        worker = runner.closed(args.run/'receipt.json')
        runner.terminal_identity(args.run/'receipt.json', args.plan, args.terminal, 'scripts/study_otto_capacity.py')
        require({p.name for p in args.run.iterdir()} == set(worker['files']) | {'receipt.json'}, 'no late or extra run artifacts')
        audit = load(ROOT/'scripts/audit_otto_capacity.py', '_capacity_independent_arithmetic')
        result = compute(args, runner, audit, plan, worker, check)
        for path, pin in ((args.plan, args.plan_sha256), (args.run/'receipt.json', args.receipt_sha256),
                          (args.terminal, args.terminal_sha256), (ROOT/'scripts/study_otto_capacity.py', args.runner_sha256),
                          (ROOT/'scripts/audit_otto_capacity.py', args.auditor_sha256)):
            check()
            require(sha(path) == pin, 'unchanged external evidence/source at completion')
        check()
        write(args.output/'summary.json', result)
        receipt.update(status='completed', agreement=True, plan_sha256=args.plan_sha256,
            input_receipt_sha256=args.receipt_sha256, terminal_sha256=args.terminal_sha256,
            clock_backend=clock.backend, wall_seconds=(clock.now_ns()-start)/1e9,
            files={'summary.json': {'sha256': sha(args.output/'summary.json'), 'bytes': (args.output/'summary.json').stat().st_size}})
        write(args.output/'receipt.json', receipt)
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(status='failed', agreement=False, error=repr(error))
        try:
            if (args.output/'receipt.json').exists():
                (args.output/'receipt.json').rename(args.output/'invalid-completed-receipt.json')
            write(args.output/'failed.json', receipt)
        except BaseException as secondary:  # noqa: BLE001 - Preserve the original failure.
            error.add_note(f'Failure publication: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'run', 'terminal', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('plan', 'receipt', 'terminal', 'runner', 'auditor'):
        parser.add_argument('--'+name+'-sha256', required=True)
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(variable) == '1', 'CPU1 review environment')
    execute(parser.parse_args())
