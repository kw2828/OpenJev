"""Synthetic cross-module contract checks, no science inputs or fitting."""
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'integration-01'
SOURCES = ('scripts/study_otto_return_value.py', 'scripts/audit_otto_return_value.py',
           'src/openjev/research/otto_return_value.py', 'src/openjev/research/otto_value_branches.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    OUT.mkdir(exist_ok=False)
    start = time.monotonic()
    before = {p: digest(ROOT / p) for p in SOURCES}
    receipt = {'status': 'started', 'sources': before, 'wall_cap_seconds': 60,
               'science_data_reads': 0, 'fitting_calls': 0, 'simulator_calls': 0, 'api_calls': 0}
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('synthetic integration cap')))
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        import numpy as np
        runner = module('_integration_runner', SOURCES[0])
        audit = module('_integration_audit', SOURCES[1])
        model = module('_integration_model', SOURCES[2])
        branch = module('_integration_branches', SOURCES[3])
        assert runner.payload_names() == audit.payload_names()
        assert [(r, s, h, a, b) for r, s, b, h, a in runner.evaluation_order()] == list(audit.evaluation_order())
        rng = np.random.default_rng(701)
        kernel = rng.uniform(0, 0.24, (4, 107, 107))
        kernel[:, 53, 53] = 0
        cases = 0
        for kind in model.KINDS:
            export = {'version': np.array(model.VERSION), 'kind': np.array(kind),
                      'input_dim': np.array(model.INPUT_DIM), 'c0': np.array(.4375, np.float32),
                      'first_weight': rng.normal(0, .1, (8, model.INPUT_DIM)).astype(np.float32)}
            if kind != 'min8':
                export['final_weight'] = rng.normal(0, .1, 8).astype(np.float32)
            if kind == 'mlp8':
                export['hidden_bias'] = rng.normal(0, .1, 8).astype(np.float32)
                export['output_bias'] = np.array(-.25, np.float32)
            frozen = model.FrozenValue(export)
            weights = audit.checkpoint(export, kind, .4375, np)
            for lam in (3., 4., 5.):
                for position in ((0, 0), (26, 26), (52, 52)):
                    belief = rng.uniform(0, 1, (53, 53))
                    belief /= belief.sum()
                    belief[position] = 0
                    allowed = [a for a in range(4) if audit.B.move(position, a) != list(position)]
                    built = branch.rl_branches(belief, position, kernel, allowed)
                    x, raw, weight = audit.branches(belief, position, kernel, lam, np)
                    np.testing.assert_array_equal(raw, built.raw_masses)
                    np.testing.assert_array_equal(weight, built.weights)
                    np.testing.assert_array_equal(x, model.value_features(built.centered_z, built.successors, lam))
                    values = 64 * frozen.normalized(x)
                    np.testing.assert_allclose(values, 64 * audit.predict(x, weights, kind, np), atol=1e-12, rtol=1e-12)
                    scores = branch.explicit_scores(built, lambda *_: values)
                    np.testing.assert_allclose(scores, audit.costs(values, weight, np), atol=1e-12, rtol=1e-12)
                    assert branch.select_action(scores, allowed) == audit.choice(scores.tolist(), allowed, True, np)
                    cases += 1
        rows = []
        for regime, seed, block, hit, arm in runner.evaluation_order():
            row = {'regime': regime, 'seed': seed, 'block': block, 'initial_hit': hit, 'arm': arm,
                   'steps': 50 + block + hit + len(arm), 'found': True, 'blocked_steps': 0,
                   'init_seconds': .002, 'choose_seconds': .003 * (1 + len(arm)),
                   'update_seconds': .001, 'setup_allocation_seconds': .0001,
                   'environment_seconds': .005, 'state_bytes': 22472}
            row['updates'] = row['steps']
            row['controller_seconds'] = sum(row[k] for k in runner.TIMES)
            rows.append(row)
        mixtures = {regime: {1: .7, 2: .2, 3: .1} for regime in runner.REGIMES}
        produced = json.loads(json.dumps(runner.summary(rows, mixtures)))
        independent = audit.aggregate(rows, mixtures)
        comparisons = audit.B.Comparisons()
        for key, value in independent.items():
            comparisons.tree(produced[key], value, 'synthetic cross-module aggregation/' + key)
        assert before == {p: digest(ROOT / p) for p in SOURCES}
        receipt.update(status='completed', branch_cases=cases, synthetic_episodes=len(rows),
                       conditions=54, aggregate_comparisons=comparisons.count)
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(wall_seconds=time.monotonic() - start, source_sha256=digest(Path(__file__)),
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                       * (1 if sys.platform == 'darwin' else 1024))
        with (OUT / 'receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
