"""Matched float64 NumPy/Rust score-kernel microbenchmark, no model inference."""

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def kernel(row, k):
    z = row[:k]
    probs = np.exp(z - z.max())
    probs /= probs.sum()
    full = np.exp(row - row.max())
    mass = full[:k].sum() / full.sum()
    positive = probs[probs > 0]
    entropy = -(positive * np.log(positive)).sum()
    return np.r_[probs, mass, entropy]


def run(binary, *, width=151936, rows=32, repeats=11, k=6):
    rng = np.random.default_rng(20260916)
    values = rng.normal(size=(rows, width)).astype('<f8')
    # Both languages receive these exact bytes, including difficult/extreme rows.
    values[0, :k] = 1000
    values[1, :k] = -1000
    values[2, :] = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'logits.bin'
        values.tofile(path)
        raw = subprocess.check_output([binary, str(path), str(width), str(k), str(repeats)], text=True)
    rust = json.loads(raw)
    for _ in range(3):
        for row in values:
            kernel(row, k)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        python_output = np.array([kernel(row, k) for row in values])
        times.append((time.perf_counter() - start) * 1000)
    difference = float(np.max(np.abs(python_output - np.array(rust['outputs']))))
    if difference > 1e-10:
        raise AssertionError(f'Parity failed: {difference}')
    def stats(batch_times):
        per_row = np.asarray(batch_times) / rows
        return {'median_ms_per_row': float(np.median(per_row)),
                'p25_ms_per_row': float(np.quantile(per_row, .25)),
                'p75_ms_per_row': float(np.quantile(per_row, .75)),
                'raw_batch_ms': batch_times}
    py, rs = stats(times), stats(rust['batch_ms'])
    return {
        'scope': 'score-kernel only; excludes model inference, HTTP, tokenization, Doom, startup and I/O',
        'implementation': 'Python uses NumPy native kernels; Rust uses scalar std f64 loops, release opt-level=3',
        'schedule': 'Rust first then Python, three warmups each; no simultaneous workload intended',
        'dtype': 'float64', 'vocabulary': width, 'candidates': k, 'rows': rows, 'repeats': repeats,
        'input_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
        'max_absolute_difference': difference,
        'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                        'numpy': np.__version__, 'rust': '1.94.0', 'cpu_limit': 2},
        'python_numpy': py, 'rust': rs,
        'python_time_divided_by_rust_time': py['median_ms_per_row']/rs['median_ms_per_row'],
        'interpretation': 'One local microbenchmark. Not end-to-end acceleration or a language-wide speed claim.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rust', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = json.dumps(run(args.rust), indent=2) + '\n'
    if args.output:
        with open(args.output, 'x') as output:
            output.write(result)
    else:
        print(result, end='')
