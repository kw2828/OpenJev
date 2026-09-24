"""One engineering-only feasibility probe of the preselected update counts."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from finite_update_learning_worker import descriptor, files, publish, require
from register_finite_update_learning import EXPOSURE


def projections(allocations, fits):
    """Double measured stage cost, then add all non-stage fitting work.

    This is only a feasibility admission, never a runtime guarantee or a
    scientific speed result. It cannot change the preselected update counts.
    """
    rows = []
    require(len(allocations) == len(fits) == 3, 'complete engineering arm roster')
    for fit in fits:
        allocation = allocations[fit['arm']]
        stages = allocation['stages']
        require(len(stages) == 2, 'both engineering stages complete')
        stage_seconds = [row['stopped_elapsed'] - row['start_elapsed'] for row in stages]
        require(all(math.isfinite(x) and x >= 0 for x in stage_seconds), 'finite positive stage time')
        overhead = fit['seconds'] - sum(stage_seconds)
        require(math.isfinite(overhead) and overhead >= 0, 'full fit includes both measured stages')
        scales = [EXPOSURE['target_counts']['prefix_updates'] / EXPOSURE['config']['prefix_updates'],
                  EXPOSURE['target_counts']['joint_updates'] / EXPOSURE['config']['joint_updates']]
        projected = 2 * sum(x * scale for x, scale in zip(stage_seconds, scales, strict=True)) + overhead
        rows.append({'arm': fit['arm'], 'seed': fit['seed'], 'stage_seconds': stage_seconds,
                     'nonstage_seconds': overhead, 'safety_multiplier': 2.,
                     'projected_seconds': projected})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'exclusive engineering exposure directory')
    output.mkdir()
    started = time.perf_counter()
    record = {'version': 'finite-update-exposure-v1', 'status': 'FAILED', **EXPOSURE,
              'purpose': 'Feasibility only; no scientific effect selection or exposure tuning.'}
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from audit_finite_update_learning import audit
        from run_finite_update_learning import run

        checks = 0

        def check(*, hard=False):
            nonlocal checks
            checks += 1
            if time.perf_counter() - started >= 150.:
                raise TimeoutError('fixed engineering exposure cap')
            if hard or checks == 1 or checks % 256 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
                require(rss_bytes <= 4 * 1024**3, 'engineering exposure child RSS bound')
                paths = list(output.rglob('*'))
                require(len(paths) <= 1024 and sum(p.stat().st_size for p in paths if p.is_file())
                        <= 512 * 1024**2, 'engineering exposure output bounds')

        check(hard=True)
        result = run(output / 'run', EXPOSURE['config'], check)
        original_files = files(output / 'run')
        verified = audit(output / 'run', check=check, profile='engineering-942201')
        publish(output / 'audit.json', verified)
        require(files(output / 'run') == original_files, 'engineering producer unchanged by audit')
        allocations = {row['arm']: json.loads((output / 'run' / row['allocation']['path']).read_text())
                       for row in result['fits']}
        record['projections'] = projections(allocations, result['fits'])
        publish(output / 'projections.json', record['projections'])
        require(all(row['projected_seconds'] < EXPOSURE['maximum_projected_seconds']
                    for row in record['projections']), 'preselected science exposure must fit engineering feasibility bound')
        record['audit'] = descriptor(output / 'audit.json')
        record['run_files'] = original_files
        check(hard=True)
        record['status'] = 'PASS'
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['seconds'] = time.perf_counter() - started
        record['files_before_receipt'] = files(output)
        publish(output / 'receipt.json', record)
        print(json.dumps({'status': record['status'], 'seconds': record['seconds']}))


if __name__ == '__main__':
    main()
