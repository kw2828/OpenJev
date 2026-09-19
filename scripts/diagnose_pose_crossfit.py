"""Post-outcome expert errors on sealed training caches; no neural calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_pose_crossfit as audit
import numpy as np

EXPECTED_COMPLETION = '3f57c8727e574a01e078ffea2e3cce5179619a72c8753e1b928d4f3115759be2'
EXPECTED_PROTOCOL = 'b9e0d4c9a43527cd14d1931242cd130fd224eb948cf6f7d8662e43cb9d2b5143'


def run(experiment, report, out):
    context = audit.validate_inputs(experiment, EXPECTED_PROTOCOL)
    source = experiment / 'run-01'
    before = audit.members(source)
    audit.require(before['completed.json']['sha256'] == EXPECTED_COMPLETION, 'completion binding')
    done = audit.read_json(source / 'completed.json')
    audit.require(done['status'] == 'completed' and done['files'] == {
        k: v for k, v in before.items() if k != 'completed.json'}, 'execution seal')
    summary = audit.read_json(report / 'summary.json')
    receipt = audit.read_json(report / 'receipt.json')
    report_members = audit.members(report)
    audit.require(summary['status'] == receipt['status'] == 'completed'
        and summary['execution_completed_sha256'] == receipt['execution_completed_sha256'] == EXPECTED_COMPLETION
        and receipt['execution_members'] == before
        and set(report_members) == {'summary.json', 'window-errors.npz', 'receipt.json'}
        and receipt['files'] == {k: v for k, v in report_members.items() if k != 'receipt.json'}, 'completed audit binding')
    out.mkdir(parents=True, exist_ok=False)
    try:
        p, r, _a, ids = audit._previous.load_public(context['data'], 'train')
        rows = {}
        for regime in ('is', 'oof', 'full'):
            for seed in audit.SEEDS:
                if regime == 'full':
                    cache = tuple(x for expert in ('fast', 'slow') for x in audit._previous.load_prediction(
                        context['prior_run'] / f'train-{expert}-{seed}.npz'))
                else:
                    with np.load(source / f'cache-{regime}-{seed}.npz', allow_pickle=False) as saved:
                        cache = tuple(saved[k] for k in ('fp', 'fR', 'sp', 'sR'))
                for name, pp, rr in (('fast', *cache[:2]), ('slow', *cache[2:])):
                    rows[f'{regime}-{name}-{seed}'] = audit.reduce_errors(
                        audit.error_arrays(pp, rr, p[:, 32:], r[:, 32:]), ids)
        families = {regime: {v: audit.pooled([rows[f'{regime}-{v}-{s}'] for s in audit.SEEDS])
                            for v in ('fast', 'slow')} for regime in ('is', 'oof', 'full')}
        audit.write_json(out / 'training-experts.json', {
            'status': 'completed', 'scope': 'post-outcome descriptive training-cache diagnosis',
            'execution_completed_sha256': EXPECTED_COMPLETION, 'protocol_sha256': EXPECTED_PROTOCOL,
            'audited_summary_sha256': audit.sha(report / 'summary.json'), 'rows': rows, 'families': families,
            'windows_per_regime_seed': 720, 'parents': 30, 'horizon': 25,
            'limits': ['OOF excludes the queried parent from expert training. IS and full include it.',
                       'Same training-domain cases across regimes; no fresh evaluation archive.',
                       'Expert training composition changes, so membership is not the sole isolated cause.',
                       'Post-outcome explanation only; no refitting, new forecasts or continuation-rule change.']})
        audit.require(audit.members(source) == before and audit.members(report) == report_members,
                      'evidence changed during diagnosis')
        audit.validate_inputs(experiment, EXPECTED_PROTOCOL)
        audit.write_json(out / 'receipt.json', {
            'status': 'completed', 'source_sha256': audit.sha(__file__),
            'inputs': {'completion_sha256': EXPECTED_COMPLETION, 'protocol_sha256': EXPECTED_PROTOCOL,
                       'audit_members': report_members, 'execution_members': before,
                       'data_hashes': context['protocol']['data_hashes'],
                       'prior_completion_sha256': context['protocol']['coordination_completed_sha256']},
            'files': audit.members(out), 'new_model_calls': 0, 'new_optimizer_calls': 0, 'new_random_draws': 0})
        print(json.dumps({k: {v: {e: row[e]['rmse'] for e in audit.ENDPOINTS}
                             for v, row in family.items()} for k, family in families.items()}))
    except BaseException as error:
        audit.write_json(out / 'failed.json', {'status': 'failed', 'error': repr(error),
                                              'source_sha256': audit.sha(__file__)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment, args.report, args.out)
