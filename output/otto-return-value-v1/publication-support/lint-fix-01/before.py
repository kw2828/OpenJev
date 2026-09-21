"""Build a complete results note from externally pinned, closed saved evidence.

This is publication support, outside the frozen scientific source closure.
It never imports scientific runners, arrays, models or simulator code. It must
not be invoked until both the worker and independent audit have completed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'output/otto-return-value-v1'
REPORT = ROOT/'research/otto-return-value-results.md'
PLAN_SHA = '92df71d5e20ca48d8485bfa0a5f50bdf0e6e7a276c8e0c097bccd8e1c095aee2'
RUNNER = 'scripts/study_otto_return_value.py'
AUDITOR = 'scripts/audit_otto_return_value.py'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_SHA = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
LIMITS = {'native_seconds': 300, 'rss_bytes': 1024**3, 'output_bytes': 16*1024**2}
REGIMES = ('lambda3', 'lambda4', 'lambda5')
SEEDS = (10101, 10102, 10103)
FAMILIES = ('min8', 'mlp8', 'homogeneous8')
ARMS = tuple(f'{family}@{seed}' for seed in SEEDS for family in FAMILIES)+('analytic_inbounds',)
GROUPS = (('competence_checks', 'Competence', 18), ('compression_checks', 'Utility/computation', 12),
          ('architecture_checks', 'Architecture', 24))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def regular(path):
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'absolute regular nonsymlink evidence')
    return path


def digest(path, check=lambda: None):
    regular(path)
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            value.update(block)
    return {'sha256': value.hexdigest(), 'bytes': path.stat().st_size}


def payloads():
    result = {'started.json', 'runtime.json', 'native-setup.json', 'qualification.json', 'qualification.jsonl',
              'work-contexts.jsonl', 'work.jsonl', 'preparation.json', 'fits.jsonl', 'fit-curves.jsonl',
              'epoch-orders.jsonl', 'parity.jsonl', 'parity.json', 'inference-setup.json', 'eval-transitions.jsonl',
              'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json', 'training-costs.json'}
    result.update(f'kernel-{regime}.npz' for regime in REGIMES)
    result.update(f'{split}-{suffix}' for split in ('train', 'valid') for suffix in ('data.npz', 'rows.jsonl'))
    result.update(f'{prefix}-{family}-{seed}.npz' for prefix in ('final', 'predictions') for family in FAMILIES for seed in SEEDS)
    return result


def numeric(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite reported number')
    return value


def relation(name):
    if name.endswith(('.success', '.positive_blocks')):
        return '>='
    return '<' if name.endswith('.every_cost') else '<='


def condition_value(name, value):
    numeric(value)
    if name.endswith('.success'):
        return f'{100*value:.2f}%'
    if name.endswith('.positive_blocks'):
        return str(int(value))
    if name.endswith(('.cost', '.cost80', '.every_cost')):
        return f'{value:.6f} s'
    return f'{value:.3f}'


class Builder:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = None
        self.published = False
        self.receipt = {'version': 'otto-return-value-publication-v1', 'status': 'started', 'limits': LIMITS,
                        'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0,
                        'scope': 'Saved summary publication only. Inherits the completed independent numerical audit and execution evidence.'}

    def check(self):
        require(self.clock.now_ns()-self.start < LIMITS['native_seconds']*10**9, 'publication native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'publication memory cap')
        size = sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
        if self.published:
            size += REPORT.stat().st_size
        require(size <= LIMITS['output_bytes'], 'publication output cap')

    def closed(self, directory, pin, expected):
        path = regular(directory/'receipt.json')
        require(digest(path, self.check)['sha256'] == pin, 'external completed receipt pin')
        receipt = read(path)
        require(receipt['status'] == 'completed' and set(receipt['files']) == expected
                and {p.name for p in directory.iterdir()} == expected | {'receipt.json'}, 'exact complete artifact closure')
        for name, recorded in receipt['files'].items():
            require(digest(directory/name, self.check) == recorded, 'closed artifact hash before summary decoding')
        return receipt

    def authenticate(self):
        a = self.args
        require(a.plan == BASE/'plan-01.json' and a.run == BASE/'run-01' and a.audit == BASE/'audit-01'
                and a.terminal == BASE/'run-process-01.terminal.json', 'fixed completed study paths')
        require(a.plan_sha256 == PLAN_SHA and digest(a.plan, self.check)['sha256'] == PLAN_SHA, 'frozen external study plan')
        require(digest(a.terminal, self.check)['sha256'] == a.terminal_sha256, 'external parent terminal pin')
        plan, terminal = read(a.plan), read(a.terminal)
        require(plan['version'] == 'otto-return-value-v1' and plan['status'] == 'frozen_before_native_run', 'frozen experiment identity')
        for name, pin in plan['sources'].items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'contained scientific source')
            require(digest(ROOT/rel, self.check)['sha256'] == pin, 'unchanged scientific source closure')
        for desc in plan['inputs'].values():
            rel = Path(desc['path'])
            require(not rel.is_absolute() and '..' not in rel.parts, 'contained prior input')
            require(digest(ROOT/rel, self.check) == {k: desc[k] for k in ('sha256', 'bytes')}, 'unchanged prior input pin')
        worker = self.closed(a.run, a.receipt_sha256, payloads())
        audit = self.closed(a.audit, a.audit_receipt_sha256, {'started.json', 'summary.json'})
        require(worker['version'] == 'otto-return-value-v1' and worker['plan_sha256'] == PLAN_SHA
                and worker['completed_fits'] == 9 and worker['completed_episodes'] == 720
                and worker['prepared_episodes'] == 240 and worker['training_updates'] == 31680
                and worker['external_model_calls'] == 0 and worker['pending'] == []
                and all(v['attempted'] == v['returned'] for v in worker['calls'].values()), 'complete original worker')
        for name in ('sources', 'inputs', 'limits'):
            require(worker[name] == plan[name], 'worker frozen configuration join')
        require(audit['version'] == 'otto-return-value-saved-audit-v1' and audit['agreement'] is True
                and audit['plan_sha256'] == PLAN_SHA and audit['worker_sha256'] == a.receipt_sha256
                and audit['terminal_sha256'] == a.terminal_sha256 and audit['producer_source_sha256'] == plan['sources'][RUNNER]
                and audit['source']['sha256'] == plan['sources'][AUDITOR]
                and audit['training_calls'] == audit['simulator_calls'] == audit['remote_model_calls'] == 0,
                'successful independent audit of this exact worker and parent')
        request = read(a.audit/'started.json')['request']
        require(request == {'plan': str(a.plan), 'plan_sha256': PLAN_SHA, 'run': str(a.run),
                            'receipt_sha256': a.receipt_sha256, 'terminal': str(a.terminal),
                            'terminal_sha256': a.terminal_sha256, 'output': str(a.audit)}, 'exact independent audit request')
        started = read(a.run/'started.json')
        launch = started['launch']
        require(digest(Path(started['request']['supervision']), self.check)['sha256'] == worker['supervision_sha256']
                and read(Path(started['request']['supervision'])) == launch, 'actual launch identity')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['reaped'] is True
                and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['errors'] == []
                and terminal['error'] is None and terminal['clock_error'] is None and terminal['timing_available'] is True,
                'successful completed supervisor')
        for key in ('pid', 'pgid', 'parent_pid', 'command', 'cwd', 'started_ns', 'deadline_ns', 'clock_backend',
                    'cap_seconds', 'clock_source_sha256', 'watchdog_sha256'):
            require(terminal[key] == launch[key], 'parent launch field equality')
        require(launch['clock_backend'] in ('mach_continuous_time', 'CLOCK_BOOTTIME')
                and launch['clock_source_sha256'] == CLOCK_SHA and launch['cap_seconds'] == plan['limits']['native_seconds']
                and launch['deadline_ns'] == launch['started_ns']+launch['cap_seconds']*10**9
                and worker['clock_backend'] == launch['clock_backend']
                and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns'] <= terminal['finished_ns'] < launch['deadline_ns']
                and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9
                and terminal['wall_seconds'] == (terminal['finished_ns']-terminal['started_ns'])/1e9, 'strict shared deadline and elapsed bounds')
        self.receipt['inputs'] = {'plan': {'path': str(a.plan), 'sha256': PLAN_SHA},
                                 'worker': {'path': str(a.run/'receipt.json'), 'sha256': a.receipt_sha256},
                                 'audit': {'path': str(a.audit/'receipt.json'), 'sha256': a.audit_receipt_sha256},
                                 'terminal': {'path': str(a.terminal), 'sha256': a.terminal_sha256},
                                 'audit_summary': {'path': str(a.audit/'summary.json'), **digest(a.audit/'summary.json', self.check)}}
        return plan, worker, audit, terminal

    def report(self, plan, worker, audit, terminal):
        a = read(self.args.audit/'summary.json')
        producer = read(self.args.run/'summary.json')
        require(a['version'] == 'otto-return-value-saved-audit-v1' and a['agreement'] is True
                and a['episodes'] == 720 and a['fits'] == 9 and a['teacher_episodes'] == 240
                and a['epoch_orders'] == 720 and a['descriptive_validation_records'] == 72 and a['parity_records'] == 144
                and a['dataset_rows']['train'] == 5589 and a['actual_work']['optimizer_update'] == 31680,
                'complete authenticated numerical readback')
        checks = []
        for key, label, count in GROUPS:
            require(len(a[key]) == len(producer[key]) == count
                    and all(type(c['passes']) is bool for c in a[key]), 'complete original gate outcomes')
            for independent, original in zip(a[key], producer[key], strict=True):
                require(independent['name'] == original['name'] and independent['passes'] is original['passes'], 'unchanged original condition decisions')
                for field in ('value', 'threshold'):
                    numeric(independent[field])
                    numeric(original[field])
                    require(abs(independent[field]-original[field]) <= 1e-9+2e-12*abs(original[field]), 'audited numerical agreement')
            checks.extend(a[key])
        passed = sum(c['passes'] for c in checks)
        require(len({(key, c['name']) for key, _, _ in GROUPS for c in a[key]}) == 54
                and a['pilot_continuation'] == producer['pilot_continuation'] == worker['pilot_continuation'] == (passed == 54), 'all54 continuation identity')
        require(set(a['regimes']) == set(REGIMES) and set(a['final_prediction_diagnostics']) == set(ARMS[:-1]), 'all settings and final fits')
        pairs = a['paired_cases']
        require(len(pairs) == 720 and len({(r['regime'], r['seed'], r['arm']) for r in pairs}) == 720, 'all paired cases')
        failed = [r for r in pairs if not r['found']]
        learned_failures = [r for r in failed if r['arm'] != 'analytic_inbounds']
        rule_rows = ['| Rule | Passed | Decision |', '| --- | ---: | --- |']
        for key, label, total in GROUPS:
            n = sum(c['passes'] for c in a[key])
            rule_rows.append(f'| {label} | {n}/{total} | {"PASS" if n == total else "FAIL"} |')
        status = 'passes all frozen pilot conditions' if passed == 54 else 'fails the frozen continuation rule'
        text = [f'# Explicit branch values: {"pilot conditions met" if passed == 54 else "the matched pilot does not qualify"}', '',
                f'**The completed pilot {status}: {passed} of 54 conditions pass.** All nine fixed models completed training and all 720 autonomous evaluations finished. '
                f'The complete cohort retains {len(failed)} censored searches, including {len(learned_failures)} learned-policy searches.', '',
                'This experiment estimates the analytic teacher policy\'s remaining search cost, not optimal value. '
                'It does not establish a new architecture, compact memory, recurrent-world-model, biological-wiring or general robotic advantage. '
                'Even a passing pilot admits stronger testing only. All earlier failures and decisions remain unchanged.', '',
                '[Frozen protocol](otto-return-value-protocol.md) · [Plan](../output/otto-return-value-v1/plan-01.json) · '
                '[Independent summary](../output/otto-return-value-v1/audit-01/summary.json) · '
                '[Worker receipt](../output/otto-return-value-v1/run-01/receipt.json)', '',
                '![All ten arms, three settings and fixed conditions](../output/otto-return-value-v1/figure-01/return-value-comparison.png)', '',
                '## Matched scalar training', '',
                f'All families use the same 192 naturally completed teacher TRAIN episodes and all **{a["dataset_rows"]["train"]:,} pre-action prefixes**. '
                f'The 48 separate teacher VALID episodes provide **{a["dataset_rows"]["valid"]:,} prefixes** for descriptive validation. '
                'Prior DAgger and EVAL trajectories are excluded from targets and fitting. The target is `(T - t) / 64`; '
                'training uses uniform row MSE, with no duration-based sampling or episode weighting. Correlated prefixes are not independent episodes.', '',
                f'The common TRAIN-only baseline is `c0={a["c0_float32"]:.9g}`. Each model takes 11,028 features: '
                'the raw centered 105x105 public belief and three mass-scaled position/sensor context values. '
                'The complete 53x53 posterior remains the state. min8 has 88,224 trainable parameters; the ordinary width-eight ReLU control has 88,241; '
                'the bias-free homogeneous control has 88,232. Predictions are signed, with no clipping or fallback.', '',
                'Fitting seeds are 10101, 10102 and 10103. Each of nine models trains for 80 epochs with Adam, learning rate 0.001, '
                'batch size 128 and gradient norm cap 5, for 31,680 optimizer updates. Families share first-layer initialization and same-seed orders. '
                'The fixed epoch-80 checkpoint is used; validation does not select a checkpoint. All 144 predetermined export-parity cases passed '
                'the scalar/branch/cost tolerance and exact eligible-action requirement.', '',
                'Deployment upcasts the stored float32 parameters to float64, constructs all sixteen action/hit branches with the original 1e-10 mass floor, '
                'and evaluates physical values as 64 times normalized predictions. All four raw costs are computed before selecting an in-bounds action. '
                'The fused homogeneous shortcut is not used.', '',
                '## Complete autonomous results', '',
                'Each setting has 24 fresh cases, eight per initial-hit stratum, with all ten arms on every case and rotated arm order. '
                'Sensing lengths three and four are training-supported; length five is an unseen supplied kernel on the same grid. '
                'Sources and channel-indexed random uniforms are paired. Every failed search contributes the full 2,188-move horizon.', '',
                'Success, capped moves and complete controller time average within each hit stratum before applying the saved native mixture. '
                'Family means average all three fitting seeds. Raw found counts are separate and need not match weighted success.', '',
                '| Setting | Family/control | Weighted success | Capped moves | Controller ms/search |',
                '| --- | --- | ---: | ---: | ---: |']
        for regime in REGIMES:
            data = a['regimes'][regime]
            require(set(data['means']) == set(ARMS) and set(data['family_means']) == set(FAMILIES)
                    and len(data['blocks']) == 8, 'complete arm/family/block coverage')
            for family in ('analytic_inbounds', *FAMILIES):
                value = data['means'][family] if family == 'analytic_inbounds' else data['family_means'][family]
                text.append(f'| {regime} | {family} | {100*numeric(value["found"]):.2f}% | {numeric(value["steps"]):.2f} | {1000*numeric(value["controller_seconds"]):.3f} |')
        text += ['', '### Every fit and control', '', '| Setting | Arm | Raw found | Weighted success | Capped moves | Controller ms/search |',
                 '| --- | --- | ---: | ---: | ---: | ---: |']
        for regime in REGIMES:
            for arm in ('analytic_inbounds', *ARMS[:-1]):
                value, raw = a['regimes'][regime]['means'][arm], a['regimes'][regime]['raw_counts'][arm]
                require(raw['episodes'] == 24, 'all24 cases per arm/setting')
                text.append(f'| {regime} | {arm} | {raw["found"]}/24 | {100*value["found"]:.2f}% | {value["steps"]:.2f} | {1000*value["controller_seconds"]:.3f} |')
        text += ['', 'All initial-hit strata, eight paired blocks per setting and all 720 paired cases are retained in the '
                 '[independent summary](../output/otto-return-value-v1/audit-01/summary.json). Block dots are descriptive, not confidence intervals.', '',
                 '## Final fit diagnostics', '',
                 'These independently reconstructed final predictions estimate the teacher-return target. They are not autonomous success measures or calibrated probabilities. '
                 'MSE is in squared normalized units, with returns divided by 64. Negative predictions are retained. Intermediate ten-epoch curves remain execution evidence.', '',
                 '| Fixed final fit | TRAIN MSE | VALID MSE | TRAIN negative | VALID negative | Checkpoint |',
                 '| --- | ---: | ---: | ---: | ---: | --- |']
        for arm in ARMS[:-1]:
            family, seed = arm.split('@')
            d = a['final_prediction_diagnostics'][arm]
            train, valid = d['train'], d['valid']
            require(train['rows'] == a['dataset_rows']['train'] and valid['rows'] == a['dataset_rows']['valid'], 'complete prediction diagnostic rows')
            text.append(f'| {arm} | {train["mse_normalized"]:.6g} | {valid["mse_normalized"]:.6g} | '
                        f'{train["negative_count"]}/{train["rows"]} | {valid["negative_count"]}/{valid["rows"]} | '
                        f'[NPZ](../output/otto-return-value-v1/run-01/final-{family}-{seed}.npz) |')
        text += ['', 'All min8 plane-usage counts and saved final prediction diagnostics remain in the independent summary; no plane or fit was selected or replaced.', '',
                 '## Failure and spatial-repetition description', '',
                 'The following are raw episode counts across three fitting seeds, not mixture-weighted success rates. '
                 'A complete alternating tail requires 256 pre-action positions and equality at all 254 lag-two comparisons. '
                 'Short tails are not counted as complete alternating tails. Position repetition does not establish repeated belief state, cause, or the efficacy of an anti-reversal intervention.', '',
                 '| Family/control | Censored searches | Complete alternating tails among censored searches | Episodes with a zero-mass pre-action belief |',
                 '| --- | ---: | ---: | ---: |']
        for family in (*FAMILIES, 'analytic_inbounds'):
            selected = [r for r in pairs if r['arm'].split('@')[0] == family]
            censored = [r for r in selected if not r['found']]
            alternating = sum(r['last256_lag2_pairs'] == 254 and r['last256_lag2_matches'] == 254 for r in censored)
            tail_text = f'{alternating}/{len(censored)}' if censored else 'Not applicable (no censored searches)'
            text.append(f'| {family} | {len(censored)}/{len(selected)} | {tail_text} | '
                        f'{sum(r["zero_mass_decisions"] > 0 for r in selected)}/{len(selected)} |')
        text += ['', 'Zero-mass counts here concern pre-action decision states only. The auditor separately verifies every final found/censored update; '
                 'this table makes no exact posterior-hash repetition claim.', '', '## Complete computation and execution', '',
                 'Controller time includes initialization, explicit branch construction and copies, context features, scalar readout, cost reduction/masking, '
                 'all public updates and actual inference setup. Each head load is allocated over 72 episodes; shared model/branch module setup is allocated over 648 learned episodes. '
                 'The inherited actor includes its allocated but unused analytic distance table. Transient branch arrays are separate from retained belief and weight storage. '
                 'Only measured nested artifact I/O is excluded. Simulator, preparation and fitting time are reported separately. One rotated CPU run does not establish repeated deployment latency.', '',
                 '| Recorded phase/resource | Value |', '| --- | ---: |']
        for key, value in a['costs']['training_phases'].items():
            text.append(f'| {key.replace("_", " ")} | {value:.3f} s |')
        text += [f'| Common setup | {a["costs"]["shared_setup_seconds"]:.6f} s |',
                 f'| Model/branch module setup, already allocated in controller cost | {a["costs"]["model_module_setup_seconds"]:.6f} s |',
                 f'| Original worker elapsed | {worker["wall_seconds"]:.3f} s |',
                 f'| Original parent elapsed, enclosing the worker | {terminal["wall_seconds"]:.3f} s |',
                 f'| Original worker peak RSS | {worker["peak_rss_bytes"]:,} bytes |',
                 f'| Original native resets / steps | {a["actual_work"]["native_reset"]:,} / {a["actual_work"]["native_step"]:,} |',
                 f'| Independent saved audit elapsed | {audit["wall_seconds"]:.3f} s |',
                 f'| Independent saved audit peak RSS | {audit["peak_rss_bytes"]:,} bytes |', '',
                 'The parent and worker intervals overlap and are not added. Audit/figure/publication time is separate from the scientific worker. '
                 'The original worker limits were 5,400 suspend-inclusive seconds, 8 GiB RSS and 6 GiB output; the independent audit limits were 1,800 seconds, 4 GiB RSS and 128 MiB output.', '',
                 f'The independent auditor agrees across **{a["comparisons"]:,} comparisons**, with maximum summary scalar difference '
                 f'`{a["maximum_scalar_difference"]:.12g}` and maximum prediction difference `{a["maximum_prediction_difference"]:.12g}`. '
                 f'It performed **{a["saved_checkpoint_readout_calls"]:,} local saved-checkpoint readouts** covering '
                 f'**{a["saved_network_rows"]:,} network rows**. Those readouts are included in the audit computation counts. '
                 'No simulator, training or remote-model call was made by the audit.', '',
                 'Public filtering, all-prefix targets/baseline, final predictions, explicit branch costs/actions, recorded work and aggregate conditions were independently reconstructed. '
                 'Original optimizer trajectories, native random execution, teacher behavior, Torch parity outputs and timing truth remain authenticated execution evidence.', '',
                 '[Audit receipt](../output/otto-return-value-v1/audit-01/receipt.json) · '
                 '[Parent terminal](../output/otto-return-value-v1/run-process-01.terminal.json) · '
                 '[Complete fit records](../output/otto-return-value-v1/run-01/fits.jsonl)', '',
                 '## Fixed illustrative replays and restoration', '',
                 'The three clips show prospectively fixed case zero, all ten arms, rather than selected wins or failures. '
                 'They are saved trajectory playback and make no claim to reproduce measured wall-clock execution.', '',
                 '[Length three, seed 11100001](../output/otto-return-value-v1/figure-01/replay-lambda3-case0.gif) · '
                 '[Length four, seed 11200001](../output/otto-return-value-v1/figure-01/replay-lambda4-case0.gif) · '
                 '[Length five, seed 11300001](../output/otto-return-value-v1/figure-01/replay-lambda5-case0.gif)', '',
                 'Download and verify the full run using the [release archive](https://github.com/kw2828/OpenJev/releases/tag/otto-return-value-v1), '
                 '[archive manifest](../output/otto-return-value-v1/release-01/manifest.json) and '
                 '[restoration instructions](../output/otto-return-value-v1/release-01/RESTORE.md).', '',
                 '## Decision boundary', '',
                 ('All prospective pilot conditions are met. This admits a separately specified stronger comparison, not a novelty or learned-memory claim.'
                  if passed == 54 else 'The prospective continuation rule is not met. Descriptive improvements cannot replace failed conditions or admit architectural escalation.'),
                 'The comparison concerns three matched full-belief value approximators trained on analytic-policy returns. It does not isolate a biological mechanism, '
                 'prove optimal control, or establish that compact recurrence would help.', '',
                 '## All frozen conditions', '', 'All 54 conditions are required. Values below are rounded for display; decisions retain the frozen full-precision comparisons.', '',
                 *rule_rows, '', '| Condition | Value | Required relation | Threshold | Result |',
                 '| --- | ---: | :---: | ---: | :---: |']
        for key, _, _ in GROUPS:
            prefix = 'utility_compute' if key == 'compression_checks' else key.removesuffix('_checks')
            for record in a[key]:
                name = record['name']
                text.append(f'| {prefix}.{name} | {condition_value(name, record["value"])} | {relation(name)} | '
                            f'{condition_value(name, record["threshold"])} | {"Pass" if record["passes"] else "Fail"} |')
        self.receipt.update(arm_setting_rows=30, family_control_rows=12, final_fit_rows=9, conditions=54,
                            conditions_passed=passed, pilot_continuation=a['pilot_continuation'], censored_searches=len(failed))
        return '\n'.join(text)+'\n'

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents), 'exclusive publication evidence directory')
        require(not REPORT.exists() and not REPORT.is_symlink(), 'results document must not already exist')
        self.out.mkdir(parents=True, exist_ok=False)
        old_alarm = signal.getsignal(signal.SIGALRM)
        try:
            require(digest(ROOT/CLOCK)['sha256'] == CLOCK_SHA, 'qualified native clock source')
            spec = importlib.util.spec_from_file_location('_return_publication_clock', ROOT/CLOCK)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.clock = module.SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('publication cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['native_seconds'])
            self.receipt['source'] = {'path': str(Path(__file__).resolve()), **digest(Path(__file__), self.check)}
            write(self.out/'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                                           'started_ns': self.start, 'clock_backend': self.clock.backend})
            context = self.authenticate()
            content = self.report(*context)
            self.check()
            self.authenticate()
            with REPORT.open('x') as stream:
                self.published = True
                stream.write(content)
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', started_ns=self.start, finished_ns=finished,
                                wall_seconds=(finished-self.start)/1e9, clock_backend=self.clock.backend,
                                report={'path': str(REPORT), **digest(REPORT, self.check)},
                                files={p.name: digest(p, self.check) for p in self.out.iterdir() if p.is_file()})
            write(self.out/'receipt.json', self.receipt)
            self.check()
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                if self.published and REPORT.exists():
                    REPORT.rename(self.out/'incomplete-report.md')
                write(self.out/'failed.json', self.receipt)
            except BaseException as secondary:
                error.add_note(f'Failure preservation: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'audit', 'terminal', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'audit-receipt-sha256', 'terminal-sha256'):
        parser.add_argument('--'+flag, required=True)
    Builder(parser.parse_args()).execute()
