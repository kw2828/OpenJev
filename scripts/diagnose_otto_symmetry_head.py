"""Post hoc trajectory descriptions of a completed, audited symmetry-head pilot.

Standard library only. Does not instantiate a policy, reconstruct a posterior,
load checkpoint/data arrays, train a model, or call the native simulator.
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
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-symmetry-head-trajectory-diagnostic-v1'
HELPER = 'scripts/render_otto_symmetry_head.py'
HELPER_PIN = '145c04b04392b2eb84f59c5c4150bd7868e69bef37df17b8174821ba4e20749a'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
LIMITS = {'seconds': 180, 'rss_bytes': 2 * 1024**3, 'output_bytes': 64 * 1024**2}
SEEDS = (9101, 9102, 9103)
ARMS = tuple(f'{family}@{seed}' for seed in SEEDS for family in ('shared', 'dense', 'dense_ensemble')) + ('analytic_inbounds',)
REGIMES = {'lambda3': 970001, 'lambda4': 980001, 'lambda5': 990001}
SCOPE = ('Post hoc descriptions of all 720 already-audited autonomous trajectories. '
         'Position repetition is not proof of a closed state cycle; equal saved posterior hashes '
         'do not identify why the state or policy repeats. Exactly zero saved mass is distinct '
         'from small positive mass and does not diagnose its cause. Distances use evaluator-only '
         'source truth. No intervention, causal claim, new efficacy gate or original-rule change.')
DEFINITIONS = {
    'visited_positions': 'Distinct public coordinates across reset and all completed updates, including final found/censored packet.',
    'last_window': 'Last min(256, steps) pre-action coordinates; terminal/final post-action packet is excluded.',
    'lag2_fraction': 'Count of p[t]==p[t-2] within that window divided by max(window_length-2,0); null for denominator zero.',
    'exact_state_repeats': 'Pre-action (public coordinate, saved posterior SHA256) occurrences beyond their first occurrence, over the whole episode.',
    'first_zero_mass': 'First public packet index 0..steps with saved posterior mass exactly 0, including a final packet with no ensuing decision.',
    'zero_mass_decisions': 'Number of pre-action packets with saved posterior mass exactly 0; no epsilon or probability floor.',
    'source_distances': 'Minimum and final Manhattan distance over reset and all updates, using evaluator-only source coordinates.',
    'group_means': 'Fixed initial-hit mixture times equal episode mean within each of three strata (eight cases each). Raw counts are separately labeled.',
    'nullable_group_means': 'For a nullable episode metric, use weight[hit]/8 per available episode and renormalize by available mixture mass; expose that denominator.',
    'pooled_lag2_fraction': 'Raw summed equal-position lag2 pairs divided by raw summed eligible lag2 pairs; separate from mixture-weighted episode fractions.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path, check=lambda: None):
    value, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            check()
            value.update(chunk)
            size += len(chunk)
    return {'sha256': value.hexdigest(), 'bytes': size}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def position(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) is int and 0 <= v < 53 for v in value), 'public grid coordinate')
    return tuple(value)


def state(value):
    require(set(value) == {'sha256', 'mass'}, 'saved posterior witness schema')
    sha, mass = value['sha256'], value['mass']
    require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'saved posterior hash')
    require(type(mass) in (int, float) and math.isfinite(mass) and mass >= 0, 'finite nonnegative saved posterior mass')
    return sha, mass


def lag2(positions):
    window = list(positions)[-256:]
    denominator = max(len(window) - 2, 0)
    numerator = sum(window[i] == window[i - 2] for i in range(2, len(window)))
    return {'window_decisions': len(window), 'equal_pairs': numerator, 'eligible_pairs': denominator,
            'fraction': numerator / denominator if denominator else None}


class Episode:
    def __init__(self, metadata, reset):
        self.metadata = metadata
        self.steps = 0
        self.source = position(metadata['source_evaluation_only'])
        self.current, self.posterior = reset['public'], reset['posterior_after']
        require(self.current['step'] == 0 and self.current['done'] is False, 'nonterminal reset')
        require(position(reset['source_evaluation_only']) == self.source, 'saved source identity')
        self.visited, self.window, self.seen = set(), deque(maxlen=256), {}
        self.repeats = self.positive_repeats = self.zero_decisions = 0
        self.first_repeat = self.first_zero = None
        self.minimum_distance = 104
        self.observe(self.current, self.posterior)

    def observe(self, packet, posterior):
        coordinate = position(packet['position'])
        _, mass = state(posterior)
        self.visited.add(coordinate)
        self.minimum_distance = min(self.minimum_distance, sum(abs(a-b) for a, b in zip(coordinate, self.source, strict=True)))
        if mass == 0 and self.first_zero is None:
            self.first_zero = packet['step']

    def update(self, row):
        require(row['kind'] == 'step' and row['episode_id'] == self.metadata['episode_id']
                and row['step'] == self.steps + 1 and row['public']['step'] == row['step']
                and self.current['done'] is False, 'chronological completed transition')
        require(row['posterior_before'] == self.posterior, 'before/after saved witness continuity')
        coordinate = position(self.current['position'])
        sha, mass = state(self.posterior)
        key = (coordinate, sha)
        if key in self.seen:
            self.repeats += 1
            self.positive_repeats += mass > 0
            if self.first_repeat is None:
                self.first_repeat = {'decision_index': self.steps, 'previous_decision_index': self.seen[key]}
        else:
            self.seen[key] = self.steps
        self.zero_decisions += mass == 0
        self.window.append(coordinate)
        self.current, self.posterior = row['public'], row['posterior_after']
        self.steps += 1
        self.observe(self.current, self.posterior)

    def finish(self):
        meta = self.metadata
        require(self.steps == meta['steps'] and self.current == meta['final_public']
                and self.current['done'] is meta['found'], 'complete final found/censored packet')
        require(meta['found'] or self.steps == 2188, 'all failures retain full horizon')
        final_distance = sum(abs(a-b) for a, b in zip(position(self.current['position']), self.source, strict=True))
        require((final_distance == 0) is meta['found'], 'found/source coordinate consistency')
        tail = lag2(self.window)
        return {k: meta[k] for k in ('episode_id', 'regime', 'seed', 'arm', 'initial_hit', 'block', 'steps', 'found')} | {
            'distinct_visited_positions': len(self.visited),
            'last_window_lag2': tail,
            'exact_state_repeat_decisions': self.repeats,
            'positive_mass_exact_state_repeat_decisions': self.positive_repeats,
            'first_exact_state_repeat': self.first_repeat,
            'first_zero_mass_public_step': self.first_zero,
            'zero_mass_decisions': self.zero_decisions,
            'final_posterior_mass': self.posterior['mass'],
            'minimum_source_distance_evaluation_only': self.minimum_distance,
            'final_source_distance_evaluation_only': final_distance,
        }


def summarize(rows, mixtures):
    require(len(rows) == 720 and len({r['episode_id'] for r in rows}) == 720, 'complete unique diagnostic population')
    groups = []
    for regime in REGIMES:
        weights = {int(h): w for h, w in mixtures[regime].items()}
        require(set(weights) == {1, 2, 3} and all(math.isfinite(v) and v > 0 for v in weights.values())
                and abs(math.fsum(weights.values()) - 1) <= 1e-12, 'fixed regime mixture')
        for arm in ARMS:
            selected = [r for r in rows if r['regime'] == regime and r['arm'] == arm]
            require(len(selected) == 24 and all(sum(r['initial_hit'] == h for r in selected) == 8 for h in weights), 'all24 cases/all3 strata in each group')

            def weighted(getter, selected=selected, weights=weights):
                available = [(getter(r), weights[r['initial_hit']] / 8) for r in selected]
                available = [(v, w) for v, w in available if v is not None]
                mass = math.fsum(w for _, w in available)
                return {'mean': math.fsum(v*w for v, w in available) / mass if mass else None,
                        'available_episodes': len(available), 'available_mixture_mass': mass}

            metrics = {name: weighted(lambda r, name=name: r[name]) for name in (
                'found', 'steps', 'distinct_visited_positions', 'exact_state_repeat_decisions',
                'positive_mass_exact_state_repeat_decisions', 'zero_mass_decisions',
                'minimum_source_distance_evaluation_only', 'final_source_distance_evaluation_only')}
            metrics['last_window_lag2_fraction'] = weighted(lambda r: r['last_window_lag2']['fraction'])
            eligible = sum(r['last_window_lag2']['eligible_pairs'] for r in selected)
            equal = sum(r['last_window_lag2']['equal_pairs'] for r in selected)
            failures = [r for r in selected if not r['found']]
            groups.append({'regime': regime, 'arm': arm, 'episodes': 24,
                'raw_successes': sum(r['found'] for r in selected), 'raw_failures': len(failures),
                'raw_zero_mass_decisions': sum(r['zero_mass_decisions'] for r in selected),
                'raw_episodes_with_zero_mass_packet': sum(r['first_zero_mass_public_step'] is not None for r in selected),
                'raw_failures_with_zero_mass_packet': sum(r['first_zero_mass_public_step'] is not None for r in failures),
                'raw_failures_without_zero_mass_packet': sum(r['first_zero_mass_public_step'] is None for r in failures),
                'raw_lag2_equal_pairs': equal, 'raw_lag2_eligible_pairs': eligible,
                'raw_pooled_lag2_fraction': equal / eligible if eligible else None,
                'mixture_weighted_episode_metrics': metrics})
    return {'episodes': len(rows), 'groups': groups, 'initial_hit_weights': mixtures,
            'definitions': DEFINITIONS, 'scope': SCOPE, 'original_rules_revised': False}


class Diagnostic:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.helper = None
        self.receipt = {'version': VERSION, 'status': 'started', 'limits': LIMITS, 'scope': SCOPE,
                        'model_calls': 0, 'simulator_calls': 0, 'training_calls': 0, 'posterior_reconstructions': 0}

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS['seconds'] * 10**9, 'diagnostic native deadline')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'diagnostic RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'diagnostic output cap')

    def authenticate(self):
        require(digest(ROOT / HELPER, self.check)['sha256'] == HELPER_PIN, 'held authentication helper before import')
        if self.helper is None:
            self.helper = load(ROOT / HELPER, '_symmetry_diagnostic_authentication')
        auth = self.helper.Render(self.args)
        auth.clock, auth.start, auth.check = self.clock, self.start, self.check
        aggregate = auth.authenticate()
        self.receipt['inputs'] = auth.receipt['inputs']
        self.receipt['authentication_helper'] = {'path': HELPER, 'sha256': HELPER_PIN,
            'scope': 'Render.authenticate only; no presentation, policy, posterior or model functions.'}
        self.receipt['original_pilot_continuation'] = aggregate['pilot_continuation']
        return {r: aggregate['regimes'][r]['weights'] for r in REGIMES}

    def records(self, name):
        with (self.args.run / name).open() as stream:
            for line in stream:
                self.check()
                yield json.loads(line)

    def compute(self, mixtures):
        metadata = {}
        expected = {f'eval:{r}:{first+i}:{arm}' for r, first in REGIMES.items() for i in range(24) for arm in ARMS}
        fields = ('episode_id', 'regime', 'seed', 'arm', 'initial_hit', 'block', 'steps', 'found', 'final_public', 'source_evaluation_only')
        for record in self.records('evaluation.jsonl'):
            row = {k: record[k] for k in fields}
            key, regime = row['episode_id'], row['regime']
            require(key in expected and key not in metadata and regime in REGIMES, 'exact unique evaluation identity')
            index = row['seed'] - REGIMES[regime]
            require(key == f"eval:{regime}:{row['seed']}:{row['arm']}" and 0 <= index < 24
                    and row['initial_hit'] == 1 + index % 3 and row['block'] == index // 3
                    and type(row['found']) is bool and type(row['steps']) is int and 1 <= row['steps'] <= 2188,
                    'fixed cohort metadata')
            metadata[key] = row
        require(set(metadata) == expected, 'all720 evaluations before trajectories')
        rows, seen, active = [], set(), None
        with (self.out / 'episodes.jsonl').open('x') as output:
            for record in self.records('eval-transitions.jsonl'):
                key = record['episode_id']
                require(key in metadata, 'trajectory joins known episode')
                if record['kind'] == 'reset':
                    if active is not None:
                        row = active.finish()
                        rows.append(row)
                        output.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                        output.flush()
                    require(key not in seen, 'unique contiguous trajectory')
                    seen.add(key)
                    active = Episode(metadata[key], record)
                else:
                    require(active is not None, 'no transition before reset')
                    active.update(record)
            require(active is not None, 'nonempty trajectories')
            row = active.finish()
            rows.append(row)
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        require(seen == expected, 'every complete trajectory retained')
        result = summarize(rows, mixtures)
        result['original_pilot_continuation'] = self.receipt['original_pilot_continuation']
        return result

    def execute(self):
        require(self.out.is_absolute() and not any(p.is_symlink() for p in (self.out, *self.out.parents)), 'absolute nonsymlink exclusive output')
        self.out.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        try:
            require(digest(ROOT / CLOCK)['sha256'] == CLOCK_PIN, 'qualified clock before import')
            self.clock = load(ROOT / CLOCK, '_symmetry_diagnostic_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('diagnostic wall cap')))
            signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
            self.receipt['source'] = {'path': str(Path(__file__).resolve()), **digest(Path(__file__), self.check)}
            write(self.out / 'started.json', {'request': {k: str(v) for k, v in vars(self.args).items()},
                  'definitions': DEFINITIONS, 'scope': SCOPE, 'limits': LIMITS,
                  'started_ns': self.start, 'clock_backend': self.clock.backend})
            mixtures = self.authenticate()
            result = self.compute(mixtures)
            require(self.authenticate() == mixtures, 'unchanged complete inputs after diagnostic')
            write(self.out / 'summary.json', result)
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status='completed', episodes=720, groups=30,
                clock_backend=self.clock.backend, started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished-self.start)/1e9,
                files={p.name: digest(p, self.check) for p in self.out.iterdir() if p.is_file()})
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            print(json.dumps({'status': 'completed', 'episodes': 720, 'groups': 30,
                              'receipt_sha256': digest(self.out / 'receipt.json', self.check)['sha256']}))
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / 'receipt.json').exists():
                    (self.out / 'receipt.json').rename(self.out / 'invalid-completed-receipt.json')
                write(self.out / 'receipt.json', self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - Keep the original diagnostic failure.
                error.add_note(f'Failure receipt publication: {secondary!r}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'audit', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    for flag in ('receipt-sha256', 'audit-receipt-sha256'):
        parser.add_argument(f'--{flag}', required=True)
    Diagnostic(parser.parse_args()).execute()
