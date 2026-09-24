"""Fresh DEV-only native collection for the prospective paired-action study."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import otto_action_effect_common as c

CALL_CAPS = {'tensorflow_construction': 1, 'tensorflow_build': 1, 'tensorflow_load': 1,
             'tensorflow_value': 2048, 'teacher_score': 2048, 'native_reset': 256,
             'native_step': 4096, 'actor_construction': 256, 'actor_update': 4096,
             'analytic_score': 4352, 'feature_build': 4352, 'backend_binding': 256,
             'sampler_table': 2, 'sampler_lookup': 6144, 'initial_source_law': 256,
             'initial_normalization': 256, 'shadow_update': 4096,
             'gap_oracle': 256, 'opposite_oracle': 256, 'normal_oracle': 2048,
             'sampler_draw_validation': 4352, 'committed_positions': 512}


def committed_actions(np, seed, horizon):
    """Separate public action stream; no observation, posterior or source argument."""
    c.require(type(seed) is int and type(horizon) is int and horizon == 8, 'declared DEV action block')
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 911]))).integers(
        0, 4, size=horizon, dtype=np.int64)


def absorbing(outcomes):
    seen = False
    for outcome in outcomes:
        c.require(0 <= int(outcome) <= 4 and (not seen or outcome == 4), 'absorbing terminal outcome')
        seen |= outcome == 4


class Ledger:
    def __init__(self, run):
        self.run, self.context, self.io_seconds = run, {}, 0.
        self.calls = {k: {'attempted': 0, 'returned': 0, 'seconds': 0.} for k in CALL_CAPS}
        self.pending, self.sequence, self.handles = [], 0, {}
        self.last_seconds = {}

    def emit(self, name, value):
        start = time.perf_counter()
        if name not in self.handles:
            raw = (self.run.out / (name + '.gz')).open('xb')
            self.handles[name] = (raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0))
        self.handles[name][1].write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())
        self.io_seconds += time.perf_counter() - start

    def flush(self):
        for raw, stream in self.handles.values():
            stream.flush()
            raw.flush()
            os.fsync(raw.fileno())

    def close(self):
        for raw, stream in self.handles.values():
            if not raw.closed:
                stream.close()
                raw.flush()
                os.fsync(raw.fileno())
                raw.close()

    def call(self, name, function):
        self.run.check()
        c.require(name in self.calls and self.calls[name]['attempted'] < CALL_CAPS[name], 'registered call cap: ' + name)
        self.sequence += 1
        row = {'id': self.sequence, 'channel': name, 'context': self.context.copy()}
        self.pending.append(row)
        self.calls[name]['attempted'] += 1
        self.emit('work.jsonl', {'event': 'attempt', **row})
        tick = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - tick
        self.emit('work.jsonl', {'event': 'return', **row, 'seconds_including_nested_io': elapsed})
        c.require(self.pending[-1] is row, 'nested call order')
        self.pending.pop()
        self.calls[name]['returned'] += 1
        self.calls[name]['seconds'] += elapsed
        self.last_seconds[name] = elapsed
        return value


class Collector(c.Run):
    def setup(self):
        self.training_reference = c.training_reference()
        c.write(self.out / 'training-reference.json', self.training_reference)
        prior, self.paths, self.reference, _ = c.original_native()
        c.require({k: c.desc(k) for k in prior['sources']} == self.plan['native_sources'], 'native source pins')
        self.plan['runtime'] = prior['runtime']  # runtime metadata expected by unchanged setup
        self.ledger = Ledger(self)
        self.runtime = self.reference.Run.setup(self)
        from openjev.research import otto_predictive_belief, otto_sampler_law
        from openjev.research.otto_query_gate import ReadOnlyBeliefView, _analytic, _features
        self.view_class, self.analytic, self.features = ReadOnlyBeliefView, _analytic, _features
        self.bayes, self.sampler = otto_predictive_belief, otto_sampler_law
        c.require('torch' not in sys.modules, 'no learner in collection')
        self.sensor_laws, self.sensor_raw, metadata = {}, {}, {}
        for regime, sensing in (('lambda3', 3.), ('lambda4', 4.)):
            self.ledger.context = {'phase': 'sensor_law_setup', 'regime': regime}
            result = self.ledger.call('sampler_table', lambda s=sensing: self.sampler.build_sensor_law(
                self.runtime.source, s, check=self.check))
            self.sensor_laws[regime], self.sensor_raw[regime] = result['probabilities'], result['raw_probabilities']
            metadata[regime] = {**result['metadata'], 'seconds': self.ledger.last_seconds['sampler_table']}
        with (self.out / 'sensor-laws.npz').open('xb') as stream:
            self.runtime.np.savez_compressed(stream, **self.sensor_laws,
                **{name + '_raw': value for name, value in self.sensor_raw.items()})
            stream.flush()
            os.fsync(stream.fileno())
        c.write(self.out / 'sensor-laws.json', metadata)

    def validate_draw(self, env, expected, channel):
        """Audit-only native draw validation; its source location never enters annotations."""
        np = self.runtime.np
        record = env.draw_log[-1]
        c.require(record['channel'] == channel, 'native draw channel')
        actual = np.asarray(record['probabilities'], np.float64)
        c.require(actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
                  'native scalar sampler probabilities equal primitive law')
        c.require(record['cdf_mass'] == float(np.cumsum(expected)[-1]), 'native CDF mass')
        self.ledger.emit('native-draws.jsonl', {'context': self.ledger.context, 'channel': channel,
                         'probabilities_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                         'cdf_mass': record['cdf_mass'], 'selected_index': record['selected_index']})

    def shadow_record(self, shadow, actor, current, *, evidence, initial=None):
        np = self.runtime.np
        legacy = actor.belief.reshape(-1)
        row = {'context': self.ledger.context, 'public_step': current['step'], 'public_done': current['done'],
               'shadow_mass': float(shadow.sum()), 'legacy_mass': float(legacy.sum()),
               'l1_difference': float(np.abs(shadow - legacy).sum()),
               'max_abs_difference': float(np.abs(shadow - legacy).max()), 'evidence': evidence,
               'shadow_sha256': hashlib.sha256(shadow.tobytes()).hexdigest()}
        if initial is not None:
            row['initial_normalization'] = {k: v for k, v in initial.items() if k != 'belief'}
        self.ledger.emit('shadow.jsonl', row)

    def case(self, identity):
        c.require(identity['split'] == 'dev' and identity['regime'] in ('lambda3', 'lambda4')
                  and identity in c.roster(), 'registered fresh DEV identity only')
        rt, call, np = self.runtime, self.ledger.call, self.runtime.np
        self.ledger.context = dict(identity)
        regime = 'base' if identity['regime'] == 'lambda3' else 'shift'
        sensing = float(identity['regime'][-1])
        H = c.CONFIG['dev_horizon']
        env = call('native_reset', lambda: rt.public.seeded_environment(rt.source, identity['seed'],
            {'Ndim': 2, 'Ngrid': 53, 'Nhits': 4, 'lambda_over_dx': sensing, 'R_dt': 2.,
             'norm_Poisson': 'Euclidean'}, initial_hit=identity['initial_hit']))
        c.require(env.p_Poisson.tobytes() == rt.kernels[regime].tobytes(), 'unchanged native kernel')
        current = self.reference.packet(rt.public.observation(env, 0))
        actor = call('actor_construction', lambda: rt.analytic(current, rt.kernels[regime], allow_stay=False))
        view = self.view_class(actor._view)
        self.reference.belief_witness(actor, env, current, np)
        # The native source draw occurs from the initial public prior. Mirror
        # that draw's discrete law, then maintain an independent public filter.
        native_initial = actor.belief.reshape(-1).copy()
        call('sampler_draw_validation', lambda: self.validate_draw(env, native_initial, 'source'))
        initial_law = call('initial_source_law', lambda: self.bayes.cdf_law(native_initial, uniform_bits=53))
        normalized = call('initial_normalization', lambda: self.bayes.normalize_prior(initial_law['probabilities']))
        initial_belief = normalized['belief'].copy()
        shadow = initial_belief.copy()
        self.shadow_record(shadow, actor, current, evidence=None, initial=normalized)
        self.ledger.emit('source-laws.jsonl', {'identity': identity,
            **{k: v for k, v in initial_law.items() if k != 'probabilities'}})
        previous = last_action = None
        prefix, prefix_actions, prefix_outcomes = [], [], []
        continuation = np.zeros((H, 31), np.float32)
        outcomes, costs, legal = np.full(H, 4, np.int64), np.zeros((H, 4), np.float32), np.zeros((H, 4), np.bool_)

        def features():
            nonlocal previous
            action, scores = call('analytic_score', lambda: self.analytic(actor._policy._value_policy()[1], current['valid_actions']))
            f, previous = call('feature_build', lambda: self.features(view, current, sensing, scores,
                last_action, current['step'] - current['step'] % 4, previous))
            return f.copy(), action

        def step(action, likelihood=None):
            nonlocal current, last_action, shadow
            c.require(action in current['valid_actions'] and not current['done'], 'legal nonterminal native transition')
            result = call('native_step', lambda: env.step(action, quiet=True))
            after = self.reference.packet(rt.public.observation(env, current['step'] + 1))
            c.require((int(result[0]), bool(result[2])) == (after['hit'], after['done']), 'native outcome identity')
            call('actor_update', lambda: actor.update(action, after))
            if likelihood is None:
                likelihood = call('sampler_lookup', lambda: self.sampler.likelihood_at(
                    self.sensor_laws[identity['regime']], np.asarray(after['position'], np.int64)))
            index = int(after['position'][0] * 53 + after['position'][1])
            update = call('shadow_update', lambda: self.bayes.observe(shadow, likelihood, index,
                None if after['done'] else after['hit'], known_found=after['done']))
            shadow = update['belief']
            self.shadow_record(shadow, actor, after, evidence=update['evidence'])
            if not after['done']:
                # Hidden source is inspected solely to audit the simulator law.
                # It cannot alter the shadow state, inputs, actions or targets.
                displacement = np.asarray(env.source) - np.asarray(after['position'])
                expected = self.sensor_raw[identity['regime']][displacement[0] + 52, displacement[1] + 52]
                call('sampler_draw_validation', lambda: self.validate_draw(env, expected, 'hit'))
            witness = self.reference.belief_witness(actor, env, after, np)
            self.ledger.emit('transitions.jsonl', {'case': identity['id'], 'action': action, 'public': after,
                'posterior': witness})
            current, last_action = after, action

        # The observed analytic prefix is the same for all learned methods.
        for t in range(9):
            self.ledger.context = {**identity, 'phase': 'prefix', 'step': t}
            if current['done']:
                return {'identity': identity, 'excluded': 'found_during_observed_prefix', 'native_steps': t}, None
            f, action = features()
            prefix.append(f)
            if t < 8:
                prefix_actions.append(action)
                step(action)
                prefix_outcomes.append(4 if current['done'] else current['hit'])
        # Every possible sequence of 8 moves from this center-origin prefix is in bounds.
        c.require(all(8 <= x <= 44 for x in current['position']), 'entire block geometrically in bounds')
        actions = committed_actions(np, identity['seed'], H)
        block = {'identity': identity, 'prefix_public': current, 'actions': actions.tolist(),
                 'prefix_actions': prefix_actions, 'commit_before_native_step': self.ledger.calls['native_step']['returned']}
        c.append(self.out / 'commitments.jsonl', block)
        prefix_position = np.asarray(current['position'], np.int64)
        positions = call('committed_positions', lambda: self.sampler.planned_positions(prefix_position, actions))
        opposite_positions = call('committed_positions', lambda: self.sampler.planned_positions(prefix_position, actions ^ 1))
        likelihoods = np.stack([call('sampler_lookup', lambda q=q: self.sampler.likelihood_at(
            self.sensor_laws[identity['regime']], q)) for q in positions])
        opposite_likelihoods = np.stack([call('sampler_lookup', lambda q=q: self.sampler.likelihood_at(
            self.sensor_laws[identity['regime']], q)) for q in opposite_positions])
        indices = positions[:, 0] * 53 + positions[:, 1]
        opposite_indices = opposite_positions[:, 0] * 53 + opposite_positions[:, 1]
        gap_oracle = call('gap_oracle', lambda: self.bayes.blind_marginals(shadow, likelihoods, indices))
        opposite_oracle = call('opposite_oracle', lambda: self.bayes.blind_marginals(
            shadow, opposite_likelihoods, opposite_indices))
        normal_oracle = np.zeros((H, 5), np.float64)
        recorder = self.reference.ForwardRecorder(rt.model, rt.policy._value_policy.__code__, self.ledger, np)
        teacher = call('backend_binding', lambda: rt.policy(env=view, model=recorder, sym_avg=True))
        for h, action in enumerate(actions):
            self.ledger.context = {**identity, 'phase': 'precommitted', 'horizon': h + 1}
            normal_oracle[h] = call('normal_oracle', lambda h=h: self.bayes.one_step(
                shadow, likelihoods[h], indices[h], known_terminal=current['done']))
            if current['done']:
                continue
            step(int(action), likelihoods[h])
            if current['done']:
                continue
            outcomes[h] = current['hit']
            continuation[h], _ = features()
            before = actor.belief.tobytes(), actor.public
            costs[h] = call('teacher_score', lambda: teacher._value_policy()[1])
            c.require(before == (actor.belief.tobytes(), actor.public), 'annotation cannot mutate public filter')
            legal[h] = [a in current['valid_actions'] for a in range(4)]
        absorbing(outcomes)
        c.require(np.isfinite(costs).all() and np.isfinite(continuation).all(), 'finite block arrays')
        arrays = {'prefix': np.stack(prefix), 'prefix_lengths': np.int64(9), 'actions': actions,
                  'continuation': continuation, 'outcomes': outcomes, 'raw_costs': costs, 'legal': legal,
                  'initial_belief': initial_belief, 'prefix_actions': np.asarray(prefix_actions, np.int64),
                  'prefix_outcomes': np.asarray(prefix_outcomes, np.int64), 'prefix_position': prefix_position,
                  'gap_oracle': gap_oracle, 'normal_oracle': normal_oracle, 'opposite_oracle': opposite_oracle}
        row = {'identity': identity, 'excluded': None, 'native_steps': current['step'],
               'found_in_block': bool(current['done']), 'source_evaluation_only': env.source.tolist(),
               'array_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
        return row, arrays

    def body(self):
        self.setup()
        data, rows = {'dev': []}, []
        for identity in c.roster():
            row, arrays = self.case(identity)
            rows.append(row)
            if arrays is not None:
                data[identity['split']].append((identity, arrays))
            self.ledger.flush()
            c.append(self.out / 'cases.jsonl', row)
            if len(rows) % 24 == 0:
                print(json.dumps({'collected_cases': len(rows), 'total': len(c.roster())}), flush=True)
        counts = {}
        for split, entries in data.items():
            np = self.runtime.np
            c.require(entries, 'nonempty split')
            arrays = {k: np.stack([a[k] for _, a in entries]) for k in entries[0][1]}
            arrays['case_ids'] = np.asarray([i['id'] for i, _ in entries])
            arrays['regimes'] = np.asarray([i['regime'] for i, _ in entries])
            with (self.out / f'{split}.npz').open('xb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            counts[split] = {regime: sum(i['regime'] == regime for i, _ in entries) for regime in ('lambda3', 'lambda4')}
        c.require(not self.ledger.pending and all(v['attempted'] == v['returned'] for v in self.ledger.calls.values()), 'all calls returned')
        c.require(self.ledger.calls['tensorflow_value']['returned'] == self.ledger.calls['teacher_score']['returned'], 'one physical forward per annotation')
        c.require(all(v >= c.CONFIG['min_dev_per_regime'] for v in counts['dev'].values()),
                  'prespecified surviving-prefix minimum')
        self.ledger.close()
        c.write(self.out / 'summary.json', {'counts': counts, 'cases': len(rows), 'calls': self.ledger.calls,
                'nested_times_not_additive': True, 'io_seconds': self.ledger.io_seconds,
                'prefix_exclusions': sum(r['excluded'] is not None for r in rows),
                'learner_calls': 0, 'replacement_cases': 0, 'future_observations_in_committed_actions': False,
                'shadow_floor': None, 'sampler_uniform_bits': 53,
                'initial_distribution': '53-bit source-draw CDF law of initial native public prior',
                'opposite_actions': 'action XOR 1 from same observed prefix',
                'training_reference': self.training_reference,
                'train_array_decodes': 0, 'new_train_cases': 0,
                'privileged_belief_targets_are_not_learner_inputs': True})
        self.receipt['calls'] = self.ledger.calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    c.require(Path(sys.executable).absolute() == c.NATIVE, 'original qualified native interpreter')
    run = Collector(args, 'collect')
    try:
        run.body()
        run.finish()
    except BaseException as error:
        if hasattr(run, 'ledger'):
            run.receipt['calls'] = run.ledger.calls
            run.receipt['pending'] = run.ledger.pending
            run.ledger.close()
        run.finish(error)
        raise


if __name__ == '__main__':
    main()
