"""Fresh public prefixes plus bounded counterfactual conditional-cost labels."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import otto_cost_information_common as c

CALL_CAPS = {'tensorflow_construction': 1, 'tensorflow_build': 1, 'tensorflow_load': 1,
             'tensorflow_value': 32768, 'teacher_score': 32768, 'native_reset': 64,
             'native_step': 512, 'actor_construction': 64, 'actor_update': 512,
             'analytic_score': 576, 'feature_build': 576, 'backend_binding': 64,
             'branch_view_construction': 64, 'sampler_table': 2, 'sampler_lookup': 1024,
             'initial_source_law': 64, 'initial_normalization': 64, 'shadow_update': 512,
             'sampler_draw_validation': 576, 'committed_positions': 64, 'root_grid_law': 64,
             'exact_tree': 64, 'mc_draw_allocation': 128, 'mc_rollout': 128,
             'legacy_tree_update': 21760, 'legacy_mc_update': 131072}


def committed_actions(np, seed):
    c.require(type(seed) is int, 'declared integer native seed')
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 911]))).integers(
        0, 4, size=8, dtype=np.int64)


def draw_integers(np, seed):
    """Allocate all source+odor draws before examining any sampled outcome."""
    c.require(type(seed) is int, 'declared integer MC seed')
    return np.random.PCG64(seed).random_raw((128, 9)) >> np.uint64(11)


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
        self.parent_reference = c.parent_reference()
        c.write(self.out / 'parent-reference.json', self.parent_reference)
        prior, self.paths, self.reference, _ = c.original_native()
        c.require({k: c.desc(k) for k in prior['sources']} == self.plan['native_sources'], 'native source pins')
        self.plan['runtime'] = prior['runtime']
        self.ledger = Ledger(self)
        self.runtime = self.reference.Run.setup(self)
        from openjev.research import otto_conditional_cost_tree, otto_predictive_belief, otto_sampler_law
        from openjev.research.otto_query_gate import ReadOnlyBeliefView, _analytic, _features
        from openjev.research.otto_released_policy import PublicBeliefView
        self.view_class, self.analytic, self.features = ReadOnlyBeliefView, _analytic, _features
        self.public_view_class = PublicBeliefView
        self.tree, self.bayes, self.sampler = otto_conditional_cost_tree, otto_predictive_belief, otto_sampler_law
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
                **{name + '_raw': value for name, value in self.sensor_raw.items()},
                legacy_lambda3=self.runtime.kernels['base'],
                legacy_lambda4=self.runtime.kernels['shift'])
            stream.flush()
            os.fsync(stream.fileno())
        c.write(self.out / 'sensor-laws.json', metadata)

    def validate_draw(self, env, expected, channel):
        """Hidden native source is used only for audit of actual prefix draws."""
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

    def shadow_record(self, shadow, actor, current, evidence):
        np = self.runtime.np
        legacy = actor.belief.reshape(-1)
        self.ledger.emit('shadow.jsonl', {'context': self.ledger.context, 'public_step': current['step'],
            'public_done': current['done'], 'shadow_mass': float(shadow.sum()), 'legacy_mass': float(legacy.sum()),
            'l1_difference': float(np.abs(shadow - legacy).sum()),
            'max_abs_difference': float(np.abs(shadow - legacy).max()), 'evidence': evidence,
            'shadow_sha256': hashlib.sha256(shadow.tobytes()).hexdigest()})

    def case(self, identity):
        c.require(identity in c.roster(), 'registered fresh diagnostic identity')
        rt, call, np = self.runtime, self.ledger.call, self.runtime.np
        self.ledger.context = dict(identity)
        regime = 'base' if identity['regime'] == 'lambda3' else 'shift'
        sensing = float(identity['regime'][-1])
        env = call('native_reset', lambda: rt.public.seeded_environment(rt.source, identity['seed'],
            {'Ndim': 2, 'Ngrid': 53, 'Nhits': 4, 'lambda_over_dx': sensing, 'R_dt': 2.,
             'norm_Poisson': 'Euclidean'}, initial_hit=identity['initial_hit']))
        c.require(env.p_Poisson.tobytes() == rt.kernels[regime].tobytes(), 'unchanged native kernel')
        current = self.reference.packet(rt.public.observation(env, 0))
        actor = call('actor_construction', lambda: rt.analytic(current, rt.kernels[regime], allow_stay=False))
        view = self.view_class(actor._view)
        self.reference.belief_witness(actor, env, current, np)
        native_initial = actor.belief.reshape(-1).copy()
        call('sampler_draw_validation', lambda: self.validate_draw(env, native_initial, 'source'))
        initial_law = call('initial_source_law', lambda: self.bayes.cdf_law(native_initial, uniform_bits=53))
        normalized = call('initial_normalization', lambda: self.bayes.normalize_prior(initial_law['probabilities']))
        initial_belief = normalized['belief'].copy()
        shadow = initial_belief.copy()
        self.shadow_record(shadow, actor, current, None)
        self.ledger.emit('source-laws.jsonl', {'identity': identity,
            **{k: v for k, v in initial_law.items() if k != 'probabilities'},
            'initial_normalization': {k: v for k, v in normalized.items() if k != 'belief'}})
        previous = last_action = None
        prefix, prefix_actions, prefix_outcomes = [], [], []
        for t in range(9):
            self.ledger.context = {**identity, 'phase': 'prefix', 'step': t}
            if current['done']:
                return {'identity': identity, 'excluded': 'found_during_observed_prefix', 'native_steps': t}, None
            action, scores = call('analytic_score', lambda current=current: self.analytic(actor._policy._value_policy()[1], current['valid_actions']))
            f, previous = call('feature_build', lambda current=current, scores=scores, last_action=last_action, previous=previous: self.features(view, current, sensing, scores,
                last_action, current['step'] - current['step'] % 4, previous))
            prefix.append(f.copy())
            if t == 8:
                break
            prefix_actions.append(action)
            result = call('native_step', lambda action=action: env.step(action, quiet=True))
            after = self.reference.packet(rt.public.observation(env, current['step'] + 1))
            c.require((int(result[0]), bool(result[2])) == (after['hit'], after['done']), 'native outcome identity')
            call('actor_update', lambda action=action, after=after: actor.update(action, after))
            likelihood = call('sampler_lookup', lambda after=after: self.sampler.likelihood_at(
                self.sensor_laws[identity['regime']], np.asarray(after['position'], np.int64)))
            index = int(after['position'][0] * 53 + after['position'][1])
            update = call('shadow_update', lambda shadow=shadow, likelihood=likelihood, index=index, after=after: self.bayes.observe(shadow, likelihood, index,
                None if after['done'] else after['hit'], known_found=after['done']))
            shadow = update['belief']
            self.shadow_record(shadow, actor, after, update['evidence'])
            if not after['done']:
                displacement = np.asarray(env.source) - np.asarray(after['position'])
                expected = self.sensor_raw[identity['regime']][displacement[0] + 52, displacement[1] + 52]
                call('sampler_draw_validation', lambda expected=expected: self.validate_draw(env, expected, 'hit'))
            witness = self.reference.belief_witness(actor, env, after, np)
            self.ledger.emit('transitions.jsonl', {'case': identity['id'], 'action': action, 'public': after, 'posterior': witness})
            current, last_action = after, action
            prefix_outcomes.append(4 if current['done'] else current['hit'])

        c.require(current['step'] == 8 and all(8 <= x <= 44 for x in current['position']), 'retained complete in-bounds prefix')
        prefix_position = np.asarray(current['position'], np.int64)
        root_strict = shadow.copy()
        law = call('root_grid_law', lambda: self.bayes.cdf_law(root_strict, uniform_bits=53))
        root_grid = law['probabilities'].copy()
        tv = .5 * float(np.abs(root_grid - root_strict).sum(dtype=np.float64))
        c.require(tv <= c.CONFIG['root_grid_max_tv'], 'prospective root-grid TV bound')
        actions = committed_actions(np, identity['seed'])
        c.append(self.out / 'commitments.jsonl', {'identity': identity, 'prefix_public': current,
            'actions': actions.tolist(), 'prefix_actions': prefix_actions,
            'native_steps_before_counterfactuals': self.ledger.calls['native_step']['returned']})
        positions = call('committed_positions', lambda: self.sampler.planned_positions(prefix_position, actions))
        likelihoods = np.stack([call('sampler_lookup', lambda p=p: self.sampler.likelihood_at(
            self.sensor_laws[identity['regime']], p)) for p in positions])
        indices = positions[:, 0] * 53 + positions[:, 1]
        legacy_root = actor.belief.reshape(-1).copy()
        branch_view = call('branch_view_construction', lambda: self.public_view_class(rt.kernels[regime]))
        recorder = self.reference.ForwardRecorder(rt.model, rt.policy._value_policy.__code__, self.ledger, np)
        teacher = call('backend_binding', lambda: rt.policy(env=self.view_class(branch_view), model=recorder, sym_avg=True))
        stream_name = 'exact'

        def context(event):
            self.ledger.context = {**identity, 'phase': 'counterfactual', 'stream': stream_name, **event}

        def update_legacy(state, index, odor):
            channel = 'legacy_tree_update' if stream_name == 'exact' else 'legacy_mc_update'
            position = tuple(divmod(index, 53))
            return call(channel, lambda: branch_view._updated(state.reshape(53, 53), position, odor, False)).reshape(-1)

        def score_legacy(state, index):
            branch_view._probability = state.reshape(53, 53).copy()
            branch_view._agent = tuple(divmod(index, 53))
            before = branch_view.p_source.tobytes(), tuple(branch_view.agent)
            scores = call('teacher_score', lambda: teacher._value_policy()[1])
            c.require(before == (branch_view.p_source.tobytes(), tuple(branch_view.agent)), 'teacher cannot mutate branch filter')
            return scores

        exact = call('exact_tree', lambda: self.tree.exact_h4(root_grid, legacy_root, likelihoods[:4], indices[:4],
            update_legacy, score_legacy, check=self.check, emit=context))
        draws, sampled = [], []
        for stream_name, seed_key in (('select', 'select_seed'), ('evaluate', 'eval_seed')):
            self.ledger.context = {**identity, 'phase': 'mc_allocation', 'stream': stream_name}
            integers = call('mc_draw_allocation', lambda key=seed_key: draw_integers(np, identity[key]))
            draws.append(integers)
            sampled.append(call('mc_rollout', lambda ints=integers: self.tree.sample_h8(
                root_grid, legacy_root, likelihoods, indices, ints, exact, update_legacy, score_legacy,
                check=self.check, emit=context)))
        c.require(current == actor.public and actor.belief.reshape(-1).tobytes() == legacy_root.tobytes()
                  and current['step'] == 8, 'counterfactuals leave actual prefix untouched')
        arrays = {'prefix': np.stack(prefix), 'prefix_lengths': np.int64(9), 'actions': actions,
                  'prefix_actions': np.asarray(prefix_actions, np.int64), 'prefix_outcomes': np.asarray(prefix_outcomes, np.int64),
                  'prefix_position': prefix_position, 'initial_belief': initial_belief, 'root_strict': root_strict,
                  'root_grid': root_grid, 'legacy_root': legacy_root, 'exact_weights': exact['weights'],
                  'exact_costs': exact['costs'], 'exact_alive': exact['supported'], 'mc_draws': np.stack(draws)}
        for key in ('source_indices', 'outcomes', 'alive4', 'alive8', 'costs4', 'costs8'):
            arrays['mc_' + key] = np.stack([row[key] for row in sampled])
        row = {'identity': identity, 'excluded': None, 'native_steps': 8,
               'root_grid': {**{k: v for k, v in law.items() if k != 'probabilities'}, 'total_variation': tv},
               'exact': {k: exact[k] for k in ('survival_mass', 'found_mass', 'work')},
               'mc': [{'stream': name, 'seed': identity[key], 'alive4': int(data['alive4'].sum()),
                       'alive8': int(data['alive8'].sum()), 'work': data['work']}
                      for (name, key), data in zip((('select', 'select_seed'), ('evaluate', 'eval_seed')), sampled, strict=True)],
               'array_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
        return row, arrays

    def body(self):
        self.setup()
        data, rows = [], []
        for identity in c.roster():
            row, arrays = self.case(identity)
            rows.append(row)
            if arrays is not None:
                data.append((identity, arrays))
            self.ledger.flush()
            c.append(self.out / 'cases.jsonl', row)
            if len(rows) % 8 == 0:
                print(json.dumps({'completed_prefixes': len(rows), 'total': len(c.roster())}), flush=True)
        np = self.runtime.np
        counts = {regime: sum(i['regime'] == regime for i, _ in data) for regime in ('lambda3', 'lambda4')}
        c.require(all(n >= c.CONFIG['min_cases'] for n in counts.values()), 'prespecified surviving-prefix minimum')
        arrays = {key: np.stack([entry[key] for _, entry in data]) for key in data[0][1]}
        arrays['case_ids'] = np.asarray([i['id'] for i, _ in data])
        arrays['regimes'] = np.asarray([i['regime'] for i, _ in data])
        with (self.out / 'cases.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        c.require(not self.ledger.pending and all(v['attempted'] == v['returned'] for v in self.ledger.calls.values()), 'all calls returned')
        c.require(self.ledger.calls['tensorflow_value']['returned'] == self.ledger.calls['teacher_score']['returned'], 'one physical forward per leaf annotation')
        c.require(self.ledger.calls['native_step']['returned'] == sum(row['native_steps'] for row in rows), 'native work is prefix only')
        self.ledger.close()
        c.write(self.out / 'summary.json', {'counts': counts, 'cases': len(rows), 'retained': len(data),
            'calls': self.ledger.calls, 'nested_times_not_additive': True, 'io_seconds': self.ledger.io_seconds,
            'prefix_exclusions': sum(row['excluded'] is not None for row in rows), 'replacement_cases': 0,
            'parent_reference': self.parent_reference, 'parent_data_array_decodes': 0, 'learner_calls': 0,
            'native_continuation_steps': 0, 'sampler_uniform_bits': 53, 'shadow_floor': None,
            'mc_stream_order': ['select', 'evaluate'], 'mc_draw_shape_per_case': [2, 128, 9],
            'root_law': 'cdf_law(strict public prefix), common to exact tree and integer-grid Monte Carlo',
            'teacher_scope': 'H4 positive exact leaves and H8 MC surviving leaves only',
            'privileged_reference_arrays_are_not_learner_inputs': True})
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
