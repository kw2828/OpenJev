"""Independent saved-output schedule-mixture headroom audit.

No producer metric, model, world generator, optimizer or RNG replay is used.
The caller authenticates sources/process closure. Historical execution order
is an authenticated producer attestation, not reconstructed by this auditor.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = 'schedule-headroom-audit-v1'
OLD_ARMS = ('unchanged', 'global', 'static_bank', 'markov_bank', 'recurrent_bank', 'reset_bank')
ARMS = OLD_ARMS + ('learned_exact', 'true_exact', 'learned_static2', 'true_static2')
CONTROLS = OLD_ARMS + ('learned_static2',)
PARAMETERS = dict(zip(OLD_ARMS, (0, 1, 0, 4, 164, 164), strict=True))
STATES = dict(zip(OLD_ARMS, (8, 8, 24, 24, 28, 28), strict=True))
FIELDS = {'transition': (4, 8, 8), 'emission': (4, 8), 'hazard': (4, 8), 'costs': (4, 8)}
CONFIG = {'cohorts': 5, 'namespace': 439260924, 'episodes': 512, 'steps': 32}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, name):
    require(np.shape(actual) == np.shape(expected) and np.allclose(actual, expected, rtol=1e-10, atol=1e-12), name)


def compare(actual, expected, name='value'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), name + ' keys')
        for key in expected:
            compare(actual[key], expected[key], name + '/' + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name + ' length')
        for i, value in enumerate(expected):
            compare(actual[i], value, name + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), name)
    else:
        require(actual == expected, name)


def descriptor(path):
    value = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}


def read(path):
    return json.loads(Path(path).read_text())


def journal(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _array(value, shape, kind, name):
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == np.dtype(kind), name + ' schema')
    if value.dtype.kind != 'U':
        require(np.isfinite(value).all(), name + ' finite')


def true_fields():
    """Literal physical law, independently assembled from its public specification."""
    t = np.full((4, 8, 8), .0025, dtype=np.float64)
    h = np.empty((4, 8), dtype=np.float64)
    c = np.empty((4, 8), dtype=np.float64)
    for action in range(4):
        for state in range(8):
            destination = (state ^ 1, (state + 1) % 8, ((state << 1) & 7) | (state >> 2), state ^ 4)[action]
            t[action, destination, state] += .98
            h[action, state] = .005 + .005 * (((state >> 2) & 1) ^ (action & 1))
            c[action, state] = -.75 if action == ((state ^ (state >> 1)) & 3) else .25
    return t, h, c


def fork_values(states, actions, transition, hazard, costs):
    current = states.copy()
    a = transition * (1 - hazard[:, :, None])
    values, masses = [], []
    for step in range(8):
        current = np.einsum('...ij,...j->...i', a[actions[..., step]], current)
        values.append(current @ costs.T)
        masses.append(current.sum(-1))
    return np.stack(values, axis=-2), np.stack(masses, axis=-1)


def reconstruct_targets(data):
    """No sampling: condition the literal law on the saved public events/noise path."""
    actions, labels = data['public/actions'], data['public/observations']
    n, length = labels.shape
    transition, hazard, costs = true_fields()
    state = np.full((n, 8), .125, dtype=np.float64)
    probabilities, states = [], []
    absorbed = np.zeros(n, dtype=bool)
    for index in range(length):
        epsilon = data['audit/epsilons'][:, index]
        emission = np.broadcast_to(epsilon[:, None, None] / 3, (n, 4, 8)).copy()
        for latent in range(8):
            emission[:, latent & 3, latent] = 1 - epsilon
        if index:
            moved = np.einsum('bij,bj->bi', transition[actions[:, index - 1]], state)
            found = (moved * hazard[actions[:, index - 1]]).sum(1)
            branches = emission * (moved * (1 - hazard[actions[:, index - 1]]))[:, None]
        else:
            found = np.zeros(n)
            branches = emission * state[:, None]
        law = np.concatenate((branches.sum(2), found[:, None]), axis=1)
        law[absorbed] = (0, 0, 0, 0, 1)
        selected = labels[:, index]
        active = ~absorbed & (selected != 4)
        state = np.zeros_like(state)
        ids = np.flatnonzero(active)
        evidence = law[ids, selected[ids]]
        require((evidence > 0).all(), 'possible saved event under true law')
        state[ids] = branches[ids, selected[ids]] / evidence[:, None]
        absorbed |= selected == 4
        probabilities.append(law)
        states.append(state.copy())
    posterior = np.stack(states, axis=1)
    forks, _ = fork_values(posterior, data['targets/fork_actions'], transition, hazard, costs)
    return {'probabilities': np.stack(probabilities, axis=1), 'post_states': posterior,
            'post_costs': posterior @ costs.T, 'fork_costs': forks}


def parameter_shapes(arm):
    result = {}
    if arm == 'global':
        result['global_logit'] = ()
    if arm in ('markov_bank', 'recurrent_bank', 'reset_bank'):
        result['prior_logits'] = (3,)
    if arm == 'markov_bank':
        result['reset_logit'] = ()
    if arm in ('recurrent_bank', 'reset_bank'):
        result.update({'gru.weight_ih': (12, 7), 'gru.weight_hh': (12, 4), 'gru.bias_ih': (12,),
                       'gru.bias_hh': (12,), 'reset_head.weight': (1, 4), 'reset_head.bias': (1,)})
    return result


def verify_state(state, fields, arm, initial=False):
    shapes = parameter_shapes(arm)
    require(state.keys() == {**FIELDS, **shapes}.keys(), 'exact model state fields')
    for name, shape in {**FIELDS, **shapes}.items():
        _array(state[name], shape, 'float64', name)
        if name in FIELDS:
            require(state[name].tobytes() == fields[name].tobytes(), 'unchanged copied frozen field/' + name)
        elif initial and name in ('global_logit', 'prior_logits', 'reset_head.weight'):
            require((state[name] == 0).all(), 'fixed zero initializer/' + name)
        elif initial and name in ('reset_logit', 'reset_head.bias'):
            require((state[name] == -4).all(), 'fixed reset initializer/' + name)
    require(sum(np.size(state[name]) for name in shapes) == PARAMETERS[arm], 'fixed parameter count')


def physical_fields():
    transition, hazard, costs = true_fields()
    emission = np.full((4, 8), .04, dtype=np.float64)
    for state in range(8):
        emission[state % 4, state] = .88
    return {'transition': transition, 'emission': emission, 'hazard': hazard, 'costs': costs}


def schedule_bank(length, kind):
    require(kind in ('exact', 'static2'), 'declared reference kind')
    schedules = [np.full(length, .12), np.full(length, .30)]
    priors = [.5, .5] if kind == 'static2' else [.25, .25]
    if kind == 'exact':
        for first, second in ((.12, .30), (.30, .12)):
            for switch in range(8, 25):
                values = np.full(length, first)
                values[switch:] = second
                schedules.append(values)
                priors.append(.25/17)
    return np.array(schedules, dtype=np.float64), np.array(priors, dtype=np.float64)


def exact_predictions(fields, public, fork_actions, kind='exact'):
    """Direct NumPy joint schedule/state filtering, predicting before the label."""
    actions, labels = public['actions'], public['observations']
    n, length = labels.shape
    schedules, prior = schedule_bank(length, kind)
    hypotheses = len(prior)
    alpha = np.broadcast_to(prior[None, :, None]/8, (n, hypotheses, 8)).copy()
    probabilities, states, posteriors = [], [], []
    absorbed = np.zeros(n, dtype=bool)
    for time in range(length):
        q = (schedules[:, time]-.12)/.63
        emission = (1-q[:, None, None])*fields['emission'][None] + q[:, None, None]/4
        if time:
            moved = np.einsum('bij,bkj->bki', fields['transition'][actions[:, time-1]], alpha)
            hazard = fields['hazard'][actions[:, time-1]]
            surviving = moved*(1-hazard[:, None])
            found = (moved*hazard[:, None]).sum((1, 2))
        else:
            surviving, found = alpha, np.zeros(n)
        branches = surviving[:, :, None]*emission[None]
        law = np.concatenate((branches.sum((1, 3)), found[:, None]), axis=1)
        law[absorbed] = (0, 0, 0, 0, 1)
        observed = labels[:, time]
        active = np.flatnonzero(~absorbed & (observed < 4))
        updated = np.zeros_like(alpha)
        evidence = law[active, observed[active]]
        require((evidence > 0).all(), 'possible exact-reference observation')
        updated[active] = branches[active, :, observed[active], :] / evidence[:, None, None]
        alpha = updated
        absorbed |= observed == 4
        probabilities.append(law)
        states.append(alpha.sum(1))
        posteriors.append(alpha.sum(2))
    posterior = np.stack(states, axis=1)
    values, masses = fork_values(posterior, fork_actions, fields['transition'], fields['hazard'], fields['costs'])
    return {'probabilities': np.stack(probabilities, axis=1), 'post_states': posterior,
            'post_costs': posterior@fields['costs'].T, 'schedule_posterior': np.stack(posteriors, axis=1),
            'fork_costs': values[:, :, (3, 7)], 'fork_survival': masses[:, :, (3, 7)]}


def validate_data(data, counts, namespace, episodes=512, steps=32):
    n, length = episodes, steps+1
    shapes = {'public/actions': ((n, steps), 'int64'), 'public/observations': ((n, length), 'int64'),
              'public/lengths': ((n,), 'int64'), 'public/case_ids': ((n,), 'U64'),
              'targets/probabilities': ((n, length, 5), 'float64'),
              'targets/post_states': ((n, length, 8), 'float64'),
              'targets/post_costs': ((n, length, 4), 'float64'),
              'targets/fork_actions': ((n, length, 8), 'int64'),
              'targets/fork_costs': ((n, length, 8, 4), 'float64'),
              'audit/epsilons': ((n, length), 'float64'),
              'audit/switch_indices': ((n,), 'int64'), 'audit/family': ((n,), 'int64')}
    require(data.keys() == shapes.keys(), 'complete saved data schema')
    for name, (shape, dtype) in shapes.items():
        _array(data[name], shape, dtype, name)
    actions, labels, lengths = data['public/actions'], data['public/observations'], data['public/lengths']
    require(((actions >= 0) & (actions < 4)).all() and ((data['targets/fork_actions'] >= 0)
            & (data['targets/fork_actions'] < 4)).all(), 'all public action bounds')
    require(((labels >= 0) & (labels <= 4)).all() and (labels[:, 0] < 4).all(), 'all event bounds')
    found = labels == 4
    require(np.array_equal(lengths, np.where(found.any(1), found.argmax(1)+1, length))
            and ((found.cumsum(1) == 0) | found).all(), 'first-found inclusive terminal grammar')
    require(data['public/case_ids'].tolist() == [f'ns{namespace}-case{i:010d}' for i in range(n)], 'all fixed case identities')
    family = data['audit/family']
    require(((family >= 0) & (family < 4)).all(), 'four supported private families')
    for case in range(n):
        first, last = ((.12, .12), (.30, .30), (.12, .30), (.30, .12))[family[case]]
        switch = int(data['audit/switch_indices'][case])
        require((8 <= switch <= 24) if family[case] >= 2 else switch == -1, 'supported single-switch schedule')
        epsilons = np.full(length, first)
        if switch >= 0:
            epsilons[switch:] = last
        require(np.array_equal(data['audit/epsilons'][case], epsilons), 'complete private schedule')
    expected_counts = {'version': 'finite-schedule-world-v1', 'namespace': namespace,
        'episodes': n, 'steps': steps, 'retained': n, 'excluded_found': 0,
        'rng_case_sequences': n, 'rng_streams': 4*n, 'episode_action_draws': n*steps,
        'fork_action_draws': n*length*8, 'family_draws': n,
        'switch_draws': int((family >= 2).sum()), 'reset_event_draws': n,
        'transition_event_draws': int(lengths.sum())-n, 'found_episodes': int(found.any(1).sum()),
        'valid_events': int(lengths.sum()), 'absorbed_suffix_events': int(n*length-lengths.sum()),
        'emission_constructions': int(lengths.sum()), 'fork_boundaries': n*length,
        'fork_transition_products': n*length*8,
        'family_counts': {str(code): int((family == code).sum()) for code in range(4)},
        'targets_privileged': True, 'learner_inputs': 'public histories and precommitted fork actions only'}
    for key, value in expected_counts.items():
        compare(counts[key], value, 'data counts/' + key)
    reference = reconstruct_targets(data)
    for key, value in reference.items():
        close(data['targets/'+key], value, 'independent private-path reference/' + key)
    return reference


def validate_predictions(data, predictions, fields, arm):
    labels = data['public/observations']
    n, length = labels.shape
    shapes = {'probabilities': (n, length, 5), 'post_states': (n, length, 8),
              'post_costs': (n, length, 4), 'fork_costs': (n, length, 2, 4),
              'fork_survival': (n, length, 2)}
    if arm in OLD_ARMS:
        shapes['reset_rates'] = (n, length-1)
    else:
        shapes['schedule_posterior'] = (n, length, 36 if arm.endswith('exact') else 2)
    require(predictions.keys() == shapes.keys(), 'complete declared prediction keys')
    for key, shape in shapes.items():
        _array(predictions[key], shape, 'float64', key)
    law, state = predictions['probabilities'], predictions['post_states']
    require((law >= 0).all() and (np.abs(law.sum(2)-1) <= 1e-12).all(), 'normalized predicted event law')
    found = (labels == 4).cumsum(1) > 0
    require((state >= 0).all() and (np.abs(state.sum(2)-(~found)) <= 1e-12).all()
            and (state[found] == 0).all(), 'normalized posterior and exact absorbing zero')
    suffix = np.arange(length)[None] >= data['public/lengths'][:, None]
    require((law[suffix] == (0, 0, 0, 0, 1)).all() and (law[:, 0, 4] == 0).all(), 'reset and absorbing event laws')
    if 'reset_rates' in predictions:
        rates = predictions['reset_rates']
        require(((rates >= 0) & (rates <= 1)).all() and (rates[found[:, :-1]] == 0).all(), 'rates and terminal chronology')
        if arm in ('unchanged', 'global', 'static_bank'):
            require((rates == 0).all(), 'no reset in nonswitching controls')
    close(predictions['post_costs'], state@fields['costs'].T, 'actual-field immediate cost')
    values, survival = fork_values(state, data['targets/fork_actions'], fields['transition'], fields['hazard'], fields['costs'])
    close(predictions['fork_costs'], values[:, :, (3, 7)], 'actual-field blind cost')
    close(predictions['fork_survival'], survival[:, :, (3, 7)], 'actual-field blind survival')
    if arm not in OLD_ARMS:
        reference = exact_predictions(fields, {'actions': data['public/actions'], 'observations': labels},
                                      data['targets/fork_actions'], 'exact' if arm.endswith('exact') else 'static2')
        for name, value in reference.items():
            close(predictions[name], value, 'independent public mixture/' + name)


def metrics(data, predictions):
    labels, lengths = data['public/observations'], data['public/lengths']
    _n, length = labels.shape
    require(length == 33, 'registered32-step fixed-denominator metric')
    valid = np.arange(length)[None] < lengths[:, None]
    chosen_prob = np.take_along_axis(predictions['probabilities'], labels[:, :, None], axis=2)[:, :, 0]
    require((chosen_prob[valid] > 0).all(), 'positive observed-event evidence')
    truth = data['targets/fork_costs'][:, :, (3, 7)]
    chosen = predictions['fork_costs'].argmin(3)
    regret = np.take_along_axis(truth, chosen[..., None], axis=3)[..., 0]-truth.min(3)
    per_horizon = regret[:, 8:].sum(1)/25
    per_episode = per_horizon.mean(1)
    post_truth = data['targets/post_costs']
    post_chosen = predictions['post_costs'].argmin(2)
    post_regret = np.take_along_axis(post_truth, post_chosen[..., None], axis=2)[..., 0]-post_truth.min(2)
    families = []
    for family in range(4):
        selected = data['audit/family'] == family
        families.append({'family': family, 'episodes': int(selected.sum()),
                         'primary_regret': float(per_episode[selected].mean()) if selected.any() else None})
    return {'primary_regret': float(per_episode.mean()), 'h4_regret': float(per_horizon[:, 0].mean()),
            'h8_regret': float(per_horizon[:, 1].mean()), 'post_regret': float(post_regret[:, 8:].sum(1).mean()/25),
            'event_nll': float(-np.log(chosen_prob[valid]).mean()),
            'event_count': int(valid.sum()), 'found_episodes': int((labels == 4).any(1).sum()),
            'per_episode_regret': per_episode.tolist(),
            'family_regret': families}


def classify(rows):
    lookup = {(r['cohort'], r['arm']): r for r in rows}
    require(len(rows) == len(lookup) == 50 and set(lookup) == {(c, a) for c in range(5) for a in ARMS}, 'all50 fixed cohort/arm rows')
    means = {arm: float(np.mean([lookup[c, arm]['primary_regret'] for c in range(5)])) for arm in ARMS}
    groups = {}
    for candidate in ('learned_exact', 'true_exact'):
        conditions, comparisons = {}, []
        for control in CONTROLS:
            wins = sum(lookup[c, candidate]['primary_regret'] < lookup[c, control]['primary_regret'] for c in range(5))
            conditions[control+'/mean_gain10pct'] = bool(means[control] > 0 and means[candidate] <= .9*means[control])
            conditions[control+'/paired_wins4of5'] = wins >= 4
            comparisons.append({'control': control, 'candidate_mean': means[candidate],
                                'control_mean': means[control], 'paired_wins': wins})
        base = {}
        for arm in (candidate, 'unchanged'):
            entries = [next(x for x in lookup[c, arm]['family_regret'] if x['family'] == 0) for c in range(5)]
            require(all(x['episodes'] > 0 and x['primary_regret'] is not None for x in entries), 'BASE-family support every cohort')
            base[arm] = float(np.mean([x['primary_regret'] for x in entries]))
        conditions['base/unchanged/noninferiority'] = base[candidate] <= 1.05*base['unchanged']+1e-6
        require(len(conditions) == 15, 'fixed15-condition group')
        groups[candidate] = {'passed': all(conditions.values()), 'conditions': conditions,
                             'comparisons': comparisons, 'candidate_base': base[candidate],
                             'unchanged_base': base['unchanged']}
    wins = sum(lookup[c, 'true_exact']['primary_regret'] < lookup[c, 'true_static2']['primary_regret'] for c in range(5))
    history = {'mean_gain10pct': bool(means['true_static2'] > 0 and means['true_exact'] <= .9*means['true_static2']),
               'paired_wins4of5': wins >= 4}
    if groups['learned_exact']['passed']:
        status = 'LEARNED_MODEL_HEADROOM'
    elif groups['true_exact']['passed'] and all(history.values()):
        status = 'MODEL_MISMATCH_HEADROOM'
    elif groups['true_exact']['passed']:
        status = 'TRUE_MODEL_ADVANTAGE_WITHOUT_SCHEDULE_MEMORY'
    else:
        status = 'NO_REGISTERED_HEADROOM'
    return {'classification': status, 'candidates': groups, 'true_history_conditions': history,
            'true_history_paired_wins': wins,
            'means': means}


def audit(run_folder: Path, output_folder: Path):
    run_folder, output_folder = Path(run_folder), Path(output_folder)
    require(output_folder.is_dir() and not any(output_folder.iterdir()), 'exclusive empty audit output')
    summary = read(run_folder / 'summary.json')
    require(summary['version'] == 'schedule-headroom-v1' and summary['config'] == CONFIG
            and summary['training_calls'] == 0, 'fixed diagnostic without training')
    rows = summary['rows']
    require(rows == journal(run_folder / 'rows.jsonl'), 'complete row journal join')
    lookup = {(r['cohort'], r['arm']): r for r in rows}
    require(len(rows) == len(lookup) == 50 and set(lookup) == {(c, a) for c in range(5) for a in ARMS}, 'exact50-row population')
    barrier = read(run_folder / 'model-copy-barrier.json')
    copied = {f'cohort-{c:02d}/{name}' for c in range(5)
              for name in ('backbone.npz', *(f'state-{a}.npz' for a in OLD_ARMS))}
    require(barrier.keys() == copied and len(copied) == 35, 'complete35-copy pre-generation barrier')
    root = Path(__file__).resolve().parents[1]
    parent = root / 'output/reliability-memory-v1'
    plan = read(run_folder.parent / 'registration.json')
    require(plan['config'] == CONFIG and plan['version'] == 'schedule-headroom-v1', 'registered numerical interface')
    for cohort in range(5):
        for arm in (None, *OLD_ARMS):
            name = 'backbone.npz' if arm is None else f'state-{arm}.npz'
            original = 'backbone.npz' if arm is None else f'{arm}-final.npz'
            relative = f'cohort-{cohort:02d}/{name}'
            source_name = f'run/cohort-{cohort:02d}/{original}'
            pin = descriptor(run_folder / relative)
            require(barrier[relative] == {'source': str(parent / source_name), **pin}
                    and pin == plan['parent']['inputs'][source_name]
                    and descriptor(parent / source_name) == pin, 'original frozen model/field byte copy')
    counts = {'npz_decodes': 0, 'model_state_decodes': 0, 'backbone_decodes': 0,
              'data_decodes': 0, 'prediction_decodes': 0,
              'private_target_reconstructions': 0, 'exact_public_reconstructions': 0,
              'prediction_checks': 0, 'copied_files_checked': 35,
              'model_calls': 0, 'generator_calls': 0, 'optimizer_calls': 0, 'rng_replays': 0}
    def load(path):
        counts['npz_decodes'] += 1
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    expected_files = {'summary.json', 'rows.jsonl', 'model-copy-barrier.json'} | copied
    audited = []
    for cohort in range(5):
        location = run_folder / f'cohort-{cohort:02d}'
        fields = load(location / 'backbone.npz')
        counts['backbone_decodes'] += 1
        require(fields.keys() == FIELDS.keys(), 'four actual learned fields')
        for name, shape in FIELDS.items():
            _array(fields[name], shape, 'float64', name)
        require((fields['transition'] >= 0).all() and (fields['emission'] >= 0).all()
                and ((fields['hazard'] >= 0) & (fields['hazard'] <= 1)).all(), 'valid frozen field probabilities')
        close(fields['transition'].sum(1), np.ones((4, 8)), 'transition stochasticity')
        close(fields['emission'].sum(0), np.ones(8), 'emission stochasticity')
        for arm in OLD_ARMS:
            state = load(location / f'state-{arm}.npz')
            counts['model_state_decodes'] += 1
            verify_state(state, fields, arm)
        data = load(location / 'data.npz')
        counts['data_decodes'] += 1
        validate_data(data, read(location / 'data.json'), CONFIG['namespace']+cohort)
        counts['private_target_reconstructions'] += 1
        require(all(int((data['audit/family'] == f).sum()) >= 64 for f in range(4)), 'fixed64-per-family support')
        expected_files.update((f'cohort-{cohort:02d}/data.npz', f'cohort-{cohort:02d}/data.json'))
        for arm in ARMS:
            row = lookup[cohort, arm]
            old = arm in OLD_ARMS
            is_true = arm.startswith('true_')
            kind = 'exact' if arm.endswith('exact') else 'static2'
            hypothesis_count = None if old else (36 if kind == 'exact' else 2)
            expected_info = {'kind': 'frozen_reliability' if old else 'exact_schedule' if kind == 'exact' else 'static_two_mode',
                'physics': 'true' if is_true else 'learned',
                'backbone': None if is_true else descriptor(location / 'backbone.npz'),
                'checkpoint': descriptor(location / f'state-{arm}.npz') if old else None,
                'schedule_hypotheses': hypothesis_count,
                'added_stored_parameters': PARAMETERS[arm] if old else 0}
            require(row['source_info'] == expected_info and row['state_scalars'] == (STATES[arm] if old else 8*hypothesis_count)
                    and math.isfinite(row['inference_seconds']) and 0 < row['inference_seconds'] < 120, 'source/size/time row metadata')
            name = f'prediction-{arm}.npz'
            prediction = load(location / name)
            counts['prediction_decodes'] += 1
            actual_fields = physical_fields() if is_true else fields
            validate_predictions(data, prediction, actual_fields, arm)
            counts['prediction_checks'] += 1
            counts['exact_public_reconstructions'] += int(not old)
            numerical = metrics(data, prediction)
            require(row.keys() == {'cohort', 'arm', 'inference_seconds', 'state_scalars', 'source_info'} | numerical.keys(), 'exact scalar row keys')
            compare({name: row[name] for name in numerical}, numerical, 'independent row arithmetic')
            audited.append({**row, **numerical})
            expected_files.add(f'cohort-{cohort:02d}/{name}')
    result = classify(audited)
    compare(summary['result'], result, 'independent registered classification')
    actual_files = {str(p.relative_to(run_folder)) for p in run_folder.rglob('*') if p.is_file()}
    require(actual_files == expected_files and len(expected_files) == 98, 'complete98-file producer inventory')
    require(counts['npz_decodes'] == 90 and counts['model_state_decodes'] == 30
            and counts['backbone_decodes'] == counts['data_decodes'] == 5
            and counts['prediction_decodes'] == counts['prediction_checks'] == 50
            and counts['private_target_reconstructions'] == 5 and counts['exact_public_reconstructions'] == 20,
            'exact saved-output check counts')
    output = {'version': VERSION, 'agreement': True, 'config': CONFIG, 'rows': audited,
              'result': result, 'checkcounts': counts,
              'scope': 'Independent saved target/public-mixture/fork/metric arithmetic. No old reliability-model, generator, RNG or training replay. The caller authenticates original phase/source closure; copy-barrier historical ordering is attested. Exactness under learned fields is model-conditional, while true-law Bayes interpretation requires the specified supported prior. Private-path oracle remains privileged; point-estimate gates are not significance tests.'}
    with (output_folder / 'audit.json').open('x') as stream:
        json.dump(output, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return {'agreement': True, 'classification': result['classification'], 'rows': 50,
            'learned_conditions_passed': sum(result['candidates']['learned_exact']['conditions'].values()),
            'true_conditions_passed': sum(result['candidates']['true_exact']['conditions'].values()),
            'true_history_conditions_passed': sum(result['true_history_conditions'].values()), 'checkcounts': counts}
