"""Independent saved-output audit of causal reliability memory.

No learner, world generator, training loop or producer metric is imported.
The caller authenticates original process/source closure before invoking this
module. Saved likelihoods are checked as probability laws, not replayed model
outputs. Historical ordering and training execution remain producer attestations.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = 'reliability-memory-audit-v1'
ARMS = ('unchanged', 'global', 'static_bank', 'markov_bank', 'recurrent_bank', 'reset_bank')
TRAINED = ('global', 'markov_bank', 'recurrent_bank', 'reset_bank')
STRATA = ('base', 'shift', 'switch', 'stress')
PARAMETERS = dict(zip(ARMS, (0, 1, 0, 4, 164, 164), strict=True))
STATES = dict(zip(ARMS, (8, 8, 24, 24, 28, 28), strict=True))
FIELDS = {'transition': (4, 8, 8), 'emission': (4, 8), 'hazard': (4, 8), 'costs': (4, 8)}
CONFIG = {'cohorts': 5, 'namespace': 438260924, 'seed': 438261001,
          'train_episodes': 512, 'dev_episodes': 512, 'steps': 32,
          'updates': 256, 'batch_size': 64, 'learning_rate': .01,
          'gradient_clip': 5., 'fit_cap_seconds': 180.}


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


def validate_data(data, counts, namespace, split, episodes=512, steps=32):
    n, length = episodes, steps + 1
    shapes = {'public/actions': ((n, steps), 'int64'), 'public/observations': ((n, length), 'int64'),
              'public/lengths': ((n,), 'int64'), 'public/case_ids': ((n,), 'U64'),
              'targets/probabilities': ((n, length, 5), 'float64'),
              'targets/post_states': ((n, length, 8), 'float64'),
              'targets/post_costs': ((n, length, 4), 'float64'),
              'targets/fork_actions': ((n, length, 8), 'int64'),
              'targets/fork_costs': ((n, length, 8, 4), 'float64'),
              'audit/epsilons': ((n, length), 'float64'),
              'audit/switch_indices': ((n,), 'int64'), 'audit/family': ((n,), 'U32')}
    require(data.keys() == shapes.keys(), 'complete saved episode keys')
    for key, (shape, dtype) in shapes.items():
        _array(data[key], shape, dtype, key)
    actions, labels, lengths = data['public/actions'], data['public/observations'], data['public/lengths']
    require(((actions >= 0) & (actions < 4)).all() and ((data['targets/fork_actions'] >= 0)
            & (data['targets/fork_actions'] < 4)).all(), 'public action bounds')
    require(((labels >= 0) & (labels <= 4)).all() and (labels[:, 0] < 4).all(), 'public event bounds')
    found = labels == 4
    expected_lengths = np.where(found.any(1), found.argmax(1) + 1, length)
    require(np.array_equal(lengths, expected_lengths)
            and ((found.cumsum(1) == 0) | found).all(), 'first found inclusive and absorbing suffix')
    codes = {'train': 0, 'base': 1, 'shift': 2, 'switch': 3, 'stress': 4}
    require(data['public/case_ids'].tolist() == [f'ns{namespace}-split{codes[split]}-case{i:010d}' for i in range(n)], 'all ordered case identities')
    families = []
    for case in range(n):
        if split == 'train':
            first, last = ((.12, .12), (.48, .48), (.12, .48), (.48, .12))[case % 4]
            family = ('static_012', 'static_048', 'switch_012_to_048', 'switch_048_to_012')[case % 4]
        elif split == 'switch':
            first, last = (.12, .30) if case % 2 == 0 else (.30, .12)
            family = 'switch_012_to_030' if case % 2 == 0 else 'switch_030_to_012'
        else:
            first = last = {'base': .12, 'shift': .30, 'stress': .60}[split]
            family = {'base': 'static_012', 'shift': 'static_030', 'stress': 'static_060'}[split]
        switch = int(data['audit/switch_indices'][case])
        require((8 <= switch <= 24) if first != last else switch == -1, 'prescribed schedule support')
        expected = np.full(length, first)
        if switch >= 0:
            expected[switch:] = last
        require(np.array_equal(data['audit/epsilons'][case], expected), 'saved private noise path')
        families.append(family)
    require(data['audit/family'].tolist() == families, 'fixed family allocation')
    expected_counts = {'version': 'finite-reliability-world-v1', 'namespace': namespace,
        'split': split, 'split_code': codes[split], 'episodes': n, 'steps': steps, 'retained': n,
        'excluded_found': 0, 'rng_case_sequences': n, 'rng_streams': 4*n,
        'episode_action_draws': n*steps, 'fork_action_draws': n*length*8,
        'switch_draws': int((data['audit/switch_indices'] >= 0).sum()), 'reset_event_draws': n,
        'transition_event_draws': int(lengths.sum())-n, 'found_episodes': int(found.any(1).sum()),
        'valid_events': int(lengths.sum()), 'absorbed_suffix_events': int(n*length-lengths.sum()),
        'emission_constructions': int(lengths.sum()), 'fork_boundaries': n*length,
        'fork_transition_products': n*length*8,
        'family_counts': {family: families.count(family) for family in set(families)},
        'targets_privileged': True, 'training_inputs': 'public only'}
    for key, value in expected_counts.items():
        compare(counts[key], value, 'episode counts/' + key)
    reference = reconstruct_targets(data)
    for key, value in reference.items():
        close(data['targets/' + key], value, 'independent true reference/' + key)
    return reference


def validate_predictions(data, predictions, fields, arm):
    labels = data['public/observations']
    n, length = labels.shape
    shapes = {'probabilities': (n, length, 5), 'post_states': (n, length, 8),
              'post_costs': (n, length, 4), 'reset_rates': (n, length-1),
              'fork_costs': (n, length, 2, 4), 'fork_survival': (n, length, 2)}
    require(predictions.keys() == shapes.keys(), 'complete prediction keys')
    for key, shape in shapes.items():
        _array(predictions[key], shape, 'float64', key)
    law, state = predictions['probabilities'], predictions['post_states']
    require((law >= 0).all() and (np.abs(law.sum(2)-1) <= 1e-12).all(), 'normalized predictive event laws')
    found = (labels == 4).cumsum(1) > 0
    require((state >= 0).all() and (np.abs(state.sum(2)-(~found)) <= 1e-12).all(), 'posterior unit/absorbed mass')
    require(np.all(state[found] == 0), 'exact absorbed posterior zero')
    suffix = np.arange(length)[None] >= data['public/lengths'][:, None]
    require(np.all(law[suffix] == (0, 0, 0, 0, 1)) and np.all(law[:, 0, 4] == 0), 'absorbed event law and reset')
    rates = predictions['reset_rates']
    require(((rates >= 0) & (rates <= 1)).all() and (rates[found[:, :-1]] == 0).all(), 'bounded causal gate and no absorbed updates')
    if arm in ('unchanged', 'global', 'static_bank'):
        require((rates == 0).all(), 'non-switching arm rates zero')
    close(predictions['post_costs'], state @ fields['costs'].T, 'linear learned immediate cost')
    forks, mass = fork_values(state, data['targets/fork_actions'], fields['transition'], fields['hazard'], fields['costs'])
    close(predictions['fork_costs'], forks[:, :, (3, 7)], 'independent learned blind costs')
    close(predictions['fork_survival'], mass[:, :, (3, 7)], 'independent learned blind survival')


def metrics(data, predictions, minimum_support=256):
    labels, lengths = data['public/observations'], data['public/lengths']
    n, length = labels.shape
    valid = np.arange(length)[None] < lengths[:, None]
    alive = valid & (labels != 4)
    late = alive & (np.arange(length)[None] >= 8)
    support = late.sum(1)
    selected = np.flatnonzero(support > 0)
    require(len(selected) >= minimum_support, 'late episode support')
    p = predictions['probabilities'][np.arange(n)[:, None], np.arange(length)[None], labels]
    require((p[valid] > 0).all(), 'strictly positive scored event probability')
    truth = data['targets/fork_costs'][:, :, (3, 7)]
    chosen = np.argmin(predictions['fork_costs'], axis=3)
    gap = np.take_along_axis(truth, chosen[..., None], axis=3)[..., 0] - np.min(truth, axis=3)
    post_truth = data['targets/post_costs']
    post_choice = np.argmin(predictions['post_costs'], axis=2)
    post_gap = np.take_along_axis(post_truth, post_choice[..., None], axis=2)[..., 0] - post_truth.min(2)
    episode = np.array([gap[i, late[i]].mean(0) for i in selected])
    post_episode = [float(post_gap[i, late[i]].mean()) for i in selected]
    result = {'primary_regret': float(episode.mean()), 'h4_regret': float(episode[:, 0].mean()),
              'h8_regret': float(episode[:, 1].mean()), 'post_regret': float(np.mean(post_episode)),
              'event_log_loss': float(np.mean(-np.log(p[valid]))), 'event_count': int(valid.sum()),
              'late_episode_support': len(selected), 'late_unsupported': n-len(selected),
              'late_boundaries': int(late.sum()), 'switch_delays': []}
    switches = data['audit/switch_indices']
    for delay in (1, 2, 4, 8):
        values = [gap[i, int(switches[i])+delay-1] for i in range(n)
                  if switches[i] >= 0 and switches[i]+delay-1 < length and alive[i, switches[i]+delay-1]]
        result['switch_delays'].append({'delay': delay, 'episodes': len(values),
                                       'regret': float(np.mean(values)) if values else None})
    return result


def continuation(rows):
    lookup = {(r['cohort'], r['arm'], r['split']): r for r in rows}
    require(len(rows) == len(lookup) == 120 and set(lookup) == {(c, a, s) for c in range(5) for a in ARMS for s in STRATA}, 'complete120-row rule population')
    means = {(a, s): float(np.mean([lookup[c, a, s]['primary_regret'] for c in range(5)])) for a in ARMS for s in STRATA}
    conditions, comparisons = {}, []
    for control in ('markov_bank', 'reset_bank'):
        for split in ('shift', 'switch'):
            candidate, baseline = means['recurrent_bank', split], means[control, split]
            wins = sum(lookup[c, 'recurrent_bank', split]['primary_regret'] < lookup[c, control, split]['primary_regret'] for c in range(5))
            conditions[f'{split}/{control}/mean_gain10pct'] = bool(baseline > 0 and candidate <= .9*baseline)
            conditions[f'{split}/{control}/paired_wins4of5'] = wins >= 4
            comparisons.append({'split': split, 'control': control, 'candidate_mean': candidate,
                                'control_mean': baseline, 'paired_wins': wins})
        conditions[f'base/{control}/noninferiority'] = bool(means['recurrent_bank', 'base'] <= 1.05*means[control, 'base']+1e-6)
    conditions['base/unchanged/noninferiority'] = bool(means['recurrent_bank', 'base'] <= 1.05*means['unchanged', 'base']+1e-6)
    candidate_time = float(np.mean([r['inference_seconds'] for r in rows if r['arm'] == 'recurrent_bank']))
    control_time = float(np.mean([r['inference_seconds'] for r in rows if r['arm'] == 'markov_bank']))
    candidate_nll = float(np.mean([lookup[c, 'recurrent_bank', 'base']['event_log_loss'] for c in range(5)]))
    control_nll = float(np.mean([lookup[c, 'markov_bank', 'base']['event_log_loss'] for c in range(5)]))
    conditions['all_strata/inference_time2x'] = candidate_time <= 2*control_time
    conditions['base/event_nll_plus001'] = candidate_nll <= control_nll+.01
    require(len(conditions) == 13, 'fixed13-condition rule')
    return {'passed': all(conditions.values()), 'status': 'PASS' if all(conditions.values()) else 'FAIL',
            'conditions': conditions, 'comparisons': comparisons,
            'means': [{'arm': a, 'split': s, 'primary_regret': means[a, s]} for a in ARMS for s in STRATA],
            'candidate_inference_seconds': candidate_time, 'markov_inference_seconds': control_time,
            'candidate_base_event_nll': candidate_nll, 'markov_base_event_nll': control_nll}


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


def audit(run_folder: Path, output_folder: Path):
    run_folder, output_folder = Path(run_folder), Path(output_folder)
    require(output_folder.is_dir() and not any(output_folder.iterdir()), 'exclusive empty audit output')
    summary = read(run_folder / 'summary.json')
    require(summary['version'] == 'reliability-memory-v1' and summary['config'] == CONFIG, 'fixed scientific configuration')
    fits, rows = summary['fits'], summary['rows']
    require(fits == journal(run_folder / 'fits.jsonl') and rows == journal(run_folder / 'results.jsonl'), 'complete summary journal joins')
    require(len(fits) == 20 and len(rows) == 120, '20 fits120 rows')
    barrier = read(run_folder / 'checkpoint-barrier.json')
    require(barrier['fits'] == fits, 'all fitted finals in barrier')
    expected_finals = {f'cohort-{c:02d}/{a}-final.npz' for c in range(5) for a in ARMS}
    require(barrier['checkpoints'].keys() == expected_finals, 'all30 final checkpoints before evaluation attestation')
    for name, pin in barrier['checkpoints'].items():
        require(descriptor(run_folder / name) == pin, 'barrier original final byte hash')
    counts = {'npz_decodes': 0, 'model_state_decodes': 0, 'optimizer_state_decodes': 0,
              'target_reconstructions': 0, 'prediction_checks': 0, 'update_records': 0,
              'model_calls': 0, 'optimizer_calls': 0, 'generator_calls': 0, 'rng_replays': 0}
    def load(path):
        counts['npz_decodes'] += 1
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    lookup = {(r['cohort'], r['arm'], r['split']): r for r in rows}
    require(len(lookup) == 120, 'unique result identities')
    audited, expected_files = [], {'fits.jsonl', 'results.jsonl', 'summary.json', 'checkpoint-barrier.json'}
    fit_index = 0
    for cohort in range(5):
        location = run_folder / f'cohort-{cohort:02d}'
        local_files = {'backbone.npz', 'backbone.json', 'train.npz', 'train.json', 'batches.npz'}
        fields = load(location / 'backbone.npz')
        require(fields.keys() == FIELDS.keys(), 'backbone keys')
        for name, shape in FIELDS.items():
            _array(fields[name], shape, 'float64', name)
        require((fields['transition'] >= 0).all() and (fields['emission'] >= 0).all()
                and ((fields['hazard'] >= 0) & (fields['hazard'] <= 1)).all(), 'backbone probabilities')
        close(fields['transition'].sum(1), np.ones((4, 8)), 'transition stochasticity')
        close(fields['emission'].sum(0), np.ones(8), 'emission stochasticity')
        backbone = read(location / 'backbone.json')
        require(backbone['exported_fields'] == descriptor(location / 'backbone.npz')
                and backbone['stored_parent_parameters'] == 352, 'backbone export descriptor')
        train = load(location / 'train.npz')
        validate_data(train, read(location / 'train.json'), CONFIG['namespace']+cohort, 'train')
        counts['target_reconstructions'] += 1
        batch_file = load(location / 'batches.npz')
        require(batch_file.keys() == {'indices'}, 'single shared batch bank')
        batches = batch_file['indices']
        _array(batches, (256, 64), 'int64', 'batch bank')
        for epoch in batches.reshape(32, 512):
            require(np.array_equal(np.sort(epoch), np.arange(512)), 'complete shuffled epoch without RNG replay')
        initials = {}
        order = TRAINED[cohort % 4:] + TRAINED[:cohort % 4]
        for arm in order:
            fit = fits[fit_index]; fit_index += 1
            require(fit['cohort'] == cohort and fit['arm'] == arm and fit['seed'] == CONFIG['seed']+cohort
                    and fit['updates'] == 256 and fit['parameters'] == PARAMETERS[arm]
                    and fit['state_scalars'] == STATES[arm] and 0 < fit['seconds'] < CONFIG['fit_cap_seconds'], 'complete ordered fit metadata')
            for key, name in (('batches', 'batches.npz'), ('initial', f'{arm}-initial.npz'), ('final', f'{arm}-final.npz')):
                require(fit[key] == descriptor(location / name), 'fit byte descriptor/' + key)
            state = load(location / f'{arm}-initial.npz')
            counts['model_state_decodes'] += 1
            verify_state(state, fields, arm, initial=True)
            initials[arm] = state
            records = journal(location / f'{arm}-updates.jsonl')
            require(len(records) == 256, 'all optimizer update records')
            for index, record in enumerate(records):
                require(record['update'] == index+1 and record['episode_exposures'] == 64
                        and record['event_exposures'] == int(train['public/lengths'][batches[index]].sum())
                        and math.isfinite(record['loss']) and record['loss'] >= -1e-12
                        and math.isfinite(record['gradient_norm_before_clip']) and record['gradient_norm_before_clip'] >= 0, 'exact paired update exposures')
            counts['update_records'] += len(records)
            optimizer = load(location / f'{arm}-adam.npz')
            counts['optimizer_state_decodes'] += 1
            groups = read(location / f'{arm}-adam.json')['param_groups']
            shapes = list(parameter_shapes(arm).values())
            require(len(groups) == 1 and groups[0]['params'] == list(range(len(shapes)))
                    and groups[0]['lr'] == .01 and groups[0]['betas'] == [.9, .999]
                    and groups[0]['eps'] == 1e-8 and groups[0]['weight_decay'] == 0
                    and not groups[0]['amsgrad'] and not groups[0]['maximize'], 'fixed Adam parameter group')
            require(optimizer.keys() == {f'{i}/{key}' for i in range(len(shapes)) for key in ('step', 'exp_avg', 'exp_avg_sq')}, 'all final Adam states')
            for i, shape in enumerate(shapes):
                require(optimizer[f'{i}/step'].shape == () and float(optimizer[f'{i}/step']) == 256, 'Adam completed step')
                for key in ('exp_avg', 'exp_avg_sq'):
                    _array(optimizer[f'{i}/{key}'], shape, 'float64', 'Adam/' + key)
                require((optimizer[f'{i}/exp_avg_sq'] >= 0).all(), 'Adam nonnegative second moment')
            local_files.update(f'{arm}-{suffix}' for suffix in ('initial.npz', 'final.npz', 'adam.npz', 'adam.json', 'updates.jsonl'))
        require(initials['recurrent_bank'].keys() == initials['reset_bank'].keys()
                and all(initials['recurrent_bank'][key].tobytes() == initials['reset_bank'][key].tobytes() for key in initials['recurrent_bank']), 'paired recurrent/reset initial bytes')
        for arm in ARMS:
            state = load(location / f'{arm}-final.npz')
            counts['model_state_decodes'] += 1
            verify_state(state, fields, arm)
            local_files.add(f'{arm}-final.npz')
        for split in STRATA:
            data = load(location / f'{split}.npz')
            validate_data(data, read(location / f'{split}.json'), CONFIG['namespace']+cohort, split)
            counts['target_reconstructions'] += 1
            local_files.update((f'{split}.npz', f'{split}.json'))
            for arm in ARMS:
                row = lookup[cohort, arm, split]
                name = f'prediction-{split}-{arm}.npz'
                require(row['parameters'] == PARAMETERS[arm] and row['state_scalars'] == STATES[arm]
                        and math.isfinite(row['inference_seconds']) and row['inference_seconds'] > 0
                        and row['prediction'] == descriptor(location / name)
                        and row['data'] == descriptor(location / f'{split}.npz')
                        and row['checkpoint'] == barrier['checkpoints'][f'cohort-{cohort:02d}/{arm}-final.npz'], 'prediction row provenance and scope')
                prediction = load(location / name)
                validate_predictions(data, prediction, fields, arm)
                numerical = metrics(data, prediction)
                compare({key: row[key] for key in numerical}, numerical, 'independent metrics')
                audited.append({**row, **numerical})
                counts['prediction_checks'] += 1
                local_files.add(name)
        expected_files.update(f'cohort-{cohort:02d}/{name}' for name in local_files)
    actual_files = {str(path.relative_to(run_folder)) for path in run_folder.rglob('*') if path.is_file()}
    require(actual_files == expected_files and len(expected_files) == 299, 'complete299-file producer roster')
    require(counts['npz_decodes'] == 225 and counts['model_state_decodes'] == 50
            and counts['optimizer_state_decodes'] == 20 and counts['update_records'] == 5120
            and counts['target_reconstructions'] == 25 and counts['prediction_checks'] == 120, 'exact independent decode/check counts')
    result = continuation(audited)
    output = {'version': VERSION, 'agreement': True, 'rows': audited, 'result': result,
              'conditions': result['conditions'], 'fits': fits, 'checkcounts': counts,
              'scope': 'Independent saved-output arithmetic and byte joins. No model/training/RNG replay; phase closure and historical all-fit barrier are authenticated by the caller. Private noise-path targets are privileged risk references. Saved inference time is producer measurement, not independently retimed.'}
    with (output_folder / 'audit.json').open('x') as stream:
        json.dump(output, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return {'agreement': True, 'status': result['status'], 'conditions_passed': sum(result['conditions'].values()),
            'conditions_total': 13, 'checkcounts': counts}
