"""Authentication and causal public labels for the separate calibration study.

Previously evaluated C trajectories are now explicitly calibration TRAINING
data. Neither their published outcome nor either original gate is revised.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import evaluate_card_controllers as prior
import numpy as np
import report_card_controllers as audit
from card_memory_common import runtime, sha, write_json

from openjev.research.card_memory_task import PublicTeacher

VERSION = 'card-calibration-v1'
POLICIES = ('baseline', 'temperature', 'hard')
MODES, REFERENCES = prior.MODES, prior.REFERENCES
CONTROLLERS = prior.CONTROLLERS
LEARNED = tuple(n for n in CONTROLLERS if n not in REFERENCES)
EVALUATION = {'episodes': 64, 'max_actions': 104, 'controllers': 20,
              'policies': list(POLICIES), 'wall_cap_seconds': 1800,
              'output_cap_bytes': 6_000_000_000}
CALIBRATION = {'fits': 18, 'episodes_per_fit': 64, 'beta_bounds': [.05, 20.0],
               'bisection_iterations': 64, 'max_oracle_evaluations': 1224,
               'wall_cap_seconds': 900, 'output_cap_bytes': 200_000_000,
               'new_model_calls': 0, 'new_native_calls': 0}
LINEAGE = {'controller_protocol', 'controller_inputs', 'controller_bindings',
           'controller_completed', 'controller_report_receipt', 'controller_summary', 'checkpoint_map'}
CORE_FILES = {'scripts/card_calibration_common.py', 'scripts/fit_card_calibration.py',
              'scripts/evaluate_card_calibration.py', 'src/openjev/research/card_probability_calibration.py'}
require, read, member = audit.require, audit.read, audit.member


def authenticate(root, protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
                 bindings_path, expected_bindings_sha256, checkpoint_map_path, expected_checkpoint_map_sha256):
    """Authenticate new recipe and inherited complete evidence without inference."""
    root = Path(root).resolve()
    paths = (protocol_path, inputs_path, bindings_path, checkpoint_map_path)
    digests = (expected_protocol_sha256, expected_inputs_sha256, expected_bindings_sha256,
               expected_checkpoint_map_sha256)
    for path, digest in zip(paths, digests, strict=True):
        require(Path(path).resolve().is_relative_to(root) and sha(path) == digest, 'External new input hash differs')
    recipe, inputs, bindings, checkpoints = map(read, paths)
    require(recipe['version'] == inputs['version'] == bindings['version'] == VERSION, 'Wrong study version')
    require(recipe['evaluation'] == EVALUATION and recipe['calibration'] == CALIBRATION, 'Frozen scope changed')
    require(bindings['runtime'] == runtime() and CORE_FILES <= set(bindings['files']), 'Runtime/core bindings differ')
    require(member(root, recipe['bindings_file']) == Path(bindings_path).resolve(), 'Binding path differs')
    for path, digest in zip(paths[:2], digests[:2], strict=True):
        require(bindings['files'].get(str(Path(path).resolve().relative_to(root))) == digest, 'Unbound new input')
    for relative, digest in bindings['files'].items():
        require(sha(member(root, relative)) == digest, 'New bound source changed: ' + relative)
    require(set(recipe['lineage']) == LINEAGE, 'Inherited lineage differs')
    lineage = {}
    for key, item in recipe['lineage'].items():
        require(set(item) == {'path', 'sha256'}, 'Malformed lineage descriptor')
        path = member(root, item['path'])
        require(sha(path) == item['sha256'], 'Inherited evidence changed: ' + key)
        lineage[key] = (path, read(path))
    require(lineage['checkpoint_map'][0] == Path(checkpoint_map_path).resolve()
            and recipe['lineage']['checkpoint_map']['sha256'] == expected_checkpoint_map_sha256,
            'Inherited checkpoint map differs')
    p = recipe['lineage']
    old_recipe, old_inputs, old_map, fits = prior.authenticate(
        lineage['controller_protocol'][0], p['controller_protocol']['sha256'],
        lineage['controller_inputs'][0], p['controller_inputs']['sha256'],
        checkpoint_map_path, expected_checkpoint_map_sha256,
        expected_bindings_sha256=p['controller_bindings']['sha256'], root=root)
    require(checkpoints == old_map and set(fits) == set(LEARNED), 'Inherited eighteen fits differ')
    complete = audit.authenticate_evaluation(lineage['controller_completed'][0].parent,
                                             p['controller_completed']['sha256'])
    require(complete['episodes'] == 3840 and complete['new_fits'] == complete['new_optimizer_steps'] == 0
            and complete['protocol_sha256'] == p['controller_protocol']['sha256']
            and complete['inputs_sha256'] == p['controller_inputs']['sha256'], 'Incomplete calibration source run')
    receipt, summary = lineage['controller_report_receipt'][1], lineage['controller_summary'][1]
    require(receipt['status'] == summary['status'] == 'complete' and receipt['continuation_passed'] is True
            and receipt['evaluation_completed_sha256'] == p['controller_completed']['sha256']
            and receipt['files']['summary.json']['sha256'] == p['controller_summary']['sha256'],
            'Previous controller report is incomplete or inconsistent')
    require(summary['original_architecture_gate']['passed'] is False, 'Original architecture outcome changed')
    require(recipe['calibration_source_role'] == 'previous_C_evaluation_repurposed_as_training', 'Data reuse must be explicit')
    cases = inputs['evaluation']
    require(len(cases) == 64, 'Exactly64 new test cases required')
    for i, case in enumerate(cases):
        require(set(case) == {'seed', 'policy_order'} and type(case['seed']) is int and 0 <= case['seed'] < 2**32
                and case['policy_order'] == list(POLICIES[i % 3:] + POLICIES[:i % 3]), 'Invalid fixed test order/seed')
    old_literal = read(member(root, old_recipe['lineage']['old_inputs']['path']))
    seeds = {case['seed'] for case in cases}
    require(len(seeds) == 64 and seeds.isdisjoint(prior._seed_inventory(old_literal)
            | prior._seed_inventory(old_inputs) | {0, 410}), 'Fresh test seeds overlap exposed streams')
    return recipe, inputs, checkpoints, fits


def public_episode(arrays):
    """Pre-action rank targets from earlier public frames, no deck reads."""
    raw, obs = arrays['raw_probabilities'], arrays['observations']
    require(raw.ndim == 3 and raw.shape[1:] == (52, 13) and obs.shape == (len(raw) + 1, 52),
            'Invalid public calibration arrays')
    teacher = PublicTeacher()
    teacher.reset(obs[0])
    targets, masks, ages = [], [], []
    for t in range(len(raw)):
        target, mask, age = teacher.targets(obs[t])
        targets.append(target); masks.append(mask); ages.append(age)
        teacher.observe(obs[t + 1])
    return {'raw_probabilities': raw.copy(), 'targets': np.asarray(targets, dtype=np.int64),
            'target_mask': np.asarray(masks, dtype=bool), 'ages': np.asarray(ages, dtype=np.int32)}


def calibration_members():
    names = {'started.json'}
    for name in LEARNED:
        names.update(f'fits/{name}/{part}' for part in ('fit.json', 'source-episodes.json', 'completed.json'))
    return names


def authenticate_calibration(calibration_dir, expected_completed_sha256, expected_protocol_sha256,
                             expected_inputs_sha256, expected_bindings_sha256, expected_checkpoint_map_sha256):
    folder = Path(calibration_dir).resolve()
    require(sha(folder / 'completed.json') == expected_completed_sha256, 'Calibration completion changed')
    done = read(folder / 'completed.json')
    require(done['status'] == 'complete' and done['version'] == VERSION
            and done['new_model_calls'] == done['new_native_calls'] == done['new_neural_weight_updates'] == 0,
            'Calibration status/work differs')
    for key, expected in [('protocol', expected_protocol_sha256), ('inputs', expected_inputs_sha256),
                          ('bindings', expected_bindings_sha256), ('checkpoint_map', expected_checkpoint_map_sha256)]:
        require(done[key + '_sha256'] == expected, 'Calibration lineage changed: ' + key)
    expected = calibration_members()
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    require(set(done['files']) == expected and actual == expected | {'completed.json'}, 'Calibration closure differs')
    for path, spec in done['files'].items():
        item = member(folder, path)
        require(item.stat().st_size == spec['bytes'] and sha(item) == spec['sha256'], 'Calibration member changed')
    require(set(done['fits']) == set(LEARNED), 'All eighteen calibration fits required')
    for name, entry in done['fits'].items():
        receipt = member(folder, entry['receipt_path'])
        require(entry['receipt_path'] == f'fits/{name}/completed.json' and sha(receipt) == entry['receipt_sha256'],
                'Calibration per-fit receipt differs')
        record = read(receipt)
        fitted = read(receipt.with_name('fit.json'))
        require(record['status'] == 'complete' and record['controller'] == name
                and record['fit_sha256'] == sha(receipt.with_name('fit.json'))
                and record['source_episodes_sha256'] == sha(receipt.with_name('source-episodes.json')),
                'Calibration fit not complete')
        require(type(entry['beta']) in (float, int) and type(entry['temperature']) in (float, int)
                and math.isfinite(entry['beta']) and math.isfinite(entry['temperature'])
                and entry['temperature'] > 0 and math.isclose(entry['beta'] * entry['temperature'], 1.0, rel_tol=1e-14, abs_tol=0.0),
                'Calibration inverse temperature is inconsistent')
        require(fitted['status'] == 'fitted' and entry['beta'] == fitted['beta']
                and entry['temperature'] == fitted['temperature'] and .05 <= entry['beta'] <= 20
                and record['checkpoint_sha256'] == entry['checkpoint_sha256'], 'Calibration parameter binding differs')
        require(type(fitted['oracle_evaluations']) is int and 3 <= fitted['oracle_evaluations'] <= 68
                and fitted['counts']['episodes'] == fitted['counts']['eligible_episodes'] == 64,
                'Calibration per-fit scope differs')
    require(sum(read(folder / f'fits/{n}/fit.json')['oracle_evaluations'] for n in LEARNED)
            == done['oracle_evaluations'] <= 1224, 'Calibration work cap differs')
    require(done['source_episodes'] == 1152 and done['fitted_parameters'] == 18, 'Calibration coverage differs')
    require(0 < done['wall_seconds'] <= CALIBRATION['wall_cap_seconds'], 'Calibration wall cap exceeded')
    payload_bytes = sum(v['bytes'] for v in done['files'].values())
    require(payload_bytes == done['payload_bytes']
            and payload_bytes + (folder / 'completed.json').stat().st_size <= CALIBRATION['output_cap_bytes'],
            'Calibration full storage exceeds cap')
    return done


def dump(path, value):
    """Exclusive finite JSON artifact writer shared by the new scripts."""
    write_json(Path(path), value)


def json_equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
