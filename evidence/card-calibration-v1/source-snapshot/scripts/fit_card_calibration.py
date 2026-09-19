"""Fit eighteen scalar temperatures on explicitly reused prior C histories.

No memory forward, memory write, optimizer, native environment, or new random
stream is invoked. The new test split is only authenticated, never evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import report_card_controllers as prior_report
from card_calibration_common import (
    CALIBRATION,
    LEARNED,
    VERSION,
    authenticate,
    dump,
    member,
    public_episode,
    read,
    require,
    sha,
)
from card_memory_common import preserve_failure

from openjev.research.card_probability_calibration import fit_temperature, score_episodes


def input_digest(episodes):
    digest = hashlib.sha256(b'card-calibration-public-episodes-v1\0')
    for index, episode in enumerate(episodes):
        for key in sorted(episode):
            value = np.ascontiguousarray(episode[key])
            digest.update(json.dumps([index, key, str(value.dtype), value.shape], separators=(',', ':')).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def collect(prior_evaluation, controller, source_inputs, *, deadline=None):
    """Recheck every inherited C choice and construct labels before each action."""
    episodes, identities = [], []
    for index, case in enumerate(source_inputs['evaluation']):
        require(deadline is None or time.monotonic() < deadline, 'Calibration replay wall cap exceeded')
        stem = Path(prior_evaluation) / 'controllers' / controller / 'C' / 'episodes' / f'{index:03d}'
        npz, receipt_path = stem.with_suffix('.npz'), stem.with_suffix('.json')
        receipt = read(receipt_path)
        require(receipt['index'] == index and receipt['seed'] == case['seed'], 'Calibration source case differs')
        prior_report.audit_episode(npz, receipt, policy='C', controller=controller)
        with np.load(npz, allow_pickle=False) as archive:
            episode = public_episode({k: archive[k] for k in ('raw_probabilities', 'observations')})
        episodes.append(episode)
        identities.append({'index': index, 'seed': case['seed'], 'npz_path': str(npz.relative_to(prior_evaluation)),
                           'npz_sha256': sha(npz), 'receipt_sha256': sha(receipt_path),
                           'layout_sha256': receipt['layout_sha256']})
    require(len(episodes) == 64, 'Every prior C episode required')
    return episodes, identities


def run(root, protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
        bindings_path, expected_bindings_sha256, checkpoint_map_path, expected_checkpoint_map_sha256, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    begin, phase, current = time.monotonic(), 'authentication', None
    deadline = begin + CALIBRATION['wall_cap_seconds']
    fits, oracles, scoring_calls = {}, 0, 0
    output_bytes, source_episodes, source_queries = 0, 0, 0
    active_fit = None
    args = (root, protocol_path, expected_protocol_sha256, inputs_path, expected_inputs_sha256,
            bindings_path, expected_bindings_sha256, checkpoint_map_path, expected_checkpoint_map_sha256)
    provenance = {'protocol_sha256': expected_protocol_sha256, 'inputs_sha256': expected_inputs_sha256,
                  'bindings_sha256': expected_bindings_sha256, 'checkpoint_map_sha256': expected_checkpoint_map_sha256}

    def guard():
        require(time.monotonic() < deadline, 'Calibration wall cap exceeded')
        require(output_bytes <= CALIBRATION['output_cap_bytes'], 'Calibration output cap exceeded')

    def write(path, value):
        nonlocal output_bytes
        dump(path, value)
        output_bytes += Path(path).stat().st_size
        guard()

    try:
        write(out / 'started.json', {'status': 'started', 'version': VERSION, **provenance,
              'scope': 'former C test episodes are calibration training for this new study', 'automatic_retry': False})
        recipe, _, checkpoints, _ = authenticate(*args)
        source_dir = member(root, recipe['lineage']['controller_completed']['path']).parent
        source_inputs = read(member(root, recipe['lineage']['controller_inputs']['path']))
        for current in LEARNED:
            guard()
            active_fit = None
            fit_begin = time.monotonic()
            phase = 'public_history_replay'
            episodes, source = collect(source_dir, current, source_inputs, deadline=deadline)
            source_episodes += len(episodes)
            data_sha256 = input_digest(episodes)
            target_dir = out / 'fits' / current
            target_dir.mkdir(parents=True)
            write(target_dir / 'source-episodes.json', {'scope': 'calibration_training_only', 'episodes': source,
                  'prior_completed_sha256': recipe['lineage']['controller_completed']['sha256'],
                  'public_dataset_sha256': data_sha256})
            phase = 'scalar_fit'
            guard()
            fit = fit_temperature(episodes)
            active_fit = fit
            oracles += fit['oracle_evaluations']
            source_queries += fit['counts']['query_cards']
            require(fit['counts']['episodes'] == fit['counts']['eligible_episodes'] == 64,
                    'Each of the64 calibration episodes must have public targets')
            require(oracles <= CALIBRATION['max_oracle_evaluations'], 'Scalar fitting work cap exceeded')
            guard()
            phase = 'calibration_scores'
            scores = {}
            for mode in ('baseline', 'temperature', 'hard'):
                scores[mode] = score_episodes(episodes, mode=mode, beta=fit['beta'] if mode == 'temperature' else 1.0)
                scoring_calls += 1
                guard()
            require(abs(scores['baseline']['all']['nll'] - fit['baseline_nll']) < 1e-12
                    and abs(scores['temperature']['all']['nll'] - fit['calibrated_nll']) < 1e-12,
                    'Hierarchical calibration score and fitting objective disagree')
            fit['calibration_scores'] = scores
            write(target_dir / 'fit.json', fit)
            receipt = {'status': 'complete', 'controller': current, 'checkpoint_sha256': checkpoints[current]['checkpoint_sha256'],
                       'fit_sha256': sha(target_dir / 'fit.json'),
                       'source_episodes_sha256': sha(target_dir / 'source-episodes.json'),
                       'public_dataset_sha256': data_sha256, 'oracle_evaluations': fit['oracle_evaluations'],
                       'scoring_calls': 3, 'wall_seconds': time.monotonic() - fit_begin,
                       'new_model_calls': 0, 'new_native_calls': 0}
            write(target_dir / 'completed.json', receipt)
            fits[current] = {'beta': fit['beta'], 'temperature': fit['temperature'],
                             'checkpoint_sha256': checkpoints[current]['checkpoint_sha256'],
                             'receipt_path': str((target_dir / 'completed.json').relative_to(out)),
                             'receipt_sha256': sha(target_dir / 'completed.json')}
            active_fit = None
            print(json.dumps({'controller': current, 'beta': fit['beta'], 'temperature': fit['temperature'],
                              'nll_before': fit['baseline_nll'], 'nll_after': fit['calibrated_nll']}), flush=True)
        phase = 'final_authentication'
        authenticate(*args)
        require(len(fits) == 18 and source_episodes == 1152 and scoring_calls == 54, 'Incomplete calibration coverage')
        guard()
        files = {str(p.relative_to(out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(out.rglob('*')) if p.is_file()}
        output_bytes = sum(item['bytes'] for item in files.values())
        done = {'status': 'complete', 'version': VERSION, **provenance, 'lineage': recipe['lineage'],
                'fits': fits, 'source_episodes': source_episodes, 'source_query_occurrences': source_queries,
                'fitted_parameters': 18, 'oracle_evaluations': oracles, 'scoring_calls': scoring_calls,
                'new_model_calls': 0, 'new_native_calls': 0, 'new_neural_weight_updates': 0,
                'wall_seconds': time.monotonic() - begin, 'files': files, 'payload_bytes': output_bytes}
        write(out / 'completed.json', done)
        guard()
        return done
    except BaseException as error:
        preserve_failure(error, [
            lambda: (out / 'completed.json').rename(out / 'invalid-completion.json')
            if (out / 'completed.json').exists() else None,
            lambda error=error: dump(out / 'failed.json', {'status': 'failed', 'phase': phase, 'current': current,
                'error': repr(error), 'completed_fits': fits, 'oracle_evaluations_returned': oracles,
                'active_returned_fit': active_fit, 'scoring_calls_returned': scoring_calls,
                'unfinished_oracle_work': 'unknown' if phase == 'scalar_fit' and active_fit is None else 'none',
                'unfinished_scoring_work': 'unknown' if phase == 'calibration_scores' else 'none',
                'work_scope': 'Counts cover returned helper calls only; an interrupted helper has unknown partial arithmetic',
                'source_episodes_returned': source_episodes, 'source_query_occurrences': source_queries,
                'wall_seconds': time.monotonic() - begin, 'automatic_retry': False})])
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'protocol-path', 'inputs-path', 'bindings-path', 'checkpoint-map-path', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('protocol', 'inputs', 'bindings', 'checkpoint-map'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({k: result[k] for k in ('status', 'fitted_parameters', 'oracle_evaluations', 'wall_seconds')}))
