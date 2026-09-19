"""Saved-data checks only: no environments, models, generators or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from openjev.research.card_memory_task import PublicCardTracker, PublicTeacher


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def audit_part(directory, part, cases, ledger, receipt):
    packed_path = directory / f'{part}.npz'
    check(sha(packed_path) == receipt['parts'][part], 'Packed data hash changed')
    packed = arrays(packed_path)
    n = len(cases)
    expected = {'positions': np.zeros((n, 104), dtype=np.int64), 'ranks': np.zeros((n, 104), dtype=np.int64),
                'valid': np.zeros((n, 104), dtype=bool), 'targets': np.full((n, 104, 52), -1, dtype=np.int64),
                'target_mask': np.zeros((n, 104, 52), dtype=bool), 'ages': np.full((n, 104, 52), -1, dtype=np.int32)}
    counts = {'episodes': n, 'actions': 0, 'query_targets': 0, 'query_boundaries': 0}
    members = {str(packed_path.relative_to(directory)): sha(packed_path)}
    for index, case in enumerate(cases):
        meta = ledger[index]
        check(meta['part'] == part and meta['index'] == index and all(meta[k] == v for k, v in case.items()),
              'Declared episode/seed/behavior identity changed')
        relative = f'{part}/{index:03d}.npz'
        check(meta['npz_path'] == relative and meta['replay'] == 'passed', 'Raw episode identity/replay status mismatch')
        path = directory / relative
        check(sha(path) == meta['npz_sha256'], 'Raw archive hash changed')
        sidecar = path.with_suffix('.json')
        check(read(sidecar) == meta, 'Raw episode sidecar differs from authenticated ledger')
        members[relative], members[str(sidecar.relative_to(directory))] = sha(path), sha(sidecar)
        raw = arrays(path)
        types = {'positions': np.int64, 'ranks': np.int64, 'rewards': np.float64,
                 'targets': np.int64, 'target_mask': bool, 'ages': np.int32,
                 'terminated': bool, 'truncated': bool, 'observations': np.int64}
        check(set(raw) == set(types), 'Raw array membership differs')
        t = len(raw['positions'])
        check(type(meta['steps']) is int and meta['steps'] == t and 0 < t <= 104, 'Raw episode length differs')
        for key, dtype in types.items():
            shape = (t + 1, 52) if key == 'observations' else (t, 52) if key in ('targets', 'target_mask', 'ages') else (t,)
            check(raw[key].shape == shape and raw[key].dtype == np.dtype(dtype), 'Raw array shape/dtype differs: ' + key)
        check(np.isfinite(raw['rewards']).all() and meta['return'] == float(raw['rewards'].sum()), 'Raw return ledger mismatch')
        check(type(meta['elapsed_seconds']) in (int, float) and math.isfinite(meta['elapsed_seconds'])
              and meta['elapsed_seconds'] >= 0, 'Invalid episode wall time')
        tracker, teacher = PublicCardTracker(), PublicTeacher()
        tracker.reset(raw['observations'][0])
        teacher.reset(raw['observations'][0])
        known, last_visible = np.full(52, -1, dtype=np.int64), np.full(52, -1, dtype=np.int32)
        for step in range(t):
            before, after = raw['observations'][step:step + 2]
            targets, mask, ages = teacher.targets(before)
            independent_mask = (known >= 0) & (before == 13)
            independent_target = np.where(independent_mask, known, -1)
            independent_age = np.where(independent_mask, step - last_visible, -1)
            for key, value, independent in (('targets', targets, independent_target),
                                             ('target_mask', mask, independent_mask), ('ages', ages, independent_age)):
                check(np.array_equal(value, independent) and np.array_equal(value, raw[key][step]),
                      'Before-action public target/mask/age mismatch')
            action = int(raw['positions'][step])
            check(0 <= action < 52 and not tracker.matched[action] and action != tracker.pending,
                  'Action violated shared public eligibility')
            event = tracker.observe(action, raw['rewards'][step], after)
            check(event.pos == action and event.rank == int(raw['ranks'][step]), 'Selected reveal mismatch')
            check(bool(raw['terminated'][step]) == bool(tracker.matched.all())
                  and bool(raw['truncated'][step]) == (step == 103), 'Public terminal flag mismatch')
            check(bool(raw['terminated'][step] or raw['truncated'][step]) == (step == t - 1), 'Terminal/prefix mismatch')
            teacher.observe(after)
            visible = after < 13
            known[visible], last_visible[visible] = after[visible], step + 1
            counts['query_targets'] += int(mask.sum())
            counts['query_boundaries'] += int(mask.any())
        expected['valid'][index, :t] = True
        for key in expected.keys() - {'valid'}:
            expected[key][index, :t] = raw[key]
        counts['actions'] += t
    check(set(packed) == set(expected), 'Packed array membership differs')
    for key, value in expected.items():
        check(packed[key].dtype == value.dtype and np.array_equal(packed[key], value),
              'Packed/raw/padding mismatch: ' + key)
    return counts, members


def audit(root, out, data_sha256, preflight_sha256, source_commit):
    root, out = Path(root).resolve(), Path(out).resolve()
    begin = time.monotonic()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        result = {'status': 'started', 'scope': 'saved public-data integrity only',
                  'native_calls': 0, 'model_calls': 0, 'rng_calls': 0, 'automatic_retry': False}
        try:
            evidence = root / 'evidence/card-memory-pilot-v1'
            bindings_path = evidence / 'source-bindings.json'
            commit = subprocess.run(['git', 'rev-parse', '--verify', source_commit + '^{commit}'],
                                    cwd=root, check=True, capture_output=True, text=True).stdout.strip()
            frozen = subprocess.run(['git', 'show', f'{commit}:evidence/card-memory-pilot-v1/source-bindings.json'],
                                    cwd=root, check=True, capture_output=True).stdout
            check(frozen == bindings_path.read_bytes(), 'Source bindings differ from published commit')
            bindings, inputs = read(bindings_path), read(evidence / 'inputs.json')
            for relative, digest in bindings['files'].items():
                path = (root / relative).resolve()
                check(path.is_relative_to(root) and sha(path) == digest, 'Frozen source/input changed: ' + relative)
            result.update(source_commit=commit, bindings_sha256=sha(bindings_path),
                          source_count=len(bindings['files']), inputs_sha256=sha(evidence / 'inputs.json'),
                          protocol_sha256=sha(evidence / 'protocol.json'), auditor_source_sha256=sha(Path(__file__).resolve()))
            reports = {}
            for name, expected_sha, cases in (('preflight-01', preflight_sha256, inputs['preflight']),
                                               ('data-01', data_sha256, inputs['data'])):
                directory = root / 'output/card-memory-pilot-v1' / name
                check(not any((directory / marker).exists() for marker in ('failed.json', 'invalid-completion.json')),
                      'Failure-marked preparation cannot pass')
                check(sha(directory / 'completed.json') == expected_sha, 'External preparation completion hash differs')
                receipt, start = read(directory / 'completed.json'), read(directory / 'started.json')
                check(receipt['status'] == 'complete' and receipt['started_sha256'] == sha(directory / 'started.json'),
                      'Preparation provenance is incomplete')
                for key in ('bindings', 'protocol', 'inputs'):
                    check(start[key + '_sha256'] == result[key + '_sha256'], 'Preparation input binding differs')
                check(sha(directory / 'episodes.json') == receipt['episodes_sha256'], 'Episode ledger hash differs')
                ledger = read(directory / 'episodes.json')
                streamed = [json.loads(line) for line in (directory / 'ledger.jsonl').read_text().splitlines()]
                check(streamed == ledger and len(ledger) == sum(len(v) for v in cases.values()) == receipt['episodes'],
                      'Complete streamed ledger differs')
                check(set(receipt['parts']) == set(cases), 'Data partition membership differs')
                member_hashes = {name: sha(directory / name) for name in
                                 ('completed.json', 'started.json', 'episodes.json', 'ledger.jsonl')}
                parts, offset = {}, 0
                for part, expected_cases in cases.items():
                    counts, members = audit_part(directory, part, expected_cases,
                                                  ledger[offset:offset + len(expected_cases)], receipt)
                    parts[part] = counts
                    member_hashes.update(members)
                    offset += len(expected_cases)
                actual_members = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
                check(actual_members == set(member_hashes), 'Preparation exact file membership differs')
                actions = sum(part['actions'] for part in parts.values())
                expected_work = {f'{phase}_{kind}_{state}': actions if kind == 'steps' else len(ledger)
                                 for phase in ('native', 'replay') for kind in ('resets', 'steps')
                                 for state in ('attempted', 'returned')}
                check(receipt['work'] == expected_work, 'Original collection/replay counters differ')
                reports[name] = {'completed_sha256': expected_sha, 'parts': parts,
                                 'recorded_native_replay_counters': expected_work, 'member_sha256': member_hashes}
            for relative, digest in bindings['files'].items():
                check(sha(root / relative) == digest, 'Frozen source changed during saved audit')
            result.update(status='passed', preparations=reports, wall_seconds=time.monotonic() - begin,
                          limits='Checks saved public frames using frozen tracker/teacher and independent label masks/ages; '
                                 'does not rerun native dynamics or regenerate behavior random streams; '
                                 'behavior seeds are authenticated identities, not a second behavior-policy replay.')
        except BaseException as error:
            result.update(status='failed', error=repr(error), wall_seconds=time.monotonic() - begin)
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
            raise
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('data-sha256', 'preflight-sha256', 'source-commit'):
        parser.add_argument('--' + name, required=True)
    audit(**vars(parser.parse_args()))
