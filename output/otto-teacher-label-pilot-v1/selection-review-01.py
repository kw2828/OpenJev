"""Independent stdlib-only fixed TRAIN metadata selection check; one invocation."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix('.json')
PLAN = ROOT / 'output/otto-teacher-label-pilot-v1/plan-01.json'
PLAN_SHA = '6956fd2658fc8920c277a6a4585ecb75d0b1639730ead5158af0cc7835269380'
PREP = ROOT / 'output/otto-teacher-label-pilot-v1/plan-preparation-01/receipt.json'
PREP_SHA = 'c33cf98ad73b6caa5cd06bbb475ecbaa2c043d85c78b0de86aebf055b75bbe6e'
META = 'output/otto-symmetry-head-v1/run-01/dagger-rows.jsonl'
META_SHA = '9935d48827e3a9b2e714383a2b3a87866bff0309205ac1316d75a0d5fdd5ce7b'
CELLS = [('lambda3', 1, 'shared@9101'), ('lambda3', 2, 'dense@9102'),
         ('lambda3', 3, 'shared@9103'), ('lambda4', 1, 'dense@9101'),
         ('lambda4', 2, 'shared@9102'), ('lambda4', 3, 'dense@9103')]
FIRST = {'lambda3': 950001, 'lambda4': 960001}
FIELDS = {'row_index', 'episode_id', 'stage', 'regime', 'seed', 'initial_hit',
          'arm', 'prefix_index', 'public', 'posterior', 'teacher_costs'}
CAP_SECONDS, OUTPUT_BYTES = 60, 256 * 1024**2
START = time.monotonic()
RECEIPT = {'version': 'otto-teacher-label-selection-review-v1', 'status': 'started',
           'agreement': False, 'limits': {'seconds': CAP_SECONDS, 'output_bytes': OUTPUT_BYTES},
           'sampler_calls': 0, 'native_calls': 0, 'model_calls': 0, 'posterior_replays': 0,
           'array_decodes': 0, 'validation_evaluation_record_decodes': 0,
           'scope': 'Opaque input/source hashes and independent TRAIN metadata selection only; no posterior replay or label generation.'}


def require(condition, name):
    if not condition:
        raise ValueError(name)


def check():
    require(time.monotonic() - START < CAP_SECONDS, '60-second independent review cap')


def timeout(_signum, _frame):
    raise TimeoutError('60-second independent review alarm')


def path(name):
    candidate = Path(name)
    require(not candidate.is_absolute() and '..' not in candidate.parts, 'relative closed path')
    candidate = ROOT / candidate
    require(not any(p.is_symlink() for p in (candidate, *candidate.parents)), 'no symlink evidence')
    require(candidate.is_file(), 'regular closed file')
    return candidate


def digest(file):
    h, size = hashlib.sha256(), 0
    with file.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            check()
            h.update(block)
            size += len(block)
    return {'sha256': h.hexdigest(), 'bytes': size}


class Number(str):
    pass


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def constant(_value):
    raise ValueError('nonfinite JSON constant')


def number(value):
    if isinstance(value, Number):
        return float(value) if any(c in value for c in '.eE') else int(value)
    if isinstance(value, list):
        return [number(x) for x in value]
    if isinstance(value, dict):
        return {k: number(v) for k, v in value.items()}
    return value


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def decode_train(line):
    lexical = json.loads(line, parse_int=Number, parse_float=Number,
                         parse_constant=constant, object_pairs_hook=pairs)
    require(isinstance(lexical, dict) and set(lexical) == FIELDS, 'exact original metadata schema')
    # No teacher score or other numerical label is decoded or used for ordering.
    row = {k: number(v) for k, v in lexical.items() if k != 'teacher_costs'}
    require(row['stage'] == 'dagger' and row['regime'] in FIRST, 'learner TRAIN only')
    first = FIRST[row['regime']]
    require(integer(row['seed'], first, first+11), 'fixed original episode seed')
    require(row['arm'] in {f'{family}@{seed}' for family in ('shared', 'dense')
                          for seed in (9101, 9102, 9103)}, 'original student collector')
    require(integer(row['initial_hit'], 1, 3) and row['initial_hit'] == 1+(row['seed']-first)%3,
            'original initial-hit assignment')
    require(row['episode_id'] == f"dagger:{row['regime']}:{row['seed']}:{row['arm']}", 'episode identity')
    require(integer(row['row_index'], 0, 4595) and integer(row['prefix_index'], 0, 2187), 'row/prefix range')
    public, witness = row['public'], row['posterior']
    require(isinstance(public, dict) and set(public) == {'position', 'hit', 'done', 'step', 'valid_actions'}, 'public schema')
    require(public['done'] is False and integer(public['hit'], 0, 3), 'nonterminal public hit')
    require(integer(public['step'], 0, 2187) and public['step'] == row['prefix_index'], 'pre-action step identity')
    q = public['position']
    require(isinstance(q, list) and len(q) == 2 and all(integer(x, 0, 52) for x in q), 'public board position')
    actions = public['valid_actions']
    require(isinstance(actions, list) and all(integer(x, 0, 3) for x in actions), 'public action types')
    require(actions == [a for a in range(4) if 0 <= q[a//2] + (-1 if a%2 == 0 else 1) < 53], 'public inbounds eligibility')
    require(isinstance(witness, dict) and set(witness) == {'sha256', 'mass'}, 'public posterior witness schema')
    require(type(witness['sha256']) is str and re.fullmatch('[0-9a-f]{64}', witness['sha256']) is not None, 'witness hash')
    require(type(witness['mass']) in (int, float) and math.isfinite(witness['mass']) and witness['mass'] >= 0, 'finite witness mass')
    return row


def run():
    require(digest(PLAN)['sha256'] == PLAN_SHA and digest(PREP)['sha256'] == PREP_SHA, 'external plan and preparation pins')
    plan, prepared = json.loads(PLAN.read_text()), json.loads(PREP.read_text())
    require(prepared['status'] == 'returned' and prepared['exit_code'] == 0
            and prepared['plan'] == digest(PLAN) and prepared['sources_unchanged'] is True, 'successful original plan preparation')
    require(plan['version'] == 'otto-teacher-label-pilot-v1' and plan['status'] == 'frozen_before_collection', 'frozen plan identity')
    config = plan['configuration']
    require(config['seed'] == 19000001 and config['anchor_ids'] == list(range(6))
            and config['replicate_ids'] == list(range(16)) and config['horizon'] == 2188
            and config['first_actions'] == 'all geometric inbounds, ascending'
            and config['familywise_alpha'] == .05 and config['no_retries'] is True
            and config['native_calls'] == config['learned_model_calls'] == config['training_calls'] == 0,
            'fixed prospective panel configuration')
    require(plan['limits'] == {'native_seconds': 1800, 'rss_bytes': 4*1024**3, 'output_bytes': 4*1024**3}, 'frozen resource allocation')
    require(plan['environment'] == dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1'), 'CPU1 declaration')
    source_count = input_count = opaque_bytes = 0
    for name, pin in plan['sources'].items():
        actual = digest(path(name))
        require(actual['sha256'] == pin, f'source changed: {name}')
        source_count += 1
        opaque_bytes += actual['bytes']
    for name, expected in plan['inputs'].items():
        actual = digest(path(name))
        require(actual == expected, f'input changed: {name}')
        input_count += 1
        opaque_bytes += actual['bytes']
    require(plan['sources']['src/openjev/research/otto_teacher_anchors.py'] ==
            '88d0793089111ac4aed0f3571a22486a0d6fbfd6b1354356632f93bdd756d39d', 'qualified extractor source')
    metadata = digest(path(META))
    require(metadata == plan['inputs'][META] and metadata['sha256'] == META_SHA, 'TRAIN metadata byte identity before parse')
    RECEIPT.update(plan={'path': str(PLAN), 'sha256': PLAN_SHA}, plan_preparation_sha256=PREP_SHA,
                   metadata={'path': META, **metadata}, source_files_verified=source_count,
                   input_files_verified=input_count, opaque_bytes_hashed=opaque_bytes)
    chosen, seen = {}, set()
    count = 0
    with path(META).open('rb') as stream:
        for line in stream:
            check()
            row = decode_train(line)
            require(row['row_index'] == count, 'complete canonical original metadata order')
            pair = row['episode_id'], row['prefix_index']
            require(pair not in seen, 'duplicate original episode-prefix')
            seen.add(pair)
            count += 1
            cell = row['regime'], row['initial_hit'], row['arm']
            if cell in CELLS and row['prefix_index'] >= 1:
                order = row['seed'], row['prefix_index'], row['row_index']
                if cell not in chosen or order < chosen[cell][0]:
                    chosen[cell] = order, copy.deepcopy(row)
    require(count == 4596 and plan['selection_rows'] == count and set(chosen) == set(CELLS), 'all original metadata and six cells')
    selected = [{'anchor_id': i, **chosen[cell][1]} for i, cell in enumerate(CELLS)]
    require(selected == plan['selected_anchors'], 'exact six independent selected records/public witnesses')
    require(digest(path(META)) == metadata and digest(PLAN)['sha256'] == PLAN_SHA, 'closing metadata/plan identity')
    RECEIPT.update(status='completed', agreement=True, metadata_rows=count, selected_anchors=selected,
                   selection_key=['seed', 'prefix_index', 'row_index'], teacher_score_values_decoded=0)


if __name__ == '__main__':
    require(not OUT.exists(), 'exclusive independent review output')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, CAP_SECONDS)
    exit_code = 0
    try:
        run()
    except BaseException as error:
        exit_code = 1
        RECEIPT.update(status='failed', agreement=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        RECEIPT.update(exit_code=exit_code, wall_seconds=time.monotonic()-START,
                       checker={'path': str(Path(__file__).resolve()),
                                'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        data = (json.dumps(RECEIPT, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
        require(len(data)+Path(__file__).stat().st_size <= OUTPUT_BYTES, 'review output bound')
        with OUT.open('xb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        print(json.dumps({'status': RECEIPT['status'], 'agreement': RECEIPT['agreement'],
                          'receipt': str(OUT), 'sha256': hashlib.sha256(data).hexdigest(),
                          'wall_seconds': RECEIPT['wall_seconds']}), flush=True)
    sys.exit(exit_code)
