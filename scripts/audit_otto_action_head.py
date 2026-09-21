"""Independent saved-output audit of the fixed OTTO action-head pilot.

No study/actor/simulator/Torch imports. NumPy replays only selected serialized
checkpoints on authenticated saved validation features. Posterior construction,
feature construction, unavailable checkpoints, RNG and timing truth are not
independently reproduced. No scientific gate uses a numerical tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import resource
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

VERSION = "otto-action-head-v1"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
FAMILIES = ("dct16_neutral", "dct16_nearest", "recent32_hard", "full_bayes")
FIT_SEEDS = (7901, 7902, 7903)
PLANNERS = ("full_bayes", "dct16_neutral", "dct16_nearest", "recent32_hard")
ARMS = tuple(f"{family}@{seed}" for seed in FIT_SEEDS for family in FAMILIES) + tuple(f"{f}@planner" for f in PLANNERS)
COHORTS = ("base", "shift")
FIRST_SEEDS = {"base": 690001, "shift": 700001}
N, NHITS, HORIZON = 53, 4, 2188
CENTER = (26, 26)
LIMITS = {"native_seconds": 5400, "rss_bytes": 8 * 1024**3, "output_bytes": 2 * 1024**3, "native_steps": 3485696}
THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
METRICS = ("steps", "found", "controller_seconds", "environment_seconds", "init_seconds", "update_seconds", "decision_seconds", "setup_allocation_seconds", "state_bytes")
CE_TOLERANCE = 2e-5
ARGMAX_MARGIN = 4e-5
SCOPE = ("Saved-only source/process/payload authentication; all 480 collection episodes, 12 selected checkpoints and 1536 autonomous episodes; independent mixture-weighted arm, stratum, block and family arithmetic plus all 40 frozen candidate conditions and full-head competence. No posterior/raw-feature reconstruction, simulator or actor calls, PRNG regeneration, timing measurement verification, optimization replay or replay of unavailable nonselected checkpoints. Near-tie numerical-library ambiguity is reported separately and cannot revise a scientific condition.")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"integer {name}")
    return value

def finite(value, name, minimum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"finite {name}")
    require(minimum is None or value >= minimum, f"minimum {name}")
    return value

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

def digest(value, name):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"SHA256 {name}")
    return value

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def safe_file(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts, "safe manifest path")
    path = root / relative
    require(path.is_file() and all(not component.is_symlink() for component in (path, *path.parents)
                                 if component != root.parent), f"regular non-symlink file {name}")
    require(path.resolve().is_relative_to(root.resolve()), f"contained file {name}")
    return path

class Comparison:
    def __init__(self):
        self.scalars = 0
        self.maximum_error = 0.0

    def close(self, actual, expected, name, tolerance=1e-9):
        finite(actual, name)
        finite(expected, name)
        difference = abs(actual - expected)
        self.scalars += 1
        self.maximum_error = max(self.maximum_error, difference)
        require(difference <= tolerance, f"arithmetic disagreement {name}: {actual!r} != {expected!r}")

    def same(self, actual, expected, name):
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), f"keys {name}")
            for key, value in expected.items():
                self.same(actual[key], value, f"{name}.{key}")
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"length {name}")
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                self.same(left, right, f"{name}[{index}]")
        elif type(expected) is float:
            self.close(actual, expected, name)
        else:
            self.scalars += 1
            require(type(actual) is type(expected) and actual == expected, f"exact {name}")

def moved(position, action):
    integer(action, "action")
    require(action < 4, "four action IDs")
    result = list(position)
    axis = action // 2
    result[axis] = max(0, min(N - 1, result[axis] + (-1 if action % 2 == 0 else 1)))
    return tuple(result)

def coordinate(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(x) is int and 0 <= x < N for x in value), "grid coordinate")
    return tuple(value)

def packet(value, step, position, initial_hit=None):
    require(isinstance(value, dict) and set(value) == {"position", "hit", "done", "step", "valid_actions"},
            "public observation whitelist")
    require(coordinate(value["position"]) == position and integer(value["step"], "public step") == step,
            "public time/position join")
    require(type(value["done"]) is bool and type(value["hit"]) is int, "typed public hit/done")
    hit, done = value["hit"], value["done"]
    require(hit == -2 if done else 0 <= hit < NHITS, "terminal sentinel or public odor")
    expected_actions = [] if done else [action for action in range(4) if moved(position, action) != position]
    require(value["valid_actions"] == expected_actions, "public valid-action IDs")
    if initial_hit is not None:
        require(step == 0 and not done and hit == initial_hit and position == CENTER, "initial conditioned observation")
    return hit, done

def selected_action(scores, position):
    require(isinstance(scores, list) and len(scores) == 4, "four recorded action scores")
    valid = [action for action in range(4) if moved(position, action) != position]
    require(all((scores[action] is None) == (action not in valid) for action in range(4)), "action score mask")
    for action in valid:
        finite(scores[action], "action score")
    best = min(scores[action] for action in valid)
    return next(action for action in valid if abs(scores[action] - best) < 1e-10)

def draw_record(row, channel, index, selected, comparison):
    """Validate a recorded categorical draw, without regenerating random numbers."""
    require(set(row) == {"channel", "index", "probabilities", "cdf_mass", "uniform", "selected_index"},
            "recorded draw fields")
    require(row["channel"] == channel and integer(row["index"], "draw index") == index
            and integer(row["selected_index"], "selected index") == selected, "draw identity")
    values = row["probabilities"]
    require(isinstance(values, list) and len(values) == (N * N if channel == "source" else NHITS), "draw support size")
    for probability in values:
        finite(probability, "categorical probability", 0)
    comparison.close(math.fsum(values), 1.0, "categorical mass", 1e-10)
    cumulative, selected_from_cdf = 0.0, None
    uniform = finite(row["uniform"], "recorded uniform", 0)
    require(uniform < 1, "recorded uniform less than one")
    mass = sum(values)
    comparison.close(row["cdf_mass"], mass, "recorded cumulative mass", 1e-10)
    for i, probability in enumerate(values):
        cumulative += probability
        if selected_from_cdf is None and cumulative / mass > uniform:
            selected_from_cdf = i
    require(selected_from_cdf == selected, "recorded inverse CDF choice")
    return uniform

def source_manifest(root, sources, budget):
    require(isinstance(sources, dict) and sources, "nonempty source manifest")
    for name, pin in sources.items():
        require(sha(safe_file(root, name)) == digest(pin, f"source {name}"), f"source bytes {name}")
        budget()


def configuration():
    return {'families': list(FAMILIES), 'fit_seeds': list(FIT_SEEDS), 'arms': list(ARMS),
            'training_cases': 384, 'validation_cases': 96, 'collection_horizon': 256,
            'training_first_seed': 670001, 'validation_first_seed': 680001, 'exploration_probability': .15,
            'evaluation_first_seeds': FIRST_SEEDS, 'cases_per_regime': 48, 'evaluation_horizon': HORIZON,
            'epochs': 40, 'batch_size': 256, 'learning_rate': .0003, 'teacher_temperature': .25,
            'checkpoint_epochs': list(range(5, 41, 5)),
            'checkpoint_rule': 'minimum finite validation cross entropy; first checkpoint wins exact ties',
            'training_weighting': 'equal decision rows across balanced initial-hit episode strata',
            'learned_architecture_claim': False}


def payload_names():
    names = {'started.json', 'imports.json', 'qualification.json', 'qualification.jsonl',
             'public-kernel-base.npz', 'public-kernel-shift.npz', 'fits.json',
             'evaluation-transitions.jsonl', 'evaluation-episodes.jsonl', 'summary.json'}
    for split in ('train', 'validation'):
        names.update({f'{split}-transitions.jsonl', f'{split}-episodes.json', f'{split}-targets.npz'})
        names.update(f'{split}-{family}.npz' for family in FAMILIES)
    for seed in FIT_SEEDS:
        for family in FAMILIES:
            names.update({f'fit-{family}-{seed}.json', f'head-{family}-{seed}.npz'})
    return names


def authenticate(args, budget):
    for path in (args.plan, args.run, args.terminal):
        require(path.is_absolute() and not path.is_symlink(), 'absolute direct external inputs')
    require(sha(args.plan) == args.plan_sha256 and sha(args.run / 'receipt.json') == args.receipt_sha256
            and sha(args.terminal) == args.terminal_sha256, 'external plan/worker/parent pins')
    plan, done, terminal = read(args.plan), read(args.run / 'receipt.json'), read(args.terminal)
    require(plan['version'] == VERSION and plan['status'] == 'frozen_before_run'
            and plan['configuration'] == configuration() and plan['limits'] == LIMITS, 'frozen scientific definition')
    require(done['status'] == 'completed' and done['phase'] == 'evaluation' and done['limits'] == LIMITS
            and done['completed_fits'] == 12 and done['completed_episodes'] == 1536
            and done['external_model_calls'] == 0 and done['plan_sha256'] == args.plan_sha256
            and done['sources'] == plan['sources'], 'complete worker identity')
    required = {'scripts/study_otto_action_head.py', 'src/openjev/research/otto_action_head.py',
                'tests/test_otto_action_head.py', 'tests/test_otto_action_head_study.py',
                'research/otto-action-head-protocol.md', 'scripts/study_otto_spectral_control.py',
                'scripts/study_otto_spectral_memory.py', 'scripts/audit_otto_large_memory.py',
                'src/openjev/research/otto_spectral_memory.py', 'src/openjev/research/otto_public.py',
                'src/openjev/research/suspend_clock.py', 'scripts/supervise_dialogue_observation_v2.py'}
    require(required <= plan['sources'].keys() and plan['sources']['src/openjev/research/suspend_clock.py'] == CLOCK_PIN,
            'prospective source closure')
    source_manifest(ROOT, plan['sources'], budget)
    require(plan['upstream_commit'] == 'a6aaef6507cffd2aff79291c1019f506f616bbef', 'upstream revision identity')
    require({'LICENSE', 'isotropic/classes/sourcetracking.py', 'isotropic/classes/heuristicpolicy.py',
             'isotropic/classes/policy.py'} <= plan['upstream_sources'].keys(), 'upstream source membership')
    source_manifest(ROOT / 'tmp/otto-source-review-01', plan['upstream_sources'], budget)
    names = payload_names()
    require(args.run.is_dir() and set(done['files']) == names
            and {p.name for p in args.run.iterdir()} == names | {'receipt.json'}, 'exact 48-payload completion')
    total_bytes = 0
    for name in sorted(names):
        path, record = safe_file(args.run, name), done['files'][name]
        require(path.stat().st_size == integer(record['bytes'], 'payload bytes')
                and sha(path) == digest(record['sha256'], 'payload pin'), f'payload authentication {name}')
        total_bytes += path.stat().st_size
        budget()
    require(total_bytes + (args.run / 'receipt.json').stat().st_size <= LIMITS['output_bytes'], 'published output cap')
    imports = read(args.run / 'imports.json')
    require(imports['versions'] == plan['runtime_versions'] and imports['python'].split()[0] == plan['python_version']
            and imports['executable'] == plan['python_executable'] and imports['threads'] == THREAD_ENV
            and imports['torch_threads'] == 1, 'source-bound single-thread qualified runtime')
    launch_path = args.terminal.with_name(args.terminal.name.replace('.terminal.json', '.launch.json'))
    require(launch_path != args.terminal and not launch_path.is_symlink()
            and sha(launch_path) == done['supervision_sha256'], 'launch external terminal/worker binding')
    launch = read(launch_path)
    for key in ('version', 'command', 'cwd', 'clock_backend', 'started_ns', 'deadline_ns', 'cap_seconds',
                'pid', 'pgid', 'parent_pid', 'watchdog_sha256', 'clock_source_sha256'):
        require(launch[key] == terminal[key], f'supervisor launch/terminal {key}')
    command = list(launch['command'])
    if command[1:2] == ['-u']:
        command.pop(1)
    require(command == [imports['executable'], str(ROOT / 'scripts/study_otto_action_head.py'),
                        '--plan', str(args.plan), '--plan-sha256', args.plan_sha256,
                        '--supervision', str(launch_path), '--output', str(args.run)], 'exact actual worker command')
    require(launch['version'] == 'dialogue-observation-supervision-v2' and Path(launch['cwd']).resolve() == ROOT
            and launch['clock_backend'] in {'mach_continuous_time', 'CLOCK_BOOTTIME'}
            and launch['clock_source_sha256'] == CLOCK_PIN
            and launch['watchdog_sha256'] == plan['sources']['scripts/supervise_dialogue_observation_v2.py'],
            'native supervisor source and clock')
    require(integer(launch['pid'], 'worker PID', 1) == launch['pgid']
            and launch['pid'] != integer(launch['parent_pid'], 'parent PID', 1), 'independent process group')
    start, finish = integer(launch['started_ns'], 'parent start'), integer(terminal['finished_ns'], 'parent finish')
    require(launch['cap_seconds'] == 5400 and launch['deadline_ns'] == start + 5400 * 10**9
            and start <= finish < launch['deadline_ns'] and terminal['elapsed_ns'] == finish - start
            and terminal['wall_seconds'] == (finish - start) / 1e9, 'successful parent within native cap')
    require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timing_available'] is True
            and terminal['timed_out'] is False and terminal['group_absent'] is True and terminal['error'] is None
            and terminal['clock_error'] is None and terminal['cleanup']['reaped'] is True
            and terminal['cleanup']['group_absent'] is True and terminal['cleanup']['errors'] == [], 'actual successful parent terminal')
    started = read(args.run / 'started.json')
    require(started['request'] == {'plan': str(args.plan), 'plan_sha256': args.plan_sha256,
                                  'supervision': str(launch_path), 'output': str(args.run)}
            and started['launch'] == launch, 'worker initial identity')
    require(done['clock_backend'] == launch['clock_backend'] and start <= started['started_ns'] == done['started_ns']
            <= done['finished_ns'] <= finish and done['wall_seconds'] == (done['finished_ns'] - done['started_ns']) / 1e9,
            'nested worker timestamps')
    require(integer(done['peak_rss_bytes'], 'worker RSS') <= LIMITS['rss_bytes']
            and integer(done['native_steps_attempted'], 'attempted calls') == done['native_steps_returned']
            <= LIMITS['native_steps'], 'completed bounded native work')
    return plan, done, terminal


def groups(path, keys):
    with path.open() as stream:
        rows = (json.loads(line) for line in stream)
        for key, values in itertools.groupby(rows, key=lambda row: tuple(row[k] for k in keys)):
            yield key, list(values)


def qualify(run, done, comparison, budget):
    qualified = read(run / 'qualification.json')
    suffixes = ('explicit_geometry', 'kernel_formula', 'initial_hit_mixture', 'replay_hit1', 'global_rng_hit1',
                'replay_hit2', 'global_rng_hit2', 'replay_hit3', 'global_rng_hit3', 'saturated_boundary_found')
    names = [f'{cohort}.{suffix}' for cohort in COHORTS for suffix in suffixes]
    require(qualified['status'] == 'completed' and [c['name'] for c in qualified['checks']] == names
            and all(c['passes'] is True for c in qualified['checks']), 'twenty passing structural witnesses')
    with (run / 'qualification.jsonl').open() as stream:
        require([json.loads(line) for line in stream] == qualified['checks'], 'qualification journal closure')
    checks, weights, replay_steps = {c['name']: c for c in qualified['checks']}, {}, 0
    for regime in COHORTS:
        weights[regime] = {int(h): finite(v, 'initial mixture', 0) for h, v in qualified['initial_hit_weights'][regime].items()}
        require(set(weights[regime]) == {1, 2, 3} and all(0 < w < 1 for w in weights[regime].values()), 'three positive strata')
        comparison.close(math.fsum(weights[regime].values()), 1., 'initial mass', 1e-12)
        comparison.same(checks[f'{regime}.initial_hit_mixture']['probabilities'],
                        [0., *(weights[regime][h] for h in (1, 2, 3))], 'qualified mixture witness')
        for suffix in ('kernel_formula', 'initial_hit_mixture'):
            require(0 <= finite(checks[f'{regime}.{suffix}']['maximum_absolute_difference'], 'qualified formula error') <= 1e-12,
                    'inherited formula qualification')
        finite(qualified['shared_model_initialization_seconds'][regime], 'shared model setup', 0)
        finite(qualified['template_initialization_seconds'][regime], 'environment template setup', 0)
        for hit in (1, 2, 3):
            witness = checks[f'{regime}.replay_hit{hit}']
            steps = integer(witness['steps'], 'qualification steps', 1)
            require(steps <= 84 and len(witness['replay']) == steps and 0 <= witness['maximum_tv'] <= 1e-10
                    and 0 <= witness['maximum_score_error'] <= 1e-8, 'bounded inherited posterior/score witness')
            position, source = CENTER, coordinate(witness['source_evaluation_only'])
            draws = iter(witness['draw_log'])
            draw_record(next(draws), 'source', 0, source[0] * N + source[1], comparison)
            odor_index = 0
            for step, event in enumerate(witness['replay'], 1):
                require(event['step'] == step, 'qualification chronology')
                position = moved(position, event['action'])
                hit_value, found = packet(event['public'], step, position)
                require(found is (position == source) and (not found or step == steps), 'qualification found/source')
                if not found:
                    draw_record(next(draws), 'hit', odor_index, hit_value, comparison)
                    odor_index += 1
            require(position == source and next(draws, None) is None, 'qualified complete discovery')
            replay_steps += steps
            budget()
        require(checks[f'{regime}.saturated_boundary_found']['native_steps'] == 110, 'fixed structural fixture work')
    require(qualified['native_steps'] == done['qualification_calls'] == 3 * replay_steps + 220 <= 2048,
            'exact qualification primitive accounting')
    return qualified, weights


def load_arrays(path, names, np):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(names), f'exact NPZ keys {path.name}')
        return {name: data[name] for name in names}


def datasets(run, done, comparison, budget, np):
    result = {}
    for split, first, count in (('train', 670001, 384), ('validation', 680001, 96)):
        arrays = load_arrays(run / f'{split}-targets.npz', ('target', 'valid', 'metadata'), np)
        target, valid, metadata = (arrays[k] for k in ('target', 'valid', 'metadata'))
        rows = len(target)
        require(target.dtype == np.float32 and target.shape == (rows, 4) and np.isfinite(target).all()
                and (target >= 0).all() and valid.dtype == np.bool_ and valid.shape == target.shape
                and metadata.dtype == np.int64 and metadata.shape == (rows, 3), 'target/mask/metadata dimensions')
        require(rows == done['collection_rows'][split] and rows > 0 and not target[~valid].any()
                and np.all(np.abs(target.astype(np.float64).sum(1) - 1) <= 1e-7), 'normalized legal teacher targets')
        episodes = read(run / f'{split}-episodes.json')
        require(len(episodes) == count, 'complete collection episode count')
        trace_groups = iter(groups(run / f'{split}-transitions.jsonl', ('seed',)))
        index = 0
        for case, episode in enumerate(episodes):
            budget()
            seed, initial_hit = first + case, 1 + case % 3
            require(episode['seed'] == seed and episode['initial_hit'] == initial_hit, 'fixed collection identities')
            steps = integer(episode['steps'], 'collection steps', 1)
            require(type(episode['found']) is bool and steps <= 256 and (episode['found'] or steps == 256), 'collection horizon')
            key, trace = next(trace_groups)
            require(key == (seed,) and len(trace) == steps + 1 and trace[0]['kind'] == 'reset'
                    and trace[0]['initial_hit'] == initial_hit, 'collection reset/steps join')
            source, position = coordinate(trace[0]['source_evaluation_only']), CENTER
            require(source != CENTER, 'conditioned source excludes center')
            packet(trace[0]['public'], 0, position, initial_hit)
            for step, event in enumerate(trace[1:], 1):
                require(event['kind'] == 'step' and event['step'] == step and index < rows, 'complete collection prefix order')
                require(tuple(metadata[index]) == (seed, initial_hit, step - 1), 'row metadata equals public pre-action prefix')
                scores = event['teacher_scores']
                require(event['teacher_action'] == selected_action(scores, position), 'teacher first-tie action')
                mask = np.array([score is not None for score in scores], dtype=bool)
                require(np.array_equal(valid[index], mask), 'teacher legal support')
                costs = np.array([score for score in scores if score is not None], dtype=np.float64)
                logits = -(costs - costs.min()) / max(float(costs.max() - costs.min()), 1e-8) / .25
                expected = np.zeros(4, dtype=np.float32)
                p = np.exp(logits)
                expected[mask] = p / p.sum()
                require(np.array_equal(target[index], expected), 'independent teacher-target construction')
                require(type(event['exploration_branch']) is bool and event['action'] in np.flatnonzero(mask)
                        and (event['exploration_branch'] or event['action'] == event['teacher_action']), 'teacher/exploration action contract')
                position = moved(position, event['action'])
                _, found = packet(event['public'], step, position)
                require(found is (position == source) and (not found or step == steps), 'collection terminal/source semantics')
                index += 1
            require(episode['found'] is found, 'collection terminal row join')
        require(next(trace_groups, None) is None and index == rows, 'no missing/extra dataset rows')
        result[split] = arrays
    train_ids = set(result['train']['metadata'][:, 0].tolist())
    valid_ids = set(result['validation']['metadata'][:, 0].tolist())
    evaluation_ids = {first + i for first in FIRST_SEEDS.values() for i in range(48)}
    require(not (train_ids & valid_ids or train_ids & evaluation_ids or valid_ids & evaluation_ids), 'disjoint collection/validation/evaluation seeds')
    require(done['training_updates'] == 12 * 40 * math.ceil(len(result['train']['target']) / 256), 'every fixed epoch/minibatch counted')
    return result


def forward(features, head, np):
    """Independent float32 MLP arithmetic, without production model helpers."""
    value = (features - head['mean']) / head['scale']
    value = np.tanh(value @ head['weight0'].T + head['bias0'])
    value = np.tanh(value @ head['weight1'].T + head['bias1'])
    return value @ head['weight2'].T + head['bias2']


def selected_validation(features, target, valid, head, np, budget):
    losses, matches, unambiguous_matches, ambiguous = [], 0, 0, 0
    for offset in range(0, len(features), 512):
        budget()
        scores = forward(features[offset:offset + 512], head, np)
        require(scores.dtype == np.float32 and np.isfinite(scores).all(), 'finite independent validation scores')
        mask = valid[offset:offset + 512]
        logits = np.where(mask, -scores.astype(np.float64), -1e9)
        maximum = logits.max(axis=1, keepdims=True)
        logprob = logits - maximum - np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
        labels = target[offset:offset + 512]
        losses.extend((-(labels.astype(np.float64) * logprob).sum(axis=1)).tolist())
        correct = logits.argmax(axis=1) == labels.argmax(axis=1)
        sorted_logits = np.sort(logits, axis=1)
        near = sorted_logits[:, -1] - sorted_logits[:, -2] <= ARGMAX_MARGIN
        matches += int(correct.sum())
        unambiguous_matches += int((correct & ~near).sum())
        ambiguous += int(near.sum())
    return {'validation_ce': math.fsum(losses) / len(losses), 'matches': matches,
            'validation_argmax_agreement': matches / len(features), 'rows': len(features),
            'near_tie_rows': ambiguous, 'certain_matches': unambiguous_matches,
            'possible_match_count_interval': [unambiguous_matches, unambiguous_matches + ambiguous]}


def checkpoint_review(run, data, comparison, budget, np):
    records = read(run / 'fits.json')
    expected_ids = [f'{family}@{seed}' for seed in FIT_SEEDS for family in FAMILIES]
    require(len(records) == 12 and [r['arm'] for r in records] == expected_ids, 'all twelve fixed fits in order')
    by_arm, verified = {r['arm']: r for r in records}, {}
    fields = {'version', 'input_dim', 'mean', 'scale', 'weight0', 'bias0', 'weight1', 'bias1', 'weight2', 'bias2'}
    for family in FAMILIES:
        dimension = 5633 if family == 'full_bayes' else 3080
        train = load_arrays(run / f'train-{family}.npz', ('features',), np)['features']
        validation = load_arrays(run / f'validation-{family}.npz', ('features',), np)['features']
        for split, features in (('train', train), ('validation', validation)):
            require(features.shape == (len(data[split]['target']), dimension) and features.dtype == np.float32
                    and np.isfinite(features).all(), 'aligned finite family features')
        # Independent moments use only TRAIN; no validation fit or held-out selection.
        mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(np.std(train, axis=0, dtype=np.float64), 1).astype(np.float32)
        del train
        budget()
        for seed in FIT_SEEDS:
            arm = f'{family}@{seed}'
            record = by_arm[arm]
            comparison.same(read(run / f'fit-{family}-{seed}.json'), record, 'individual/combined fit receipt')
            require(record['family'] == family and record['seed'] == seed and record['input_dim'] == dimension
                    and record['parameters'] == 64 * dimension + 2276
                    and record['training_rows'] == len(data['train']['target'])
                    and record['validation_rows'] == len(validation), 'fit geometry/data exposure')
            curve = record['curve']
            require(len(curve) == 8 and [c['epoch'] for c in curve] == list(range(5, 41, 5)), 'all eight validation checkpoints')
            for item in curve:
                finite(item['training_ce'], 'recorded training CE', 0)
                finite(item['validation_ce'], 'recorded validation CE', 0)
                require(0 <= finite(item['validation_argmax_agreement'], 'recorded validation agreement') <= 1,
                        'bounded recorded validation agreement')
            best = min(curve, key=lambda row: row['validation_ce'])
            require(record['selected_epoch'] == best['epoch'] and record['validation_ce'] == best['validation_ce'],
                    'exact first minimum of all eight recorded checkpoint losses')
            filename = f'head-{family}-{seed}.npz'
            require(record['checkpoint'] == filename and record['checkpoint_sha256'] == sha(run / filename), 'selected checkpoint bytes')
            head = load_arrays(run / filename, fields, np)
            require(head['version'].shape == head['input_dim'].shape == () and head['version'].item() == VERSION
                    and head['input_dim'].item() == dimension, 'checkpoint scalar identity')
            shapes = {'mean': (dimension,), 'scale': (dimension,), 'weight0': (64, dimension), 'bias0': (64,),
                      'weight1': (32, 64), 'bias1': (32,), 'weight2': (4, 32), 'bias2': (4,)}
            for name, shape in shapes.items():
                require(head[name].shape == shape and head[name].dtype == np.float32 and np.isfinite(head[name]).all(),
                        f'finite checkpoint array {name}')
            require(np.array_equal(head['mean'], mean) and np.array_equal(head['scale'], scale), 'independent TRAIN-only population moments')
            replay = selected_validation(validation, data['validation']['target'], data['validation']['valid'], head, np, budget)
            comparison.close(replay['validation_ce'], record['validation_ce'], 'independent selected-checkpoint CE', CE_TOLERANCE)
            recorded_matches = round(best['validation_argmax_agreement'] * len(validation))
            comparison.close(best['validation_argmax_agreement'], recorded_matches / len(validation), 'integral recorded match count', 1e-12)
            lower, upper = replay['possible_match_count_interval']
            require(lower <= recorded_matches <= upper, 'selected-checkpoint argmax discrepancy confined to predeclared numerical near ties')
            replay.update(arm=arm, selected_epoch=best['epoch'], recorded_validation_ce=record['validation_ce'],
                          ce_absolute_error=abs(replay['validation_ce'] - record['validation_ce']),
                          recorded_matches=recorded_matches, match_count_exact=replay['matches'] == recorded_matches,
                          matched_first_minimum_recorded_epoch=True,
                          unavailable_checkpoint_curves_independently_replayed=False)
            parameter_bytes = sum(head[k].nbytes for k in shapes if k not in ('mean', 'scale'))
            standardizer_bytes = head['mean'].nbytes + head['scale'].nbytes
            storage = record['storage']
            require(storage['parameter_count'] == 64 * dimension + 2276 and storage['parameter_bytes'] == parameter_bytes
                    and storage['standardizer_bytes'] == standardizer_bytes
                    and storage['total_array_bytes'] == parameter_bytes + standardizer_bytes, 'head storage accounting')
            finite(record['fit_seconds'], 'fit wall time', 0)
            require(0 <= finite(record['checkpoint_load_validation_seconds'], 'head load cost') <= record['fit_seconds'], 'head setup enclosed in fit')
            verified[arm] = replay
        del validation
    return records, [verified[arm] for arm in expected_ids]


def compact_draw(row, channel, index, selected, comparison):
    require(set(row) == {'channel', 'index', 'uniform', 'selected_index', 'cdf_mass'}, 'compact categorical draw schema')
    require(row['channel'] == channel and integer(row['index'], 'draw index') == index
            and integer(row['selected_index'], 'selected category') == selected, 'recorded categorical identity')
    uniform = finite(row['uniform'], 'recorded uniform', 0)
    require(uniform < 1, 'uniform in half-open unit interval')
    comparison.close(row['cdf_mass'], 1., 'recorded categorical normalization', 1e-10)
    return uniform


def evaluation_order():
    for regime_index, regime in enumerate(COHORTS):
        for case in range(48):
            rotation = (regime_index * 48 + case) % 16
            for arm in ARMS[rotation:] + ARMS[:rotation]:
                yield regime, FIRST_SEEDS[regime] + case, case // 6, 1 + (case % 6) // 2, arm


def check_evaluation(run, records, qualified, comparison, budget):
    fits = {r['arm']: r for r in records}
    rows, shared_cases = [], {}
    traces = iter(groups(run / 'evaluation-transitions.jsonl', ('regime', 'seed', 'arm')))
    with (run / 'evaluation-episodes.jsonl').open() as stream:
        for identity in evaluation_order():
            budget()
            line = next(stream, None)
            require(line is not None, 'every expected evaluation episode present')
            row = json.loads(line)
            regime, seed, block, hit, arm = identity
            require(tuple(row[k] for k in ('regime', 'seed', 'block', 'initial_hit', 'arm')) == identity,
                    'exact balanced rotated evaluation order')
            steps = integer(row['steps'], 'evaluation steps', 1)
            require(type(row['found']) is bool and steps <= HORIZON and (row['found'] or steps == HORIZON)
                    and integer(row['updates'], 'actor updates') == steps - int(row['found']), 'complete censored evaluation')
            key, trace = next(traces)
            require(key == (regime, seed, arm) and len(trace) == steps + 1 and trace[0]['kind'] == 'reset', 'evaluation trace closure')
            reset, source, position = trace[0], coordinate(row['source_evaluation_only']), CENTER
            require(source != CENTER and reset['source_evaluation_only'] == row['source_evaluation_only']
                    and reset['block'] == block and reset['initial_hit'] == hit, 'evaluation reset/source metadata')
            packet(reset['public'], 0, position, hit)
            draws = iter(row['draws'])
            source_uniform = compact_draw(next(draws), 'source', 0, source[0] * N + source[1], comparison)
            odor_uniforms = []
            costs = {name: [] for name in ('decision_seconds', 'update_seconds', 'environment_seconds')}
            for step, event in enumerate(trace[1:], 1):
                if step % 128 == 0:
                    budget()
                require(event['kind'] == 'step' and integer(event['step'], 'evaluation primitive step', 1) == step,
                        'contiguous autonomous actions')
                require(event['action'] == selected_action(event['scores'], position), 'recorded first-tie autonomous choice')
                after = moved(position, event['action'])
                require(after != position, 'legal nonblocked action')
                reading, found = packet(event['public'], step, after)
                require(found is (after == source) and (not found or step == steps), 'source discovery and no post-terminal action')
                if not found:
                    odor_uniforms.append(compact_draw(next(draws), 'hit', len(odor_uniforms), reading, comparison))
                else:
                    require(event['update_seconds'] == 0, 'terminal sentinel is never assimilated')
                for name, values in costs.items():
                    values.append(finite(event[name], name, 0))
                position = after
            require(row['found'] is found and next(draws, None) is None, 'terminal episode and complete draw stream')
            for name in ('decision_seconds', 'update_seconds'):
                comparison.close(row[name], math.fsum(costs[name]), f'primitive/episode {name}')
            env_initial = finite(row['environment_initialization_seconds'], 'environment initialization', 0)
            comparison.close(row['environment_seconds'], env_initial + math.fsum(costs['environment_seconds']), 'complete environment cost')
            finite(row['init_seconds'], 'actor initialization', 0)
            family, mode = arm.split('@')
            allocation = qualified['shared_model_initialization_seconds'][regime] / 48 if family.startswith('dct') else 0.
            if mode != 'planner':
                allocation += fits[arm]['checkpoint_load_validation_seconds'] / 48
            comparison.close(row['setup_allocation_seconds'], allocation, 'logical model/head setup charge')
            comparison.close(row['controller_seconds'], math.fsum(row[k] for k in
                             ('init_seconds', 'update_seconds', 'decision_seconds', 'setup_allocation_seconds')), 'complete public controller cost')
            state_bytes = {'full_bayes': 25281, 'recent32_hard': 3577, 'dct16_neutral': 4857, 'dct16_nearest': 4857}[family]
            require(integer(row['state_bytes'], 'evolving state bytes', 1) == state_bytes, 'fixed evolving-state geometry')
            storage = row['storage']
            if 'state_array_bytes' in storage:
                require(storage['state_array_bytes'] == storage['evidence_array_bytes'] + storage['support_mask_bytes'] == state_bytes,
                        'DCT coefficients plus support-mask storage')
            else:
                require(storage['mutable_array_bytes'] == sum(storage['mutable_arrays'].values()) == state_bytes, 'full/recent memory storage')
            pair = shared_cases.get((regime, seed))
            if pair is None:
                shared_cases[(regime, seed)] = (source, source_uniform, odor_uniforms)
            else:
                require(pair[0] == source and pair[1] == source_uniform, 'paired source and source-channel draw')
                length = min(len(pair[2]), len(odor_uniforms))
                require(pair[2][:length] == odor_uniforms[:length], 'paired completed-hit uniform channels')
                if len(odor_uniforms) > len(pair[2]):
                    shared_cases[(regime, seed)] = (source, source_uniform, odor_uniforms)
            rows.append(row)
        require(next(stream, None) is None and next(traces, None) is None and len(shared_cases) == 96,
                'exact evaluation cohort without extra episodes or transitions')
    return rows


def aggregate(rows, weights):
    """Independent grouped arithmetic; fixed seed means and comparison rules."""
    result = {'episodes': len(rows), 'regimes': {}, 'inherited_gate_revised': False,
              'learned_architecture_advantage_established': False}
    for regime in COHORTS:
        selected = [r for r in rows if r['regime'] == regime]
        strata = {}
        for hit in (1, 2, 3):
            strata[str(hit)] = {}
            for arm in ARMS:
                cell = [r for r in selected if r['initial_hit'] == hit and r['arm'] == arm]
                require(len(cell) == 16, 'sixteen episodes in every evaluation arm/stratum')
                strata[str(hit)][arm] = {metric: math.fsum(float(r[metric]) for r in cell) / 16 for metric in METRICS}
        means = {arm: {metric: math.fsum(weights[regime][hit] * strata[str(hit)][arm][metric]
                                        for hit in (1, 2, 3)) for metric in METRICS} for arm in ARMS}
        blocks = []
        for block in range(8):
            values = {}
            for arm in ARMS:
                components = []
                for hit in (1, 2, 3):
                    cell = [r for r in selected if r['arm'] == arm and r['block'] == block and r['initial_hit'] == hit]
                    require(len(cell) == 2, 'two episodes in every block/arm/stratum')
                    components.append(weights[regime][hit] * math.fsum(r['steps'] for r in cell) / 2)
                values[arm] = math.fsum(components)
            blocks.append(values)
        family = {f: {m: math.fsum(means[f'{f}@{seed}'][m] for seed in FIT_SEEDS) / 3 for m in METRICS} for f in FAMILIES}
        criteria = {}
        full = means['full_bayes@planner']
        for candidate in FAMILIES[:2]:
            c = family[candidate]
            gains = [math.fsum(b[f'recent32_hard@{seed}'] - b[f'{candidate}@{seed}'] for seed in FIT_SEEDS) / 3 for b in blocks]
            checks = {
                'every_fit_success_at_least_95pct': all(means[f'{candidate}@{seed}']['found'] >= .95 for seed in FIT_SEEDS),
                'mean_success_no_worse_than_full_planner': c['found'] >= full['found'],
                'mean_moves_at_most_105pct_full_planner': c['steps'] <= full['steps'] * 1.05,
                'mean_controller_at_most_80pct_full_planner': c['controller_seconds'] <= full['controller_seconds'] * .8,
                'every_fit_controller_cheaper_than_full_planner': all(means[f'{candidate}@{seed}']['controller_seconds'] < full['controller_seconds'] for seed in FIT_SEEDS),
                'mean_moves_at_most_95pct_recent_head': c['steps'] <= family['recent32_hard']['steps'] * .95,
                'positive_blocks_vs_recent_head_at_least_six': sum(x > 0 for x in gains) >= 6,
                'mean_moves_at_most_105pct_full_head': c['steps'] <= family['full_bayes']['steps'] * 1.05,
                'mean_controller_no_more_than_full_head': c['controller_seconds'] <= family['full_bayes']['controller_seconds'],
                'state_at_most_20pct_full': c['state_bytes'] <= full['state_bytes'] * .2}
            criteria[candidate] = {'checks': checks, 'passes': all(checks.values()), 'block_gains_vs_recent_head': gains}
        competent = all(means[f'full_bayes@{seed}']['found'] >= .95
                        and means[f'full_bayes@{seed}']['steps'] <= 1.05 * full['steps'] for seed in FIT_SEEDS)
        result['regimes'][regime] = {'weights': {str(h): w for h, w in weights[regime].items()}, 'means': means,
                                     'strata': strata, 'blocks': blocks, 'family_means': family,
                                     'criteria': criteria, 'full_head_competent': competent}
    result['readout_pilot_passes'] = all(r['full_head_competent'] and all(c['passes'] for c in r['criteria'].values())
                                       for r in result['regimes'].values())
    return result


def verify_summary(run, rows, weights, records, qualified, done, comparison):
    computed, published = aggregate(rows, weights), read(run / 'summary.json')
    for name, value in computed.items():
        comparison.same(published[name], value, f'independent summary {name}')
    comparison.same(published['head_setup_seconds'], {r['arm']: r['checkpoint_load_validation_seconds'] for r in records}, 'head setup metadata')
    comparison.same(published['head_storage'], {r['arm']: r['storage'] for r in records}, 'immutable head storage metadata')
    comparison.same(published['shared_model_initialization_seconds'], qualified['shared_model_initialization_seconds'], 'shared model initialization metadata')
    require(set(published['shared_model_storage']) == set(COHORTS) and isinstance(published['timing_scope'], str)
            and published['timing_scope'], 'explicit shared storage and timing disclosure')
    for storage in published['shared_model_storage'].values():
        require(storage['shared_array_bytes'] == storage['kernel_array_bytes'] + storage['initial_log_prior_array_bytes'], 'immutable shared-model arrays')
    require(done['readout_pilot_passes'] is computed['readout_pilot_passes'], 'worker gate identity')
    checks = [value for regime in computed['regimes'].values() for candidate in regime['criteria'].values()
              for value in candidate['checks'].values()]
    require(len(checks) == 40, 'all forty original candidate conditions')
    computed['conditions_passed'] = sum(checks)
    computed['conditions_total'] = len(checks)
    computed['full_head_competence_details'] = {
        regime: {str(seed): {
            'success_at_least_95pct': cell['means'][f'full_bayes@{seed}']['found'] >= .95,
            'moves_at_most_105pct_full_planner': cell['means'][f'full_bayes@{seed}']['steps']
            <= 1.05 * cell['means']['full_bayes@planner']['steps']}
            for seed in FIT_SEEDS} for regime, cell in computed['regimes'].items()}
    return computed


def self_check():
    """Small unequal-stratum arithmetic and masked-network closed forms only."""
    import numpy as np
    head = {'mean': np.zeros(2, dtype=np.float32), 'scale': np.ones(2, dtype=np.float32),
            'weight0': np.zeros((64, 2), dtype=np.float32), 'bias0': np.zeros(64, dtype=np.float32),
            'weight1': np.zeros((32, 64), dtype=np.float32), 'bias1': np.zeros(32, dtype=np.float32),
            'weight2': np.zeros((4, 32), dtype=np.float32), 'bias2': np.zeros(4, dtype=np.float32)}
    target = np.array([[.5, .5, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    mask = np.array([[True, True, False, False], [False, False, True, False]])
    closed = selected_validation(np.zeros((2, 2), dtype=np.float32), target, mask, head, np, lambda: None)
    require(abs(closed['validation_ce'] - math.log(2) / 2) < 1e-15 and closed['matches'] == 2
            and closed['near_tie_rows'] == 1, 'masked CE and first-argmax closed form')
    rows = []
    for regime, seed, block, hit, arm in evaluation_order():
        family, mode = arm.split('@')
        base = 20 if family.startswith('dct') else 30
        values = {name: 0. for name in METRICS}
        values.update(steps=base + hit, found=True, controller_seconds=1. if family.startswith('dct') else 2.,
                      state_bytes=4857 if family.startswith('dct') else 25281,
                      regime=regime, seed=seed, block=block, initial_hit=hit, arm=arm)
        if family == 'full_bayes' and mode == 'planner':
            values['controller_seconds'] = 3.
        rows.append(values)
    weights = {regime: {1: .5, 2: .25, 3: .25} for regime in COHORTS}
    result = aggregate(rows, weights)
    require(result['regimes']['base']['means']['dct16_neutral@7901']['steps'] == 21.75
            and result['readout_pilot_passes'] is True, 'unequal-stratum weighting and all candidate decisions')
    altered = [dict(row) for row in rows]
    for row in altered:
        if row['arm'] == 'dct16_neutral@7901':
            row['controller_seconds'] = 10.
    failed = aggregate(altered, weights)
    require(failed['readout_pilot_passes'] is False
            and failed['regimes']['base']['criteria']['dct16_neutral']['checks']['every_fit_controller_cheaper_than_full_planner'] is False,
            'retain a failed fit rather than average it away')
    require(len(payload_names()) == 48 and len(list(evaluation_order())) == 1536,
            'exact complete artifact and autonomous cohort cardinality')
    print(json.dumps({'status': 'completed', 'synthetic_checks': 5, 'scientific_inputs_read': False,
                      'native_calls': 0, 'torch_imported': 'torch' in sys.modules}))


def execute(args):
    args.output.mkdir(parents=True, exist_ok=False)
    clock, start = None, None
    source_pin = sha(Path(__file__))
    receipt = {'status': 'started', 'source_sha256': source_pin, 'request': {k: str(v) for k, v in vars(args).items()},
               'scope': SCOPE, 'simulator_calls': 0, 'actor_calls': 0, 'training_updates': 0,
               'checkpoint_replay': 'Independent NumPy implementation on saved validation features',
               'limits': {'seconds': 600, 'rss_bytes': 8 * 1024**3, 'output_bytes': 64 * 1024**2},
               'ce_tolerance': CE_TOLERANCE, 'argmax_margin': ARGMAX_MARGIN}
    try:
        require(sha(ROOT / 'src/openjev/research/suspend_clock.py') == CLOCK_PIN, 'audit clock source')
        clock = SuspendClock()
        start = clock.now_ns()

        def budget():
            require(clock.now_ns() - start < 600 * 10**9, 'audit native elapsed cap')
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            require(rss <= receipt['limits']['rss_bytes'], 'audit RSS cap')
            receipt['peak_rss_bytes'] = rss
            require(sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) <= 64 * 1024**2, 'audit output cap')

        plan, done, terminal = authenticate(args, budget)
        comparison = Comparison()
        qualified, weights = qualify(args.run, done, comparison, budget)
        # Numerical dependency is imported only after every scientific payload and
        # actual successful parent completion is authenticated. No Torch import.
        for name, value in THREAD_ENV.items():
            os.environ[name] = value
        import numpy as np
        receipt['audit_runtime'] = {'python': sys.version, 'numpy': np.__version__, 'threads': dict(THREAD_ENV)}
        data = datasets(args.run, done, comparison, budget, np)
        records, checkpoints = checkpoint_review(args.run, data, comparison, budget, np)
        rows = check_evaluation(args.run, records, qualified, comparison, budget)
        collection_steps = sum(len(value['target']) for value in data.values())
        evaluation_steps = sum(row['steps'] for row in rows)
        native = qualified['native_steps'] + collection_steps + evaluation_steps
        require(done['native_steps_attempted'] == done['native_steps_returned'] == native, 'all qualification/collection/evaluation native calls')
        computed = verify_summary(args.run, rows, weights, records, qualified, done, comparison)
        computed.update(scope=SCOPE, collection_cases={'train': 384, 'validation': 96},
                        collection_rows={k: len(v['target']) for k, v in data.items()},
                        evaluation_steps=evaluation_steps, qualification_steps=qualified['native_steps'],
                        native_steps=native, fits=12, numerical_comparisons=comparison.scalars,
                        maximum_absolute_comparison_error=comparison.maximum_error)
        # Inputs and source must remain identical throughout the saved-only audit.
        source_manifest(ROOT, plan['sources'], budget)
        for name, item in done['files'].items():
            require(sha(args.run / name) == item['sha256'], f'unchanged payload {name}')
            budget()
        require(sha(Path(__file__)) == source_pin and sha(args.plan) == args.plan_sha256
                and sha(args.run / 'receipt.json') == args.receipt_sha256 and sha(args.terminal) == args.terminal_sha256,
                'unchanged external identity pins')
        write(args.output / 'summary.json', computed)
        write(args.output / 'selected-checkpoints.json', checkpoints)
        budget()
        receipt.update(status='completed', agreement=True, plan_sha256=args.plan_sha256,
                       worker_receipt_sha256=args.receipt_sha256, supervisor_terminal_sha256=args.terminal_sha256,
                       original_payloads=done['files'], original_sources=plan['sources'],
                       supervisor_wall_seconds=terminal['wall_seconds'], episodes=len(rows), fits=12,
                       native_steps_verified=native, numerical_comparisons=comparison.scalars,
                       maximum_absolute_comparison_error=comparison.maximum_error,
                       conditions_passed=computed['conditions_passed'], conditions_total=40,
                       readout_pilot_passes=computed['readout_pilot_passes'],
                       files={p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in args.output.iterdir() if p.is_file()})
        receipt.update(clock_backend=clock.backend, started_ns=start, finished_ns=clock.now_ns())
        receipt['wall_seconds'] = (receipt['finished_ns'] - start) / 1e9
        write(args.output / 'receipt.json', receipt)
        budget()
        print(json.dumps({'status': 'completed', 'receipt_sha256': sha(args.output / 'receipt.json'),
                          'summary_sha256': sha(args.output / 'summary.json'), 'episodes': len(rows),
                          'fits': 12, 'conditions_passed': computed['conditions_passed'], 'conditions_total': 40,
                          'readout_pilot_passes': computed['readout_pilot_passes']}))
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if clock is not None and start is not None:
            try:
                receipt['failure_elapsed_seconds'] = (clock.now_ns() - start) / 1e9
            except BaseException as timing_error:  # noqa: BLE001 - preserve the original audit failure.
                receipt['failure_clock_error'] = repr(timing_error)
        try:
            path = args.output / ('failed.json' if (args.output / 'receipt.json').exists() else 'receipt.json')
            write(path, receipt)
        except BaseException as publication_error:  # noqa: BLE001 - a second I/O failure must not mask the first.
            error.add_note(f'Audit failure receipt could not be published: {publication_error!r}')
        raise


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ('plan', 'run', 'terminal', 'output'):
            parser.add_argument(f'--{name}', type=Path, required=True)
        for name in ('plan-sha256', 'receipt-sha256', 'terminal-sha256'):
            parser.add_argument(f'--{name}', required=True)
        execute(parser.parse_args())
