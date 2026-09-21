"""Saved scalar/branch arithmetic only; no model, posterior or simulator replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import resource
import signal
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-return-consistency-v1'
SELF = 'scripts/diagnose_otto_return_consistency.py'
TEST = 'tests/test_otto_return_consistency.py'
DESIGN = 'research/otto-return-consistency-design.md'
HELPER = 'scripts/render_otto_return_value.py'
HELPER_PIN = 'f390c890014fa97d269c13c7bd3621a982d23b3da987dbb32cb51dfaeabdf0f0'
CLOCK = 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
SOURCES = {SELF, TEST, DESIGN, HELPER, CLOCK}
ROLES = {'study_plan', 'worker_receipt', 'terminal', 'audit_receipt', 'prior_receipt'}
LIMITS = {'seconds': 180, 'rss_bytes': 2 * 1024**3, 'output_bytes': 64 * 1024**2}
CONFIGURATION = {'fits': 9, 'prefixes_per_split': 8, 'splits': ['train', 'valid'], 'rows': 144,
                 'atol': 1e-10, 'rtol': 1e-10, 'selection_epsilon': 1e-10}
FAMILIES, SEEDS, SPLITS = ('min8', 'mlp8', 'homogeneous8'), (10101, 10102, 10103), ('train', 'valid')
SELECTED = {'train': ('train:lambda3:910001:teacher', 910001, 79),
            'valid': ('valid:lambda3:930001:teacher', 930001, 14)}
EPSILON = 1e-10
SCOPE = ('Descriptive saved arithmetic on 144 model-prefix rows from 16 mechanically selected prefixes '
         'in two lambda3/initial-hit1 teacher episodes, steps 0..7. No inference, posterior reconstruction, '
         'training, policy or simulator calls. Monte Carlo teacher returns are not optimal or counterfactual '
         'values. Floored-backup residuals measure internal consistency, not observed action-value error. '
         'Teacher disagreement is not automatically policy error. No efficacy gate or original decision change.')
METRICS = ('current_return_error', 'teacher_action_residual', 'greedy_action_residual')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite nonboolean number')
    return value


def integer(value, low=0):
    require(type(value) is int and value >= low, 'bounded integer')
    return value


def vector(value, length):
    require(type(value) is list and len(value) == length, 'declared vector shape')
    return [number(v) for v in value]


def matrix(value):
    require(type(value) is list and len(value) == 4, 'four action rows')
    return [vector(v, 4) for v in value]


def near(actual, expected):
    number(actual)
    number(expected)
    require(abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected), 'saved arithmetic tolerance')


def allowed(value):
    require(type(value) is list and value and all(type(a) is int and 0 <= a < 4 for a in value)
            and value == sorted(set(value)), 'sorted unique eligible actions')
    return value


def select(costs, eligible):
    costs, eligible = vector(costs, 4), allowed(eligible)
    minimum = min(costs[a] for a in eligible)
    ties = [a for a in eligible if abs(costs[a] - minimum) < EPSILON]
    action = ties[0]
    return {'action': action, 'minimum': minimum, 'tie_ids': ties,
            'chosen_minus_minimum': costs[action] - minimum}


def witness(value):
    require(type(value) is dict and set(value) == {'sha256', 'mass'}, 'posterior witness fields')
    pin = value['sha256']
    require(type(pin) is str and len(pin) == 64 and all(c in '0123456789abcdef' for c in pin), 'posterior hash')
    require(number(value['mass']) >= 0, 'posterior nonnegative mass')
    return value


def packet(value):
    require(type(value) is dict and set(value) == {'position', 'hit', 'done', 'step', 'valid_actions'},
            'public packet whitelist')
    pos = value['position']
    require(type(pos) is list and len(pos) == 2 and all(type(x) is int and 0 <= x < 53 for x in pos),
            'grid position')
    integer(value['step'])
    require(type(value['done']) is bool and type(value['hit']) is int
            and (value['hit'] == -2 if value['done'] else 0 <= value['hit'] < 4), 'public hit/done')
    actions = [a for a in range(4) if 0 <= pos[a // 2] + 2 * (a % 2) - 1 < 53]
    require(allowed(value['valid_actions']) == actions, 'public movement eligibility')
    return value


def validate_metadata(row, split, index):
    require(type(row) is dict and set(row) == {'row_index', 'episode_id', 'stage', 'regime', 'seed',
            'initial_hit', 'prefix_index', 'total_steps', 'public', 'posterior', 'target'}, 'metadata schema')
    episode, seed, total = SELECTED[split]
    require(integer(row['row_index']) == index and integer(row['prefix_index']) == index
            and row['stage'] == split and row['episode_id'] == episode and row['regime'] == 'lambda3'
            and integer(row['seed']) == seed and integer(row['initial_hit']) == 1
            and integer(row['total_steps'], 1) == total, 'fixed selected prefix identity')
    public = packet(row['public'])
    require(not public['done'] and public['step'] == index, 'current pre-action public packet')
    witness(row['posterior'])
    require(number(row['target']) == (total - index) / 64, 'exact realized teacher return target')
    return row


def teacher_join(events, metadata):
    """Stop at the last required TRAIN/VALID step, never consume a later stage."""
    wanted = {(m['episode_id'], m['prefix_index'] + 1): m for m in metadata.values()}
    require(len(wanted) == len(metadata), 'unique requested teacher prefixes')
    found, resets, active = {}, set(), None
    for event in events:
        require(type(event) is dict and event.get('kind') in ('reset', 'step'), 'teacher event kind')
        if event['kind'] == 'reset':
            require(event['stage'] in SPLITS and event['arm'] == 'teacher', 'TRAIN/VALID teacher stream only')
            eid = event['episode_id']
            require(eid not in resets and eid.startswith(event['stage'] + ':'), 'unique teacher reset')
            resets.add(eid)
            current, state = packet(event['public']), witness(event['posterior_after'])
            require(current['step'] == 0 and not current['done'] and current['hit'] == event['initial_hit'],
                    'teacher initial public packet')
            active = eid
            continue
        require(active is not None and event['episode_id'] == active and not current['done']
                and integer(event['step'], 1) == current['step'] + 1, 'chronological teacher step')
        require(witness(event['posterior_before']) == state, 'teacher posterior chronology')
        require(allowed(event['allowed_actions']) == current['valid_actions']
                and type(event['action']) is int and event['action'] in current['valid_actions'], 'teacher eligible action')
        after = packet(event['public'])
        position = list(current['position'])
        position[event['action'] // 2] += 2 * (event['action'] % 2) - 1
        require(after['step'] == event['step'] and after['position'] == position, 'teacher returned packet join')
        key = active, event['step']
        if key in wanted:
            m = wanted[key]
            require(key not in found and current == m['public'] and state == m['posterior'], 'exact t+1 teacher prefix join')
            found[key] = {'episode_id': active, 'step': event['step'], 'action': event['action'],
                          'allowed_actions': event['allowed_actions'], 'public_before': current,
                          'public_after': after, 'posterior_before': state,
                          'posterior_after': witness(event['posterior_after'])}
        current, state = after, witness(event['posterior_after'])
        if len(found) == len(wanted):
            return found
    raise ValueError('missing required teacher transition')


def analyze_row(row, metadata, teacher):
    fields = {'fit_id', 'split', 'row_index', 'episode_id', 'prefix_index', 'posterior_sha256', 'scalar_numpy',
              'scalar_torch', 'raw_masses', 'weights', 'values_numpy', 'values_torch', 'costs_numpy', 'costs_torch',
              'action_numpy', 'action_torch', 'allowed_actions', 'passed'}
    require(type(row) is dict and set(row) == fields and row['passed'] is True, 'complete saved qualified parity row')
    require(row['episode_id'] == metadata['episode_id'] and integer(row['prefix_index']) == metadata['prefix_index']
            and row['posterior_sha256'] == metadata['posterior']['sha256']
            and row['allowed_actions'] == metadata['public']['valid_actions'], 'parity metadata identity')
    require(teacher['episode_id'] == row['episode_id'] and teacher['step'] == row['prefix_index'] + 1
            and teacher['posterior_before'] == metadata['posterior']
            and teacher['public_before'] == metadata['public']
            and teacher['allowed_actions'] == row['allowed_actions'], 'teacher current-prefix identity')
    raw, weights = matrix(row['raw_masses']), matrix(row['weights'])
    values, costs = vector(row['values_numpy'], 16), vector(row['costs_numpy'], 4)
    reference_values, reference_costs = vector(row['values_torch'], 16), vector(row['costs_torch'], 4)
    near(number(row['scalar_numpy']), number(row['scalar_torch']))
    for actual, expected in zip(values + costs, reference_values + reference_costs, strict=True):
        near(actual, expected)
    selected = select(costs, row['allowed_actions'])
    require(type(row['action_numpy']) is int and type(row['action_torch']) is int
            and row['action_numpy'] == selected['action'] == row['action_torch'], 'exact saved near-tie action')
    a_teacher, a_greedy = teacher['action'], selected['action']
    require(type(a_teacher) is int and a_teacher in row['allowed_actions'], 'teacher action eligibility')
    branches, products, reconstructed = [], [], []
    for action in range(4):
        action_products = []
        for hit in range(4):
            p, w, v = raw[action][hit], weights[action][hit], values[4 * action + hit]
            require(p >= 0 and w == max(p, EPSILON), 'exact raw-mass floor; no renormalization')
            product = number(w * v)
            action_products.append(product)
            branches.append({'action': action, 'hit': hit, 'eligible': action in row['allowed_actions'],
                             'raw_mass': p, 'weight': w, 'physical_value': v, 'weighted_value': product})
        products.append(action_products)
        reconstructed.append(number(1 + math.fsum(action_products)))
        near(reconstructed[-1], costs[action])
    current, target = number(64 * row['scalar_numpy']), number(64 * metadata['target'])
    r_teacher, r_greedy = number(costs[a_teacher] - current), number(costs[a_greedy] - current)
    advantage = number(costs[a_teacher] - costs[a_greedy])
    contributions = [number(products[a_teacher][h] - products[a_greedy][h]) for h in range(4)]
    near(math.fsum(contributions), advantage)
    near(r_teacher - r_greedy, advantage)
    return {'fit_id': row['fit_id'], 'split': row['split'], 'row_index': row['row_index'],
            'episode_id': row['episode_id'], 'prefix_index': row['prefix_index'],
            'metadata': metadata, 'teacher_transition': teacher, 'saved_parity': row, 'branches': branches,
            'current_physical_value': current, 'teacher_realized_return': target,
            'reconstructed_costs': reconstructed, 'cost_reconstruction_differences': [a - b for a, b in zip(reconstructed, costs, strict=True)],
            'selection': selected, 'teacher_action': a_teacher, 'teacher_disagreement': a_teacher != a_greedy,
            'current_return_error': number(current - target), 'teacher_action_residual': r_teacher,
            'greedy_action_residual': r_greedy, 'predicted_switching_advantage': advantage,
            'switching_contributions_by_hit': contributions}


def expected_order():
    return [(f'{family}@{seed}', split, index) for seed in SEEDS for family in FAMILIES
            for split in SPLITS for index in range(8)]


def diagnose(parity, metadata, teachers):
    require(set(metadata) == {(s, i) for s in SPLITS for i in range(8)}, 'sixteen metadata prefixes')
    for (split, index), value in metadata.items():
        validate_metadata(value, split, index)
    require(len(parity) == 144, 'all144 parity rows')
    result = []
    for row, key in zip(parity, expected_order(), strict=True):
        require((row['fit_id'], row['split'], integer(row['row_index'])) == key, 'exact parity order and coverage')
        m = metadata[key[1:]]
        result.append(analyze_row(row, m, teachers[(m['episode_id'], m['prefix_index'] + 1)]))
    return result


def summarize(rows):
    require(len(rows) == 144 and [(r['fit_id'], r['split'], r['row_index']) for r in rows] == expected_order(),
            'summary complete ordered rows')
    panels = {}
    for seed in SEEDS:
        for family in FAMILIES:
            fit = f'{family}@{seed}'
            panels[fit] = {}
            for split in SPLITS:
                group = [r for r in rows if r['fit_id'] == fit and r['split'] == split]
                statistics = {}
                for metric in METRICS:
                    values = [number(r[metric]) for r in group]
                    statistics[metric] = {'signed_mean': math.fsum(values) / 8,
                                          'mean_absolute': math.fsum(abs(x) for x in values) / 8,
                                          'rms': math.sqrt(math.fsum(x * x for x in values) / 8)}
                panels[fit][split] = {'prefixes': 8, 'unique_episodes': 1, 'statistics': statistics,
                                     'mean_switching_advantage': math.fsum(r['predicted_switching_advantage'] for r in group) / 8,
                                     'teacher_disagreement_count': sum(r['teacher_disagreement'] for r in group)}
    family_means = {}
    for family in FAMILIES:
        family_means[family] = {}
        for split in SPLITS:
            groups = [panels[f'{family}@{seed}'][split] for seed in SEEDS]
            family_means[family][split] = {
                'seeds': list(SEEDS), 'prefixes_per_seed': 8,
                'statistics': {m: {s: math.fsum(g['statistics'][m][s] for g in groups) / 3
                                   for s in ('signed_mean', 'mean_absolute', 'rms')} for m in METRICS},
                'mean_switching_advantage': math.fsum(g['mean_switching_advantage'] for g in groups) / 3,
                'mean_teacher_disagreement_count': math.fsum(g['teacher_disagreement_count'] for g in groups) / 3}
    return {'version': VERSION, 'scope': SCOPE, 'model_prefix_rows': 144, 'fits': 9, 'unique_prefixes': 16,
            'unique_episodes': 2, 'coverage': {s: {'episode_id': v[0], 'total_steps': v[2], 'regime': 'lambda3',
            'initial_hit': 1, 'prefix_indices': list(range(8))} for s, v in SELECTED.items()},
            'complete_validation_states': 1109, 'selected_validation_states': 8, 'fit_split': panels,
            'family_means': family_means, 'family_aggregation': 'Equal mean of all three per-seed statistics, including RMS; not a pooled RMS.',
            'new_efficacy_gate': None, 'original_decisions_changed': False}


def parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return parse(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(path, check=lambda: None):
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            check()
            h.update(chunk)
            size += len(chunk)
    return {'sha256': h.hexdigest(), 'bytes': size}


def regular(path):
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'absolute regular nonsymlink file')
    return path


def relative(name):
    p = Path(name)
    require(p.parts and not p.is_absolute() and '..' not in p.parts, 'contained relative path')
    return ROOT / p


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prior_names():
    names = {'started.json', 'runtime.json', 'native-setup.json', 'qualification.json', 'qualification.jsonl',
             'collection-transitions.jsonl', 'collection-episodes.jsonl', 'work-contexts.jsonl', 'work.jsonl',
             'fits.jsonl', 'fit-curves.jsonl', 'epoch-orders.jsonl', 'inference-setup.json', 'eval-transitions.jsonl',
             'eval-episodes.jsonl', 'evaluation.jsonl', 'summary.json', 'pooled-weights.npz', 'pooling.json', 'training-costs.json'}
    names.update(f'kernel-lambda{n}.npz' for n in (3, 4, 5))
    names.update(f'{split}-{suffix}' for split in ('train', 'valid', 'dagger') for suffix in ('data.npz', 'rows.jsonl'))
    names.update(f'{phase}-{family}-{seed}.npz' for phase in ('initial', 'final')
                 for family in ('shared', 'dense') for seed in (9101, 9102, 9103))
    return names


class Diagnostic:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.finished = None
        self.receipt = {'version': VERSION, 'status': 'started', 'limits': LIMITS, 'scope': SCOPE,
                        'model_calls': 0, 'simulator_calls': 0, 'training_calls': 0, 'policy_calls': 0}

    def check(self):
        now = self.clock.now_ns()
        require(now - self.start < LIMITS['seconds'] * 10**9, 'diagnostic native deadline')
        self.finished = now
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(0 < rss <= LIMITS['rss_bytes'], 'diagnostic RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'],
                'diagnostic output cap')

    def authenticate(self):
        args = self.args
        require(digest(regular(args.plan), self.check)['sha256'] == args.plan_sha256, 'external diagnostic plan pin')
        plan = read(args.plan)
        require(plan['version'] == VERSION and plan['status'] == 'frozen_before_saved_read'
                and plan['limits'] == LIMITS and plan['configuration'] == CONFIGURATION
                and relative(plan['output']) == self.out and set(plan['sources']) == SOURCES
                and set(plan['inputs']) == ROLES, 'exact diagnostic plan contract')
        require(plan['sources'][HELPER] == HELPER_PIN and plan['sources'][CLOCK] == CLOCK_PIN, 'qualified helper pins')
        for name, pin in plan['sources'].items():
            require(digest(regular(relative(name)), self.check)['sha256'] == pin, 'unchanged diagnostic source')
        require(sys.executable == plan['python_executable'] and sys.version.split()[0] == plan['python_version']
                and {d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == plan['all_distributions'],
                'diagnostic runtime identity')
        paths = {}
        for role, item in plan['inputs'].items():
            require(set(item) == {'path', 'sha256', 'bytes'}, 'input descriptor schema')
            paths[role] = regular(relative(item['path']))
            require(digest(paths[role], self.check) == {'sha256': item['sha256'], 'bytes': item['bytes']}, 'external input bytes')
        helper = load(ROOT / HELPER, '_return_consistency_auth')
        auth = helper.Render(SimpleNamespace(run=paths['worker_receipt'].parent, audit=paths['audit_receipt'].parent,
            receipt_sha256=plan['inputs']['worker_receipt']['sha256'],
            audit_receipt_sha256=plan['inputs']['audit_receipt']['sha256'], output=self.out))
        auth.check = self.check
        aggregate = auth.authenticate()
        for role, helper_role in (('study_plan', 'plan'), ('terminal', 'terminal')):
            record = auth.receipt['inputs'][helper_role]
            require(record['path'] == str(paths[role]) and record['sha256'] == plan['inputs'][role]['sha256'],
                    'external audited plan/terminal join')
        study = read(paths['study_plan'])
        require(study['inputs']['prior_receipt'] == plan['inputs']['prior_receipt'], 'exact inherited teacher receipt')
        prior = auth.manifest(paths['prior_receipt'].parent, plan['inputs']['prior_receipt']['sha256'], prior_names())
        require(prior['version'] == 'otto-symmetry-head-v1'
                and prior['plan_sha256'] == study['inputs']['prior_plan']['sha256']
                and all(study['sources'].get(k) == v for k, v in prior['sources'].items()), 'frozen teacher source lineage')
        self.receipt.update(plan_sha256=args.plan_sha256, sources=plan['sources'], inputs=plan['inputs'],
                            authenticated_current_payloads=44, authenticated_audit_payloads=2,
                            authenticated_teacher_payloads=41,
                            original_pilot_continuation=aggregate['pilot_continuation'])
        return paths

    def lines(self, path):
        with path.open() as stream:
            for text in stream:
                self.check()
                require(text.strip(), 'no blank record')
                yield parse(text)

    def compute(self, paths):
        run = paths['worker_receipt'].parent
        metadata = {}
        for split in SPLITS:
            records = self.lines(run / f'{split}-rows.jsonl')
            try:
                for index in range(8):
                    metadata[split, index] = validate_metadata(next(records), split, index)
            finally:
                records.close()
        transitions = self.lines(paths['prior_receipt'].parent / 'collection-transitions.jsonl')
        try:
            teachers = teacher_join(transitions, metadata)
        finally:
            transitions.close()
        parity = list(self.lines(run / 'parity.jsonl'))
        rows = diagnose(parity, metadata, teachers)
        with (self.out / 'rows.jsonl').open('x') as stream:
            for row in rows:
                self.check()
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        result = summarize(rows)
        result['inputs'] = self.receipt['inputs']
        result['original_pilot_continuation'] = self.receipt['original_pilot_continuation']
        write(self.out / 'summary.json', result)
        self.receipt.update(model_prefix_rows=len(rows), unique_prefixes=16, unique_episodes=2)

    def execute(self):
        require(self.out.is_absolute() and '..' not in self.out.parts
                and not any(p.is_symlink() for p in (self.out, *self.out.parents)), 'exclusive nonsymlink output')
        self.out.mkdir(parents=True, exist_ok=False)
        old_handler = signal.getsignal(signal.SIGALRM)
        def timeout(_signum, _frame):
            raise TimeoutError('diagnostic alarm')
        try:
            require(digest(regular(ROOT / CLOCK))['sha256'] == CLOCK_PIN, 'clock source pin')
            self.clock = load(ROOT / CLOCK, '_return_consistency_clock').SuspendClock()
            self.start = self.clock.now_ns()
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
            write(self.out / 'started.json', {'version': VERSION, 'request': {k: str(v) for k, v in vars(self.args).items()},
                  'started_ns': self.start, 'clock_backend': self.clock.backend, 'limits': LIMITS, 'scope': SCOPE})
            paths = self.authenticate()
            self.compute(paths)
            self.authenticate()  # Byte identity again after all saved arithmetic.
            files = {name: digest(self.out / name, self.check) for name in ('started.json', 'rows.jsonl', 'summary.json')}
            self.check()
            self.receipt.update(status='completed', files=files, started_ns=self.start, finished_ns=self.finished,
                                clock_backend=self.clock.backend, wall_seconds=(self.finished - self.start) / 1e9,
                                timing_scope='Native start through final pre-receipt check; publication rechecked before return.')
            write(self.out / 'receipt.json', self.receipt)
            self.check()
            return 0
        except BaseException as error:  # noqa: BLE001 - Preserve interrupts and all failed attempts.
            signal.setitimer(signal.ITIMER_REAL, 0)
            trace = traceback.format_exc()
            timing = None
            try:
                if self.clock is not None and self.start is not None:
                    timing = (self.clock.now_ns() - self.start) / 1e9
            except Exception:  # noqa: BLE001 - A broken clock must yield unavailable timing.
                timing = None
            try:
                receipt = self.out / 'receipt.json'
                if receipt.exists():
                    receipt.rename(self.out / 'invalid-completed-receipt.json')
                self.receipt.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=trace,
                                    wall_seconds=timing, timing_available=timing is not None)
                write(receipt, self.receipt)
            except Exception as preservation_error:  # noqa: BLE001 - Retain the primary failure.
                print(f'Failure preservation error: {preservation_error}', file=sys.stderr)
            print(trace, file=sys.stderr)
            return 1
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    return Diagnostic(parser.parse_args()).execute()


if __name__ == '__main__':
    raise SystemExit(main())
