"""Independent saved arithmetic and original-record joins; no producer imports."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import resource
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
PLAN_PIN = '9bc2015d762c8d63df4f6bc51ee687f9281be7d855815ac3f475498dd0ced16f'
RECEIPT_PIN = '64cbe0ad150bc574d336d8d4472949ca80f136dfcfb2117d66b795234a06c212'
CLOCK = ROOT / 'src/openjev/research/suspend_clock.py'
CLOCK_PIN = 'cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124'
LIMITS = {'seconds': 120, 'rss_bytes': 1024**3, 'output_bytes': 16*1024**2}
SEEDS = (10101, 10102, 10103)
FAMILIES = ('min8', 'mlp8', 'homogeneous8')
SPLITS = ('train', 'valid')
METRICS = ('current_return_error', 'teacher_action_residual', 'greedy_action_residual')
SELECTED = {'train': ('train:lambda3:910001:teacher', 910001, 79),
            'valid': ('valid:lambda3:930001:teacher', 930001, 14)}
SOURCES = {'research/otto-return-consistency-design.md', 'research/otto-return-consistency-repair.md',
           'scripts/diagnose_otto_return_consistency.py', 'scripts/diagnose_otto_return_consistency_v2.py',
           'scripts/render_otto_return_value.py', 'src/openjev/research/suspend_clock.py',
           'tests/test_otto_return_consistency.py', 'tests/test_otto_return_consistency_v2.py'}
SCOPE = ('Independent original-record identity, teacher t+1 joins and saved arithmetic on all144 '
         'model-prefix rows,18 fit/split panels and6 equal-seed family panels. No posterior, network, '
         'policy, simulator or training execution. Original value predictions, posterior witnesses, '
         'native runtime and timing truth are authenticated upstream evidence. Two lambda3/hit1 '
         'episodes, first8 prefixes each; no efficacy gate or original decision change.')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return parse(path.read_text())


def write(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path, check=lambda: None):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink file')
    hasher, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024**2), b''):
            check()
            hasher.update(data)
            size += len(data)
    return {'sha256': hasher.hexdigest(), 'bytes': size}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def numeric(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite nonboolean number')
    return value


def public(packet):
    require(type(packet) is dict and set(packet) == {'position', 'hit', 'done', 'step', 'valid_actions'}, 'public whitelist')
    pos = packet['position']
    require(type(pos) is list and len(pos) == 2 and all(type(x) is int and 0 <= x < 53 for x in pos), 'position')
    require(type(packet['step']) is int and packet['step'] >= 0 and type(packet['done']) is bool, 'step and done')
    require(type(packet['hit']) is int and (packet['hit'] == -2 if packet['done'] else 0 <= packet['hit'] < 4), 'hit sentinel')
    expected = [] if packet['done'] else [a for a in range(4) if 0 <= pos[a//2]+(-1 if a%2 == 0 else 1) < 53]
    require(type(packet['valid_actions']) is list and packet['valid_actions'] == expected
            and all(type(a) is int for a in packet['valid_actions']), 'terminal or inbounds eligibility')
    return packet


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = None
        self.start = None
        self.comparisons = 0
        self.maximum_difference = 0.0
        self.receipt = {'version': 'otto-return-consistency-independent-audit-v1', 'status': 'started',
                        'scope': SCOPE, 'limits': LIMITS, 'model_calls': 0, 'training_calls': 0,
                        'policy_calls': 0, 'simulator_calls': 0}

    def check(self):
        require(self.clock.now_ns()-self.start < LIMITS['seconds']*10**9, 'native elapsed cap')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        self.receipt['peak_rss_bytes'] = rss
        require(rss <= LIMITS['rss_bytes'], 'RSS cap')
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS['output_bytes'], 'output cap')

    def equal(self, actual, expected, name):
        self.comparisons += 1
        require(canonical(actual) == canonical(expected), 'exact identity: ' + name)

    def compare(self, actual, expected, name='result'):
        self.comparisons += 1
        if type(expected) is dict:
            require(type(actual) is dict and set(actual) == set(expected), 'keys: ' + name)
            for key in expected:
                self.compare(actual[key], expected[key], name+'.'+key)
        elif type(expected) is list:
            require(type(actual) is list and len(actual) == len(expected), 'length: ' + name)
            for i, value in enumerate(expected):
                self.compare(actual[i], value, name+'.'+str(i))
        elif type(expected) is float:
            delta = abs(numeric(actual)-numeric(expected))
            self.maximum_difference = max(self.maximum_difference, delta)
            require(delta <= 1e-12+1e-12*abs(expected), 'numerical agreement: ' + name)
        else:
            require(type(actual) is type(expected) and actual == expected, 'exact field: ' + name)

    def bound(self, actual, expected):
        self.comparisons += 1
        require(abs(numeric(actual)-numeric(expected)) <= 1e-10+1e-10*abs(expected), 'original arithmetic bound')

    def manifest(self, directory, receipt, count):
        require(len(receipt['files']) == count and set(p.name for p in directory.iterdir()) == set(receipt['files']) | {'receipt.json'}, 'closed payload membership')
        for name, desc in receipt['files'].items():
            require(Path(name).name == name, 'flat payload path')
            self.equal(digest(directory/name, self.check), desc, 'payload '+name)

    def authenticate(self):
        a = self.args
        require(a.plan_sha256 == PLAN_PIN and a.receipt_sha256 == RECEIPT_PIN, 'fixed external pins')
        self.equal(digest(a.plan, self.check)['sha256'], PLAN_PIN, 'plan')
        self.equal(digest(a.run/'receipt.json', self.check)['sha256'], RECEIPT_PIN, 'completed diagnostic')
        plan, done = read(a.plan), read(a.run/'receipt.json')
        require(plan['version'] == done['version'] == 'otto-return-consistency-v2'
                and done['status'] == 'completed' and done['plan_sha256'] == PLAN_PIN, 'completed exact version')
        require(ROOT/plan['output'] == a.run and set(plan['sources']) == SOURCES, 'fixed run and eight sources')
        self.equal(done['sources'], plan['sources'], 'source map')
        self.equal(done['inputs'], plan['inputs'], 'input map')
        require(platform.python_version() == plan['python_version'] and str(Path(sys.executable).absolute()) == plan['python_executable'], 'declared Python runtime')
        for path, pin in plan['sources'].items():
            self.equal(digest(ROOT/path, self.check)['sha256'], pin, 'source '+path)
        self.manifest(a.run, done, 3)
        inputs = {}
        for name, desc in plan['inputs'].items():
            path = ROOT/desc['path']
            self.equal(digest(path, self.check), {k:desc[k] for k in ('sha256', 'bytes')}, 'input '+name)
            inputs[name] = read(path)
        worker, previous, audit = (inputs[k] for k in ('worker_receipt', 'prior_receipt', 'audit_receipt'))
        terminal, study = inputs['terminal'], inputs['study_plan']
        require(all(r['status'] == 'completed' for r in (worker, previous, audit)) and audit['agreement'] is True, 'upstream completion and audit')
        require(terminal['status'] == 'completed' and terminal['returncode'] == 0 and terminal['timed_out'] is False
                and terminal['group_absent'] is True and terminal['cleanup']['errors'] == []
                and terminal['cleanup']['reaped'] is True and terminal['error'] is None and terminal['clock_error'] is None, 'successful original supervisor')
        self.equal(worker['plan_sha256'], plan['inputs']['study_plan']['sha256'], 'worker plan')
        self.equal(worker['sources'], study['sources'], 'original source closure')
        self.equal(study['inputs']['prior_receipt'], plan['inputs']['prior_receipt'], 'teacher lineage')
        self.equal(audit['worker_sha256'], plan['inputs']['worker_receipt']['sha256'], 'audited worker')
        self.equal(audit['terminal_sha256'], plan['inputs']['terminal']['sha256'], 'audited terminal')
        require(worker['pilot_continuation'] is False and done['original_pilot_continuation'] is False, 'original decision preserved')
        self.worker_dir = (ROOT/plan['inputs']['worker_receipt']['path']).parent
        self.teacher_dir = (ROOT/plan['inputs']['prior_receipt']['path']).parent
        self.manifest(self.worker_dir, worker, 44)
        self.manifest(self.teacher_dir, previous, 41)
        self.manifest((ROOT/plan['inputs']['audit_receipt']['path']).parent, audit, 2)
        self.receipt.update(plan_sha256=PLAN_PIN, input_receipt_sha256=RECEIPT_PIN,
                            diagnostic_receipt_sha256=RECEIPT_PIN, sources=plan['sources'], inputs=plan['inputs'])
        return plan, done

    def lines(self, path):
        with path.open() as stream:
            for text in stream:
                self.check()
                yield parse(text)

    def teachers(self, metadata):
        wanted = {(r['episode_id'], r['prefix_index']+1) for r in metadata.values()}
        found, state, current, active = {}, None, None, None
        terminals = 0
        for row in self.lines(self.teacher_dir/'collection-transitions.jsonl'):
            if row['kind'] == 'reset':
                require(row['stage'] in SPLITS and row['arm'] == 'teacher', 'stop before DAgger')
                current, state, active = public(row['public']), row['posterior_after'], row['episode_id']
                require(current['step'] == 0 and not current['done'], 'teacher reset')
                continue
            require(row['kind'] == 'step' and row['episode_id'] == active and not current['done']
                    and row['step'] == current['step']+1, 'chronological teacher transition')
            self.equal(row['posterior_before'], state, 'teacher posterior chronology')
            self.equal(row['allowed_actions'], current['valid_actions'], 'teacher pre-action mask')
            action, after = row['action'], public(row['public'])
            require(type(action) is int and action in current['valid_actions'], 'eligible teacher action')
            position = list(current['position']); position[action//2] += -1 if action%2 == 0 else 1
            require(after['position'] == position and after['step'] == row['step'], 'teacher action/packet join')
            key = active, row['step']
            if key in wanted:
                require(key not in found, 'unique teacher match')
                found[key] = {'episode_id': active, 'step': row['step'], 'action': action,
                              'allowed_actions': row['allowed_actions'], 'public_before': current, 'public_after': after,
                              'posterior_before': row['posterior_before'], 'posterior_after': row['posterior_after']}
            terminals += after['done']
            current, state = after, row['posterior_after']
            if set(found) == wanted:
                self.receipt['traversed_teacher_terminals'] = terminals
                return found
        raise ValueError('missing original teacher records')

    def row(self, original, meta, teacher):
        current = float(original['scalar_numpy'])*64
        target = float(meta['target'])*64
        allowed = meta['public']['valid_actions']
        costs = [float(x) for x in original['costs_numpy']]
        require(len(costs) == 4 and all(math.isfinite(x) for x in costs), 'four finite raw costs')
        low = min(costs[a] for a in allowed)
        ties = [a for a in range(4) if a in allowed and abs(costs[a]-low) < 1e-10]
        greedy, teach = ties[0], teacher['action']
        self.equal(greedy, original['action_numpy'], 'strict eligible saved action')
        self.equal(original['action_torch'], greedy, 'qualified exact Torch action witness')
        require(teach in allowed and original['passed'] is True, 'qualified teacher eligibility')
        require(len(original['raw_masses']) == len(original['weights']) == 4 and len(original['values_numpy']) == 16, 'branch geometry')
        branches, terms, reconstructed = [], [], []
        for a in range(4):
            require(len(original['raw_masses'][a]) == len(original['weights'][a]) == 4, 'four hits')
            part = []
            for h in range(4):
                p = numeric(original['raw_masses'][a][h]); w = numeric(original['weights'][a][h])
                value = numeric(original['values_numpy'][4*a+h])
                require(p >= 0 and w == (p if p > 1e-10 else 1e-10), 'raw mass and exact floor')
                product = w*value; part.append(product)
                branches.append({'action':a, 'hit':h, 'eligible':a in allowed, 'raw_mass':p,
                                 'weight':w, 'physical_value':value, 'weighted_value':product})
            terms.append(part); reconstructed.append(1+sum(part))
            self.bound(reconstructed[-1], costs[a])
        advantage = costs[teach]-costs[greedy]
        contributions = [terms[teach][h]-terms[greedy][h] for h in range(4)]
        self.bound(sum(contributions), advantage)
        self.bound((costs[teach]-current)-(costs[greedy]-current), advantage)
        return {'fit_id': original['fit_id'], 'split': original['split'], 'row_index': original['row_index'],
                'episode_id': original['episode_id'], 'prefix_index': original['prefix_index'],
                'metadata':meta, 'teacher_transition':teacher, 'saved_parity':original, 'branches':branches,
                'current_physical_value':current, 'teacher_realized_return':target,
                'reconstructed_costs':reconstructed, 'cost_reconstruction_differences':[x-y for x,y in zip(reconstructed,costs,strict=True)],
                'selection':{'action':greedy,'minimum':low,'tie_ids':ties,'chosen_minus_minimum':costs[greedy]-low},
                'teacher_action':teach,'teacher_disagreement':teach != greedy,'current_return_error':current-target,
                'teacher_action_residual':costs[teach]-current,'greedy_action_residual':costs[greedy]-current,
                'predicted_switching_advantage':advantage,'switching_contributions_by_hit':contributions}

    def compute(self, plan):
        metadata = {}
        for split in SPLITS:
            rows = self.lines(self.worker_dir/f'{split}-rows.jsonl')
            for i in range(8):
                r = next(rows); episode, seed, total = SELECTED[split]
                require(r['row_index'] == r['prefix_index'] == r['public']['step'] == i
                        and (r['episode_id'], r['seed'], r['total_steps']) == (episode, seed, total)
                        and r['regime'] == 'lambda3' and r['initial_hit'] == 1 and r['stage'] == split
                        and r['target'] == (total-i)/64 and not public(r['public'])['done'], 'fixed metadata identity')
                metadata[split,i] = r
            rows.close()
        teachers = self.teachers(metadata)
        parity = list(self.lines(self.worker_dir/'parity.jsonl'))
        saved = list(self.lines(self.args.run/'rows.jsonl'))
        order = [(f'{family}@{seed}', split, i) for seed in SEEDS for family in FAMILIES for split in SPLITS for i in range(8)]
        require(len(parity) == len(saved) == len(order) == 144, 'all144 complete rows')
        rebuilt = []
        for p, row, key in zip(parity, saved, order, strict=True):
            self.equal((p['fit_id'],p['split'],p['row_index']), key, 'ordered fit/split/index')
            m = metadata[key[1:]]; teacher = teachers[m['episode_id'],m['prefix_index']+1]
            self.equal(row['saved_parity'], p, 'copied parity exact canonical bytes')
            self.equal(row['metadata'], m, 'copied metadata exact canonical bytes')
            self.equal(row['teacher_transition'], teacher, 'original teacher transition')
            self.equal((p['episode_id'],p['prefix_index']), (m['episode_id'],m['prefix_index']), 'parity current row identity')
            self.equal(teacher['public_before'], m['public'], 'current public prefix')
            self.equal(teacher['posterior_before'], m['posterior'], 'current posterior witness')
            self.equal(p['posterior_sha256'], m['posterior']['sha256'], 'parity posterior')
            self.equal(p['allowed_actions'], m['public']['valid_actions'], 'parity eligibility')
            r = self.row(p,m,teacher); self.compare(row,r,'row'); rebuilt.append(r)
        panels, families = {}, {}
        mean = lambda v: sum(v)/len(v)
        for seed in SEEDS:
            for family in FAMILIES:
                fit = f'{family}@{seed}'; panels[fit] = {}
                for split in SPLITS:
                    group = [r for r in rebuilt if r['fit_id'] == fit and r['split'] == split]
                    require(len(group) == 8, 'complete eight-prefix panel')
                    stats = {}
                    for metric in METRICS:
                        v = [r[metric] for r in group]
                        stats[metric] = {'signed_mean':mean(v),'mean_absolute':mean([abs(x) for x in v]),
                                         'rms':math.sqrt(mean([x*x for x in v]))}
                    panels[fit][split] = {'prefixes':8,'unique_episodes':1,'statistics':stats,
                        'mean_switching_advantage':mean([r['predicted_switching_advantage'] for r in group]),
                        'teacher_disagreement_count':sum(r['teacher_disagreement'] for r in group)}
        for family in FAMILIES:
            families[family] = {}
            for split in SPLITS:
                group = [panels[f'{family}@{seed}'][split] for seed in SEEDS]
                families[family][split] = {'seeds':list(SEEDS),'prefixes_per_seed':8,
                    'statistics':{metric:{stat:mean([p['statistics'][metric][stat] for p in group])
                        for stat in ('signed_mean','mean_absolute','rms')} for metric in METRICS},
                    'mean_switching_advantage':mean([p['mean_switching_advantage'] for p in group]),
                    'mean_teacher_disagreement_count':mean([p['teacher_disagreement_count'] for p in group])}
        reported = read(self.args.run/'summary.json')
        self.compare(reported['fit_split'],panels,'18 fit/split panels')
        self.compare(reported['family_means'],families,'six family panels')
        for key,value in {'model_prefix_rows':144,'fits':9,'unique_prefixes':16,'unique_episodes':2,
                          'complete_validation_states':1109,'selected_validation_states':8,'new_efficacy_gate':None,
                          'original_decisions_changed':False,'original_pilot_continuation':False,
                          'inputs':plan['inputs'],'version':'otto-return-consistency-v2'}.items():
            self.equal(reported[key],value,'summary '+key)
        coverage = {split:{'episode_id':v[0],'total_steps':v[2],'regime':'lambda3','initial_hit':1,'prefix_indices':list(range(8))}
                    for split,v in SELECTED.items()}
        self.equal(reported['coverage'],coverage,'actual two-episode scope')
        return {'version':self.receipt['version'],'agreement':True,'model_prefix_rows':144,'branch_records':2304,
                'fit_split_panels':18,'family_panels':6,'unique_teacher_joins':16,'coverage':coverage,
                'fit_split':panels,'family_means':families,'scope':SCOPE,'original_decisions_changed':False}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists() and not any(p.is_symlink() for p in self.out.parents), 'exclusive output')
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            require(digest(CLOCK)['sha256'] == CLOCK_PIN, 'qualified clock before evidence')
            spec = importlib.util.spec_from_file_location('_consistency_independent_clock', CLOCK)
            module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
            self.clock = module.SuspendClock(); self.start = self.clock.now_ns()
            def alarm(signum, frame):
                raise TimeoutError('awake emergency timer')
            signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, LIMITS['seconds'])
            own = digest(Path(__file__), self.check)
            require(own['sha256'] == self.args.source_sha256, 'external held audit source')
            self.receipt.update(source=own, python=platform.python_version(), request={k:str(v) for k,v in vars(self.args).items()})
            write(self.out/'started.json', self.receipt)
            plan, done = self.authenticate()
            summary = self.compute(plan)
            self.equal(digest(self.args.plan,self.check)['sha256'],PLAN_PIN,'unchanged plan')
            self.equal(digest(self.args.run/'receipt.json',self.check)['sha256'],RECEIPT_PIN,'unchanged receipt')
            self.manifest(self.args.run,done,3)
            summary.update(comparisons=self.comparisons,maximum_difference=self.maximum_difference)
            write(self.out/'summary.json',summary); self.check()
            files = {p.name:digest(p,self.check) for p in (self.out/'started.json',self.out/'summary.json')}
            finished = self.clock.now_ns()
            self.receipt.update(status='completed',agreement=True,comparisons=self.comparisons,
                maximum_difference=self.maximum_difference,clock_backend=self.clock.backend,started_ns=self.start,
                finished_ns=finished,wall_seconds=(finished-self.start)/1e9,files=files,
                model_prefix_rows=144,branch_records=2304,fit_split_panels=18,family_panels=6,unique_teacher_joins=16)
            write(self.out/'receipt.json',self.receipt); self.check()
            return 0
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL,0)
            self.receipt.update(status='failed',agreement=False,error=repr(error),traceback=traceback.format_exc())
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'invalid-completed-receipt.json')
                write(self.out/'failed.json',self.receipt)
            except Exception as secondary:  # Failure preservation must not replace the original exception.
                error.add_note('Could not preserve secondary receipt: '+repr(secondary))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('plan','run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('plan-sha256','receipt-sha256','source-sha256'):
        parser.add_argument('--'+name,required=True)
    raise SystemExit(Audit(parser.parse_args()).execute())
