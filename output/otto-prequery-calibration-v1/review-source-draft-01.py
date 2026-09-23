"""Bounded prospective metadata review. No scientific imports or payload reads."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/otto-prequery-calibration-v1'
CELLS = {'innovation_aux': ('innovation', 'query_aux'), 'innovation_mse': ('innovation', 'mse'),
         'innovation_gru_mse': ('innovation_gru', 'mse'), 'innovation_gru_aux': ('innovation_gru', 'query_aux')}
SEEDS = (275000001, 275000002, 275000003)
PREFIX = 'src/openjev/research/'
TRAINER = 'scripts/train_otto_prequery_calibration.py'
CAPACITY = 'scripts/qualify_otto_prequery_capacity.py'
AUDITOR = 'scripts/audit_otto_prequery_calibration.py'
COLLECTOR = 'scripts/collect_otto_prequery_calibration.py'
ROLES = {'seed_review', 'collection_plan', 'components_qualification', 'metrics_qualification',
         'audit_qualification', 'training_qualification', 'fit_audit_qualification',
         'capacity_plan', 'capacity_receipt', 'capacity_terminal'}


class Review:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.checked, self.checks = {}, 0
        self.started = time.time_ns()
        self.tick = time.monotonic()

    def require(self, ok, label):
        self.checks += 1
        if not ok:
            raise ValueError(label)
        if time.monotonic() - self.tick > 120:
            raise TimeoutError('bounded 120-second metadata review')

    def path(self, value):
        p = Path(value)
        p = p if p.is_absolute() else ROOT / p
        self.require(p.is_file() and p.is_relative_to(ROOT) and '..' not in p.parts
                     and not any(q.is_symlink() for q in (p, *p.parents)), 'contained regular metadata/source')
        self.require(p.suffix not in ('.npz', '.npy', '.h5', '.hdf5', '.gz', '.pkl', '.pt'),
                     'no empirical arrays, model files or compressed journals')
        return p

    def descriptor(self, value, expected=None):
        p = self.path(value)
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        result = {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}
        if isinstance(expected, str):
            self.require(result['sha256'] == expected, 'source/external hash ' + result['path'])
        elif expected is not None:
            self.require(all(result[k] == expected[k] for k in ('bytes', 'sha256')), 'descriptor ' + result['path'])
        self.checked[result['path']] = result
        return result

    def read(self, value):
        self.require(self.path(value).suffix == '.json', 'JSON metadata only')
        return json.loads(self.path(value).read_text())

    def write(self, name, value):
        with (self.out / name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

    def closed(self, value, receipt):
        directory = self.path(value).parent
        self.require({p.name for p in directory.iterdir()} == set(receipt['files']) | {'receipt.json'},
                     'closed metadata inventory ' + directory.name)
        for name, pin in receipt['files'].items():
            self.require(Path(name).name == name, 'flat declared payload')
            self.descriptor(directory / name, pin)

    def joined_sources(self, mapping):
        for name, pin in mapping.items():
            self.require(self.sources.get(name) == pin, 'source union consistency ' + name)

    def qualification(self, role):
        item = self.inputs[role]
        receipt = self.read(item['path'])
        self.require(receipt['status'] == 'passed' and receipt['sources_before'] == receipt['sources_after'],
                     'original successful unchanged qualification ' + role)
        self.joined_sources(receipt['sources_after'])
        self.closed(item['path'], receipt)
        directory = self.path(item['path']).parent
        passed, commands = 0, []
        self.require(bool(receipt['results']), 'qualification contains actual commands')
        for result in receipt['results']:
            self.require(type(result['exit_code']) is int and result['exit_code'] == 0
                         and result.get('timed_out') is False and result.get('group_absent') is True
                         and result.get('reaped') is True, 'original command successful and closed')
            self.require(result['started_unix_ns'] <= result['finished_unix_ns']
                         and math.isfinite(result['elapsed_seconds']) and result['elapsed_seconds'] >= 0,
                         'recorded command timing')
            log = result['log']
            self.require(log in receipt['files'], 'original bound command log')
            text = (directory / log).read_text()
            command = result['command']
            count = 0
            if 'pytest' in command:
                matches = re.findall(r'(\d+) passed in ', text)
                self.require(len(matches) == 1 and ' failed' not in text and 'ERROR' not in text,
                             'one recorded complete pytest count')
                count = int(matches[0]); passed += count
            else:
                self.require(Path(command[0]).name == 'ruff' and 'All checks passed!' in text,
                             'recorded static lint success')
            commands.append({'command': command, 'passed_cases': count, 'log': self.descriptor(directory / log)})
        return {'receipt': item, 'passed_cases': passed, 'commands': commands,
                'sources': receipt['sources_after']}

    def source_api(self):
        """Inspect source syntax only; no module imports or execution."""
        def assignments(name):
            tree = ast.parse(self.path(name).read_text())
            return {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}

        trainer, auditor = assignments(TRAINER), assignments(AUDITOR)
        metrics = assignments(PREFIX + 'otto_prequery_metrics.py')
        capacity = assignments(CAPACITY)
        def static_set(node):
            if isinstance(node, ast.Name):
                return static_set(trainer[node.id])
            if isinstance(node, ast.Set):
                return {static_set(value) for value in node.elts}
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                return static_set(node.left) | static_set(node.right)
            return ast.literal_eval(node)
        self.trainer_new = static_set(trainer['NEW'])
        for mapping in (trainer, auditor, metrics):
            name = 'KINDS' if 'KINDS' in mapping else 'FAMILIES'
            self.require(ast.literal_eval(mapping[name]) == tuple(CELLS)
                         and ast.literal_eval(mapping['SEEDS']) == SEEDS, 'same four cells and three fit seeds')
        self.require(ast.literal_eval(trainer['CELLS']) == ast.literal_eval(auditor['CELLS'])
                     == ast.literal_eval(capacity['CELLS']) == CELLS, 'same architecture/objective routing')

        class Normalize(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.copy_location(ast.Name(id='KINDS' if node.id == 'FAMILIES' else node.id,
                                                  ctx=node.ctx), node)
        left, right = (ast.dump(Normalize().visit(mapping['CONFIG']), include_attributes=False)
                       for mapping in (trainer, auditor))
        self.require(left == right, 'independent auditor exact training CONFIG syntax')
        config = trainer['CONFIG']
        fields = {ast.literal_eval(key): value for key, value in zip(config.keys, config.values, strict=True)}
        for name, expected in {'required_conditions': 55, 'epochs': 80, 'batch_episodes': 6, 'chunk': 32,
                               'scale': 64., 'prior_coefficient': 1., 'training_episodes': 54,
                               'validation_episodes': 36, 'selection_seed_start': 276000001}.items():
            self.require(ast.literal_eval(fields[name]) == expected, 'fixed training declaration ' + name)
        self.require(ast.literal_eval(assignments(PREFIX + 'otto_prequery_scores.py')['KINDS'])
                     == ('innovation', 'innovation_gru'), 'two unchanged underlying architectures')
        self.require(ast.literal_eval(assignments(PREFIX + 'otto_prequery_loss.py')['OBJECTIVES'])
                     == ('mse', 'query_aux'), 'two objective branches')
        self.require(ast.literal_eval(assignments(PREFIX + 'otto_prequery_data.py')['SELECTION_START'])
                     == 276000001, 'selection namespace matches projector')
        self.require(ast.literal_eval(trainer['ROLES']) == ('collection_plan', 'collection_receipt',
            'collection_terminal', 'engineering', 'capacity_plan', 'capacity_receipt', 'capacity_terminal'),
            'seven prospective original fit roles')
        return {'cells': {k: {'architecture': v[0], 'objective': v[1]} for k, v in CELLS.items()},
                'fit_seeds': list(SEEDS), 'fits': 12, 'updates_per_fit': 720, 'updates_total': 8640,
                'required_conditions': 1+6+12+18+12+4+2, 'training_payloads': 49,
                'collection_payloads': 18, 'saved_prediction_files': 25,
                'scope': 'Literal configuration and declared API joins; algorithms separately peer-reviewed and qualified.'}

    def capacity(self):
        plan = self.read(self.inputs['capacity_plan']['path'])
        worker = self.read(self.inputs['capacity_receipt']['path'])
        terminal = self.read(self.inputs['capacity_terminal']['path'])
        directory = self.path(self.inputs['capacity_receipt']['path']).parent
        self.require(plan['version'] == worker['version'] == 'otto-prequery-capacity-v1'
                     and plan['status'] == 'frozen_before_synthetic_work' and worker['status'] == 'completed'
                     and worker['complete'] is worker['admitted'] is True
                     and worker['pending'] is worker['pending_emission'] is None
                     and worker['completed_families'] == list(CELLS) and worker['optimizer_updates'] == 4,
                     'complete admitted four-cell original capacity')
        self.require(worker['sources'] == plan['sources'] and worker['engineering'] == plan['engineering']
                     and worker['plan_sha256'] == self.inputs['capacity_plan']['sha256'], 'capacity frozen identities')
        self.joined_sources(plan['sources']); self.closed(self.inputs['capacity_receipt']['path'], worker)
        self.descriptor(plan['engineering']['path'], plan['engineering'])
        command = list(terminal['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        self.require(command[:3] == [str(ROOT / '.venv/bin/python'), str(ROOT / CAPACITY), 'run']
                     and len(command) == 11 and len(set(command[3::2])) == 4, 'canonical actual capacity command')
        options = dict(zip(command[3::2], command[4::2], strict=True))
        self.require(set(options) == {'--plan', '--plan-sha256', '--supervision', '--output'}
                     and options['--plan'] == str(self.path(self.inputs['capacity_plan']['path']))
                     and options['--plan-sha256'] == worker['plan_sha256']
                     and options['--output'] == str(directory), 'original capacity option bindings')
        self.descriptor(options['--supervision'], worker['supervision_sha256'])
        launch = self.read(options['--supervision'])
        self.require(all(terminal.get(k) == v for k, v in launch.items())
                     and launch['pid'] == launch['pgid'] and launch['pid'] != launch['parent_pid']
                     and launch['cwd'] == str(ROOT) and launch['cap_seconds'] == 120,
                     'original launch and terminal process identity')
        self.require(terminal['status'] == 'completed' and terminal['returncode'] == 0
                     and terminal['timed_out'] is False and terminal['error'] is terminal['clock_error'] is None
                     and terminal['group_absent'] is terminal['cleanup']['group_absent'] is True
                     and terminal['cleanup']['reaped'] is True and terminal['cleanup']['errors'] == [],
                     'original successful parent and complete cleanup')
        self.require(launch['clock_source_sha256'] == self.sources[PREFIX + 'suspend_clock.py']
                     and launch['watchdog_sha256'] == self.sources['scripts/supervise_dialogue_observation_v2.py']
                     and launch['deadline_ns'] == launch['started_ns'] + 120*10**9
                     and launch['started_ns'] <= worker['started_ns'] <= worker['finished_ns']
                     <= terminal['finished_ns'] <= launch['deadline_ns'], 'original capacity deadline')
        self.require(terminal['elapsed_ns'] == terminal['finished_ns']-terminal['started_ns']
                     and terminal['wall_seconds'] == terminal['elapsed_ns']/1e9
                     and worker['wall_seconds'] == (worker['finished_ns']-worker['started_ns'])/1e9,
                     'physical original clock intervals')
        self.require(self.read(directory / 'started.json') == {'launch': launch, 'started_ns': worker['started_ns']},
                     'saved worker original launch')
        self.require(worker['limits'] == plan['limits'] == {'seconds': 120, 'rss_bytes': 4*1024**3,
                                                           'output_bytes': 128*1024**2}
                     and 0 < worker['peak_rss_bytes'] <= 4*1024**3
                     and sum(d['bytes'] for d in worker['files'].values())
                     + self.path(self.inputs['capacity_receipt']['path']).stat().st_size <= 128*1024**2,
                     'capacity complete resource accounting')
        runtime = self.read(directory / 'runtime.json')
        self.require(all(runtime[k] == v for k, v in plan['runtime'].items()) and runtime['threads'] == 1
                     and runtime['interop_threads'] == 1 and runtime['cuda_used'] is runtime['mps_used'] is False,
                     'original CPU1 runtime metadata')
        summary = self.read(directory / 'summary.json')
        rows = summary['families']
        self.require([r['family'] for r in rows] == list(CELLS), 'all capacity cells in declared order')
        for row in rows:
            architecture, objective = CELLS[row['family']]
            self.require(row['architecture'] == architecture and row['objective'] == objective
                         and row['forward_chunks'] == row['chunks'] == 69 and row['forward_rows'] == 6*2188
                         and row['prior_rows'] == 6*546 and row['optimizer_updates'] == row['optimizer_step'] == 1
                         and row['backward_chunks'] + row['no_grad_chunks'] == 69
                         and (row['backward_chunks'] == 69 if objective == 'query_aux' else row['prior_loss'] == 0)
                         and any(n.startswith('output.') for n in row['changed_parameters'])
                         and math.isfinite(row['batch_seconds']) and row['batch_seconds'] > 0,
                         'complete synthetic actual batch and objective accounting')
        seconds = math.fsum(r['batch_seconds'] for r in rows)
        projected = 1.5*3*720*seconds + 120
        self.require(summary['sum_batch_seconds'] == seconds and summary['projected_seconds'] == projected
                     and projected <= summary['threshold_seconds'] == plan['configuration']['admission_seconds'] == 8100.
                     and plan['configuration']['training_cap_seconds'] == 10800
                     and summary['technical_complete'] is summary['admitted'] is True,
                     'unchanged prospective capacity admission')
        self.require(worker['new_teacher_calls'] == worker['new_native_calls'] == worker['empirical_payloads_read'] == 0,
                     'no empirical capacity inputs or teacher/native calls')
        return {'worker_seconds': worker['wall_seconds'], 'parent_seconds': terminal['wall_seconds'],
                'batch_seconds': {r['family']: r['batch_seconds'] for r in rows},
                'projected_seconds': projected, 'admission_seconds': 8100., 'training_cap_seconds': 10800,
                'original_parent_successful': True, 'admitted': True}

    def main(self):
        frozen = self.descriptor(self.args.freeze, self.args.freeze_sha256)
        manifest = self.read(self.args.freeze)
        self.sources, self.inputs = manifest['sources'], manifest['inputs']
        self.require(manifest['status'] == 'frozen_before_collection' and set(self.inputs) == ROLES
                     and manifest['capacity_admitted'] is True and manifest['no_retry_or_replacement_seeds'] is True
                     and manifest['requires_original_supervisors'] is True
                     and manifest['empirical_arrays_decoded'] == manifest['native_calls'] == manifest['teacher_calls'] == 0,
                     'complete prospective freeze state')
        self.require(set(manifest['prospective_outputs']) == {'collection','training','audit'}, 'three prospective phases')
        self.require(all(not (ROOT / value).exists() for value in manifest['prospective_outputs'].values()),
                     'no empirical allocation started before review')
        for name, pin in self.sources.items():
            self.descriptor(name, pin)
        for item in self.inputs.values():
            self.descriptor(item['path'], item)
        collection = self.read(self.inputs['collection_plan']['path'])
        self.require(collection['status'] == 'frozen_before_collection'
                     and collection['version'] == 'otto-prequery-calibration-collection-v1', 'new frozen collection')
        self.joined_sources(collection['sources'])
        for descriptor in collection['inputs'].values():
            self.descriptor(descriptor['path'], descriptor)
        inherited = self.read(collection['inputs']['collection_plan']['path'])
        self.joined_sources(inherited['sources'])
        self.require(collection['native_inputs'] == inherited['native_inputs'], 'unchanged native descriptors without opening arrays')
        self.require(collection['limits'] == {'native_seconds':7200,'rss_bytes':4*1024**3,'output_bytes':2*1024**3},
                     'full fresh collection allocation')
        cohort, case_index = [], 0
        arms = ('analytic','neural','period4_hold')
        for stage, size, starts in (('train',9,(271000001,272000001)),('valid',6,(273000001,274000001))):
            for regime, first in zip(('lambda3','lambda4'),starts,strict=True):
                for case in range(size):
                    shift = case_index % 3
                    for arm in arms[shift:] + arms[:shift]:
                        cohort.append({'stage':stage,'episode_index':len(cohort),'regime':regime,'seed':first+case,
                            'case':case,'initial_hit':1+case%3,'arm':arm,'episode_id':f'{stage}:{regime}:{first+case}:{arm}'})
                    case_index += 1
        self.require(collection['cohort'] == cohort, 'all90 exact fresh case/controller identities')
        self.require(collection['call_caps']['teacher_score'] == collection['call_caps']['tensorflow_value']
                     == 18*2188 + 36*(547+8*3) + 36*2188 == 138708, 'complete teacher call cap')
        seed = self.read(self.inputs['seed_review']['path'])
        seeds = [value for first, size in ((271000001,9),(272000001,9),(273000001,6),(274000001,6),
                 (275000001,3),(276000001,54)) for value in range(first, first+size)]
        self.require(seed['status'] == 'reserved_before_run' and seed['hits'] == [] and seed['seeds'] == seeds,
                     'exact87 reserved seeds; saved scan not rerun')
        self.require(collection['inputs']['seed_review'] == self.inputs['seed_review']
                     and collection['inputs']['engineering'] == self.inputs['components_qualification'],
                     'collection admission pins join direct review inputs')
        qualifications = {role: self.qualification(role) for role in
            ('components_qualification','metrics_qualification','audit_qualification','training_qualification')}
        cases = sum(q['passed_cases'] for q in qualifications.values())
        self.require(cases == manifest['new_completed_fabricated_tests'], 'direct qualification cases counted once')
        aggregate = self.read(self.inputs['fit_audit_qualification']['path'])
        self.require(aggregate['status'] == 'passed' and aggregate['sources_before'] == aggregate['sources_after']
                     and all(r['exit_code'] == 0 for r in aggregate['results']), 'successful complete fit/audit aggregate')
        self.joined_sources(aggregate['sources_after'])
        self.closed(self.inputs['fit_audit_qualification']['path'], aggregate)
        rolemap = {'components':'components_qualification','metrics':'metrics_qualification',
                   'audit':'audit_qualification','training':'training_qualification'}
        self.require({entry['role'] for entry in aggregate['inputs']} == set(rolemap)
                     and len(aggregate['inputs']) == 4 and aggregate['direct_tests_passed'] == cases,
                     'aggregate contains all four direct qualifications once')
        for entry in aggregate.get('inputs', []):
            self.descriptor(entry['path'], entry)
            self.require(all(entry[k] == self.inputs[rolemap[entry['role']]][k] for k in ('path','bytes','sha256')),
                         'aggregate receipt joins direct qualification')
            if 'copied_receipt' in entry:
                copied = self.path(self.inputs['fit_audit_qualification']['path']).parent / entry['copied_receipt']
                self.require(copied.read_bytes() == self.path(entry['path']).read_bytes(), 'aggregate exact copied original receipt')
                original = self.read(entry['path'])
                prefix = entry['copied_receipt'].removesuffix('receipt.json')
                selected = [r for r in aggregate['results'] if r['qualification_receipt'] == entry['copied_receipt']]
                self.require(len(selected) == len(original['results']), 'all original aggregate commands')
                for old, new in zip(original['results'], selected, strict=True):
                    expected = {**old,'log':prefix+old['log'],'qualification_receipt':entry['copied_receipt']}
                    self.require(new == expected and (copied.parent/new['log']).read_bytes()
                                 == (self.path(entry['path']).parent/old['log']).read_bytes(),
                                 'aggregate exact original command and log bytes')
        capacity = self.capacity()
        api = self.source_api()
        required = {TRAINER,AUDITOR,COLLECTOR,CAPACITY,
            'tests/test_train_otto_prequery_calibration.py','tests/test_audit_otto_prequery_calibration.py',
            'tests/test_collect_otto_prequery_calibration.py','tests/test_qualify_otto_prequery_capacity.py',
            'research/otto-prequery-calibration-protocol.md',str(Path(__file__).relative_to(ROOT))}
        for stem in ('scores','data','loss','metrics'):
            required.update({PREFIX+f'otto_prequery_{stem}.py',f'tests/test_otto_prequery_{stem}.py'})
        self.require(required <= self.sources.keys(), 'all current implementation/test/protocol/review sources held')
        capacity_sources = self.read(self.inputs['capacity_plan']['path'])['sources']
        expected_sources = set(collection['sources']) | set(capacity_sources) | set(aggregate['sources_after'])
        expected_sources |= self.trainer_new | {str(Path(__file__).relative_to(ROOT))}
        self.require(set(self.sources) == expected_sources, 'exact prospective source union without omitted dependencies')
        for name,pin in self.sources.items():
            self.descriptor(name,pin)
        for item in self.inputs.values():
            self.descriptor(item['path'],item)
        self.descriptor(self.args.freeze,self.args.freeze_sha256)
        self.require(all(not (ROOT / value).exists() for value in manifest['prospective_outputs'].values()),
                     'review closes before scientific phases')
        return {'version':'otto-prequery-precollection-review-v1','status':'passed','clear':True,
            'manifest':frozen,'checks_passed':self.checks,'source_count':len(self.sources),'input_count':len(self.inputs),
            'qualification_cases':{k:q['passed_cases'] for k,q in qualifications.items()},
            'new_completed_fabricated_tests':cases,'qualifications':qualifications,'capacity':capacity,'api':api,
            'qualification_count_scope':'Direct newly recorded passed case executions, including unchanged helper tests; aggregate copies not recounted.',
            'scope':'Saved metadata/source-byte/qualification-log/original process review only; no source module execution.',
            'limits':['No empirical arrays or model files opened; numerical native input descriptors inherited unchanged.',
                      'Saved test execution and synthetic timing truth inherited; neither rerun.',
                      'Source algorithms separately peer-reviewed; this verifies frozen configuration and API joins.',
                      'Capacity projection is prospective planning, not a runtime guarantee or scientific efficacy.'],
            'empirical_arrays_decoded':0,'model_calls':0,'native_calls':0,'teacher_calls':0,'optimizer_calls':0,
            'started_unix_ns':self.started,'finished_unix_ns':time.time_ns()}

    def execute(self):
        self.require(self.out.is_absolute() and self.out.is_relative_to(BASE) and '..' not in self.out.parts,
                     'exclusive contained review output')
        self.out.mkdir(exist_ok=False)
        try:
            receipt = self.main()
            self.write('checked-files.json',dict(sorted(self.checked.items())))
            receipt['files'] = {'checked-files.json':self.descriptor(self.out/'checked-files.json')}
            receipt['review_source'] = self.descriptor(__file__)
            receipt['finished_unix_ns'] = time.time_ns()
            self.write('receipt.json',receipt)
            print(json.dumps({'status':'passed','receipt':self.descriptor(self.out/'receipt.json')},sort_keys=True),flush=True)
        except BaseException as error:
            try:
                if (self.out/'receipt.json').exists():
                    (self.out/'receipt.json').rename(self.out/'receipt.invalid.json')
                self.write('receipt.json',{'status':'failed','clear':False,'error':repr(error),
                    'checks_completed':self.checks,'started_unix_ns':self.started,'finished_unix_ns':time.time_ns(),
                    'scope':'Metadata review only; no empirical or scientific execution.'})
            except BaseException as publication:  # noqa: BLE001 - retain the primary review failure
                error.add_note('Failure receipt publication also failed: '+repr(publication))
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    Review(parser.parse_args()).execute()
