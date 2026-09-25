"""Publish closed observer development evidence from audited scalars and opaque bytes.

No array decoder, model inference, fitting, scoring or timing is called here.
The four measured target NPZs and all external recordings/raw media stay external.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import audit_robot_observer_study as auditor

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-observer-publication-v1'
STUDY = 'output/robot-observer-study-v1'
ENGINEERING = 'output/robot-observer-engineering-v1'
AUDIT = 'output/robot-observer-audit-v1/audit.json'
OUTPUT = 'research/robot-observer-results'
DELIVERY_SOURCES = ('scripts/publish_robot_observer_study.py', 'tests/test_publish_robot_observer_study.py')
PUBLICATION_ENGINEERING = 'output/robot-observer-publication-engineering-v1'
LAUNCHER = 'scripts/launch_robot_observer_study.py'
FAMILIES = (*auditor.ARMS, *auditor.REFS)
LABELS = ('Local affine', 'Temporal affine', 'Observer learned', 'Last two', 'Observer fixed', 'Observer zero',
          'Joint local', 'Joint temporal', 'Dense bounded', 'Dense unbounded', 'GRU32', 'Legacy instant',
          'GRU10', 'Causal ridge 1', 'Causal ridge 100', 'Linear AR2', 'Persistence')
require, read, descriptor = auditor.require, auditor.read, auditor.descriptor


def safe_path(folder, name):
    part = Path(name)
    require(type(name) is str and name and not part.is_absolute() and '..' not in part.parts
            and str(part) == name and name != '.', 'safe relative evidence path')
    path = folder / part
    require(not any(p.is_symlink() for p in (folder, path, *path.parents)), 'no evidence symlink')
    return path


def regular_tree(folder, names):
    require(folder.is_dir() and not folder.is_symlink(), 'regular evidence directory')
    paths = list(folder.rglob('*'))
    require(not any(p.is_symlink() for p in paths)
            and {str(p.relative_to(folder)) for p in paths if p.is_file()} == set(names), 'complete evidence file roster')
    for name in names:
        safe_path(folder, name)


def study_roster(inventory, plan, attempts):
    """Exact dynamic child roster; external recordings never enter delivery."""
    require(set(plan['sources']) == set(auditor.SOURCES), 'frozen45 source roster')
    required = {'registration.json', 'runtime.json', 'fits.json', 'checkpoint-barrier.json', 'parameter-checks.json',
                'prediction-attempts.json', 'resources.json', 'results.json', 'parent-parity.json'}
    required |= {'sources/'+name for name in plan['sources']}
    parent = {name for name in plan['inputs'] if name.startswith('parent/')}
    require(len(parent) == 34, 'all34 inherited payloads')
    required |= parent
    identities = auditor.identities()
    for number, identity in enumerate(identities[:48], 1):
        required.add(identity['key']+'/final.npz')
        if identity['origin'] == 'fresh':
            required |= {identity['key']+'/'+name for name in ('initial.npz', 'optimizer.npz', 'trace.json', 'fit-receipt.json')}
            required.add(f'completed-fit-{number:02d}.json')
    required |= {f'completed-prediction-{i:03d}.json' for i in range(1, 209)}
    required |= {f'completed-timing-{i:02d}.json' for i in range(1, 44)}
    excluded = {'dev-windows-'+name+'.npz' for name in auditor.EXPOSED}
    required |= excluded
    expected = {(name, row['key'], row['arm'], row['seed'], row['learning_rate'])
                for name in auditor.EXPOSED for row in identities}
    actual = [(r['recording'], r['fit_key'], r['arm'], r['seed'], r['learning_rate']) for r in attempts]
    require(len(actual) == len(set(actual)) == 208 and set(actual) == expected, 'all208 audited forecast attempts')
    for row in attempts:
        name = f"prediction-{row['recording']}-{row['fit_key']}.npz"
        require(row['status'] in ('PASS', 'FAILED') and len(row['errors']) == 2, 'forecast attempt status')
        if row['prediction_file'] is None:
            require(row['status'] == 'FAILED' and all(error is not None for error in row['errors']),
                    'absent bank is an explicit numerical failure')
        else:
            require(row['prediction_file'] == name, 'audited forecast bank identity')
            required.add(name)
    require(set(inventory) == required, 'exact published study roster')
    for name in required:
        safe_path(Path('/opaque-evidence'), name)
    return {name: inventory[name] for name in sorted(excluded)}


def qualification_attempts(engineering):
    attempts = sorted(engineering.glob('qualification-*'))
    require(attempts and [p.name for p in attempts] == [f'qualification-{i:02d}' for i in range(1, len(attempts)+1)]
            and all(p.is_dir() and not p.is_symlink() for p in attempts), 'all consecutive original qualifications')
    return attempts


def qualification_files(engineering, plan):
    """All original attempts and snapshots, including any failed commands."""
    commands = [['.venv/bin/ruff', 'check', *[p for p in auditor.QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_observer_initializer.py',
                 'tests/test_robot_observer_study.py', 'tests/test_audit_robot_observer_study.py']]
    attempts = qualification_attempts(engineering)
    files = {}
    for folder in attempts:
        launch, receipt = read(folder/'launch.json'), read(folder/'receipt.json')
        require(set(receipt) == set(launch) | {'status', 'sources_unchanged'} and launch['commands'] == []
                and all(receipt[k] == v for k, v in launch.items() if k != 'commands')
                and set(receipt['sources']) == set(auditor.SOURCES) and receipt['sources_unchanged'] is True
                and receipt['thread_env'] == dict.fromkeys(auditor.THREADS, '1')
                and receipt['command_cap_seconds'] == 180, 'qualification original launch/source closure')
        rows = receipt['commands']; codes = [r['returncode'] for r in rows]
        require(len(rows) == 2 and [r['command'] for r in rows] == commands and all(type(c) is int for c in codes)
                and receipt['status'] == ('PASS' if codes == [0, 0] else 'FAILED'), 'original qualification outcomes')
        names = {'launch.json', 'receipt.json', 'qualification-helper.py'}
        descriptor(folder/'qualification-helper.py')
        for name, expected in {**receipt['sources'], LAUNCHER: receipt['launcher']}.items():
            relative = 'sources/'+name
            require(descriptor(safe_path(folder, relative)) == expected, 'qualification snapshot bytes')
            names.add(relative)
        for i, row in enumerate(rows, 1):
            name, process = f'command-{i:02d}.log', f'process-{i:02d}.json'
            require(read(folder/process) == row and Path(row['log']) == folder/name
                    and descriptor(folder/name)['sha256'] == row['sha256'] and type(row['seconds']) in (float, int)
                    and math.isfinite(row['seconds']) and row['seconds'] > 0, 'original qualification log/process/duration')
            names |= {name, process}
        if folder == attempts[-1]:
            require(receipt['status'] == 'PASS' and receipt['sources'] == plan['sources'] and receipt['launcher'] == plan['launcher']
                    and Path(plan['qualification']['path']) == folder/'receipt.json'
                    and descriptor(folder/'receipt.json') == {k: plan['qualification'][k] for k in ('sha256', 'bytes')},
                    'registered final qualification')
        regular_tree(folder, names)
        for name in names:
            files[f'engineering/{folder.name}/{name}'] = folder/name
    return files


def publication_qualification_files():
    root = ROOT / PUBLICATION_ENGINEERING
    attempts = qualification_attempts(root)
    commands = [['.venv/bin/ruff', 'check', *DELIVERY_SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', DELIVERY_SOURCES[1]]]
    result = {}
    for folder in attempts:
        pre, receipt = read(folder/'preflight.json'), read(folder/'receipt.json')
        require(set(pre['sources']) == set(DELIVERY_SOURCES) and pre['sources'] == receipt['sources']
                and receipt['sources_unchanged'] is True and receipt['thread_env'] == dict.fromkeys(auditor.THREADS, '1')
                and pre['commands'] == commands, 'publication qualification source/command closure')
        rows = receipt['commands']; codes = [r['returncode'] for r in rows]
        require(0 < len(rows) <= 2 and [r['command'] for r in rows] == commands[:len(rows)]
                and all(type(code) is int for code in codes) and all(code == 0 for code in codes[:-1])
                and ((receipt['status'] == 'PASS' and codes == [0, 0]) or (receipt['status'] == 'FAIL' and codes[-1] != 0)),
                'original publication qualification outcome and executed prefix')
        names = {'preflight.json', 'receipt.json'}
        for name, expected in receipt['sources'].items():
            require(descriptor(safe_path(folder/'sources', name)) == expected, 'publication qualification snapshot')
            if folder == attempts[-1]:
                require(receipt['status'] == 'PASS' and descriptor(ROOT/name) == expected, 'current publisher/test qualification')
            names.add('sources/'+name)
        for i, row in enumerate(rows, 1):
            name = f'command-{i:02d}.log'
            require(Path(row['log']) == folder/name and descriptor(folder/name)['sha256'] == row['sha256']
                    and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and 0 < row['seconds'] <= 180,
                    'publication qualification original log/duration')
            names.add(name)
        regular_tree(folder, names)
        for name in names:
            result['publication-qualification/'+folder.name+'/'+name] = folder/name
    return result


def authenticate(study, audit_path, engineering):
    """Original terminal admission first; then scalar-only audit/result joins."""
    study, audit_path, engineering = [Path(p).resolve() for p in (study, audit_path, engineering)]
    require((study, audit_path, engineering) == (ROOT / STUDY, ROOT / AUDIT, ROOT / ENGINEERING), 'canonical original evidence paths')
    process_path = engineering / 'audit-process-01.json'
    process = read(process_path)
    argv = ['.venv/bin/python', 'scripts/audit_robot_observer_study.py', '--study', STUDY,
            '--run-receipt', ENGINEERING + '/run-process-01.json', '--output', AUDIT]
    require(process['command'] == argv and type(process['returncode']) is int and process['returncode'] == 0
            and process['external_timeout'] is False and process['cap_seconds'] == 180
            and process['thread_env'] == dict.fromkeys(auditor.THREADS, '1')
            and type(process['elapsed_seconds']) in (float, int) and math.isfinite(process['elapsed_seconds'])
            and 0 < process['elapsed_seconds'] <= process['cap_seconds'], 'original successful audit process required')
    source = ROOT / 'scripts/audit_robot_observer_study.py'
    launch_path, closure_path = engineering / 'audit-launch-01.json', engineering / 'audit-closure-01.json'
    launch, closure = read(launch_path), read(closure_path)
    require(set(launch) == {'command', 'started_utc', 'thread_env', 'cap_seconds'}
            and all(process[k] == v for k, v in launch.items())
            and descriptor(engineering / 'audit-process-01.log') == process['log'], 'original audit launch/log join')
    require(closure['status'] == 'PASS' and closure['source_matches_registration'] is True
            and type(closure['scope']) is str and closure['scope'], 'explicit postterminal audit closure')
    for key, path in {'auditor': source, 'audit_output': audit_path, 'audit_manifest': audit_path.parent / 'manifest.json',
                      'process': process_path, 'launch': launch_path, 'log': engineering / 'audit-process-01.log',
                      'registration': ROOT / 'research/robot-observer-registration.json'}.items():
        require(closure[key] == auditor.pin(path), 'postterminal original byte join: ' + key)
    regular_tree(audit_path.parent, {'audit.json', 'manifest.json'})
    require(read(audit_path.parent / 'manifest.json') == {'files': {'audit.json': descriptor(audit_path)}}, 'exact audit manifest')
    plan, inputs = auditor.authenticate(study, engineering / 'run-process-01.json')
    value = read(audit_path)
    require(value['status'] == 'PASS' and value['agreement'] is True and value['study'] == str(study)
            and value['registration_sha256'] == inputs['registration']['sha256']
            and value['source_pins'] == plan['sources'] and value['inputs'] == inputs
            and value['auditor'] == auditor.pin(source), 'completed audit of these exact original inputs')
    auditor.close(value['results'], read(study / 'results.json'), 'audited saved results')
    require(value['resources'] == read(study / 'resources.json')
            and value['prediction_attempts'] == read(study / 'prediction-attempts.json'), 'audited resource/attempt evidence')
    require(value['results']['config'] == plan['config'], 'audit uses registered config')
    validate_scalars(value)
    inventory = read(study / 'manifest.json')['files']
    excluded = study_roster(inventory, plan, value['prediction_attempts'])
    qualifications = qualification_files(engineering, plan)
    qualifications.update(publication_qualification_files())
    preflight_path = engineering/'registration-preflight-01.json'
    preflight = read(preflight_path)
    require(preflight['status'] == 'PASS' and preflight['registration'] == inputs['registration']
            and preflight['qualification'] == plan['qualification']
            and [preflight[k] for k in ('sources', 'inputs', 'model_calls', 'numeric_decodes', 'optimizer_updates')]
            == [45, 45, 0, 0, 0], 'original metadata-only registration preflight')
    qualifications['engineering/registration-preflight-01.json'] = preflight_path
    pins = {str(path): descriptor(path) for path in (process_path, launch_path, closure_path, engineering / 'audit-process-01.log', audit_path,
                                                  audit_path.parent / 'manifest.json')}
    return {'plan': plan, 'inputs': inputs, 'audit': value, 'inventory': inventory, 'excluded': excluded,
            'qualification_files': {k: str(v) for k, v in qualifications.items()}, 'publication_inputs': pins}


def selected(row, selection):
    return row['arm'] not in auditor.LEARNED or row['learning_rate'] == selection['selected_rates'][row['arm']]


def validate_scalars(value):
    results, resources = value['results'], value['resources']
    require(results['version'] == 'robot-observer-study-v1' and results['config']['all_evaluation_data_exposed'] is True,
            'exposed-development scope')
    selection = results['selection']
    require(set(selection['selected_rates']) == set(auditor.LEARNED)
            and all(rate in (*auditor.RATES, None) for rate in selection['selected_rates'].values()), 'declared original DEV rate choices')
    keys = [(r['recording'], r['fit_key'], r['arm'], r['seed'], r['learning_rate'], r['horizon']) for r in results['rows']]
    expected = {(name, r['key'], r['arm'], r['seed'], r['learning_rate'], h)
                for name in auditor.EXPOSED for r in auditor.identities() for h in (64, 128)}
    require(len(keys) == len(set(keys)) == 416 and set(keys) == expected, 'all416 audited score rows')
    for row in results['rows']:
        require(row['status'] in ('PASS', 'FAILED'), 'declared score status')
        if row['status'] == 'FAILED':
            require(row['metrics'] is None and row['error'] is not None, 'failed score evidence')
        else:
            m = row['metrics']
            require(row['error'] is None and len(m['per_joint_rmse_deg']) == 6
                    and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in
                            [m['standardized_rmse'], m['standardized_sse'], m['physical_rmse_deg'], *m['per_joint_rmse_deg']]),
                    'finite audited metrics')
    require(len(resources) == 43 and all(all(row[k] == v for k, v in identity.items())
            for row, identity in zip(resources, auditor.resource_identities(selection), strict=True)), 'all43 exact resource attempts')
    for row in resources:
        require(row['status'] in ('PASS', 'FAILED', 'UNAVAILABLE'), 'resource outcome')
        if row['status'] == 'PASS':
            require(row['error'] is None and type(row['timing']['median_seconds']) in (float, int)
                    and math.isfinite(row['timing']['median_seconds']) and row['timing']['median_seconds'] > 0,
                    'positive finite saved request median')
        else:
            require(row['timing'] is None and row['error'] is not None, 'retained failed or unavailable timing')
        require(all(type(row[k]) is int and row[k] >= 0 for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes'))
                and persistent_bytes(row) > 0, 'positive actual persistent storage')
    gate = results['result']
    require(tuple(c['name'] for c in gate['conditions']) == auditor.CONDITIONS
            and all(type(c['passed']) is bool for c in gate['conditions'])
            and gate['passed'] == sum(c['passed'] for c in gate['conditions']) and gate['total'] == 5
            and gate['status'] == ('OBSERVER_DEVELOPMENT_PASS' if gate['passed'] == 5 else 'OBSERVER_DEVELOPMENT_FAIL'),
            'literal five-condition scientific status; no outcome promotion')
    require(set(gate['details']) == set(auditor.EXPOSED) and set(gate['costs']) == set(FAMILIES), 'complete family/file summary')
    for cost in gate['costs'].values():
        require(cost is None or all(type(cost[k]) in (int, float) and math.isfinite(cost[k]) and cost[k] > 0 for k in ('latency', 'bytes')),
                'positive finite complete family costs')


def persistent_bytes(row):
    return sum(row[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes'))


def write_tables(output, value):
    rows, gate = value['results']['rows'], value['results']['result']
    selection = value['results']['selection']
    fields = ['recording', 'arm', 'seed', 'learning_rate', 'horizon', 'status', 'selected', 'error_type', 'error_message',
              'standardized_rmse', 'standardized_sse', 'scalars', 'physical_rmse_deg', 'windows',
              *[f'joint_{i}_rmse_deg' for i in range(1, 7)]]
    with (output/'scores.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            result = {k: row[k] for k in fields[:6]}
            result.update(selected=selected(row, selection), error_type=(row['error'] or {}).get('type', ''),
                          error_message=(row['error'] or {}).get('message', ''))
            if row['metrics'] is not None:
                result.update({k: row['metrics'][k] for k in fields[9:14]})
                result.update(dict(zip(fields[14:], row['metrics']['per_joint_rmse_deg'], strict=True)))
            writer.writerow(result)
    fields = ['arm', 'seed', 'learning_rate', 'origin', 'status', 'parameter_bytes', 'buffer_bytes', 'state_bytes',
              'normalizer_bytes', 'persistent_bytes', 'median_seconds', 'p95_seconds', 'error']
    with (output/'costs.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in value['resources']:
            result = {k: row[k] for k in fields[:9]}
            result.update(persistent_bytes=persistent_bytes(row), error=json.dumps(row['error'], sort_keys=True))
            if row['timing'] is not None:
                result.update({k: row['timing'][k] for k in ('median_seconds', 'p95_seconds')})
            writer.writerow(result)
    fmt = lambda x: 'FAILED / incomplete' if x is None else f'{x:.6g}'
    lines = ['# Frozen-backbone observer development', '', f"**{gate['status']}**, {gate['passed']}/5 conditions.", '',
             '| Criterion | Passed |', '|---|---|']
    lines += [f"| {r['name']} | {r['passed']} |" for r in gate['conditions']]
    lines += ['', '| Family | Selected/fixed rate | File 1 RMSE | File 2 RMSE | File 3 RMSE | File 4 RMSE | Equal-file mean | Request ms | Persistent bytes |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, label in zip(FAMILIES, LABELS, strict=True):
        errors = [gate['details'][n]['means'].get(arm) for n in auditor.EXPOSED]
        cost = gate['costs'][arm]
        rate = selection['selected_rates'].get(arm, auditor.CACHED_RATES.get(arm, '-'))
        lines.append(f"| {label} | {rate} | "+' | '.join(fmt(v) for v in errors)+f" | {fmt(gate['equal_file_means'].get(arm))} | "
                     f"{fmt(None if cost is None else cost['latency']*1000)} | {'incomplete' if cost is None else cost['bytes']} |")
    lines += ['', ('All 416 H64/H128 rows (both learned rates), 208 forecast attempts and 43 cost attempts are retained. '
              'Table means and family costs are saved audited scalars. Rates use only the original two DEV files. '
              'All four files are exposed development data. Frozen backbone and inherited training costs remain historical. '
              'New training is limited to the affine head or observer gain; no model is declared novel by this result.'), '']
    (output/'table.md').write_text('\n'.join(lines))
    (output/'audited-summary.json').write_text(json.dumps({'results': value['results'], 'resources': value['resources'],
        'prediction_attempts': value['prediction_attempts'], 'counts': value['counts']}, indent=2, sort_keys=True, allow_nan=False)+'\n')


def render(output, value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, gate, selection = value['results']['rows'], value['results']['result'], value['results']['selection']
    fig, axes = plt.subplots(3, 2, figsize=(21, 18))
    colors = ['#64879b', '#8b76aa', '#c34b28', '#a1a8ae', '#c59a47', '#82927c', '#5c95a2', '#977459', '#749c84',
              '#927caf', '#b58147', '#8d8b85', '#637da0', '#ab8282', '#996c4d', '#767cae', '#6c8978']
    accuracy = [r['metrics']['standardized_rmse'] for r in rows if r['horizon'] == 128 and r['status'] == 'PASS']
    upper = max(accuracy, default=0.)*1.08 or 1.
    for axis, name in zip(list(axes.flat)[:4], auditor.EXPOSED, strict=True):
        for i, arm in enumerate(FAMILIES):
            instances = [r for r in rows if r['recording'] == name and r['arm'] == arm and r['horizon'] == 128]
            for j, row in enumerate(instances):
                x = i+(j-(len(instances)-1)/2)*.09
                if row['status'] == 'PASS':
                    chosen = selected(row, selection)
                    axis.scatter(x, row['metrics']['standardized_rmse'], color=colors[i], s=30,
                                 marker='o' if chosen else 'x', alpha=1 if chosen else .45, zorder=3)
                else:
                    axis.text(x, .015, 'failed', rotation=90, ha='center', fontsize=6, transform=axis.get_xaxis_transform())
            mean = gate['details'][name]['means'].get(arm)
            if mean is not None:
                axis.plot([i-.27, i+.27], [mean, mean], color='black', linewidth=1.5)
        axis.set_title(name.removeprefix('recording_2021_12_15_').removesuffix('.mat')+' (exposed development)')
        axis.set_ylabel('H128 standardized RMSE (lower is better)'); axis.set_ylim(0, upper)
    for axis, key, scale, label in ((axes[2, 0], 'latency', 1000, 'Complete request latency (ms)'),
                                   (axes[2, 1], 'bytes', 1, 'Persistent numeric bytes')):
        for i, arm in enumerate(FAMILIES):
            cost = gate['costs'][arm]
            if cost is not None:
                axis.plot([i-.27, i+.27], [cost[key]*scale]*2, color=colors[i], linewidth=3)
            else:
                axis.text(i, .02, 'incomplete', rotation=90, ha='center', fontsize=7, transform=axis.get_xaxis_transform())
            instances = [r for r in value['resources'] if r['arm'] == arm]
            for j, row in enumerate(instances):
                scalar = persistent_bytes(row) if key == 'bytes' else (None if row['timing'] is None else row['timing']['median_seconds']*1000)
                if scalar is not None:
                    axis.scatter(i+(j-(len(instances)-1)/2)*.15, scalar, color=colors[i], s=30, zorder=3)
        axis.set_ylabel(label+' (log scale)'); axis.set_yscale('log')
    for axis in axes.flat:
        axis.set_xticks(range(len(FAMILIES)), LABELS, rotation=55, ha='right')
        axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
    fig.suptitle(f"Frozen-backbone observer development: {gate['passed']} of 5 criteria pass (all 5 required)", fontsize=15)
    fig.text(.5, .009, 'Accuracy: all finite seed/rate points, selected circles and unselected crosses; black lines are saved selected means. '
             'Four panels share a linear scale. Costs use log scales and all instance points.\n'
             'Full request includes normalization, casting, prefix assimilation, forecast, validation and deadline checks. '
             'Same robot/day and realized future torque; no independent confirmation or control claim.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .037, 1, .97))
    fig.savefig(output/'benchmark.png', dpi=150); fig.savefig(output/'benchmark.svg'); plt.close(fig)


def publish(study, audit_path, engineering, output):
    study, audit_path, engineering, output = [Path(p).resolve() for p in (study, audit_path, engineering, output)]
    auth = authenticate(study, audit_path, engineering)
    require(output == ROOT / OUTPUT and not output.exists(), 'exclusive canonical publication folder')
    script_pin = descriptor(__file__)
    output.mkdir(parents=True, exist_ok=False)
    mapping = {}
    def copy(source, name):
        source = Path(source)
        pin = descriptor(source)
        target = safe_path(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        require(descriptor(source) == descriptor(target) == pin, 'byte-exact evidence copy')
        mapping[name] = {'path': str(source.resolve()), **pin}
    for name in (*auth['inventory'], 'manifest.json', 'receipt.json'):
        if name not in auth['excluded']:
            copy(study / name, 'study/' + name)
    copy(audit_path, 'audit.json'); copy(audit_path.parent / 'manifest.json', 'audit-manifest.json')
    for name in ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log', 'audit-launch-01.json',
                 'audit-process-01.json', 'audit-process-01.log', 'audit-closure-01.json'):
        copy(engineering / name, 'engineering/' + name)
    for name, path in sorted(auth['qualification_files'].items()):
        copy(path, name)
    for name in (*DELIVERY_SOURCES, LAUNCHER):
        copy(ROOT / name, 'delivery-source/' + name)
    for name in ('parent_audit', 'parent_publication_manifest'):
        copy(auth['plan'][name]['path'], 'parent-closure/'+name+'.json')
    write_tables(output, auth['audit']); render(output, auth['audit'])
    gate = auth['audit']['results']['result']
    (output/'README.md').write_text(
        '# Frozen-backbone observer development evidence\n\n'
        f"**{gate['status']}**, {gate['passed']}/5 frozen conditions. Evidence audit: PASS.\n\n"
        'All 48 model checkpoints, 18 fresh initial checkpoints and optimizer/trace bundles, 34 inherited payloads, '
        'all 208 forecast attempts, 416 scores, 43 cost attempts and 45 source snapshots are retained. '
        'Every original scientific and publication qualification attempt is preserved with its own sources and logs.\n\n'
        'Exactly four measured target-window NPZs and all external measured recordings/raw archives are excluded, '
        'with descriptors in manifest.json. Licensed local inputs and the '
        '[parent confirmation evidence](../robot-history-confirmation-results.md) are required for full replay. '
        'Repository licensing does not relicense [Industrial Robot measurements](https://doi.org/10.26204/data/5).\n\n'
        'The 590-parameter backbone is frozen; only new affine heads or learned observer gains are trained. '
        'Inherited jointly trained heads and recurrent models remain separate controls. '
        'Rates use only the original two DEV files; all four evaluation recordings were previously exposed. '
        'This is a development result, not renewed confirmation, a novelty claim, native speedup or control performance. '
        'The observer gain is not a Kalman gain and its latent lower coordinates are not physical velocity.\n\n'
        'Full request costs include normalization, casts, prefix assimilation, rollout, validation and deadline checks. '
        'Persistent storage includes all parameters, buffers, state and normalization; temporary workspace, Python '
        'object overhead and model loading are excluded. Historical training costs remain labeled in fits.json. '
        'Future inputs are realized measured torques; official TEST remains closed.\n\n'
        'See [scores](scores.csv), [costs](costs.csv), [table](table.md), [report](../robot-observer-results.md) and '
        '[protocol](../robot-observer-protocol.md).\n')
    require(authenticate(study, audit_path, engineering) == auth and descriptor(__file__) == script_pin, 'unchanged publication inputs/source')
    for name, item in mapping.items():
        expected = {k: item[k] for k in ('sha256', 'bytes')}
        require(descriptor(item['path']) == descriptor(output / name) == expected, 'complete copied-byte roundtrip')
    files = {str(p.relative_to(output)): descriptor(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'version': VERSION, 'files': files, 'copy_sources': mapping, 'excluded_measurement_payloads': auth['excluded'],
                'external_measurement_inputs': {k: v for k, v in auth['plan']['inputs'].items() if k.startswith('data/')},
                'parent_lineage': {k: auth['plan'][k] for k in ('parent_audit', 'parent_publication_manifest')},
                'original_inputs': auth['inputs'], 'publication_inputs': auth['publication_inputs'], 'publisher': script_pin,
                'scope': 'Saved audited scalars and opaque derived evidence; four measured target NPZs and all external recordings/raw media excluded'}
    with (output / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    receipt = {'version': VERSION, 'status': 'PASS', 'scientific_status': gate['status'], 'passed': gate['passed'], 'total': 5,
               'publisher': script_pin, 'manifest': descriptor(output / 'manifest.json'), 'files': len(files),
               'excluded_measurement_files': 4, 'model_calls': 0, 'array_decodes': 0, 'rescoring_calls': 0, 'timing_calls': 0}
    with (output / 'receipt.json').open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=ROOT / STUDY)
    parser.add_argument('--audit', type=Path, default=ROOT / AUDIT)
    parser.add_argument('--engineering', type=Path, default=ROOT / ENGINEERING)
    parser.add_argument('--output', type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.audit, args.engineering, args.output)), flush=True)
