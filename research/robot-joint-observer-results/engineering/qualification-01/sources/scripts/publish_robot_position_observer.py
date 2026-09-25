"""Publish closed position-observer scalars and opaque derived evidence.

No measurement decoder, model, backward, scoring or timing call is made here.
The parent failure and all inherited rows remain visible and separately linked.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import audit_robot_position_observer as auditor

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-position-observer-publication-v1'
STUDY = 'output/robot-position-observer-study-v1'
ENGINEERING = 'output/robot-position-observer-engineering-v1'
AUDIT = 'output/robot-position-observer-audit-v1/audit.json'
OUTPUT = 'research/robot-position-observer-results'
DELIVERY_SOURCES = ('scripts/publish_robot_position_observer.py', 'tests/test_publish_robot_position_observer.py')
PUBLICATION_ENGINEERING = 'output/robot-position-observer-publication-engineering-v1'
LAUNCHER = 'scripts/launch_robot_position_observer_study.py'
FAMILIES = auditor.DISPLAY
LABELS = ('Position learned', 'Position fixed', 'Local affine', 'Temporal affine', 'Prior observer learned',
          'Last two', 'Observer fixed', 'Observer zero', 'Joint local', 'Joint temporal', 'Dense bounded',
          'Dense unbounded', 'GRU32', 'Legacy instant', 'GRU10', 'Causal ridge 1', 'Causal ridge 100',
          'Linear AR2', 'Persistence')
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


def qualification_attempts(engineering):
    attempts = sorted(engineering.glob('qualification-*'))
    require(attempts and [p.name for p in attempts] == [f'qualification-{i:02d}' for i in range(1, len(attempts)+1)]
            and all(p.is_dir() and not p.is_symlink() for p in attempts), 'all consecutive original qualifications')
    return attempts


def qualification_files(engineering, plan):
    """All original attempts and snapshots, including any failed commands."""
    commands = [['.venv/bin/ruff', 'check', *[p for p in auditor.QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_position_observer.py',
                 'tests/test_robot_position_observer_study.py', 'tests/test_audit_robot_position_observer.py']]
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


def persistent_bytes(row):
    return sum(row[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes'))


def study_roster(inventory, plan, attempts):
    """Child-only banks and complete inherited scalar/evidence lineage."""
    require(set(plan['sources']) == set(auditor.SOURCES) and len(plan['sources']) == 54, 'all54 scientific sources')
    required = {'registration.json', 'runtime.json', 'fits.json', 'checkpoint-barrier.json', 'parameter-checks.json',
                'prediction-attempts.json', 'resources.json', 'results.json', 'probes.json'}
    required |= {'sources/'+name for name in plan['sources']}
    copied = {name for name in plan['inputs'] if not name.startswith('data/')}
    require(len(plan['inputs']) == 61 and len(copied) == 50, '50 copied evidence inputs and11 external recordings')
    required |= copied
    new = auditor.identities()
    inherited = auditor.inherited_identities(plan['config']['inherited_rates'])
    for identity in (*new, *(r for r in inherited if r['arm'] not in auditor.REFS)):
        required.add(identity['key']+'/final.npz')
    for number, identity in enumerate(new[:6], 1):
        required |= {identity['key']+'/'+name for name in ('initial.npz', 'optimizer.npz', 'trace.json',
                    'fit-receipt.json', 'diagnostics.json', 'diagnostic-summary.json', 'last-gradient.npz')}
        required.add(f'completed-fit-{number:02d}.json')
    for probe in auditor.probe_schedule():
        required |= {'probes/'+probe['key']+'/'+name for name in ('receipt.json', 'preclip.npz')}
    required |= {f'completed-prediction-{i:03d}.json' for i in range(1, 37)}
    required |= {f'completed-timing-{i:02d}.json' for i in range(1, 47)}
    excluded = {'dev-windows-'+name+'.npz' for name in auditor.EXPOSED}
    required |= excluded
    require(len(attempts) == 244, 'all244 inherited and new forecast attempts')
    for subset, identities in ((attempts[:208], auditor.parent_audit.identities()), (attempts[208:], new)):
        expected = {(name, r['key'], r['arm'], r['seed'], r['learning_rate'])
                    for name in auditor.EXPOSED for r in identities}
        actual = [(r['recording'], r['fit_key'], r['arm'], r['seed'], r['learning_rate']) for r in subset]
        require(len(actual) == len(set(actual)) and set(actual) == expected, 'separate complete parent/new attempt rosters')
    for row in attempts[208:]:
        name = f"prediction-{row['recording']}-{row['fit_key']}.npz"
        require(row['status'] in ('PASS', 'FAILED') and len(row['errors']) == 2, 'new forecast outcome')
        if row['prediction_file'] is None:
            require(row['status'] == 'FAILED' and all(e is not None for e in row['errors']), 'explicit absent failed new bank')
        else:
            require(row['prediction_file'] == name, 'new forecast bank identity')
            required.add(name)
    require(set(inventory) == required, 'exact child evidence inventory; no parent forecast regeneration')
    for name in required:
        safe_path(Path('/opaque-evidence'), name)
    return {name: inventory[name] for name in sorted(excluded)}


def selected(row, selection, inherited_rates):
    arm = row['arm']
    if arm == auditor.CANDIDATE:
        rate = selection['selected_rates'][arm]
        return rate is not None and row['learning_rate'] == rate
    if arm == auditor.FIXED or arm in auditor.REFS:
        return True
    return arm in inherited_rates and row['learning_rate'] == inherited_rates[arm]


def validate_scalars(value):
    results, resources = value['results'], value['resources']
    cfg, selection, gate = results['config'], results['selection'], results['result']
    require(results['version'] == auditor.STUDY_VERSION and cfg['all_evaluation_data_exposed'] is True,
            'exposed development only')
    require(set(selection['selected_rates']) == {auditor.CANDIDATE}
            and selection['selected_rates'][auditor.CANDIDATE] in (*auditor.RATES, None), 'one DEV-selected new rate')
    recipes = (*auditor.parent_audit.identities(), *auditor.identities())
    expected = {(name, r['key'], r['arm'], r['seed'], r['learning_rate'], h)
                for name in auditor.EXPOSED for r in recipes for h in (64, 128)}
    keys = [(r['recording'], r['fit_key'], r['arm'], r['seed'], r['learning_rate'], r['horizon']) for r in results['rows']]
    require(len(keys) == len(set(keys)) == 488 and set(keys) == expected, 'all488 exact audited rows')
    for row in results['rows']:
        require(row['status'] in ('PASS', 'FAILED'), 'declared score status')
        if row['status'] == 'FAILED':
            require(row['error'] is not None and row['metrics'] is None, 'failed scores never plotted as zero')
        else:
            m = row['metrics']
            require(row['error'] is None and m['horizon'] == row['horizon'] and m['windows'] == 22
                    and m['scalars'] == 22*row['horizon']*6 and len(m['per_joint_rmse_deg']) == 6
                    and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in
                            [m['standardized_rmse'], m['standardized_sse'], m['physical_rmse_deg'], *m['per_joint_rmse_deg']]),
                    'complete finite score geometry')
    roster = auditor.resource_identities(selection, cfg['inherited_rates'])
    require(len(resources) == 46 and all(all(row[k] == v for k, v in identity.items())
            for row, identity in zip(resources, roster, strict=True)), 'all46 current request cost slots')
    for row in resources:
        require(row['status'] in ('PASS', 'FAILED', 'UNAVAILABLE'), 'declared cost status')
        if row['status'] == 'PASS':
            require(row['error'] is None and isinstance(row['timing'], dict)
                    and all(type(row['timing'][k]) in (int, float) and math.isfinite(row['timing'][k])
                            and row['timing'][k] > 0 for k in ('median_seconds', 'p95_seconds')), 'finite saved cost summaries')
        else:
            require(row['error'] is not None and row['timing'] is None, 'missing cost stays missing')
        require(all(type(row[k]) is int and row[k] >= 0 for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes'))
                and persistent_bytes(row) > 0, 'positive full logical storage')
    require(tuple(c['name'] for c in gate['conditions']) == auditor.CONDITIONS
            and all(type(c['passed']) is bool for c in gate['conditions'])
            and gate['passed'] == sum(c['passed'] for c in gate['conditions']) and gate['total'] == 5
            and gate['status'] == ('POSITION_OBSERVER_DEVELOPMENT_PASS' if gate['passed'] == 5 else 'POSITION_OBSERVER_DEVELOPMENT_FAIL'),
            'literal all-five rule; no partial promotion')
    require(set(gate['details']) == set(auditor.EXPOSED) and set(gate['costs']) == set(auditor.ELIGIBLE)
            and 'observer_learned' not in gate['equal_file_means'], '18 eligible families; failed parent remains diagnostic')
    for cost in gate['costs'].values():
        require(cost is None or all(type(cost[k]) in (int, float) and math.isfinite(cost[k]) and cost[k] > 0 for k in ('latency', 'bytes')),
                'finite complete family costs')
    require(len(value['probes']) == 7 and all(all(row[k] == v for k, v in identity.items())
            for row, identity in zip(value['probes'], auditor.probe_schedule(), strict=True)), 'seven fixed audited probe identities')
    require(set(value['fit_diagnostics']) == {r['key'] for r in auditor.identities()[:6]}, 'six retained fit diagnostic summaries')
    required_counts = {'fresh_fits': 6, 'fixed_models': 3, 'inherited_models': 36, 'final_checkpoints': 45,
                       'diagnostic_probes': 7, 'raw_gradient_banks': 13, 'metric_rows': 488,
                       'inherited_metric_rows': 416, 'new_metric_rows': 72, 'prediction_attempts': 244,
                       'inherited_prediction_attempts': 208, 'new_prediction_attempts': 36,
                       'resource_rows': 46, 'condition_rows': 5, 'parent_forecast_replays': 0}
    require(all(value['counts'][k] == v for k, v in required_counts.items()), 'explicit audit scope counters')


def authenticate(study, audit_path, engineering):
    """Only opaque admission and saved JSON; never call auditor.audit."""
    study, audit_path, engineering = [Path(p).resolve() for p in (study, audit_path, engineering)]
    require((study, audit_path, engineering) == (ROOT/STUDY, ROOT/AUDIT, ROOT/ENGINEERING), 'canonical original inputs')
    process_path = engineering/'audit-process-01.json'; process = read(process_path)
    argv = ['.venv/bin/python', 'scripts/audit_robot_position_observer.py', '--study', STUDY,
            '--run-receipt', ENGINEERING+'/run-process-01.json', '--output', AUDIT]
    require(process['command'] == argv and type(process['returncode']) is int and process['returncode'] == 0
            and process['external_timeout'] is False and process['cap_seconds'] == 180
            and process['thread_env'] == dict.fromkeys(auditor.THREADS, '1')
            and type(process['elapsed_seconds']) in (int, float) and math.isfinite(process['elapsed_seconds'])
            and 0 < process['elapsed_seconds'] <= 180, 'successful original independent audit closure required')
    launch_path, closure_path = engineering/'audit-launch-01.json', engineering/'audit-closure-01.json'
    launch, closure = read(launch_path), read(closure_path)
    require(set(launch) == {'command', 'started_utc', 'thread_env', 'cap_seconds'}
            and all(process[k] == v for k, v in launch.items())
            and descriptor(engineering/'audit-process-01.log') == process['log'], 'original audit launch/process/log join')
    source = ROOT/'scripts/audit_robot_position_observer.py'
    require(closure['status'] == 'PASS' and closure['source_matches_registration'] is True
            and type(closure['scope']) is str and closure['scope'], 'postterminal original audit output closure')
    for key, path in {'auditor': source, 'audit_output': audit_path, 'audit_manifest': audit_path.parent/'manifest.json',
                      'process': process_path, 'launch': launch_path, 'log': engineering/'audit-process-01.log',
                      'registration': ROOT/'research/robot-position-observer-registration.json'}.items():
        require(closure[key] == auditor.pin(path), 'closed audit byte identity: '+key)
    regular_tree(audit_path.parent, {'audit.json', 'manifest.json'})
    require(read(audit_path.parent/'manifest.json') == {'files': {'audit.json': descriptor(audit_path)}}, 'exact audit output manifest')
    plan, inputs = auditor.authenticate(study, engineering/'run-process-01.json')
    value = read(audit_path)
    require(value['status'] == 'PASS' and value['agreement'] is True and value['study'] == str(study)
            and value['registration_sha256'] == inputs['registration']['sha256'] and value['inputs'] == inputs
            and value['source_pins'] == plan['sources'] and value['auditor'] == auditor.pin(source), 'same closed run and agreeing audit')
    auditor.close(value['results'], read(study/'results.json'), 'audited saved scalar results')
    require(value['resources'] == read(study/'resources.json')
            and value['prediction_attempts'] == read(study/'prediction-attempts.json')
            and value['probes'] == read(study/'probes.json'), 'same cost, attempt and diagnostic receipts')
    for key, summary in value['fit_diagnostics'].items():
        require(summary == read(study/key/'diagnostic-summary.json'), 'audited fit diagnostics')
    require(value['results']['config'] == plan['config'], 'registered config in saved report')
    validate_scalars(value)
    inventory = read(study/'manifest.json')['files']
    excluded = study_roster(inventory, plan, value['prediction_attempts'])
    files = qualification_files(engineering, plan)
    files.update(publication_qualification_files())
    preflight_path = engineering/'registration-preflight-01.json'
    preflight = read(preflight_path)
    require(preflight['status'] == 'PASS' and preflight['registration'] == inputs['registration']
            and preflight['qualification'] == plan['qualification']
            and [preflight[k] for k in ('sources', 'inputs', 'model_calls', 'numeric_decodes', 'optimizer_updates')]
            == [54, 61, 0, 0, 0] and type(preflight['scope']) is str and preflight['scope'],
            'original metadata-only registration preflight')
    files['engineering/registration-preflight-01.json'] = preflight_path
    pins = {str(path): descriptor(path) for path in (process_path, launch_path, closure_path,
            engineering/'audit-process-01.log', audit_path, audit_path.parent/'manifest.json')}
    return {'plan': plan, 'inputs': inputs, 'audit': value, 'inventory': inventory, 'excluded': excluded,
            'qualification_files': {k: str(v) for k, v in files.items()}, 'publication_inputs': pins}


def render(output, value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, gate, selection = value['results']['rows'], value['results']['result'], value['results']['selection']
    colors = ['#176b45', '#4b8c68', '#64879b', '#8b76aa', '#c34b28', '#a1a8ae', '#c59a47', '#82927c', '#5c95a2', '#977459', '#749c84',
              '#927caf', '#b58147', '#8d8b85', '#637da0', '#ab8282', '#996c4d', '#767cae', '#6c8978']
    accuracy = [r['metrics']['standardized_rmse'] for r in rows if r['horizon'] == 128 and r['status'] == 'PASS']
    means = [v for detail in gate['details'].values() for v in detail['means'].values() if v is not None]
    upper = max([*accuracy, *means], default=0.)*1.08 or 1.
    for detail_view in (False, True):
        fig, axes = plt.subplots(3, 2, figsize=(21, 18))
        for axis, name in zip(list(axes.flat)[:4], auditor.EXPOSED, strict=True):
            outside_points, outside_means = 0, 0
            for i, arm in enumerate(FAMILIES):
                instances = [r for r in rows if r['recording'] == name and r['arm'] == arm and r['horizon'] == 128]
                for j, row in enumerate(instances):
                    x = i+(j-(len(instances)-1)/2)*.09
                    if row['status'] == 'PASS':
                        chosen = selected(row, selection, value['results']['config']['inherited_rates'])
                        raw = row['metrics']['standardized_rmse']
                        outside = detail_view and raw > 2.
                        axis.scatter(x, 2. if outside else raw, color=colors[i], s=30,
                                     marker='o' if chosen else 'x', alpha=1 if chosen else .45, zorder=3, clip_on=False)
                        if outside:
                            outside_points += 1
                            axis.annotate('↑', (x, 2.), xytext=(0, 3), textcoords='offset points',
                                          ha='center', va='bottom', fontsize=8, color=colors[i], annotation_clip=False)
                    else:
                        axis.text(x, .015, 'failed', rotation=90, ha='center', fontsize=6, transform=axis.get_xaxis_transform())
                mean = gate['details'][name]['means'].get(arm)
                if mean is not None:
                    outside = detail_view and mean > 2.
                    shown = 2. if outside else mean
                    axis.plot([i-.27, i+.27], [shown, shown], color='black', linewidth=1.5,
                              linestyle='--' if outside else '-', clip_on=False)
                    if outside:
                        outside_means += 1
                        axis.annotate('mean ↑', (i, 2.), xytext=(0, 15), textcoords='offset points',
                                      ha='center', va='bottom', fontsize=7, annotation_clip=False)
            caption = name.removeprefix('recording_2021_12_15_').removesuffix('.mat')+' (exposed development)'
            if detail_view:
                caption += f'\nDetail 0–2: {outside_points} points / {outside_means} means above range, marked at top'
            axis.set_title(caption, pad=36 if detail_view else 6)
            axis.set_ylabel('H128 standardized RMSE ('+('linear detail 0–2' if detail_view else 'symlog; linear threshold 0.1')+')')
            axis.set_ylim(0, 2. if detail_view else upper)
            if not detail_view:
                axis.set_yscale('symlog', linthresh=.1)
        for axis, key, scale, label in ((axes[2, 0], 'latency', 1000, 'Complete request latency (ms)'),
                                       (axes[2, 1], 'bytes', 1, 'Persistent numeric bytes')):
            for i, arm in enumerate(FAMILIES):
                cost = gate['costs'].get(arm)
                if cost is not None:
                    axis.plot([i-.27, i+.27], [cost[key]*scale]*2, color=colors[i], linewidth=3)
                else:
                    axis.text(i, .02, 'prior diagnostic' if arm == 'observer_learned' else 'incomplete',
                              rotation=90, ha='center', fontsize=7, transform=axis.get_xaxis_transform())
                instances = [r for r in value['resources'] if r['arm'] == arm]
                for j, row in enumerate(instances):
                    scalar = persistent_bytes(row) if key == 'bytes' else (None if row['timing'] is None else row['timing']['median_seconds']*1000)
                    if scalar is not None:
                        axis.scatter(i+(j-(len(instances)-1)/2)*.15, scalar, color=colors[i], s=30, zorder=3)
            axis.set_ylabel(label+' (log scale)'); axis.set_yscale('log')
        for axis in axes.flat:
            axis.set_xticks(range(len(FAMILIES)), LABELS, rotation=55, ha='right')
            axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
        view = 'DETAIL 0–2' if detail_view else 'FULL RANGE'
        fig.suptitle(f"Position-only observer development: {gate['passed']} of 5 criteria pass (all 5 required) | {view}", fontsize=15)
        explanation = ('Detail only: all four accuracy axes are linear 0–2. Every larger point is placed at 2 with an upward arrow; '
                       'larger means have dashed bars and mean arrows. See benchmark.png/SVG for the complete range.' if detail_view else
                       'Full range: all four accuracy axes share a symlog scale, linear from 0 to 0.1 and logarithmic above 0.1. '
                       'All finite errors remain at their actual values. See benchmark-detail.png/SVG for the labeled 0–2 detail.')
        fig.text(.5, .009, explanation+'\nSelected circles / unselected crosses; black lines are saved selected means; failures remain labeled. '
                 'Cost panels use log scales and all instance points.\nFull request includes normalization, casting, prefix assimilation, forecast, validation and deadline checks. '
                 'Prior learned observer remains diagnostic, with no complete recipe or current timing. Same robot/day; no confirmation or control claim.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, .97))
        stem = 'benchmark-detail' if detail_view else 'benchmark'
        fig.savefig(output/(stem+'.png'), dpi=150); fig.savefig(output/(stem+'.svg')); plt.close(fig)


def render_probes(output, value):
    """Only saved audited diagnostics; this never computes a model gradient."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    probes = value['probes']
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    labels = [f"{r['gain_origin']}\nseed {r['seed']} / batch {r['batch_index']}" for r in probes]
    for axis, field, title in ((axes[0], 'gradient', 'Pre-clipping gradient L2 norm (float64 diagnostic)'),
                               (axes[1], 'state', 'Maximum absolute finite prefix-state coordinate')):
        for i, row in enumerate(probes):
            gradient, native = row.get('gradient'), row.get('native_norm')
            scalar = (None if gradient is None else gradient['norm64']) if field == 'gradient' else row['prefix'].get('max_state_abs')
            overflow = native is not None and native['kind'] != 'finite'
            color = '#bd382f' if overflow else '#216b4d'
            if scalar is not None and scalar > 0:
                axis.scatter(i, scalar, color=color, marker='X' if overflow else 'o', s=65, zorder=3)
            else:
                axis.text(i, .03, 'zero' if scalar == 0 else 'not captured / nonfinite', rotation=90,
                          ha='center', fontsize=8, transform=axis.get_xaxis_transform())
            if overflow:
                axis.text(i, .95, 'native float32\n'+native['kind'], color=color, rotation=90,
                          ha='center', va='top', fontsize=8, transform=axis.get_xaxis_transform())
        axis.set_yscale('log'); axis.set_title(title); axis.set_xticks(range(7), labels, rotation=30, ha='right')
        axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
    fig.suptitle('Seven predeclared FIT-only probes: diagnostics before unchanged native clipping', fontsize=14)
    fig.text(.5, .01, 'Each probe is one forward/backward/clip attempt with zero optimizer steps. Red X marks a nonfinite native norm.\n'
             'Finite entries can still overflow the float32 clipping norm. Missing or zero values are labeled, never plotted as positive values.\n'
             'No learning-rate effect is inferred before an optimizer update; these probes do not decide the five scientific criteria.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    fig.savefig(output/'diagnostic-probes.png', dpi=150); fig.savefig(output/'diagnostic-probes.svg'); plt.close(fig)


def write_tables(output, value):
    results, gate = value['results'], value['results']['result']
    fields = ['recording', 'fit_key', 'arm', 'seed', 'learning_rate', 'horizon', 'status', 'selected',
              'error_type', 'error_message', 'standardized_rmse', 'standardized_sse', 'scalars',
              'physical_rmse_deg', 'windows', *[f'joint_{i}_rmse_deg' for i in range(1, 7)]]
    with (output/'scores.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in results['rows']:
            record = {k: row[k] for k in fields[:7]}
            record.update(selected=selected(row, results['selection'], results['config']['inherited_rates']),
                          error_type=(row['error'] or {}).get('type', ''), error_message=(row['error'] or {}).get('message', ''))
            if row['metrics'] is not None:
                record.update({k: row['metrics'][k] for k in fields[10:15]})
                record.update(dict(zip(fields[15:], row['metrics']['per_joint_rmse_deg'], strict=True)))
            writer.writerow(record)
    fields = ['key', 'arm', 'seed', 'learning_rate', 'origin', 'status', 'parameter_bytes', 'buffer_bytes',
              'state_bytes', 'normalizer_bytes', 'persistent_bytes', 'median_seconds', 'p95_seconds', 'error']
    with (output/'costs.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in value['resources']:
            record = {k: row[k] for k in fields[:10]}
            record.update(persistent_bytes=persistent_bytes(row), error=json.dumps(row['error'], sort_keys=True))
            if row['timing'] is not None:
                record.update({k: row['timing'][k] for k in ('median_seconds', 'p95_seconds')})
            writer.writerow(record)
    fmt = lambda x: 'FAILED / incomplete' if x is None else f'{x:.6g}'
    lines = ['# Position-only observer development', '', f"**{gate['status']}**, {gate['passed']}/5 conditions; all five required.", '',
             '| Criterion | Passed |', '|---|---|']
    lines += [f"| {r['name']} | {r['passed']} |" for r in gate['conditions']]
    lines += ['', '| Family | Rate | File 1 | File 2 | File 3 | File 4 | Equal-file mean | Request ms | Persistent bytes |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, label in zip(FAMILIES, LABELS, strict=True):
        if arm == 'observer_learned':
            lines.append(f'| {label} | no complete parent recipe | diagnostic | diagnostic | diagnostic | diagnostic | ineligible | not retimed | see parent |')
            continue
        errors = [gate['details'][name]['means'].get(arm) for name in auditor.EXPOSED]
        cost = gate['costs'][arm]
        rate = results['selection']['selected_rates'].get(arm, results['config']['inherited_rates'].get(arm))
        lines.append(f'| {label} | {rate} | '+' | '.join(fmt(v) for v in errors)
                     +f" | {fmt(gate['equal_file_means'].get(arm))} | {fmt(None if cost is None else cost['latency']*1000)} | "
                     +('incomplete' if cost is None else str(cost['bytes']))+' |')
    lines += ['', ('All 488 score rows retain both horizons and every declared rate: 416 unchanged parent rows plus 72 new rows. '
              'All 244 attempts and 46 current costs remain visible. No parent recipe is reselected. '
              'Only the new learned gain uses the two original DEV files for rate selection; all four files are exposed development.'), '',
              ('Seven FIT-only no-update probes and six per-fit diagnostic bundles are retained under study/. '
              'Finite gradient entries and a nonfinite native clipping norm are distinct outcomes. '
              'Diagnostics do not establish a learning-rate effect before any optimizer update.'), '']
    (output/'table.md').write_text('\n'.join(lines))
    (output/'audited-summary.json').write_text(json.dumps({k: value[k] for k in
        ('results', 'resources', 'prediction_attempts', 'probes', 'fit_diagnostics', 'counts')},
        indent=2, sort_keys=True, allow_nan=False)+'\n')


def publish(study, audit_path, engineering, output):
    study, audit_path, engineering, output = [Path(p).resolve() for p in (study, audit_path, engineering, output)]
    auth = authenticate(study, audit_path, engineering)
    require(output == ROOT/OUTPUT and not output.exists(), 'exclusive canonical publication folder')
    source_pin = descriptor(__file__)
    output.mkdir(parents=True, exist_ok=False)
    mapping = {}
    def copy(source, name):
        source = Path(source); expected = descriptor(source); target = safe_path(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
        require(descriptor(source) == descriptor(target) == expected, 'byte-exact opaque evidence copy')
        mapping[name] = {'path': str(source.resolve()), **expected}
    for name in (*auth['inventory'], 'manifest.json', 'receipt.json'):
        if name not in auth['excluded']: copy(study/name, 'study/'+name)
    copy(audit_path, 'audit.json'); copy(audit_path.parent/'manifest.json', 'audit-manifest.json')
    for name in ('run-launch-01.json', 'run-process-01.json', 'run-process-01.log', 'audit-launch-01.json',
                 'audit-process-01.json', 'audit-process-01.log', 'audit-closure-01.json'):
        copy(engineering/name, 'engineering/'+name)
    for name, path in sorted(auth['qualification_files'].items()): copy(path, name)
    for name in (*DELIVERY_SOURCES, LAUNCHER): copy(ROOT/name, 'delivery-source/'+name)
    for name in ('parent_audit', 'parent_publication_manifest'):
        copy(auth['plan'][name]['path'], 'parent-closure/'+name+'.json')
    write_tables(output, auth['audit']); render(output, auth['audit']); render_probes(output, auth['audit'])
    gate = auth['audit']['results']['result']
    (output/'README.md').write_text(
        '# Position-only observer development evidence\n\n'
        f"**{gate['status']}**, {gate['passed']}/5 required conditions. Independent evidence audit: PASS.\n\n"
        'The 488 scores include all 416 unchanged parent scores; 244 attempts include all 208 parent attempts. '
        'Nineteen families remain displayed, including the failed prior learned observer, which has no eligible recipe. '
        'Eighteen families enter the prospective comparison and 46 current request costs are retained.\n\n'
        'All 45 model states, six fresh initial/final/optimizer and diagnostic bundles, seven FIT-only no-update probes, '
        '50 inherited evidence inputs and 54 scientific source snapshots are retained. The earlier failed observer study '
        'remains authoritative. Its forecast banks are linked through the [parent publication](../robot-observer-results.md) '
        'and exact parent audit/manifest pins; they were not regenerated here.\n\n'
        'The intervention starts the 72-value gain at [I;0] and freezes the same 590-value transition. '
        'The lower latent coordinates are not verified physical velocity. This is a conventional observer development '
        'experiment, not a Kalman posterior, stability guarantee, novelty result or renewed confirmation. '
        'All four files are exposed recordings from the same robot and day; official TEST remains closed.\n\n'
        'Every scientific/publication qualification attempt, original run/audit closure and diagnostic failure is preserved. '
        'Exactly four target-window NPZ files and all external measurement arrays stay excluded under their descriptors. '
        'Licensed local measurements are needed for full replay. Repository licensing does not relicense '
        '[Industrial Robot measurements](https://doi.org/10.26204/data/5).\n\n'
        'Latency is the complete physical request with normalization, casts, prefix assimilation, rollout, checks and '
        'denormalization. Persistent numeric bytes include weights, buffers, state and normalizers, excluding temporary '
        'workspace, Python object overhead and loading. Realized future torques are supplied; no closed-loop control claim.\n\n'
        'See [all scores](scores.csv), [all current costs](costs.csv), [table](table.md), '
        '[protocol](../robot-position-observer-protocol.md), full-range benchmark.png/SVG, benchmark-detail.png/SVG '
        'and the seven-case diagnostic-probes.png/SVG.\n')
    require(authenticate(study, audit_path, engineering) == auth and descriptor(__file__) == source_pin,
            'all evidence and publisher unchanged after delivery')
    for name, item in mapping.items():
        expected = {k: item[k] for k in ('sha256', 'bytes')}
        require(descriptor(item['path']) == descriptor(output/name) == expected, 'complete opaque copy roundtrip')
    files = {str(p.relative_to(output)): descriptor(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'version': VERSION, 'files': files, 'copy_sources': mapping, 'publisher': source_pin,
                'excluded_measurement_payloads': auth['excluded'],
                'external_measurement_inputs': {k: v for k, v in auth['plan']['inputs'].items() if k.startswith('data/')},
                'parent_lineage': {k: auth['plan'][k] for k in ('parent_audit', 'parent_publication_manifest')},
                'original_inputs': auth['inputs'], 'publication_inputs': auth['publication_inputs'],
                'scope': 'Audited scalar presentation and opaque derived evidence; no array decoding, replay or metric recomputation'}
    with (output/'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    receipt = {'version': VERSION, 'status': 'PASS', 'scientific_status': gate['status'],
               'passed': gate['passed'], 'total': 5, 'publisher': source_pin,
               'manifest': descriptor(output/'manifest.json'), 'files': len(files), 'excluded_measurement_files': 4,
               'model_calls': 0, 'array_decodes': 0, 'rescoring_calls': 0, 'timing_calls': 0}
    with (output/'receipt.json').open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=ROOT/STUDY)
    parser.add_argument('--audit', type=Path, default=ROOT/AUDIT)
    parser.add_argument('--engineering', type=Path, default=ROOT/ENGINEERING)
    parser.add_argument('--output', type=Path, default=ROOT/OUTPUT)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.audit, args.engineering, args.output)), flush=True)
