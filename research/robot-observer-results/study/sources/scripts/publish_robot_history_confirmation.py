"""Publish closed confirmation evidence from audited scalars and opaque bytes.

No array decoder, model inference, fitting, scoring or timing is called here.
The four measured recording/target NPZs and original raw media stay external.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import audit_robot_history_confirmation as auditor

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-history-confirmation-publication-v1'
STUDY = 'output/robot-history-confirmation-v1'
ENGINEERING = 'output/robot-history-confirmation-engineering-v1'
AUDIT = 'output/robot-history-confirmation-audit-v1/audit.json'
OUTPUT = 'research/robot-history-confirmation-results'
DELIVERY_SOURCES = ('scripts/publish_robot_history_confirmation.py', 'tests/test_publish_robot_history_confirmation.py')
PUBLICATION_ENGINEERING = 'output/robot-history-confirmation-publication-engineering-v1'
LAUNCHER = 'scripts/launch_robot_history_confirmation.py'
FAMILIES = (*auditor.ARMS, *auditor.REFS)
LABELS = ('Last two', 'Local affine', 'Temporal affine', 'Dense bounded', 'Dense unbounded',
          'GRU32', 'Legacy instant', 'GRU10', 'Causal ridge 1', 'Causal ridge 100', 'Linear AR2', 'Persistence')
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
    """Exact derived roster; no alternate measurement filename can enter delivery."""
    require(plan['config'] == auditor.config() and set(plan['sources']) == set(auditor.SOURCES), 'frozen design/source roster')
    required = {'registration.json', 'runtime.json', 'models.json', 'checkpoint-barrier.json', 'parameter-checks.json',
                'prediction-attempts.json', 'results.json', 'resources.json', *auditor.COMMON_PAYLOADS}
    required |= {'sources/' + name for name in plan['sources']}
    required |= {auditor.model_key(a, s) + '/final.npz' for a in auditor.ARMS for s in auditor.SEEDS}
    required |= {f'completed-prediction-{i:02d}.json' for i in range(1, 57)}
    required |= {f'completed-timing-{i:02d}.json' for i in range(1, 29)}
    excluded = {f'{prefix}-{name}.npz' for name in auditor.CONFIRM for prefix in ('confirm-data', 'confirm-windows')}
    required |= excluded
    identities = [(name, auditor.model_key(a, s), a, s, auditor.FIXED_RATES[a])
                  for name in auditor.CONFIRM for a in auditor.ARMS for s in auditor.SEEDS]
    identities += [(name, a, a, None, None) for name in auditor.CONFIRM for a in auditor.REFS]
    expected = set(identities)
    actual = [(r['recording'], r['key'], r['arm'], r['seed'], r['learning_rate']) for r in attempts]
    require(len(actual) == len(set(actual)) == 56 and set(actual) == expected, 'all56 audited forecast attempts')
    for row in attempts:
        name = f"prediction-{row['recording']}-{row['key']}.npz"
        require(row['status'] in ('PASS', 'FAILED') and len(row['errors']) == 2, 'forecast attempt status')
        if row['prediction_file'] is None:
            require(row['status'] == 'FAILED' and row['prediction_pin'] is None
                    and all(error is not None for error in row['errors']), 'absent bank is an explicit numerical failure')
        else:
            require(row['prediction_file'] == name and inventory.get(name) == row['prediction_pin'], 'audited forecast bank pin')
            required.add(name)
    require(set(inventory) == required, 'exact published study roster')
    for name in required:
        safe_path(Path('/opaque-evidence'), name)
    return {name: inventory[name] for name in sorted(excluded)}


def qualification_files(engineering, plan):
    """Both original attempts, including the failed fixture and its old sources."""
    commands = [['.venv/bin/ruff', 'check', *auditor.QUALIFICATION_SOURCES],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q',
                 'tests/test_robot_history_confirmation.py', 'tests/test_audit_robot_history_confirmation.py']]
    require(sorted(p.name for p in engineering.glob('qualification-*') if p.is_dir())
            == ['qualification-01', 'qualification-02'], 'both original qualification attempts only')
    files = {}
    for number, status in ((1, 'FAIL'), (2, 'PASS')):
        folder = engineering / f'qualification-{number:02d}'
        pre, receipt = read(folder / 'preflight.json'), read(folder / 'receipt.json')
        require(pre['sources'] == receipt['sources'] and set(receipt['sources']) == set(auditor.SOURCES)
                and receipt['sources_unchanged'] is True and pre['launcher'] == receipt['launcher']
                and receipt['thread_env'] == dict.fromkeys(auditor.THREADS, '1'), 'qualification exact source closure')
        require(pre['commands'] == commands and [r['command'] for r in receipt['commands']] == commands
                and [r['returncode'] for r in receipt['commands']] == ([0, 1] if number == 1 else [0, 0])
                and receipt['status'] == status, 'original failed/pass qualification outcomes')
        names = {'preflight.json', 'receipt.json'}
        for name, pin in {**receipt['sources'], LAUNCHER: receipt['launcher']}.items():
            relative = 'sources/' + name
            require(descriptor(safe_path(folder, relative)) == pin, 'qualification snapshot bytes')
            names.add(relative)
        for i, row in enumerate(receipt['commands'], 1):
            name = f'command-{i:02d}.log'
            require(Path(row['log']) == folder / name and descriptor(folder / name)['sha256'] == row['sha256']
                    and type(row['seconds']) in (float, int) and math.isfinite(row['seconds']) and row['seconds'] > 0,
                    'original qualification log and duration')
            names.add(name)
        if number == 2:
            require(receipt['sources'] == plan['sources'] and receipt['launcher'] == plan['launcher']
                    and Path(plan['qualification']['path']) == folder / 'receipt.json'
                    and descriptor(folder / 'receipt.json') == {k: plan['qualification'][k] for k in ('sha256', 'bytes')},
                    'registered final qualification')
        regular_tree(folder, names)
        for name in names:
            files[f'engineering/qualification-{number:02d}/{name}'] = folder / name
    return files


def publication_qualification_files():
    root = ROOT / PUBLICATION_ENGINEERING
    attempts = sorted(p for p in root.glob('qualification-*') if p.is_dir())
    require(attempts and [p.name for p in attempts] == [f'qualification-{i:02d}' for i in range(1, len(attempts)+1)],
            'all consecutive original publication qualifications')
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
    argv = ['.venv/bin/python', 'scripts/audit_robot_history_confirmation.py', '--study', STUDY,
            '--run-receipt', ENGINEERING + '/run-process-01.json', '--output', AUDIT]
    require(process['command'] == argv and type(process['returncode']) is int and process['returncode'] == 0
            and process['external_timeout'] is False and process['cap_seconds'] == 180
            and process['thread_env'] == dict.fromkeys(auditor.THREADS, '1')
            and type(process['elapsed_seconds']) in (float, int) and math.isfinite(process['elapsed_seconds'])
            and 0 < process['elapsed_seconds'] <= process['cap_seconds'], 'original successful audit process required')
    source = ROOT / 'scripts/audit_robot_history_confirmation.py'
    launch_path, closure_path = engineering / 'audit-launch-01.json', engineering / 'audit-closure-01.json'
    launch, closure = read(launch_path), read(closure_path)
    require(set(launch) == {'command', 'started_utc', 'thread_env', 'cap_seconds'}
            and all(process[k] == v for k, v in launch.items())
            and descriptor(engineering / 'audit-process-01.log') == process['log'], 'original audit launch/log join')
    require(closure['status'] == 'PASS' and closure['source_matches_registration'] is True
            and type(closure['scope']) is str and closure['scope'], 'explicit postterminal audit closure')
    for key, path in {'auditor': source, 'audit_output': audit_path, 'audit_manifest': audit_path.parent / 'manifest.json',
                      'process': process_path, 'launch': launch_path, 'log': engineering / 'audit-process-01.log',
                      'registration': ROOT / 'research/robot-history-confirmation-registration.json'}.items():
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
    validate_scalars(value)
    inventory = read(study / 'manifest.json')['files']
    excluded = study_roster(inventory, plan, value['prediction_attempts'])
    qualifications = qualification_files(engineering, plan)
    qualifications.update(publication_qualification_files())
    pins = {str(path): descriptor(path) for path in (process_path, launch_path, closure_path, engineering / 'audit-process-01.log', audit_path,
                                                  audit_path.parent / 'manifest.json')}
    return {'plan': plan, 'inputs': inputs, 'audit': value, 'inventory': inventory, 'excluded': excluded,
            'qualification_files': {k: str(v) for k, v in qualifications.items()}, 'publication_inputs': pins}


def validate_scalars(value):
    results, resources = value['results'], value['resources']
    require(results['config'] == auditor.config() and results['fixed_rates'] == auditor.FIXED_RATES, 'literal frozen confirmation recipe')
    keys = [(r['recording'], r['arm'], r['seed'], r['learning_rate'], r['horizon']) for r in results['rows']]
    expected = {(n, a, s, auditor.FIXED_RATES[a], h) for n in auditor.CONFIRM for a in auditor.ARMS for s in auditor.SEEDS for h in (64, 128)}
    expected |= {(n, a, None, None, h) for n in auditor.CONFIRM for a in auditor.REFS for h in (64, 128)}
    require(len(keys) == len(set(keys)) == 112 and set(keys) == expected, 'all112 audited score rows')
    for row in results['rows']:
        require(row['status'] in ('PASS', 'FAILED'), 'declared score status')
        if row['status'] == 'FAILED':
            require(row['metrics'] is None and row['error'] is not None, 'failed score evidence')
        else:
            m = row['metrics']
            require(row['error'] is None and len(m['per_joint_rmse_deg']) == 6
                    and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in
                            [m['standardized_rmse'], m['standardized_sse'], m['physical_rmse_deg'], *m['per_joint_rmse_deg']]), 'finite audited metrics')
    cost_keys = [(r['arm'], r['seed'], r['learning_rate']) for r in resources]
    expected_costs = {(a, s, auditor.FIXED_RATES[a]) for a in auditor.ARMS for s in auditor.SEEDS} | {(a, None, None) for a in auditor.REFS}
    require(len(cost_keys) == len(set(cost_keys)) == 28 and set(cost_keys) == expected_costs, 'all28 audited resource attempts')
    gate = results['result']
    require(tuple(c['name'] for c in gate['conditions']) == auditor.CONDITIONS
            and all(type(c['passed']) is bool for c in gate['conditions'])
            and gate['passed'] == sum(c['passed'] for c in gate['conditions']) and gate['total'] == 5
            and gate['status'] == ('CONFIRMED_HISTORY_INITIALIZATION' if gate['passed'] == 5 else 'DO_NOT_CONFIRM_HISTORY_INITIALIZATION'),
            'literal five-condition scientific status; no outcome promotion')


def write_tables(output, value):
    rows, gate = value['results']['rows'], value['results']['result']
    fields = ['recording', 'arm', 'seed', 'learning_rate', 'horizon', 'status', 'error_type', 'error_message',
              'standardized_rmse', 'standardized_sse', 'scalars', 'physical_rmse_deg', 'windows',
              *[f'joint_{i}_rmse_deg' for i in range(1, 7)]]
    with (output / 'scores.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            result = {k: row[k] for k in fields[:6]}
            result.update(error_type=(row['error'] or {}).get('type', ''), error_message=(row['error'] or {}).get('message', ''))
            if row['metrics'] is not None:
                result.update({k: row['metrics'][k] for k in fields[8:13]})
                result.update(dict(zip(fields[13:], row['metrics']['per_joint_rmse_deg'], strict=True)))
            writer.writerow(result)
    fmt = lambda x: 'FAILED / incomplete' if x is None else f'{x:.6g}'
    lines = ['# Fixed-checkpoint confirmation', '', f"**{gate['status']}**, {gate['passed']}/5 conditions.", '',
             '| Criterion | Passed |', '|---|---|']
    lines += [f"| {r['name']} | {r['passed']} |" for r in gate['conditions']]
    lines += ['', '| Family | Fixed rate | File 1 H128 RMSE | File 2 H128 RMSE | Equal-file mean | Request ms | Persistent bytes |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, label in zip(FAMILIES, LABELS, strict=True):
        errors = [gate['details'][n]['means'].get(arm) for n in auditor.CONFIRM]
        cost = gate['costs'][arm]
        lines.append(f"| {label} | {auditor.FIXED_RATES.get(arm, '-')} | {fmt(errors[0])} | {fmt(errors[1])} | "
                     f"{fmt(gate['equal_file_means'].get(arm))} | {fmt(None if cost is None else cost['latency']*1000)} | "
                     f"{'incomplete' if cost is None else cost['bytes']} |")
    lines += ['', ('All 112 H64/H128 score rows, all 56 forecast attempts and all 28 cost attempts remain in the evidence. '
              'Means and family costs above are saved audited scalars. Failed cases are retained. '
              'Training costs in models.json are historical; this confirmation performs zero fitting.'), '']
    (output / 'table.md').write_text('\n'.join(lines))
    (output / 'audited-summary.json').write_text(json.dumps({'results': value['results'], 'resources': value['resources'],
        'prediction_attempts': value['prediction_attempts']}, indent=2, sort_keys=True, allow_nan=False)+'\n')


def render(output, value):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, gate = value['results']['rows'], value['results']['result']
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    colors = ['#b9c2ca', '#64879b', '#c34b28', '#749c84', '#927caf', '#c59a47', '#8d8b85', '#5c95a2', '#ab8282', '#977459', '#767cae', '#82927c']
    accuracy = [r['metrics']['standardized_rmse'] for r in rows if r['horizon'] == 128 and r['status'] == 'PASS']
    upper = max(accuracy, default=0.) * 1.08 or 1.
    for axis, name in zip(axes[0], auditor.CONFIRM, strict=True):
        for i, arm in enumerate(FAMILIES):
            values = [r['metrics']['standardized_rmse'] for r in rows if r['recording'] == name and r['arm'] == arm
                      and r['horizon'] == 128 and r['status'] == 'PASS']
            for j, scalar in enumerate(values):
                axis.scatter(i + (j-(len(values)-1)/2)*.15, scalar, color=colors[i], s=35, zorder=3)
            mean = gate['details'][name]['means'].get(arm)
            if mean is not None:
                axis.plot([i-.27, i+.27], [mean, mean], color='black', linewidth=1.5)
            else:
                axis.text(i, .02, 'failed', rotation=90, ha='center', transform=axis.get_xaxis_transform())
        axis.set_title(name.removeprefix('recording_2021_12_15_').removesuffix('.mat'))
        axis.set_ylabel('H128 standardized RMSE (lower is better)'); axis.set_ylim(0, upper)
    for axis, key, scale, label in ((axes[1, 0], 'latency', 1000, 'Complete request latency (ms)'),
                                   (axes[1, 1], 'bytes', 1, 'Persistent numeric bytes')):
        for i, arm in enumerate(FAMILIES):
            cost = gate['costs'][arm]
            if cost is not None:
                axis.plot([i-.27, i+.27], [cost[key]*scale]*2, color=colors[i], linewidth=3)
            else:
                axis.text(i, .02, 'failed', rotation=90, ha='center', transform=axis.get_xaxis_transform())
            if key == 'latency':
                points = [r['timing']['median_seconds']*1000 for r in value['resources']
                          if r['arm'] == arm and r['status'] == 'PASS']
                for j, scalar in enumerate(points):
                    axis.scatter(i+(j-(len(points)-1)/2)*.15, scalar, color=colors[i], s=30, zorder=3)
        axis.set_ylabel(label + ' (log scale)'); axis.set_yscale('log')
    for axis in axes.flat:
        axis.set_xticks(range(len(FAMILIES)), LABELS, rotation=55, ha='right')
        axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
    fig.suptitle(f"Fixed-checkpoint confirmation: {gate['passed']} of 5 criteria pass (all 5 required)", fontsize=14)
    fig.text(.5, .012, 'Accuracy: all finite seed dots, complete family means, common linear scale. Cost panels: log scales; '
             'latency dots show every finite instance median.\nSame robot/day and realized future torque; not a control test. '
             'Costs include normalization and validation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(output / 'benchmark.png', dpi=150)
    fig.savefig(output / 'benchmark.svg')
    plt.close(fig)


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
    for name, item in auth['plan']['parent_closure'].items():
        copy(item['path'], 'parent-closure/' + name + '.json')
    write_tables(output, auth['audit']); render(output, auth['audit'])
    gate = auth['audit']['results']['result']
    (output / 'README.md').write_text(
        '# Fixed-checkpoint confirmation evidence\n\n'
        f"**{gate['status']}**, {gate['passed']}/5 frozen conditions. Evidence audit: PASS.\n\n"
        'All 24 selected checkpoints, inherited references and normalization, all 56 forecast attempts, 112 scores, '
        '28 cost attempts and 36 source snapshots are retained. Both original qualification attempts and their exact '
        'source snapshots remain visible. No confirmation training or recipe selection occurred.\n\n'
        'Four measured recording/target NPZ files and all raw media are excluded, with hashes in manifest.json. '
        'Licensed local inputs and the [parent evidence](../robot-history-initialization-results.md) are required for full replay. '
        'Repository licensing does not relicense [Industrial Robot measurements](https://doi.org/10.26204/data/5).\n\n'
        'Costs cover normalization, casting, conditioning, rollout, validation and deadline callbacks. '
        'Persistent bytes include parameters, recurrent state, buffers and normalizers. Temporary workspace and Python '
        'object overhead are not measured; model loading is excluded. Historical training remains labeled historical.\n\n'
        'These two recordings share the same robot and day. Future inputs are realized measured torques. '
        'This internal confirmation does not establish architecture novelty, independent-environment generalization, '
        'native speedup or control performance. Official TEST remains closed.\n\n'
        'See [scores](scores.csv), [table](table.md), [report](../robot-history-confirmation-results.md) and '
        '[protocol](../robot-history-confirmation-protocol.md).\n')
    require(authenticate(study, audit_path, engineering) == auth and descriptor(__file__) == script_pin, 'unchanged publication inputs/source')
    for name, item in mapping.items():
        expected = {k: item[k] for k in ('sha256', 'bytes')}
        require(descriptor(item['path']) == descriptor(output / name) == expected, 'complete copied-byte roundtrip')
    files = {str(p.relative_to(output)): descriptor(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'version': VERSION, 'files': files, 'copy_sources': mapping, 'excluded_measurement_payloads': auth['excluded'],
                'external_raw_media': {k: auth['plan'][k] for k in ('archive', 'raw_recordings', 'extraction')},
                'original_inputs': auth['inputs'], 'publication_inputs': auth['publication_inputs'], 'publisher': script_pin,
                'scope': 'Saved audited scalars and opaque derived evidence; all four measurement NPZs and raw media excluded'}
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
