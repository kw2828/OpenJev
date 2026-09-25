"""Present closed, independently audited history results without numerical replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-history-initialization-plot-v1'
STUDY_VERSION = 'robot-history-initialization-study-v1'
PLAN_SHA = '628b038c910bf37246e939b104d70530f6783a7fbc67f79afeb79b8813738994'
PREFIT_COMMIT = '80fc2369b60d8b8434832af904a1f50c099f8031'
PRIMARY = ('last_two', 'local_affine', 'temporal_affine')
CACHED = ('dense_bounded', 'dense_unbounded', 'gru32', 'legacy_instant', 'gru10')
ARMS = (*PRIMARY, *CACHED)
REFERENCES = ('causal_ridge_1', 'causal_ridge_100', 'linear_frozen', 'persistence')
ORDER = ('temporal_affine', 'local_affine', 'last_two', *CACHED, *REFERENCES)
LABELS = {
    'temporal_affine': 'Temporal affine (fresh)', 'local_affine': 'Local affine (fresh)',
    'last_two': 'Last two (fresh)', 'dense_bounded': 'Bounded dense', 'dense_unbounded': 'Unbounded dense',
    'gru32': 'GRU32 residual', 'legacy_instant': 'Legacy instant', 'gru10': 'GRU10 residual',
    'causal_ridge_1': 'Causal ridge (1)', 'causal_ridge_100': 'Causal ridge (100)',
    'linear_frozen': 'Frozen linear AR2', 'persistence': 'Persistence',
}
CONDITIONS = ('primary_recipes_complete', 'equal_file_mean_5pct_vs_both_locals',
              'each_file_within_2pct_best_local', 'latency_within_125pct_last_two',
              'complete_frontier_not_dominated')
CAPTION = (
    'Same two previously exposed DEV recordings; one pooled learning rate per family, with all three seeds retained.\n'
    '18 fresh initializer fits and30 unchanged control fits; dots are individual fits, diamonds summarize their errors or latencies.\n'
    'Temporal/local models each have962 parameters:590 transition plus372 initializer; last-two has590.\n'
    '128-step (12.8 s) forecasts condition on realized measured torque, not verified issued commands.\n'
    'All selected controls are retimed on the current host; cached fitting times remain historical.\n'
    'CONFIRM and official TEST stay closed. A pass supports only a separately registered confirmation, not novelty or robot control.'
)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)

def relative_path(folder, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "safe relative path required")
    path = folder / name
    require(path.resolve().is_relative_to(folder.resolve()), "path escapes evidence root")
    return path

def finite(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0

def scalar_agreement(actual, expected):
    """The audit's frozen roundoff tolerance; identities and gates stay exact."""
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(actual) == set(expected), 'audited scalar keys')
        for key in actual:
            scalar_agreement(actual[key], expected[key])
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), 'audited scalar list')
        for left, right in zip(actual, expected, strict=True):
            scalar_agreement(left, right)
    elif isinstance(actual, float):
        require(type(expected) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), 'audited scalar roundoff')
    else:
        require(type(actual) is type(expected) and actual == expected, 'audited identity or gate')

def axis_specs(values):
    """Both accuracy panels share scale and limits; costs retain their units."""
    def spec(entries):
        numbers = [point['value'] for entry in entries for point in entry['points']]
        positive = [value for value in numbers if value > 0]
        maximum = max(numbers, default=0.)
        if positive and max(positive) / min(positive) >= 100:
            if len(positive) == len(numbers):
                return {'scale': 'log', 'limits': [min(positive) / 1.2, maximum * 1.2]}
            return {'scale': 'symlog', 'linthresh': min(positive) / 10,
                    'limits': [0., maximum * 1.2]}
        return {'scale': 'linear', 'limits': [0., maximum * 1.12 if maximum > 0 else 1.]}
    accuracy = spec([entry for panel in values['panels'] for entry in panel['entries']])
    return [dict(accuracy), dict(accuracy), spec(values['latency_ms']), spec(values['storage_kib'])]

def fmt(value):
    return "n/a" if value is None else f"{value:.6g}"

def clean(value):
    return str(value).replace("|", "/").replace("\n", " ")


def authenticate(study, audit_path, engineering):
    """Opaque byte admission; never imports the model or an array decoder."""
    study, audit_path, engineering = [Path(p).resolve() for p in (study, audit_path, engineering)]
    require(study == ROOT / 'output/robot-history-initialization-study-v1', 'registered study path')
    require(engineering == ROOT / 'output/robot-history-initialization-engineering-v1', 'original engineering path')
    inputs = {}
    def bind(path, expected=None):
        path = Path(path)
        actual = pin(path)
        require(expected is None or actual == expected, 'evidence changed: ' + str(path))
        inputs[str(path)] = actual
        return actual
    def desc(item):
        require(set(item) == {'path', 'sha256', 'bytes'} and Path(item['path']).is_absolute(), 'absolute descriptor')
        return bind(item['path'], {k: item[k] for k in ('sha256', 'bytes')})
    def absolute(value):
        value = Path(value)
        return value.resolve() if value.is_absolute() else (ROOT / value).resolve()
    registration = ROOT / 'research/robot-history-initialization-registration.json'
    require(bind(registration)['sha256'] == PLAN_SHA, 'frozen registration hash')
    plan = read_json(registration)
    require(plan['version'] == STUDY_VERSION, 'registered study version')
    run_path = engineering / 'run-process-01.json'
    audit_process_path = engineering / 'audit-process-01.json'
    for path in (run_path, audit_process_path, engineering / 'run-launch-01.json', audit_path, audit_path.parent / 'manifest.json'):
        bind(path)
    process, audit_process = read_json(run_path), read_json(audit_process_path)
    launch = read_json(engineering / 'run-launch-01.json')
    require(process['returncode'] == audit_process['returncode'] == 0 and process['external_timeout'] is False,
            'original run and audit must be terminal successes')
    require(audit_process['source_unchanged'] is True, 'original audit source must remain unchanged')
    require(process['prefit_commit'] == PREFIT_COMMIT and process['registration_sha256'] == PLAN_SHA
            and process['launcher'] == plan['launcher'], 'original run identity')
    require(process['command'] == ['.venv/bin/python', '-u', 'scripts/robot_history_initialization_study.py',
            '--registration', 'research/robot-history-initialization-registration.json',
            '--output', 'output/robot-history-initialization-study-v1'], 'original run command')
    require(all(process[k] == v for k, v in launch.items()), 'original launch/terminal join')
    argv = audit_process['command']
    require(len(argv) == 8 and argv[0] == '.venv/bin/python'
            and argv[2::2] == ['--study', '--run-receipt', '--output'], 'original audit command')
    auditor = ROOT / 'scripts/audit_robot_history_initialization.py'
    require(absolute(argv[1]) == auditor and [absolute(argv[i]) for i in (3, 5, 7)] == [study, run_path, audit_path],
            'original audit paths')
    require(bind(auditor)['sha256'] == audit_process['auditor_sha256'], 'original auditor source')
    bind(engineering / 'run-process-01.log', process['log'])
    bind(engineering / 'audit-process-01.log', audit_process['log'])
    require(audit_process['audit_output'] == pin(audit_path), 'original audit output')
    require(read_json(audit_path.parent / 'manifest.json') == {'files': {audit_path.name: pin(audit_path)}}, 'audit manifest')
    receipt_path, manifest_path = study / 'receipt.json', study / 'manifest.json'
    bind(receipt_path); bind(manifest_path)
    receipt = read_json(receipt_path)
    require(receipt['status'] == 'PASS' and not (study / 'failure.json').exists(), 'producer closure required')
    require(receipt['registration_sha256'] == PLAN_SHA, 'producer registration')
    count_keys = ('fits', 'fresh_fit_attempts', 'cached_fit_records', 'rows', 'permutation_metric_rows',
                  'raw_decodes', 'reference_refits', 'parent_refits', 'saved_fit_loads', 'saved_dev_loads')
    require([receipt[k] for k in count_keys] == [48, 18, 30, 208, 36, 0, 0, 0, 7, 2], 'producer count roster')
    require(receipt['confirmation_access'] is False and receipt['official_test_access'] is False, 'closed partitions')
    require(read_json(study / 'registration.json') == plan, 'saved registration join')
    manifest = read_json(manifest_path)
    require(set(manifest) == {'files'}, 'manifest schema')
    inventory = manifest['files']
    paths = list(study.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'no evidence symlinks')
    require({str(p.relative_to(study)) for p in paths if p.is_file()} == set(inventory) | {'manifest.json', 'receipt.json'},
            'complete study inventory')
    for name, expected in inventory.items():
        bind(relative_path(study, name), expected)
    for name, expected in plan['sources'].items():
        bind(relative_path(ROOT, name), expected)
        bind(relative_path(study / 'sources', name), expected)
    desc(plan['qualification'])
    q = read_json(plan['qualification']['path'])
    require(q['status'] == 'PASS' and q['sources_unchanged'] is True and q['sources'] == plan['sources']
            and q['launcher'] == plan['launcher'] and len(q['commands']) == 2, 'qualified frozen sources')
    for row in q['commands']:
        require(row['returncode'] == 0 and bind(row['log'])['sha256'] == row['sha256'], 'original qualification command/log')
    bind(ROOT / 'scripts/launch_robot_history_initialization.py', plan['launcher'])
    for group, expected_sha in plan['parent_registration_sha256'].items():
        require(bind(ROOT / f'research/robot-{group}-registration.json')['sha256'] == expected_sha, 'parent registration')
        for item in plan['parent_closure'][group].values():
            desc(item)
        for item in plan['parent_payloads'][group].values():
            desc(item)
    for item in plan['data'].values():
        desc(item)
    audit = read_json(audit_path)
    require(audit['status'] == 'PASS' and audit['agreement'] is True and audit['study'] == str(study)
            and audit['registration_sha256'] == PLAN_SHA, 'completed independent audit')
    require(audit['source_pins'] == plan['sources'], 'audited source pins')
    for key, path in (('manifest', manifest_path), ('producer_receipt', receipt_path),
                      ('run_receipt', run_path), ('run_log', engineering / 'run-process-01.log')):
        require(audit['inputs'][key] == {'path': str(path), **bind(path)}, 'audit original input join: ' + key)
    return {'study': str(study), 'plan': plan, 'receipt': receipt, 'inputs': inputs,
            'audit': audit, 'audit_path': str(audit_path), 'engineering': str(engineering)}


def validate_metric(row):
    require(row['status'] in ('PASS', 'FAILED'), 'metric status')
    if row['status'] == 'FAILED':
        require(row['metrics'] is None and row.get('error') is not None, 'failed score must preserve error')
        return
    m = row['metrics']
    require(row.get('error') is None and m['horizon'] == row['horizon'] and len(m['per_joint_rmse_deg']) == 6,
            'successful metric shape')
    require(all(finite(v) for v in (m['standardized_rmse'], m['standardized_sse'], m['physical_rmse_deg'],
                                   *m['per_joint_rmse_deg'])), 'finite metric scalars')


def extract(results, resources, fits, permutation):
    cfg, rows, selection, gate = [results[k] for k in ('config', 'rows', 'selection', 'result')]
    require(results['version'] == cfg['version'] == STUDY_VERSION and tuple(cfg['arms']) == PRIMARY
            and tuple(cfg['comparison_arms']) == ARMS, 'study version/arms')
    require(cfg['context'] == 32 and cfg['dev_horizon'] == 128 and cfg['horizons'] == [64, 128]
            and cfg['seeds'] == [8101, 8102, 8103] and cfg['learning_rates'] == [.001, .003], 'complete design')
    dev = cfg['partitions']['dev']
    require(len(dev) == len(set(dev)) == 2, 'two development recordings')
    expected = {(f, a, s, r, h) for f in dev for a in ARMS for s in cfg['seeds'] for r in cfg['learning_rates'] for h in (64, 128)}
    expected |= {(f, a, None, None, h) for f in dev for a in REFERENCES for h in (64, 128)}
    def key(row):
        return tuple(row[k] for k in ('recording', 'arm', 'seed', 'learning_rate', 'horizon'))
    keyed = {key(row): row for row in rows}
    require(len(rows) == len(keyed) == 208 and set(keyed) == expected, 'all208 unique ordinary rows')
    for row in rows:
        validate_metric(row)
    expected_fits = {(a, s, r) for a in ARMS for s in cfg['seeds'] for r in cfg['learning_rates']}
    require(len(fits) == 48 and {(f['arm'], f['seed'], f['learning_rate']) for f in fits} == expected_fits, 'all48 fits')
    require(all(f['origin'] == ('fresh' if f['arm'] in PRIMARY else 'cached_parent') for f in fits), 'fresh/cached fit identity')
    require(set(selection['selected_rates']) == set(ARMS) and set(selection['options']) == set(ARMS), 'all rate selections')
    require(gate['total'] == len(gate['conditions']) == 5 and tuple(c['name'] for c in gate['conditions']) == CONDITIONS
            and all(type(c['passed']) is bool for c in gate['conditions'])
            and gate['passed'] == sum(c['passed'] for c in gate['conditions']), 'five unchanged conditions')
    require(gate['status'] == ('QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL' if gate['passed'] == 5
                               else 'DO_NOT_ADVANCE_HISTORY_INITIALIZATION'), 'outcome consistency')
    panels = []
    for recording in dev:
        entries = []
        for arm in ORDER:
            rate = selection['selected_rates'].get(arm)
            require(arm not in ARMS or rate in (*cfg['learning_rates'], None), 'selected rate')
            seeds = cfg['seeds'] if arm in ARMS else [None]
            chosen = [keyed[(recording, arm, seed, rate, 128)] for seed in seeds] if arm not in ARMS or rate is not None else []
            points = [{'seed': r['seed'], 'value': r['metrics']['standardized_rmse']} for r in chosen if r['status'] == 'PASS']
            complete = len(points) == len(seeds)
            entries.append({'arm': arm, 'points': points, 'learning_rate': rate,
                            'center': statistics.mean(p['value'] for p in points) if complete else None,
                            'status': 'PASS' if complete else 'INELIGIBLE' if not chosen else 'FAILED'})
        panels.append({'recording': recording, 'entries': entries})
    resource_keys = {(r['arm'], r['seed']): r for r in resources}
    allowed = {(a, s) for a in ARMS if selection['selected_rates'][a] is not None for s in cfg['seeds']}
    expected_costs = allowed | {(a, None) for a in REFERENCES
                               if all(keyed[(f, a, None, None, 128)]['status'] == 'PASS' for f in dev)}
    require(len(resource_keys) == len(resources) and set(resource_keys) == expected_costs, 'complete eligible resource rows')
    latency, storage = [], []
    for arm in ORDER:
        seeds = cfg['seeds'] if arm in ARMS else [None]
        points = []
        for seed in seeds:
            r = resource_keys.get((arm, seed))
            if r is not None:
                require(r.get('learning_rate') == selection['selected_rates'].get(arm), 'selected cost rate')
                value = r['timing']['median_seconds']
                require(finite(value) and value > 0, 'positive request latency')
                points.append({'seed': seed, 'value': value * 1000})
        complete = len(points) == len(seeds)
        latency.append({'arm': arm, 'points': points, 'center': statistics.median(p['value'] for p in points) if complete else None,
                        'status': 'PASS' if complete else 'NOT TIMED'})
        candidates = [f['resources'] for f in fits if f['arm'] == arm] if arm in ARMS else [resource_keys[(arm, None)]] if (arm, None) in resource_keys else []
        totals = [sum(r[k] for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes')) for r in candidates]
        require(all(type(v) is int and v >= 0 for v in totals) and len(set(totals)) <= 1, 'consistent numeric storage')
        storage.append({'arm': arm, 'points': [{'seed': None, 'value': totals[0] / 1024}] if totals else [],
                        'center': totals[0] / 1024 if totals else None, 'bytes': totals[0] if totals else None,
                        'status': 'PASS' if totals else 'UNAVAILABLE'})
    require(permutation['scientific_gate'] is False and permutation['permutation'] == cfg['older_permutation'], 'fixed nongating permutation')
    prows = permutation['rows']
    pexpected = {(f, a, s, selection['selected_rates'][a], h) for f in dev for a in PRIMARY for s in cfg['seeds'] for h in (64, 128)}
    require(len(prows) == len({key(r) for r in prows}) == 36 and {key(r) for r in prows} == pexpected, 'all36 diagnostic score attempts')
    for row in prows:
        validate_metric(row)
    checks = permutation['checks']
    require(len(checks) == 18 and {(r['recording'], r['arm'], r['seed']) for r in checks}
            == {(f, a, s) for f in dev for a in PRIMARY for s in cfg['seeds']}, 'all18 diagnostic attempts')
    return {'version': VERSION, 'scope': CAPTION, 'config': cfg, 'panels': panels, 'latency_ms': latency, 'storage_kib': storage,
            'accuracy_center': 'arithmetic mean of individual fit RMSEs', 'latency_center': 'median of per-fit request medians',
            'selection': selection, 'result': gate, 'all_rows': rows, 'resources': resources, 'fits': fits,
            'permutation': permutation, 'order': list(ORDER)}


def figure(values, folder, *, fabricated=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(21, 10), sharey=True)
    datasets = [p["entries"] for p in values["panels"]] + [values["latency_ms"], values["storage_kib"]]
    specifications = axis_specs(values)
    scales = []
    seed_offsets = {8101: -.15, 8102: 0., 8103: .15, None: 0.}
    for panel, (ax, entries) in enumerate(zip(axes, datasets, strict=True)):
        specification = specifications[panel]
        scale = specification['scale']
        ax.set_xscale(scale, **({'linthresh': specification['linthresh']} if scale == 'symlog' else {}))
        ax.set_xlim(specification['limits'])
        scales.append(scale)
        for y, entry in enumerate(entries):
            color = "#126e82" if entry["arm"] == "temporal_affine" else "#8856a7" if entry["arm"] == "local_affine" else "#68788b" if entry["arm"] in ARMS else "#b16b29"
            for point in entry["points"]:
                ax.scatter(point["value"], y + seed_offsets[point["seed"]], s=30, alpha=.85, color=color, zorder=3)
            if entry["center"] is not None:
                ax.scatter(entry["center"], y, marker="D", s=64, facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
            if entry["status"] != "PASS":
                ax.text(.98, y, entry["status"], transform=ax.get_yaxis_transform(), ha="right", va="center", color="#a33030", fontsize=9)
        ax.set_ylim(len(entries) - .5, -.5)
        ax.set_yticks(range(len(entries)), [LABELS[a] for a in values["order"]])
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        if panel < 2:
            name = values["panels"][panel]["recording"].removeprefix("recording_2021_12_15_").removesuffix(".mat")
            ax.set_title(f"Exposed DEV {panel + 1}: {name}\nH128 accuracy; lower is better")
            ax.set_xlabel("FIT-standardized RMSE" + (f" ({scale})" if scale != "linear" else ""))
        elif panel == 2:
            ax.set_title("Full request latency; lower is better\n20 repeats per fit after 3 warmups")
            ax.set_xlabel("Median request latency, ms" + (f" ({scale})" if scale != "linear" else ""))
        else:
            ax.set_title("Persistent numeric storage\nWeights + state + buffers + normalizers")
            ax.set_xlabel("KiB" + (f" ({scale})" if scale != "linear" else ""))
    result = values["result"]
    status = 'QUALIFIES FOR NEW CONFIRMATION PROTOCOL' if result['passed'] == 5 else 'DO NOT ADVANCE'
    title = f"Observed-history initialization: {status} ({result['passed']}/5 criteria)"
    fig.suptitle('FABRICATED LAYOUT CHECK: ' + title if fabricated else title, fontsize=15, y=.97)
    fig.legend([Line2D([], [], marker="o", linestyle="", color="#68788b"),
                Line2D([], [], marker="D", linestyle="", markerfacecolor="white", color="#68788b")],
               ["Individual seed fit / deterministic reference", "Accuracy mean / latency median"],
               loc="upper center", bbox_to_anchor=(.5, .925), ncol=2, frameon=False)
    fig.text(.035, .025, CAPTION + "\nLatency includes normalization, casting, conditioning, gate/operator preparation, forecast, denormalization and a uniform deadline check; excludes model loading.", fontsize=9.5, va="bottom")
    fig.subplots_adjust(left=.15, right=.98, bottom=.27, top=.82, wspace=.20)
    fig.savefig(folder / "benchmark.png", dpi=160)
    fig.savefig(folder / "benchmark.pdf")
    plt.close(fig)
    return scales


def metric_table(rows):
    lines = ['| Recording | Model | Rate | Seed | Horizon | Status | Standardized RMSE | Physical RMSE (deg) | Per-joint RMSE (deg) | Error |',
             '|---|---|---:|---:|---:|---|---:|---:|---|---|']
    for row in rows:
        m = row['metrics'] or {}
        recording = row['recording'].removeprefix('recording_2021_12_15_').removesuffix('.mat')
        joints = ', '.join(fmt(v) for v in m.get('per_joint_rmse_deg', []))
        lines.append(f"| {recording} | {LABELS[row['arm']]} | {fmt(row['learning_rate'])} | {row['seed']} | {row['horizon']} | {row['status']} | {fmt(m.get('standardized_rmse'))} | {fmt(m.get('physical_rmse_deg'))} | {joints} | {clean(row.get('error') or '')} |")
    return lines


def tables(values):
    lines = ['# History initialization: all saved results', '', CAPTION.replace('\n', ' '), '',
             f"Scientific outcome: **{values['result']['status']}**, {values['result']['passed']}/5 criteria.", '',
             '## All208 ordinary scores', '',
             'All seeds and both rates are retained. H64 is descriptive; rate selection uses pooled H128 SSE. Both ridge penalties remain comparison controls.', '']
    lines += metric_table(values['all_rows'])
    lines += ['', '## Rate selection', '', '| Family | Selected rate | All options |', '|---|---:|---|']
    for arm in ARMS:
        options = '; '.join(f"{r['rate']:g}: {'eligible' if r['eligible'] else 'INELIGIBLE'}, pooled RMSE {fmt(r['pooled_rmse'])}" for r in values['selection']['options'][arm])
        lines.append(f"| {LABELS[arm]} | {fmt(values['selection']['selected_rates'][arm])} | {options} |")
    lines += ['', '## All48 fits and request costs', '',
              'All18 fresh attempts and30 unchanged cached fits are retained. Cached fitting times are historical. A missing latency measurement is not zero.', '',
              '| Model | Rate | Seed | Origin | Effective / helper status | Updates | Optimizer loop (s) | Parameters | Inactive | Weight / state / buffer / normalizer (B) | Request median (ms) |',
              '|---|---:|---:|---|---|---:|---:|---:|---:|---|---:|']
    timing = {(r['arm'], r['seed'], r.get('learning_rate')): r['timing'] for r in values['resources']}
    costs = [{**f['resources'], **{k: f[k] for k in ('arm', 'seed', 'learning_rate', 'fit', 'origin')},
              'effective_status': f.get('effective_status', f['fit']['status']),
              'timing': timing.get((f['arm'], f['seed'], f['learning_rate']))} for f in values['fits']]
    costs += [{**r, 'fit': None} for r in values['resources'] if r['arm'] in REFERENCES]
    for row in costs:
        fit, duration = row['fit'] or {}, row.get('timing')
        storage = ' / '.join(str(row[k]) for k in ('parameter_bytes', 'state_bytes', 'buffer_bytes', 'normalizer_bytes'))
        lines.append(f"| {LABELS[row['arm']]} | {fmt(row.get('learning_rate'))} | {row['seed']} | {row.get('origin', 'reference')} | {row.get('effective_status', 'reference')} / {fit.get('status', 'reference')} | {fit.get('completed_updates', 'n/a')} | {fmt(fit.get('optimizer_seconds'))} | {row['parameters']} | {row.get('inactive_parameters', 0)} | {storage} | {fmt(duration['median_seconds'] * 1000 if duration else None)} |")
    lines += ['', 'Storage includes numeric parameters, state, buffers and normalizers. Request arrays are separate; temporary workspace and Python overhead are not measured. Legacy instant LPV retains192 inactive recurrent weights. Model sizes and precision differ.', '',
              '## Five frozen criteria', '', '| Criterion | Passed |', '|---|---|']
    lines += [f"| {clean(c['name'])} | {c['passed']} |" for c in values['result']['conditions']]
    lines += ['', '## Older-order corruption: all36 descriptive scores', '',
              'The selected primary models receive the paired reversal29..0,30,31. Later positions and future inputs do not change. Local predictions must remain exactly invariant. Temporal numerical failures remain failed diagnostic attempts and cannot substitute for, or change, the five primary criteria.', '']
    lines += metric_table(values['permutation']['rows'])
    lines += ['', '## All18 diagnostic attempt receipts', '', '| Recording | Model | Rate | Seed | Status | Check / error |', '|---|---|---:|---:|---|---|']
    for row in values['permutation']['checks']:
        lines.append(f"| {row['recording']} | {LABELS[row['arm']]} | {fmt(row['learning_rate'])} | {row['seed']} | {row['status']} | {clean(row.get('check') or row.get('error') or row.get('reason') or '')} |")
    lines += ['', 'History could encode preprocessing state, physical state or a statistical regularity. This study does not identify that mechanism. A pass only supports a separately registered confirmation study. No biological, calibration, novel-architecture, official-benchmark or robot-control claim follows.', '']
    return '\n'.join(lines)


def write_csv(path, rows):
    metric_keys = ['standardized_rmse', 'physical_rmse_deg', 'standardized_sse', 'scalars', 'windows', 'per_joint_rmse_deg']
    keys = ['recording', 'arm', 'seed', 'learning_rate', 'horizon', 'status']
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[*keys, *metric_keys, 'error'], lineterminator='\n')
        writer.writeheader()
        for row in rows:
            metrics = {k: (row['metrics'] or {}).get(k) for k in metric_keys}
            metrics['per_joint_rmse_deg'] = json.dumps(metrics['per_joint_rmse_deg'])
            writer.writerow({**{k: row[k] for k in keys}, **metrics, 'error': json.dumps(row.get('error'), sort_keys=True)})


def render(study, output, audit_path, engineering):
    auth = authenticate(study, audit_path, engineering)
    study, output = Path(auth['study']), Path(output).resolve()
    require(not output.is_relative_to(study), 'publication outside immutable study')
    before = pin(__file__)
    results = read_json(study / 'results.json')
    resources = read_json(study / 'resources.json')
    permutation = read_json(study / 'permutation.json')
    require(results['config'] == auth['plan']['config'] and results['result']['status'] == auth['receipt']['scientific_result'],
            'result/config/receipt join')
    scalar_agreement(auth['audit']['results'], {k: results[k] for k in ('rows', 'selection', 'result')})
    scalar_agreement(auth['audit']['permutation'], permutation)
    require(auth['audit']['resources'] == resources, 'audited resource join')
    values = extract(results, resources, read_json(study / 'fits.json'), permutation)
    output.mkdir(parents=True, exist_ok=False)
    values['axis_specs'] = axis_specs(values)
    values['axis_scales'] = figure(values, output)
    (output / 'table.md').write_text(tables(values))
    write_csv(output / 'all-candidates.csv', values['all_rows'])
    write_csv(output / 'permutation.csv', permutation['rows'])
    (output / 'plotted-values.json').write_text(json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + '\n')
    require(authenticate(study, audit_path, engineering) == auth and pin(__file__) == before, 'presentation inputs/source changed')
    receipt = {'version': VERSION, 'study': str(study), 'registration_sha256': PLAN_SHA,
               'inputs': auth['inputs'], 'script': before, 'scope': 'Closed audited scalars only; no models or numerical replay.',
               'outputs': {p.name: pin(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('study', 'output', 'audit', 'engineering'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    render(args.study, args.output, args.audit, args.engineering)
