"""Present a closed TRAIN-only sensor screen without decoding scientific arrays."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'sensor-screen-plot-v1'
METHODS = ('s2linear', 's2quadratic', 'linear7', 'quadratic7',
           'rls-linear7-lambda1', 'rls-linear7-lambda.995',
           'rls-quadratic7-lambda1', 'rls-quadratic7-lambda.995', 'persistence')
LABELS = ('S2 linear', 'S2 quadratic', 'Seven-input linear', 'Seven-input quadratic',
          'Linear RLS, lambda 1', 'Linear RLS, lambda 0.995',
          'Quadratic RLS, lambda 1', 'Quadratic RLS, lambda 0.995', 'Persistence')
CONDITIONS = ('fit_rows_at_least_512', 'may_rows_at_least_256', 'june_rows_at_least_256',
              'may_static_rmse_above_1pct_fit_std', 'june_static_rmse_above_1pct_fit_std',
              'same_rls_beats_best_static_by_10pct_both_months')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'ordinary evidence file required')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def read(path):
    descriptor(path)
    return json.loads(Path(path).read_text())


def absolute(value):
    require(type(value) is str, 'string command argument required')
    path = Path(value)
    return path.absolute() if path.is_absolute() else (ROOT / path).absolute()


def closed(path):
    receipt = read(path)
    require(receipt['state'] == 'EXITED' and type(receipt['returncode']) is int
            and receipt['returncode'] == 0, 'original clean successful process required')
    seconds = receipt['elapsed_seconds']
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
            'finite original process duration required')
    require(type(receipt['argv']) is list and all(type(x) is str for x in receipt['argv']),
            'original command required')
    return receipt


def authenticate(study, audit_path, run_receipt, audit_receipt):
    """Only JSON, source bytes and opaque hashes; never call the numerical audit."""
    study, audit_path = Path(study).resolve(), Path(audit_path).resolve()
    run_receipt, audit_receipt = Path(run_receipt).resolve(), Path(audit_receipt).resolve()
    run_process, audit_process = closed(run_receipt), closed(audit_receipt)
    command = run_process['argv']
    require(len(command) == 11 and absolute(command[0]) == ROOT / '.venv/bin/python'
            and absolute(command[1]) == ROOT / 'scripts/sensor_screen.py'
            and command[2:4] == ['run', '--csv'], 'original screen command required')
    csv_path = absolute(command[4])
    command = audit_process['argv']
    require(len(command) == 10 and absolute(command[0]) == ROOT / '.venv/bin/python'
            and absolute(command[1]) == ROOT / 'scripts/audit_sensor_screen.py'
            and command[2] == '--study' and absolute(command[3]) == study
            and command[4] == '--output' and absolute(command[5]) == audit_path
            and command[6] == '--run-receipt' and absolute(command[7]) == run_receipt
            and command[8] == '--csv' and absolute(command[9]) == csv_path,
            'original independent audit command and evidence paths required')
    plan = read(study / 'registration.json')
    checker_path = ROOT / 'scripts/audit_sensor_screen.py'
    require(descriptor(checker_path) == plan['sources']['scripts/audit_sensor_screen.py'],
            'registered metadata admission source required before importing it')
    spec = importlib.util.spec_from_file_location('sensor_plot_admission', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    admission = checker.authenticate(study, csv_path, run_receipt)
    require(checker.closed_receipt(audit_receipt, plan['sources']) == audit_process,
            'original audit source, log and process pins required')
    audit = read(audit_path)
    require(audit['version'] == 'sensor-screen-audit-v1' and audit['agreement'] is True
            and audit['admission'] == admission, 'original independent agreement and evidence joins required')
    require(audit['counts']['npz_decodes'] == 3 and audit['counts']['numerical_segments'] == ['train']
            and audit['counts']['held_out_numeric_values'] == 0
            and audit['counts']['model_calls'] == audit['counts']['producer_calls'] == 0,
            'declared TRAIN-only independent audit scope required')
    require(type(audit['seconds']) in (int, float) and math.isfinite(audit['seconds'])
            and 0 < audit['seconds'] <= audit_process['elapsed_seconds'], 'original audit duration join')
    # Read presentation numbers only after both original closures and provenance admission.
    result, resources = read(study / 'results.json'), read(study / 'resources.json')
    checker.compare(result, audit['results'], 'saved audited screen result')
    checker.compare(resources, audit['resources'], 'saved audited logical storage')
    require(result['total'] == 6 and [c['name'] for c in result['conditions']] == list(CONDITIONS)
            and all(type(c['passed']) is bool for c in result['conditions'])
            and result['passed'] == sum(c['passed'] for c in result['conditions']), 'complete six-condition result')
    expected = 'QUALIFIES_LEARNED_MEMORY_SCREEN' if result['passed'] == 6 else 'REJECT_BENZENE_MEMORY_BENCHMARK'
    require(result['outcome'] == expected and set(resources['logical_persistent_bytes']) == set(METHODS),
            'unchanged recorded outcome and all nine resource rows')
    require(set(result['metrics']) == {'may', 'june'}, 'both recorded months required')
    for month in ('may', 'june'):
        require(set(result['metrics'][month]) == set(METHODS), 'all nine methods required')
        for row in result['metrics'][month].values():
            require(set(row) == {'rmse', 'mae'} and all(v is None or
                    (type(v) in (int, float) and math.isfinite(v) and v >= 0) for v in row.values()),
                    'finite nonnegative saved errors or explicit unsupported rows')
    paths = [p for p in sorted(study.rglob('*')) if p.is_file()]
    paths += [audit_path, run_receipt, audit_receipt, csv_path, Path(__file__).resolve()]
    # External qualification log and receipt are already joined by the auditor.
    run_command = run_process['argv']
    qualification_path = absolute(run_command[8])
    qualification = read(qualification_path)
    paths += [absolute(run_command[6]), qualification_path,
              qualification_path.parent / qualification['log_path'],
              run_receipt.parent / run_process['log_path'], audit_receipt.parent / audit_process['log_path'],
              ROOT / 'output/sensor-screen-engineering-v1/invoke.py']
    inputs = {str(path): descriptor(path) for path in paths}
    return inputs, {'result': result, 'resources': resources, 'admission': admission,
                    'audit_counts': audit['counts'], 'run_seconds': run_process['elapsed_seconds'],
                    'audit_seconds': audit_process['elapsed_seconds']}


def value(number):
    return 'n/a' if number is None else f'{number:.6g}'


def figure(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    threshold = .01 * result['fit_target_std']
    positive = [row['rmse'] for month in result['metrics'].values() for row in month.values()
                if row['rmse'] is not None]
    use_log = bool(positive) and min(positive) > 0 and threshold > 0 and (
        max(positive + [threshold]) / min(positive + [threshold]) > 30)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7), sharey=True, constrained_layout=True)
    colors = ['#245b91'] * 4 + ['#087f6c'] * 4 + ['#777777']
    for ax, month in zip(axes, ('may', 'june'), strict=True):
        for index, (method, color) in enumerate(zip(METHODS, colors, strict=True)):
            score = result['metrics'][month][method]['rmse']
            if score is None:
                ax.text(.02, index, 'n/a', transform=ax.get_yaxis_transform(), va='center')
            else:
                ax.scatter(score, index, color=color, s=50, zorder=3)
                ax.annotate(value(score), (score, index), xytext=(7, 5),
                            textcoords='offset points', fontsize=9)
        if use_log:
            ax.set_xscale('log')
        else:
            ax.set_xlim(left=0)
        ax.axvline(threshold, color='#b44c36', linestyle='--', linewidth=1.2)
        ax.set_yticks(range(9), LABELS, fontsize=10)
        ax.tick_params(labelleft=True)
        ax.grid(axis='x', color='#d9dde1', linewidth=.6)
        ax.axhline(3.5, color='#aeb6bf', linewidth=.6)
        ax.axhline(7.5, color='#aeb6bf', linewidth=.6)
        ax.margins(x=.32, y=.10)
        ax.set_title(f"{month.title()} 2004 | {result['month_rows'][month]:,} scored rows", loc='left')
        ax.set_xlabel('Benzene RMSE (original units; lower is better)' + ('\nLog scale' if use_log else ''))
    axes[0].invert_yaxis()
    fig.suptitle(f"TRAIN-only delayed sensor screen: {result['passed']}/6 conditions\n{result['outcome']}",
                 fontsize=15)
    handles = [Line2D([], [], marker='o', linestyle='', color='#245b91', label='Static calibration'),
               Line2D([], [], marker='o', linestyle='', color='#087f6c', label='Adaptive RLS'),
               Line2D([], [], marker='o', linestyle='', color='#777777', label='Persistence'),
               Line2D([], [], linestyle='--', color='#b44c36', label=f'1% FIT target std = {value(threshold)}')]
    fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False, fontsize=9)
    fig.savefig(output / 'benchmark.png', dpi=180)
    fig.savefig(output / 'benchmark.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return 'log' if use_log else 'linear'


def document(data):
    result, resources = data['result'], data['resources']
    qualifiers = ', '.join(result['adaptive_qualifiers']) or 'None'
    rows = ['# TRAIN-only delayed sensor-memory screen', '',
            f"**{result['outcome']}: {result['passed']}/6 conditions passed.**", '',
            ('This is a benchmark usefulness screen on March-June 2004 TRAIN data. '
             'It is not held-out performance or a learned-architecture result.'), '',
            '![All nine methods, both TRAIN months](benchmark.png)', '',
            (f"FIT contains {result['fit_rows']:,} complete rows whose labels were strictly released before "
            'May 1. Seven input normalizers were fitted only on those rows. '
            'Each screen label arrived 24 calendar hours after its input; predictions were made before '
            'the current label was available. Missing hours and measurements did not shorten the delay.'), '',
            (f"The near-solution threshold is 1% of FIT target population standard deviation: "
            f"{value(.01 * result['fit_target_std'])} in original target units. "
            f"Scored rows: May {result['month_rows']['may']:,}; June {result['month_rows']['june']:,}."), '',
            '| Method | May RMSE | May MAE | June RMSE | June MAE | Logical bytes |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method, label in zip(METHODS, LABELS, strict=True):
        may, june = result['metrics']['may'][method], result['metrics']['june'][method]
        rows.append(f"| {label} | {value(may['rmse'])} | {value(may['mae'])} | "
                    f"{value(june['rmse'])} | {value(june['mae'])} | "
                    f"{resources['logical_persistent_bytes'][method]:,} |")
    rows += ['', '| Frozen condition | Result |', '| --- | --- |']
    rows += [f"| `{c['name']}` | {'PASS' if c['passed'] else 'FAIL'} |" for c in result['conditions']]
    rows += ['', f'RLS configurations satisfying the same-configuration, both-month comparison: {qualifiers}.', '',
             ('The best static comparator is the lowest RMSE among all four static models separately '
             'in each month. Persistence is reported but excluded from that gate. '
             'A successful screen only authorizes designing a separately registered learned-memory '
             'comparison. Failure rejects this benzene-memory benchmark under the frozen rule; '
             'it does not authorize dropping a sensor, changing the target or relaxing a condition.'), '',
             ('There is no fixed byte cap or matched-storage architecture comparison here. Each RLS '
             'model includes a 24 by 7 float64 pending-input queue (1,344 bytes), its actual dense '
             'precision/information state, duplicated normalizers, decay, clock and design tag. '
             'Python/native workspace and fit-only or audit arrays are outside this logical count. '
             'The screen makes no latency, calibrated uncertainty, novelty or generalization claim. '
             'The 24-hour delay is imposed by this experiment, not a measured analyzer latency.'), '',
             ('No held-out numerical values were accessed. Missing labels are scored only where present; '
             'all nine methods use the same complete-input and available-target scoring rows. '
             'A strong S2 calibration fit alone would not establish how the reference target was produced.'), '',
             (f"Original run process: {data['run_seconds']:.6f} seconds. "
             f"Original independent audit process: {data['audit_seconds']:.6f} seconds. "
             'These are whole-process durations, not model latency comparisons. '
             'The independent audit replayed TRAIN equations from saved evidence; this presentation '
             'helper performed no array decoding, fitting or scientific replay.'), '',
             (f"Registration SHA256: `{data['admission']['registration']['sha256']}`. "
             f"Registered commit: `{data['admission']['registration_commit']}`."), '',
             ('[Saved plotted values and all conditions](plotted-values.json) | '
             '[PDF figure](benchmark.pdf) | [Presentation provenance](plot-receipt.json)'), '',
             ('Source: [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality). '
             'Raw measurements and retained TRAIN arrays are not copied into this presentation. '
             'The source page contains conflicting license language; no unrestricted commercial-rights claim is made.'), '']
    return '\n'.join(rows)


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit', 'run-receipt', 'audit-receipt', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    inputs, data = authenticate(args.study, args.audit, args.run_receipt, args.audit_receipt)
    output = args.output.resolve()
    require(not output.is_relative_to(args.study.resolve()), 'presentation output must be outside original study')
    output.mkdir(parents=True, exist_ok=False)
    axis = figure(output, data['result'])
    with (output / 'report.md').open('x') as stream:
        stream.write(document(data))
    write_json(output / 'plotted-values.json', {'version': VERSION, 'results': data['result'],
               'resources': data['resources'], 'rmse_axis': axis,
               'threshold': .01 * data['result']['fit_target_std']})
    after, after_data = authenticate(args.study, args.audit, args.run_receipt, args.audit_receipt)
    require(after == inputs and after_data == data, 'original evidence changed during presentation')
    names = ('benchmark.png', 'benchmark.pdf', 'report.md', 'plotted-values.json')
    write_json(output / 'plot-receipt.json', {'version': VERSION, 'inputs': inputs,
               'renderer': descriptor(Path(__file__)), 'admission': data['admission'],
               'result_reads_after_original_closure': True, 'source_pins_verified_before_and_after': True,
               'npz_decodes': 0, 'model_calls': 0, 'scientific_replays': 0,
               'outputs': {name: descriptor(output / name) for name in names}})
    print(json.dumps({'outcome': data['result']['outcome'], 'output': str(output)}))


if __name__ == '__main__':
    main()
