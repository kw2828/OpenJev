"""Present a closed Silverbox development study from audited scalar JSON only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'phase-study-plot-v1'
SEEDS = (7301, 7302, 7303)
ARMS = ('fixed_phase', 'energy_phase', 'nonlinear_readout', 'gru16', 'cubic_ar2')
REFERENCES = ('static_cubic', 'fir128', 'fir512', 'cubic_ar2_frozen')
METHODS = (*ARMS, *REFERENCES)
LABELS = ('Fixed phase', 'Energy phase', 'Nonlinear readout', 'GRU16', 'Refined cubic AR2',
          'Static cubic', 'FIR128', 'FIR512', 'Frozen cubic AR2')
PARTITIONS = ('dev_a', 'dev_b')


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
    require(type(receipt['elapsed_seconds']) in (int, float)
            and math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] > 0,
            'finite original process duration required')
    require(type(receipt['argv']) is list and all(type(x) is str for x in receipt['argv']),
            'original command required')
    return receipt


def condition_names():
    names = ['all_completed_predictions_finite']
    for partition in PARTITIONS:
        for control in ('fixed_phase', 'nonlinear_readout'):
            names.append(f'{partition}_mean_energy_improves_{control}_10pct')
            names.extend(f'{partition}_seed{seed}_energy_not_worse_{control}' for seed in SEEDS)
        names.extend((f'{partition}_within_5pct_best_conventional',
                      f'{partition}_competent_below_10pct_fit_std'))
    return names


def authenticate(study, audit_path, run_receipt, audit_receipt):
    """Metadata admission plus opaque byte hashes, never numerical audit replay."""
    study, audit_path = Path(study).resolve(), Path(audit_path).resolve()
    run_receipt, audit_receipt = Path(run_receipt).resolve(), Path(audit_receipt).resolve()
    run_process, audit_process = closed(run_receipt), closed(audit_receipt)
    command = run_process['argv']
    require(len(command) == 11 and absolute(command[0]) == ROOT / '.venv/bin/python'
            and absolute(command[1]) == ROOT / 'scripts/phase_study.py'
            and command[2:4] == ['run', '--csv'], 'original study command required')
    csv_path = absolute(command[4])
    command = audit_process['argv']
    require(len(command) == 10 and absolute(command[0]) == ROOT / '.venv/bin/python'
            and absolute(command[1]) == ROOT / 'scripts/audit_phase_study.py'
            and command[2] == '--study' and absolute(command[3]) == study
            and command[4] == '--output' and absolute(command[5]) == audit_path
            and command[6] == '--run-receipt' and absolute(command[7]) == run_receipt
            and command[8] == '--csv' and absolute(command[9]) == csv_path,
            'original independent audit command and paths required')
    plan = read(study / 'registration.json')
    checker_path = ROOT / 'scripts/audit_phase_study.py'
    require(descriptor(checker_path) == plan['sources']['scripts/audit_phase_study.py'],
            'registered metadata admission source required before import')
    spec = importlib.util.spec_from_file_location('phase_plot_admission', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    admission = checker.authenticate(study, csv_path, run_receipt)
    require(checker.closed_receipt(audit_receipt, plan['sources'], cap=600) == audit_process,
            'original audit source, wrapper, log and cap joins required')
    audit_manifest_path = audit_path.parent / 'manifest.json'
    audit_pin = descriptor(audit_path)
    require(read(audit_manifest_path) == {'files': {audit_path.name: audit_pin}},
            'exact original audit output manifest required before scalar reads')
    audit = read(audit_path)
    require(audit['version'] == 'phase-study-audit-v1' and audit['agreement'] is True
            and audit['admission'] == admission, 'original independent agreement and evidence joins required')
    expected_counts = {'npz_decodes': 64, 'npy_decodes': 41, 'array_loads': 504,
        'checkpoint_decodes': 30, 'optimizer_decodes': 15, 'prediction_replays': 38,
        'independent_ridge_solves': 4, 'ridge_certificates': 4,
        'ridge_solution_prediction_comparisons': 12, 'window_order_reconstructions': 3,
        'official_test_values_read': 0, 'producer_calls': 0, 'torch_calls': 0, 'optimizer_updates': 0}
    require(audit['counts'] == expected_counts, 'complete independent audit scope required')
    require(type(audit['seconds']) in (int, float) and math.isfinite(audit['seconds'])
            and 0 < audit['seconds'] <= audit_process['elapsed_seconds'], 'original audit duration join')
    require(audit['tolerance']['normalized_rtol'] == audit['tolerance']['normalized_atol'] == 1e-4,
            'fixed normalized inference tolerance')
    require(set(audit['ridge_certificates']) == set(REFERENCES), 'all four saved ridge certificates')
    # Presentation values are admitted only after both original closures and full provenance joins.
    result, resources, run = (read(study / name) for name in ('results.json', 'resources.json', 'run.json'))
    checker.compare(result, audit['results'], 'saved audited result')
    checker.compare(resources, audit['resources'], 'saved audited logical storage')
    require(result['total'] == 21 and [c['name'] for c in result['conditions']] == condition_names()
            and all(type(c['passed']) is bool for c in result['conditions'])
            and result['passed'] == sum(c['passed'] for c in result['conditions']), 'complete 21-condition result')
    expected_outcome = 'ADVANCE_PHASE_MECHANISM' if result['passed'] == 21 else 'DO_NOT_ADVANCE_PHASE_MECHANISM'
    require(result['outcome'] == expected_outcome, 'unchanged frozen outcome')
    keys = {f'{arm}/{seed}' for arm in ARMS for seed in SEEDS} | set(REFERENCES)
    require(set(result['scores']) == set(PARTITIONS), 'both internal DEV partitions required')
    for partition in PARTITIONS:
        require(set(result['scores'][partition]) == keys, 'all 19 model/reference rows required')
        for row in result['scores'][partition].values():
            require(set(row) == {'rmse_mv', 'mae_mv', 'scored_rows'} and row['scored_rows'] == 7680
                    and all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                            for k in ('rmse_mv', 'mae_mv')), 'finite complete saved score rows')
        require(set(result['summary'][partition]['family_mean_rmse_mv']) == set(ARMS)
                and set(result['summary'][partition]['reference_rmse_mv']) == set(REFERENCES),
                'complete saved mean and reference rows')
    fit_keys = {f'{arm}/{seed}' for arm in ARMS for seed in SEEDS}
    require(set(resources['trained_models']) == set(run['fit_seconds']) == fit_keys
            and set(resources['references']) == set(REFERENCES), 'all saved costs and fit times required')
    qualification_path = absolute(run_process['argv'][8])
    qualification = read(qualification_path)
    paths = [p for p in sorted(study.rglob('*')) if p.is_file()]
    paths += [audit_path, audit_manifest_path, run_receipt, audit_receipt, csv_path,
              absolute(run_process['argv'][6]), qualification_path,
              qualification_path.parent / qualification['log_path'],
              run_receipt.parent / run_process['log_path'], audit_receipt.parent / audit_process['log_path'],
              ROOT / 'output/phase-engineering-v1/invoke.py', Path(__file__).resolve()]
    paths += [ROOT / name for name in plan['sources']]
    inputs = {str(path): descriptor(path) for path in paths}
    require(inputs[str(audit_path)] == audit_pin, 'audit bytes unchanged during admission')
    return inputs, {'result': result, 'resources': resources, 'fit_seconds': run['fit_seconds'],
                    'admission': admission, 'audit_counts': audit['counts'], 'audit_tolerance': audit['tolerance'],
                    'ridge_certificates': audit['ridge_certificates'],
                    'audit_limitations': audit['limitations'], 'run_seconds': run_process['elapsed_seconds'],
                    'audit_seconds': audit_process['elapsed_seconds'],
                    'qualification_seconds': qualification['elapsed_seconds']}


def number(value):
    return f'{value:.6g}'


def figure(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    positive = all(row['rmse_mv'] > 0 for rows in result['scores'].values() for row in rows.values())
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 8), sharey=True, constrained_layout=True)
    colors = ('#5278a4', '#0e8370', '#a170aa')
    for ax, partition, title in zip(axes, PARTITIONS, ('DEV A', 'DEV B'), strict=True):
        summary = result['summary'][partition]
        for index, method in enumerate(METHODS):
            if method in ARMS:
                for offset, seed, color in zip((-.18, 0, .18), SEEDS, colors, strict=True):
                    score = result['scores'][partition][f'{method}/{seed}']['rmse_mv']
                    ax.scatter(score, index + offset, color=color, s=32, alpha=.9, zorder=3)
                score = summary['family_mean_rmse_mv'][method]
                ax.scatter(score, index, marker='D', color='#172b3a', s=50, zorder=4)
            else:
                score = summary['reference_rmse_mv'][method]
                ax.scatter(score, index, marker='s', color='#69737b', s=45, zorder=3)
            ax.annotate(number(score), (score, index), xytext=(7, -15), textcoords='offset points', fontsize=9)
        if positive:
            ax.set_xscale('log')
        else:
            ax.set_xlim(left=0)
        ax.set_yticks(range(len(METHODS)), LABELS, fontsize=10)
        ax.tick_params(labelleft=True)
        ax.axhline(4.5, color='#adb7bf', linewidth=.7)
        ax.grid(axis='x', color='#d9dde1', linewidth=.6)
        ax.margins(x=.24, y=.07)
        ax.set_title(f'{title} | 7,680 scored samples after 512 warmup', loc='left', fontsize=11)
        ax.set_xlabel('RMSE (mV), lower is better' + (' | logarithmic axis' if positive else ' | linear axis (zero retained)'))
    axes[0].invert_yaxis()
    fig.suptitle(f"Silverbox internal development: {result['passed']}/21 conditions\n"
                 f"{result['outcome']} | Official TEST unused", fontsize=15)
    handles = [Line2D([], [], marker='o', linestyle='', color=color, label=f'Seed {seed}')
               for seed, color in zip(SEEDS, colors, strict=True)]
    handles += [Line2D([], [], marker='D', linestyle='', color='#172b3a', label='Family mean (not ensemble)'),
                Line2D([], [], marker='s', linestyle='', color='#69737b', label='Single FIT ridge reference')]
    fig.legend(handles=handles, loc='outside lower center', ncols=3, frameon=False, fontsize=9)
    fig.savefig(output / 'benchmark.png', dpi=180)
    fig.savefig(output / 'benchmark.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return 'log' if positive else 'linear; an exact zero was retained without a display floor'


def document(data):
    result, resources = data['result'], data['resources']
    rows = ['# Small nonlinear phase memory on Silverbox', '',
        f"**{result['outcome']}: {result['passed']}/21 frozen conditions passed.**", '',
        ('This is an internal development comparison on one measured electronic oscillator. '
         'Official benchmark TEST outputs remain numerically unparsed. It is not an official benchmark '
         'score, a new-architecture claim, or evidence about biological memory or control performance.'), '',
        '![Every trained seed and classical reference on both internal DEV sequences](benchmark.png)', '',
        ('FIT contains 43,296 samples. All 15 final fits completed before either 8,192-sample DEV '
         'sequence was loaded. Each prediction starts from zero state and uses inputs only, including '
         'the common 512-sample warmup; the remaining 7,680 samples are scored. '
         'Training uses 2,048 updates per fit with the same per-seed 32-window batches, length 256 '
         'and 64-sample loss burn. Measured outputs enter FIT losses and AR2 initialization, '
         'but never a scored rollout or DEV checkpoint selection.'), '',
        ('Each phase model has four recurrent state scalars. The energy mechanism changes rotation '
         'using previous state energy; the 18-parameter nonlinear-readout control changes only the '
         'output map, while fixed phase has 14 parameters. All common initial tensors are paired '
         'and the three phase models initially implement the same function. GRU16 and the classical '
         'controls are not parameter- or compute-matched.'), '',
        '| Model / seed | DEV A RMSE mV | DEV A MAE mV | DEV B RMSE mV | DEV B MAE mV | Loop seconds | Logical bytes |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method, label in zip(METHODS, LABELS, strict=True):
        for seed in SEEDS if method in ARMS else (None,):
            key = f'{method}/{seed}' if seed is not None else method
            a, b = result['scores']['dev_a'][key], result['scores']['dev_b'][key]
            footprint = resources['trained_models'][key] if seed is not None else resources['references'][method]
            loop = number(data['fit_seconds'][key]) if seed is not None else 'n/a'
            name = f'{label} / {seed}' if seed is not None else label
            rows.append(f"| {name} | {number(a['rmse_mv'])} | {number(a['mae_mv'])} | "
                        f"{number(b['rmse_mv'])} | {number(b['mae_mv'])} | {loop} | {footprint['logical_total_bytes']:,} |")
    rows += ['', ('Logical bytes include deployed parameters, recurrent state or FIR input queues, and '
        '32 bytes of FIT normalizers. They exclude Python/native workspace, optimizer and training/audit '
        'arrays. Frozen AR2 stores its original float64 ridge coefficient file but simulates with '
        'float32 coefficients; the logical count uses that actual inference dtype. '
        'The figures do not establish deployment latency or a matched-compute advantage.'), '',
        ('Loop seconds begin after model/Adam construction and FIT tensor conversion, and include '
        'the optimizer loop plus final Adam-slot extraction. They exclude checkpoint serialization. '
        'They are not inference timings. AR2 has a predeclared smaller learning rate of 0.0001; '
        'phase/GRU models use 0.003. The frozen ridge AR2 remains a separate comparator, so '
        'refinement cannot conceal a stronger classical initialization.'), '',
        '| Frozen condition | Result |', '| --- | --- |']
    rows += [f"| `{condition['name']}` | {'PASS' if condition['passed'] else 'FAIL'} |"
             for condition in result['conditions']]
    rows += ['', ('The rule requires 10% mean improvements over both phase controls on both DEV '
        'sequences, no worse result for any paired seed, competence below 10% of FIT output '
        'standard deviation, and a result within 5% of the strongest conventional comparator. '
        'Three optimization seeds share the same physical observations; they are not independent '
        'datasets, and no significance claim is made. A pass only supports a separately frozen next '
        'experiment. A failed condition cannot be rescued by choosing a favorable seed, DEV sequence '
        'or comparator.'), '',
        ('The independent audit checked all 135 manifest-listed files, decoded 64 NPZ and 41 NPY '
        'files containing 504 saved arrays, and replayed 38 final predictions using independent '
        'NumPy equations. It checked all 30 initial/final model files, 15 Adam files and four ridge '
        'solutions. For each ridge solution it checked a regularized normal-equation backward error '
        'bounded by 64 times gamma_d, where gamma_d=d*u/(1-d*u), d is coefficient count and u is '
        'float64 unit roundoff. Saved-versus-independent predictions were compared on FIT and both '
        'DEV sequences (12 comparisons); coefficient differences remain diagnostics because '
        'ill-conditioned directions can have different coefficients but nearly identical predictions. '
        'All four certificates are retained in the plotted-values JSON. '
        'It did not replay 30,720 optimizer updates. Its fixed normalized per-point '
        'tolerance is rtol=atol=1e-4; this is an engineering guard, not a proof of incremental '
        'stability or arbitrary-parameter floating-point equivalence. The cell has a bounded-input, '
        'bounded-state argument, which does not imply bounded training gradients.'), '',
        (f"Original qualification process: {data['qualification_seconds']:.6f} seconds. "
         f"Original run: {data['run_seconds']:.6f} seconds. "
         f"Original independent audit: {data['audit_seconds']:.6f} seconds. "
         'These are whole-process durations. This presentation helper decoded no scientific arrays '
         'and made no model, optimizer or numerical audit calls.'), '',
        (f"Registration SHA256: `{data['admission']['registration']['sha256']}`. "
         f"Pre-fit source commit: `{data['admission']['registration_commit']}`."), '',
        ('[All plotted values, costs and conditions](plotted-values.json) | '
         '[PDF figure](benchmark.pdf) | [Presentation provenance](plot-receipt.json)'), '',
        ('Source: [Silverbox benchmark](https://www.nonlinearbenchmark.org/benchmarks/silverbox). '
         'Related mechanisms include [coRNN](https://arxiv.org/abs/2010.00951), '
         '[ReLiNet](https://www.ijcai.org/proceedings/2023/0385.pdf), and '
         '[polynomial nonlinear state-space identification](https://doi.org/10.1016/j.automatica.2010.01.001). '
         'The official initialization and TEST protocol differs from this internal comparison. '
         'Raw measurements and retained FIT/DEV arrays stay local because explicit redistribution '
         'permission has not been established; software licensing is a separate question.'), '']
    return '\n'.join(rows)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit', 'run-receipt', 'audit-receipt', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    inputs, data = authenticate(args.study, args.audit, args.run_receipt, args.audit_receipt)
    output = args.output.resolve()
    require(not output.is_relative_to(args.study.resolve()) and not output.is_relative_to(args.audit.parent.resolve()),
            'exclusive presentation folder outside original study and audit evidence required')
    output.mkdir(parents=True, exist_ok=False)
    axis = figure(output, data['result'])
    with (output / 'report.md').open('x') as stream:
        stream.write(document(data))
    write_json(output / 'plotted-values.json', {'version': VERSION, **data, 'rmse_axis': axis,
        'seed_display': 'Individual fits and saved family means, not ensemble predictions',
        'scope': 'Internal DEV only; official TEST unused'})
    after, after_data = authenticate(args.study, args.audit, args.run_receipt, args.audit_receipt)
    require(after == inputs and after_data == data, 'original evidence changed during presentation')
    outputs = ('benchmark.png', 'benchmark.pdf', 'report.md', 'plotted-values.json')
    write_json(output / 'plot-receipt.json', {'version': VERSION, 'inputs': inputs,
        'renderer': descriptor(Path(__file__)), 'admission': data['admission'],
        'result_reads_after_original_closure': True, 'source_pins_verified_before_and_after': True,
        'scientific_array_decodes': 0, 'model_calls': 0, 'optimizer_calls': 0, 'scientific_replays': 0,
        'visual_review_required': True, 'outputs': {name: descriptor(output / name) for name in outputs}})
    print(json.dumps({'outcome': data['result']['outcome'], 'output': str(output)}))


if __name__ == '__main__':
    main()
