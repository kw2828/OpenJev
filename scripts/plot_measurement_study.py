"""Render only closed and independently audited measurement-memory results."""
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location('measurement_plot_admission', ROOT/'scripts/audit_measurement_study.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    study = ROOT/'output/measurement-v1'
    delivery = ROOT/'output/measurement-delivery-v1'
    admission = checker.authenticate(study, delivery/'run-process.json')
    audit = json.loads((delivery/'audit.json').read_text())
    closure = json.loads((delivery/'audit-process.json').read_text())
    command = [checker.absolute_argument(v) if i in (0, 1, 3, 5, 7) else v
               for i, v in enumerate(closure['command'])]
    expected = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/audit_measurement_study.py'),
                '--study', str(study), '--out', str(delivery/'audit.json'),
                '--process', str(delivery/'run-process.json')]
    if (closure['state'] != 'EXITED' or closure['returncode'] != 0 or command != expected
            or not math.isfinite(closure['elapsed_seconds']) or closure['elapsed_seconds'] <= 0
            or audit['agreement'] is not True or audit['admission'] != admission):
        raise ValueError('independent audit and original process closure required')
    out = ROOT/'research/measurement-results'
    out.mkdir(exist_ok=False)
    rows, resources = audit['rows'], audit['resources']
    methods = ['spectral118', 'dct118', 'bins118', 'coverage118', 'coverage98', 'recent98', 'full']
    populations = ['base', 'shift', 'long', 'long_shift']
    labels = ['Spectral118 (primary)', 'DCT118', 'Bins118', 'Coverage118', 'Coverage98', 'Recent98', 'Full-history GP']
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)
    for ax, metric, title in zip(axes.flat[:3], ['regret', 'nll', 'median_ms'],
                                ['Decision regret (lower is better)', 'Latent-exposure NLL (nats/path)',
                                 'Complete stream + four requests (ms)'], strict=True):
        values = np.empty((len(methods), len(populations)))
        for i, method in enumerate(methods):
            for j, population in enumerate(populations):
                if metric == 'median_ms':
                    values[i, j] = next(r[metric] for r in resources if (r['population'], r['method']) == (population, method))
                else:
                    values[i, j] = np.mean([r['metrics'][metric] for r in rows if (r['population'], r['method']) == (population, method)])
        ax.imshow(values, cmap='YlOrBr', aspect='auto')
        for (i, j), value in np.ndenumerate(values):
            norm = (value-values.min())/(values.max()-values.min()) if values.max() != values.min() else 0
            ax.text(j, i, f'{value:.4g}', ha='center', va='center', color='white' if norm > .55 else 'black', fontsize=10)
        ax.set_xticks(range(4), [p.upper() for p in populations], fontsize=10)
        ax.set_yticks(range(7), labels, fontsize=10)
        ax.set_title(title, loc='left', fontsize=13)
        ax.axhline(2.5, color='#617181', lw=.7)
        ax.axhline(5.5, color='#617181', lw=.7)
    ax = axes[1, 1]
    y = np.arange(len(methods))
    for index, (population, color) in enumerate([('base', '#286fa2'), ('long', '#168871')]):
        values = [next(r['logical_bytes'] for r in resources if (r['population'], r['method']) == (population, method)) for method in methods]
        position = y+(index-.5)*.28
        ax.barh(position, values, height=.25, label=population.upper(), color=color)
        for yi, value in zip(position, values, strict=True):
            ax.text(value+25, yi, f'{value:,}', va='center', fontsize=9)
    ax.axvline(1024, color='#c13628', linestyle='--', label='1,024-byte cap')
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 5400)
    ax.set_title('Persistent logical state (bytes/context)', loc='left', fontsize=13)
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlabel('SHIFT matches BASE; LONG_SHIFT matches LONG')
    figure.suptitle(f"Online observation consolidation: {audit['passed']}/{audit['total']} conditions\n"
                   +audit['gate'], fontsize=17)
    figure.supxlabel('Three cohorts per score. Timings: first context, median of ten probes. '
                       'Logical bytes exclude Python overhead and temporary workspace.\n'
                       'BASE/SHIFT are exact-prefix checks. Fixed known grid and kernel; no trained architecture claim.', fontsize=10)
    figure.savefig(out/'benchmark.png', dpi=180)
    figure.savefig(out/'benchmark.pdf')
    plt.close(figure)
    for name, original in [('audit.json', delivery/'audit.json'), ('summary.json', study/'summary.json'),
                           ('bootstrap.json', study/'bootstrap.json')]:
        with (out/name).open('xb') as stream:
            stream.write(original.read_bytes())
    receipt = {'version': 'measurement-plot-v1', 'admission': admission,
               'audit': checker.descriptor(delivery/'audit.json'),
               'audit_process': checker.descriptor(delivery/'audit-process.json'),
               'renderer': checker.descriptor(Path(__file__)), 'model_calls': 0, 'npz_decodes': 0,
               'outputs': {name: checker.descriptor(out/name) for name in
                           ('benchmark.png', 'benchmark.pdf', 'audit.json', 'summary.json', 'bootstrap.json')}}
    with (out/'plot-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'gate': audit['gate'], 'outputs': receipt['outputs']}))


if __name__ == '__main__':
    main()
