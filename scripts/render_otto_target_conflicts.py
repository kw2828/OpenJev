"""Render completed, independently checked saved-label conflicts; no scientific calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'otto-target-conflicts-render-v1'
FITS = tuple(f'{kind}@{seed}' for seed in (30101, 30102, 30103) for kind in ('short', 'long'))
PAYLOADS = {'started.json', 'work.jsonl', 'views.npz', 'memberships.jsonl', 'groups.jsonl',
            'comparisons.jsonl', 'summary.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink file')
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def authenticate(args, bound):
    def bind(path, *, pin=None, expected=None):
        actual = digest(path)
        require(pin is None or actual['sha256'] == pin, 'external identity: ' + path.name)
        require(expected is None or actual == expected, 'payload identity: ' + path.name)
        bound[str(path)] = actual

    for path, pin in ((args.plan, args.plan_sha256), (args.run / 'receipt.json', args.worker_sha256),
                      (args.terminal, args.terminal_sha256), (args.audit, args.audit_sha256),
                      (args.audit_terminal, args.audit_terminal_sha256)):
        bind(path, pin=pin)
    plan, worker, audit = (read(path) for path in (args.plan, args.run / 'receipt.json', args.audit))
    require(plan['version'] == worker['version'] == 'otto-target-conflicts-v1'
            and worker['status'] == audit['status'] == 'completed' and audit['agreement'] is True
            and worker['plan_sha256'] == args.plan_sha256
            and audit['version'] == 'otto-target-conflicts-saved-audit-v1', 'completed agreeing diagnostic')
    require(audit['producer_inputs'] == {role: {'path': str(path), 'sha256': pin} for role, path, pin in (
            ('plan', args.plan, args.plan_sha256), ('worker', args.run / 'receipt.json', args.worker_sha256),
            ('terminal', args.terminal, args.terminal_sha256))}, 'independent audit joins')
    for directory, receipt, names in ((args.run, worker, PAYLOADS), (args.audit.parent, audit, {'started.json', 'audit.json'})):
        require(set(receipt['files']) == names and {p.name for p in directory.iterdir()} == names | {'receipt.json'},
                'exact completed file inventory')
        for name, desc in receipt['files'].items():
            bind(directory / name, expected=desc)
    for directory, receipt, terminal_path in ((args.run, worker, args.terminal), (args.audit.parent, audit, args.audit_terminal)):
        started, terminal = read(directory / 'started.json'), read(terminal_path)
        launch = started['launch']
        launch_path = Path(started['request']['supervision'])
        bind(launch_path, pin=receipt['supervision_sha256'])
        require(read(launch_path) == launch and all(terminal[k] == v for k, v in launch.items())
                and terminal['returncode'] == 0 and terminal['status'] == 'completed' and terminal['timed_out'] is False
                and terminal['error'] is terminal['clock_error'] is None and terminal['cleanup']['errors'] == []
                and terminal['group_absent'] is terminal['cleanup']['group_absent'] is terminal['cleanup']['reaped'] is True
                and launch['started_ns'] <= receipt['started_ns'] <= receipt['finished_ns']
                <= terminal['finished_ns'] < launch['deadline_ns'], 'original successful process')
    summary, checked = read(args.run / 'summary.json'), read(args.audit.parent / 'audit.json')
    require(checked['agreement'] is True and checked['summary'] == summary and checked['scientific_gate'] is None
            and worker['scientific_gate'] is audit['scientific_gate'] is None, 'audited descriptive summary')
    require(summary['rows'] == 558 and summary['occurrences'] == summary['denominator'] == 4464
            and [r['fit_id'] for r in summary['fits']] == list(FITS), 'complete fixed cohort and all six fits')
    for row in summary['fits']:
        require(math.isfinite(row['L']) and row['B'] == summary['B']
                and row['L_minus_B'] == row['L'] - row['B'], 'unchanged signed arithmetic')
    return summary, worker, audit


def render(summary, worker, audit, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = summary['fits']
    labels = [r['fit_id'].replace('short@', '80 / ').replace('long@', '320 / ') for r in rows]
    colors = ['#2878A8' if r['fit_id'].startswith('short') else '#CA641C' for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    fig.subplots_adjust(left=.08, right=.985, bottom=.28, top=.77, wspace=.26)
    for ax, key, title in zip(axes, ('L', 'L_minus_B'), ('Saved final TRAIN loss', 'Signed loss above the common floor'), strict=True):
        ax.bar(range(6), [r[key] for r in rows], color=colors, width=.7)
        ax.set_xticks(range(6), labels, rotation=35, ha='right')
        ax.set_xlabel('Epochs / paired seed')
        ax.set_ylabel('Weighted centered mean squared error')
        ax.set_title(title)
        ax.axhline(0, color='#333333', linewidth=.6)
        ax.grid(axis='y', alpha=.17)
        ax.set_axisbelow(True)
        ax.spines[['right', 'top']].set_visible(False)
    axes[0].axhline(summary['B'], color='#252525', linestyle='--', linewidth=1.6, label=f"Common floor B = {summary['B']:.6f}")
    axes[0].legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle('Exact-input target conflicts in the fixed TRAIN objective', fontsize=16, y=.97)
    fig.text(.5, .88, '558 anchors × 8 views | all six saved final models | no new model or simulator calls', ha='center', fontsize=10)
    fig.text(.5, .045, 'Blue: 80 epochs. Orange: 320 epochs. Byte-identical feature + mask groups only; signed excess is not clipped.', ha='center', fontsize=9)
    fig.text(.5, .01, 'An empirical relaxed floor, not population noise or a control result. Weighted-mean pooling adds no new information.', ha='center', fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'target-conflicts.{suffix}', dpi=180, bbox_inches='tight', pad_inches=.18, facecolor='white')
    plt.close(fig)
    fields = ('fit_id', 'L', 'B', 'L_minus_B', 'maximum_within_group_centered_score_spread',
              'groups_with_nonzero_score_spread', 'saved_L_difference')
    with (output / 'comparisons.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    means = {kind: {key: math.fsum(r[key] for r in rows if r['fit_id'].startswith(kind + '@')) / 3
                   for key in ('L', 'L_minus_B')} for kind in ('short', 'long')}
    for row in means.values():
        row['B'] = summary['B']
        row['B_fraction_of_L'] = summary['B'] / row['L'] if row['L'] else None
    write(output / 'plotted-values.json', {'summary': summary, 'family_means': means,
          'diagnostic_worker_seconds': worker['wall_seconds'], 'audit_worker_seconds': audit['wall_seconds'],
          'scope': 'Saved arithmetic only; no new score, gradient, trajectory, confidence threshold or efficacy gate.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'run', 'terminal', 'audit', 'audit-terminal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('plan', 'worker', 'terminal', 'audit', 'audit-terminal'):
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    require(all(getattr(args, name).is_absolute() for name in ('plan', 'run', 'terminal', 'audit', 'audit_terminal', 'output')),
            'absolute paths required')
    args.output.mkdir(parents=False, exist_ok=False)
    source = digest(Path(__file__).resolve())
    bound = {}
    receipt = {'version': VERSION, 'status': 'started', 'source': source,
               'request': {k: str(v) for k, v in vars(args).items()}, 'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0}
    try:
        summary, worker, audit = authenticate(args, bound)
        render(summary, worker, audit, args.output)
        for name, desc in bound.items():
            require(digest(Path(name)) == desc, 'unchanged rendered evidence')
        require(digest(Path(__file__).resolve()) == source, 'unchanged renderer source')
        receipt.update(status='completed', inputs=bound, files={p.name: digest(p) for p in args.output.iterdir()},
                       verification_scope='Full current diagnostic and audit output bytes; historical inputs and numerical checks inherited from the completed audit.')
        write(args.output / 'receipt.json', receipt)
        print(json.dumps({'status': 'completed', 'receipt': digest(args.output / 'receipt.json')}), flush=True)
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        try:
            target = args.output / 'receipt.json'
            if target.exists():
                target.rename(args.output / 'receipt.invalid.json')
            receipt['partial_files'] = {p.name: {'bytes': p.stat().st_size} for p in args.output.iterdir() if p.is_file()}
            write(target, receipt)
        except BaseException as secondary:  # noqa: BLE001 - preserve primary publication failure
            error.add_note(f'Failure receipt publication: {secondary!r}')
        raise


if __name__ == '__main__':
    main()
