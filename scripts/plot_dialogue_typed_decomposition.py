"""Render a completed branch/value diagnostic from its authenticated summary only.

This is a descriptive visualization, not an independent audit. In particular,
reading saved contribution partitions does not independently verify them. No
model, weights, prediction arrays, corpus, or producer module is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

METHODS = ('flat_stratum', 'flat_balanced', 'typed_stratum', 'typed_balanced')
SEEDS = (6101, 6102, 6103)
LABELS = ('Flat stratum', 'Flat balanced', 'Typed stratum', 'Typed balanced')
FITS = {f'{method}-{seed}' for method in METHODS for seed in SEEDS}
GROUP = 'heldout_service/changed'
COMPONENTS = ('branch', 'value')
PARTITIONS = ('CC', 'CW', 'WC', 'WW')
COLORS = ('#3d6d99', '#d39c45')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Finite metric required')
    return value


def count(value):
    require(type(value) is int and value >= 0, 'Nonnegative integer count required')
    return value


def near(left, right):
    require(math.isclose(left, right, abs_tol=1e-10, rel_tol=1e-10), 'Saved arithmetic inconsistency')


def cell(summary, method, seed):
    return summary['fits'][f'{method}-{seed}']['groups'][GROUP]


def pair(summary, seed):
    return summary['pairs']['typing_balanced'][str(seed)]


def validate(summary):
    require(summary['status'] == 'completed' and summary['version'] == 'dialogue-typed-decomposition-v1',
            'Completed diagnostic required')
    require(set(summary['fits']) == FITS, 'All twelve fits required')
    require(summary['original_continuation_allowed'] is False and summary['original_checks_passed'] == 5
            and summary['original_checks_total'] == 9, 'Closed original FAIL, five of nine passed, required')
    for key in ('execution_completed_sha256', 'published_summary_sha256'):
        pin = summary[key]
        require(type(pin) is str and len(pin) == 64 and all(c in '0123456789abcdef' for c in pin), 'Invalid input identity')
    require(set(summary['pairs']['typing_balanced']) == {str(s) for s in SEEDS}, 'All three paired seeds required')
    for method in METHODS:
        for seed in SEEDS:
            c = cell(summary, method, seed)
            require(c['rows'] == 578 and c['services'] == 6, 'Fixed primary population changed')
            losses = {k: finite(c['loss'][k]['equal_service']) for k in ('total', *COMPONENTS)}
            near(losses['total'], losses['branch']+losses['value'])
            # Preserve permitted tiny negative branch NLL instead of clipping it.
            require(losses['branch'] >= -2e-6 and losses['value'] >= -1e-10, 'Invalid component NLL')
            require(sum(count(c['decisions'][k]) for k in ('correct', 'wrong_selected_branch', 'wrong_value')) == 578,
                    'Actual error classes must partition all rows')
    for seed in SEEDS:
        p = pair(summary, seed)
        require(p['base'] == f'flat_balanced-{seed}' and p['candidate'] == f'typed_balanced-{seed}', 'Paired orientation')
        g = p['groups'][GROUP]
        counts = g['correctness_counts']
        require(set(counts) == set(PARTITIONS) and sum(count(v) for v in counts.values()) == 578, 'Paired partition')
        near(finite(g['accuracy_difference']), (counts['WC']-counts['CW'])/578)
        require(cell(summary, 'flat_balanced', seed)['decisions']['correct'] == counts['CC']+counts['CW']
                and cell(summary, 'typed_balanced', seed)['decisions']['correct'] == counts['CC']+counts['WC'], 'Paired correctness join')
        contributions = p['primary_contributions']['by_correctness']
        require(set(contributions) == set(PARTITIONS), 'Four contribution groups required')
        for key in PARTITIONS:
            part = contributions[key]
            require(part['rows'] == counts[key], 'Contribution population join')
            values = {k: finite(part['loss_difference'][k]['equal_service']) for k in ('total', *COMPONENTS)}
            near(values['total'], values['branch']+values['value'])
        for component in ('total', *COMPONENTS):
            delta = finite(g['loss_difference'][component]['equal_service'])
            near(delta, cell(summary, 'typed_balanced', seed)['loss'][component]['equal_service']
                 -cell(summary, 'flat_balanced', seed)['loss'][component]['equal_service'])
            near(delta, math.fsum(contributions[k]['loss_difference'][component]['equal_service'] for k in PARTITIONS))


def mean(values):
    return math.fsum(values)/len(values)


def render(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 13,
                         'axes.titleweight': 'bold', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(left=.075, right=.975, bottom=.17, top=.81, wspace=.24, hspace=.58)
    delta = {k: mean([pair(summary, s)['groups'][GROUP]['loss_difference'][k]['equal_service'] for s in SEEDS])
             for k in COMPONENTS}
    flips = {k: sum(pair(summary, s)['groups'][GROUP]['correctness_counts'][k] for s in SEEDS) for k in ('CW', 'WC')}
    fig.suptitle('Lower NLL, but more correct choices lost than repaired', x=.075, y=.975,
                 ha='left', fontsize=20, fontweight='bold')
    fig.text(.075, .937, 'Original continuation: FAIL (5 of 9 checks passed). Posthoc decomposition; no rule or decision changed.', fontsize=11)
    fig.text(.075, .907, f"Typed balanced minus flat balanced: mean branch NLL {delta['branch']:+.3f}, value NLL {delta['value']:+.3f} nats.\n"
             f"Across three paired seeds: {flips['CW']} correct-to-wrong choices versus {flips['WC']} wrong-to-correct choices.",
             fontsize=11, linespacing=1.5)
    positions = [i*4+j for i in range(4) for j in range(3)]
    names = [(method, seed) for method in METHODS for seed in SEEDS]
    ticks = [str(seed) for _, seed in names]
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#dddddd', linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    def fit_labels(ax):
        ax.set_xticks(positions, ticks, fontsize=8)
        ax.set_xlim(-.7, 14.7)
        for i, label in enumerate(LABELS):
            ax.text(i*4+1, -.14, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)

    ax = axes[0, 0]
    bottoms = [0.]*12
    for component, color, label in zip(COMPONENTS, COLORS, ('Target-branch NLL', 'Within-branch value NLL'), strict=True):
        values = [cell(summary, m, s)['loss'][component]['equal_service'] for m, s in names]
        ax.bar(positions, values, bottom=bottoms, width=.76, color=color, label=label)
        bottoms = [b+v for b, v in zip(bottoms, values, strict=True)]
    ax.set_ylabel('Equal-service NLL (nats)')
    ax.set_title('A  All 12 fits: exact loss decomposition', loc='left')
    fit_labels(ax)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_ylim(min(0., min(cell(summary, m, s)['loss']['branch']['equal_service'] for m, s in names)), max(bottoms)*1.23)

    ax = axes[0, 1]
    bottoms = [0]*12
    for key, color, label in (('correct', '#5b8d73', 'Correct'), ('wrong_selected_branch', '#b95850', 'Wrong selected branch'),
                               ('wrong_value', '#b19b76', 'Wrong value in correct branch')):
        values = [cell(summary, m, s)['decisions'][key] for m, s in names]
        ax.bar(positions, values, bottom=bottoms, width=.76, color=color, label=label)
        bottoms = [b+v for b, v in zip(bottoms, values, strict=True)]
    fit_labels(ax)
    ax.set_ylim(0, 740)
    ax.set_ylabel('Actual candidate decisions (578 per fit)')
    ax.set_title('B  The candidate argmax determines the error', loc='left')
    ax.legend(frameon=False, fontsize=9, loc='upper center')

    ax = axes[1, 0]
    maximum = 0
    for key, offset, color, label in (('CW', -.19, '#b95850', 'Correct to wrong (harm)'),
                                     ('WC', .19, '#5b8d73', 'Wrong to correct (repair)')):
        values = [pair(summary, s)['groups'][GROUP]['correctness_counts'][key] for s in SEEDS]
        maximum = max(maximum, *values)
        bars = ax.bar([i+offset for i in range(3)], values, width=.34, color=color, label=label)
        ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xticks(range(3), [str(s) for s in SEEDS])
    ax.set_ylim(0, maximum*1.35)
    ax.set_ylabel('Paired choices (same 578 rows per seed)')
    ax.set_title('C  Typed balanced versus flat balanced', loc='left')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.text(0, -.16, f"Sum across repeated fits: {flips['CW']} harms, {flips['WC']} repairs.\n"
            'These are counts, not rates with different denominators.', transform=ax.transAxes, va='top', fontsize=9)

    ax = axes[1, 1]
    for component, offset, color, label in zip(COMPONENTS, (-.19, .19), COLORS, ('Branch component', 'Value component'), strict=True):
        values = [mean([pair(summary, s)['primary_contributions']['by_correctness'][k]['loss_difference'][component]['equal_service']
                        for s in SEEDS]) for k in PARTITIONS]
        ax.bar([i+offset for i in range(4)], values, width=.34, color=color, label=label)
    ax.axhline(0, color='#444444', linewidth=.9)
    ax.set_xticks(range(4), ['Both correct\nCC', 'Correct to wrong\nCW', 'Wrong to correct\nWC', 'Both wrong\nWW'], fontsize=9)
    ax.set_ylabel('Contribution to mean NLL difference (nats)')
    ax.set_title('D  Loss can fall on rows that stay wrong', loc='left')
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    ax.text(0, -.22, 'Descriptive mean of three seeds; full primary equal-service weights.\n'
            'Bars sum to the total loss difference, not subgroup mean differences.', transform=ax.transAxes, va='top', fontsize=9)
    footer = ('Primary: 578 changed rows, 6 held-out services in historically exposed TRAIN; correct previous value supplied.\n'
              'All methods use the same rows. Repeated seeds are not independent data; counts and equal-service losses use different weights.\n'
              'Wrong branch means the branch of the selected candidate, not the largest summed branch mass.\n'
              'Saved-summary visualization only. Contribution bars are descriptive, not independently audited by this plotting script.')
    fig.text(.075, .025, footer, fontsize=9, va='bottom', linespacing=1.5)
    try:
        fig.savefig(out/'typed-decomposition.png', dpi=170, facecolor='white')
        fig.savefig(out/'typed-decomposition.pdf', facecolor='white')
    finally:
        plt.close(fig)


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_pin = None
    try:
        source_pin = digest(__file__)
        require(digest(args.summary) == args.summary_sha256, 'Externally pinned summary hash')
        summary = read(args.summary)
        validate(summary)
        render(summary, out)
        require(digest(args.summary) == args.summary_sha256 and digest(__file__) == source_pin, 'End source/input stability')
        write(out/'receipt.json', {'status': 'completed', 'version': 'dialogue-typed-decomposition-figure-v1',
              'source_sha256': source_pin, 'summary_sha256': args.summary_sha256,
              'summary_path': str(Path(args.summary).resolve()), 'execution_completed_sha256': summary['execution_completed_sha256'],
              'original_continuation_allowed': False, 'original_checks_passed': 5, 'original_checks_total': 9,
              'files': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()},
              'scope': 'Summary-only descriptive rendering. No independent contribution audit or new quality decision.',
              'model_calls': 0, 'prediction_array_reads': 0, 'wall_seconds': time.monotonic()-started})
        return {'status': 'completed', 'receipt_sha256': digest(out/'receipt.json'), 'source_sha256': source_pin}
    except BaseException as error:
        try:
            write(out/'failed.json', {'status': 'failed', 'error': repr(error), 'source_sha256': source_pin,
                                    'summary_sha256': args.summary_sha256, 'wall_seconds': time.monotonic()-started})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original rendering failure
            if callable(getattr(error, 'add_note', None)):
                error.add_note('Could not preserve rendering failure: '+repr(secondary))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(execute(parser.parse_args())))
