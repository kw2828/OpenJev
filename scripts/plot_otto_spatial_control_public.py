"""Display-only layout of one pinned, completed spatial-control report.

The sole scientific input is report-01/derivation.json at DATA_PIN. This file
does not import research code, load models, run native environments, train,
re-audit evidence, change a criterion, or choose a checkpoint. Values, scales,
labels and criterion text match report_otto_spatial_control.render. Only the
figure layout separates its previously overlapping legend and footer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

DATA_PIN = '703d884d27dc3b656254d3267598e249f08d8ed707f2e52d9b922ba7e420e222'
VERSION = 'otto-spatial-control-public-figure-v1'
KINDS = ('spatial', 'neighbor_free', 'cnn', 'dense128', 'statistics')
SEEDS = (10101, 10102, 10103)
REGIMES = ('lambda3', 'lambda4', 'lambda5')
GROUPS = ('competence_checks', 'improvement_checks', 'control_competence_checks')
SCOPE = ('Display-only plotting of a pinned saved derivation. No model, native '
         'environment, training, audit replay, statistical test, selection or '
         'change to the scientific evidence or continuation criteria.')


def descriptor(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def render(out, saved):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 3, figsize=(20, 15), layout=None)
    fig.subplots_adjust(left=.065, right=.985, bottom=.17, top=.895,
                        wspace=.18, hspace=.25)
    labels = ('Spatial\nsums', 'Neighbor-\nfree', 'Ordinary\nCNN',
              'Dense128', 'Statistics', 'Analytic\ninbounds')
    specs = (('found', 'Weighted success', 'linear'),
             ('steps', 'Mean capped moves', 'linear'),
             ('controller_seconds', 'Complete controller seconds/search', 'log'))
    bounds = []
    plotted_cells = set()
    for ri, (metric, ylabel, scale) in enumerate(specs):
        for ci, regime in enumerate(REGIMES):
            axis = axes[ri, ci]
            panel = saved['regimes'][regime]
            panel_bounds = saved['figure_panels'][ri * 3 + ci]
            require(panel_bounds['metric'] == metric
                    and panel_bounds['regime'] == regime
                    and panel_bounds['scale'] == scale, 'original panel order and scales')
            lower, upper = panel_bounds['limits']
            for si, (seed, color, marker) in enumerate(zip(
                    SEEDS, ('#2468a2', '#c06c22', '#448b50'), ('o', 's', '^'), strict=True)):
                arms = [f'{kind}@{seed}' for kind in KINDS]
                axis.scatter([i + (si - 1) * .16 for i in range(5)],
                             [panel['means'][arm][metric] for arm in arms],
                             color=color, marker=marker, s=61,
                             label=f'Fit seed {seed}', zorder=3)
                plotted_cells.update((regime, arm) for arm in arms)
            for ki, kind in enumerate(KINDS):
                value = panel['family_means'][kind][metric]
                axis.plot([ki - .27, ki + .27], [value, value], color='#252525',
                          linewidth=1.8, label='Equal-fit-seed mean' if ki == 0 else None,
                          zorder=2)
            axis.scatter([5], [panel['means']['analytic_inbounds'][metric]],
                         marker='D', s=66, color='#252525', label='Analytic control', zorder=3)
            plotted_cells.add((regime, 'analytic_inbounds'))
            axis.set(xticks=range(6), xticklabels=labels, xlim=(-.5, 5.5),
                     yscale=scale, ylim=(lower, upper))
            axis.set_ylabel(ylabel + (' (log scale)' if scale == 'log' else ''))
            if metric == 'found':
                axis.yaxis.set_major_formatter(PercentFormatter(1.))
                setting = ('training-supported setting' if regime != 'lambda5'
                           else 'supplied-model parameter shift')
                axis.set_title(f'{regime}: {setting}', fontsize=12)
            axis.grid(axis='y', which='both', alpha=.18)
            axis.tick_params(axis='x', labelsize=10)
            bounds.append({'regime': regime, 'metric': metric, 'scale': scale,
                           'limits': [lower, upper]})
    require(bounds == saved['figure_panels'], 'all original panel bounds retained')
    require(plotted_cells == {(c['regime'], c['arm']) for c in saved['cells']}
            and len(plotted_cells) == 48, 'all 48 saved cells displayed')
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, legend_labels, loc='center', bbox_to_anchor=(.525, .105),
                        ncol=5, frameon=False, fontsize=11)
    counts = saved['criterion_counts']
    competence, improvement, descriptive = (counts[k]['passed'] for k in GROUPS)
    status = 'PASS' if saved['pilot_continuation'] else 'NOT PASSED'
    title = fig.suptitle(
        'OTTO spatial controllers: all 15 saved heads and analytic control\n'
        f'Prespecified spatial screen: {status} ({competence + improvement}/66) | '
        f'Competence {competence}/18; paired controls {improvement}/48 | '
        f'Other-head competence {descriptive}/72 (descriptive)', fontsize=15, y=.968)
    header = fig.text(.525, .99, 'Display-only layout of the pinned saved report',
                      ha='center', va='top', fontsize=9, color='#555555')
    footer = fig.text(
        .525, .045,
        '72 paired cases; 24 per setting, 8 per initial-hit stratum. Mixture-weighted means; '
        'failures retain all 2,188 moves.\n'
        'Cost includes initialization, branches/features/readout/selection, every update and '
        'allocated head/module load; excludes measured artifact I/O.\n'
        'Native execution, prior fitting, qualification and audit are separate. '
        'One timing pass; no architectural advantage claim.',
        ha='center', va='center', fontsize=10, linespacing=1.5)

    # Inspect the rendered geometry before emitting either artifact. These checks
    # concern layout only and do not re-evaluate any scientific result.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {name: artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
             for name, artist in (('legend', legend), ('footer', footer),
                                  ('title', title), ('display_header', header))}
    bottom_axes = [axis.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                   for axis in axes[2]]
    top_axes = [axis.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                for axis in axes[0]]
    require(boxes['footer'].y1 < boxes['legend'].y0
            and boxes['legend'].y1 < min(b.y0 for b in bottom_axes),
            'footer, legend and plot labels occupy separate bands')
    require(max(b.y1 for b in top_axes) < boxes['title'].y0
            and boxes['title'].y1 < boxes['display_header'].y0,
            'panel titles, main title and display header do not overlap')
    require(all(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= 1 and b.y1 <= 1 for b in boxes.values()),
            'all figure-level text fits inside the canvas')
    for suffix in ('png', 'svg'):
        with (out / f'spatial-control-comparison.{suffix}').open('xb') as stream:
            fig.savefig(stream, format=suffix, dpi=180)
    plt.close(fig)
    return {'figure_panels': bounds, 'displayed_cells': len(plotted_cells),
            'legend_footer_separated': True,
            'figure_text_bounds': {name: list(box.extents) for name, box in boxes.items()},
            'matplotlib_version': matplotlib.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.input.is_absolute() and args.output.is_absolute(), 'absolute paths required')
    require(not args.input.is_symlink() and not args.output.exists()
            and not any(p.is_symlink() for p in args.output.parents), 'exclusive nonsymlink output')
    raw = args.input.read_bytes()
    require(descriptor(raw)['sha256'] == DATA_PIN, 'exact saved derivation pin before decode')
    saved = json.loads(raw)
    require(saved['version'] == 'otto-spatial-control-report-v1'
            and len(saved['cells']) == 48 and len(saved['figure_panels']) == 9,
            'complete saved display input')
    source = Path(__file__).resolve()
    source_before = descriptor(source.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    layout = render(args.output, saved)
    require(descriptor(source.read_bytes()) == source_before, 'plot source unchanged')
    files = {p.name: descriptor(p.read_bytes()) for p in sorted(args.output.iterdir())}
    receipt = {
        'version': VERSION, 'status': 'completed', 'scope': SCOPE,
        'source': {'path': str(source), **source_before},
        'input': {'path': str(args.input), **descriptor(raw)},
        'files': files, 'layout': layout, 'criterion_counts': saved['criterion_counts'],
        'pilot_continuation': saved['pilot_continuation'],
        'learned_architecture_advantage_established': saved['learned_architecture_advantage_established'],
        'model_calls': 0, 'training_calls': 0, 'simulator_calls': 0, 'audit_calls': 0,
        'wall_seconds': time.perf_counter() - started,
    }
    with (args.output / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'completed', 'output': str(args.output),
                      'source_sha256': source_before['sha256'],
                      'receipt_sha256': descriptor((args.output / 'receipt.json').read_bytes())['sha256']}))


if __name__ == '__main__':
    main()
