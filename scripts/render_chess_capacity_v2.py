"""Audited layout-only revision; preserve the original publisher and figures."""

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / 'scripts/publish_chess_capacity.py'


def render(publication, prefix, prior_figure):
    spec = importlib.util.spec_from_file_location('_frozen_capacity_publisher', PUBLISHER)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    publication, prefix, prior_figure = map(Path, (publication, prefix, prior_figure))
    verification = frozen.audit(publication)
    prior = frozen.read(prior_figure)
    publisher_hash = frozen.sha(PUBLISHER)
    frozen.require(prior['status'] == 'completed' and prior['verification'] == verification
                   and prior['publisher_sha256'] == publisher_hash,
                   'Prior figure does not bind the same verified evidence and frozen publisher')
    for ext in ('png', 'svg'):
        frozen.require(prior['plots'][ext] == frozen.sha(prior_figure.with_suffix('.'+ext)),
                       'Prior figure artifact changed')
    paths = {ext: prefix.with_suffix('.'+ext) for ext in ('png', 'svg', 'json')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Revised figure output already exists')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = frozen.figure(frozen.read(publication/'plan.json'), frozen.read(publication/'summary.json'))
    ax = fig.axes[0]
    legend = ax.get_legend()
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]
    legend.remove()
    ax.legend(handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(0, 1.13),
              ncol=3, borderaxespad=0, fontsize=8.5, frameon=False)
    fig.savefig(paths['png'], dpi=180)
    fig.savefig(paths['svg'], metadata={'Date': None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    frozen.write_new(paths['json'], {
        'status': 'completed', 'revision': 'Legend location only; all data and axes unchanged.',
        'verification': verification, 'wrapper_sha256': frozen.sha(__file__),
        'frozen_publisher_sha256': publisher_hash,
        'summary_sha256': frozen.sha(publication/'summary.json'),
        'prior_figure_receipt_sha256': frozen.sha(prior_figure),
        'prior_plots': prior['plots'],
        'plots': {ext: frozen.sha(paths[ext]) for ext in ('png', 'svg')},
    })
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--prior-figure', type=Path, default=ROOT/'docs/assets/chess-capacity-results.json')
    args = parser.parse_args()
    for ext, path in render(args.publication, args.prefix, args.prior_figure).items():
        print(f'{ext}: {path}')


if __name__ == '__main__':
    main()
