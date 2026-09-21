"""Stage standalone LaTeX and figures from closed, audited Bellman evidence only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/report_otto_bellman_control.py'
HELPER_PIN = '8fe5f1db28250e41c41291198ddb064dae93a8f1dc8a6f3c4cdd1537a2c0aa6c'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_PIN:
    raise ValueError('pinned read-only report authenticator')
_spec = importlib.util.spec_from_file_location('_bellman_paper_auth', HELPER)
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)
require, write, digest = R.require, R.write, R.digest
VERSION = 'otto-bellman-control-paper-v1'
IMAGE = 'sha256:4984977ccf5afe883cb382d0163f267de0d029d140bb7a9e8f4c19f0b781d57b'
FIT_IDS = tuple(f'{family}@{seed}' for family in ('backup', 'mc') for seed in R.SEEDS)


def tex(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                    '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def label(arm):
    family, _, seed = arm.partition('@')
    return R.LABELS[family] + (' ' + seed if seed else '')


def records(summary):
    """Validate complete display membership without rescoring or changing a gate."""
    R.rules(summary)
    require(summary['episodes'] == 1440 and summary['paired_cases'] == 144
            and summary['dataset_rows'] == {'train': 5589, 'valid': 1109}, 'complete declared cohorts')
    require(set(summary['costs']['training']['fits']) == set(FIT_IDS)
            and set(summary['final_prediction_diagnostics']) == set(FIT_IDS), 'all six continuations')
    rows = []
    for regime in R.REGIMES:
        panel = summary['regimes'][regime]
        require(set(panel['means']) == set(R.ARMS) and set(panel['family_means']) == set(R.FAMILIES), 'all arms and families')
        for arm in R.ARMS:
            m, raw = panel['means'][arm], panel['raw_counts'][arm]
            require(raw['episodes'] == 48 and type(raw['found']) is int and 0 <= raw['found'] <= 48, 'raw coverage')
            values = {k: m[k] for k in ('found', 'steps', 'controller_seconds')}
            am = summary['amortization'][regime][arm]['seconds_per_search']
            require(set(am) == {'1', '100', '10000'} and all(type(v) in (float, int) and math.isfinite(v) and v >= 0
                    for v in (*values.values(), *am.values())) and values['found'] <= 1, 'finite reported metrics')
            rows.append({'regime': regime, 'arm': arm, **values, 'raw_found': raw['found'], 'amortized': am})
    return rows


def table(headers, rows, alignment):
    return '\n'.join([r'\begin{tabular}{' + alignment + '}', r'\toprule',
                      ' & '.join(tex(v) for v in headers) + r' \\ \midrule',
                      *(' & '.join(tex(v) for v in row) + r' \\' for row in rows),
                      r'\bottomrule', r'\end{tabular}'])


def outcome(summary):
    comparisons = []
    for regime in R.REGIMES:
        f = summary['regimes'][regime]['family_means']
        comparisons.append(f'{regime}: {f["backup"]["steps"]:.3f} versus {f["mc"]["steps"]:.3f} moves '
                           rf'and {100*f["backup"]["found"]:.2f}\% versus {100*f["mc"]["found"]:.2f}\% success')
    shifted = summary['regimes']['lambda5']['family_means']
    after, before = shifted['backup'], shifted['reference']
    relation = 'higher than' if after['steps'] > before['steps'] else 'lower than' if after['steps'] < before['steps'] else 'equal to'
    analytic = [summary['regimes'][r]['means']['analytic_inbounds']['found'] for r in R.REGIMES]
    baseline = (r'The analytic controller achieved 100\% weighted success in all three settings.' if all(v == 1 for v in analytic)
                else 'Analytic weighted success was ' + ', '.join(rf'{r}: {100*v:.2f}\%' for r, v in zip(R.REGIMES, analytic, strict=True)) + '.')
    return ('Bellman versus Monte Carlo family results were ' + '; '.join(comparisons) + '. '
            f'Under lambda5, Bellman capped moves were {relation} the unchanged reference '
            f'({after["steps"]:.3f} versus {before["steps"]:.3f}); weighted success was '
            rf'{100*after["found"]:.2f}\% versus {100*before["found"]:.2f}\%. ' + baseline)


def document(summary, inputs):
    rows = records(summary)
    checks = summary['competence_checks'] + summary['improvement_checks']
    status = 'PASS' if summary['pilot_continuation'] else 'FAIL'
    competence, improvement = (sum(c['passes'] for c in summary[k]) for k in ('competence_checks', 'improvement_checks'))
    families = []
    for regime in R.REGIMES:
        panel = summary['regimes'][regime]
        for family in (*R.FAMILIES, 'analytic_inbounds'):
            m = panel['means'][family] if family == 'analytic_inbounds' else panel['family_means'][family]
            families.append([regime, R.LABELS[family], f'{100*m["found"]:.2f}', f'{m["steps"]:.3f}', f'{m["controller_seconds"]:.4f}'])
    all_rows = [[r['regime'], label(r['arm']), f'{r["raw_found"]}/48', f'{100*r["found"]:.2f}', f'{r["steps"]:.3f}',
                 f'{r["controller_seconds"]:.4f}', *(f'{r["amortized"][str(h)]:.4f}' for h in (1, 100, 10000))] for r in rows]
    training, diagnostics = summary['costs']['training'], summary['final_prediction_diagnostics']
    fit_rows = [[label(arm), f'{training["fits"][arm]:.3f}',
                 *(f'{diagnostics[arm][split][metric]:.5g}' for split in ('train', 'valid')
                   for metric in ('mse_normalized', 'mae_physical'))] for arm in FIT_IDS]
    lines = [r'\documentclass[10pt]{article}', r'\usepackage[margin=0.72in]{geometry}',
             r'\usepackage[T1]{fontenc}\usepackage{lmodern,booktabs,graphicx,longtable,pdflscape,hyperref,xurl}',
             r'\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}',
             r'\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{5pt}',
             r'\begin{document}\begin{center}{\Large\bfseries Bellman versus Monte Carlo Continuation in Olfactory Search\par}',
             r'\vspace{3pt}OpenJev development report\end{center}',
             r'\begin{abstract}',
             (f'This paired target-control pilot is {status}: {sum(c["passes"] for c in checks)}/42 frozen conditions pass '
              f'({competence}/18 competence; {improvement}/24 paired improvement). Six continuations use the same MLP '
              'architecture and teacher-visited training states. All three fit seeds, three unchanged references, and the '
              'analytic controller are retained across 1,440 fresh policy episodes. This tests a learning procedure; '
              'it does not establish an architectural or biological-memory advantage.'), r'\end{abstract}',
             r'\section{Question and fixed comparison}',
             ('Can delayed Bellman targets improve autonomous control beyond matched Monte Carlo continuation and '
              'its unchanged starting checkpoint? The model is a width-eight biased ReLU scalar MLP on 11,028 features, '
              'including the full exact public posterior. It is not compact learned memory. Each continuation restores '
              'one of three prior checkpoints and receives 40 epochs, batch size 128, fresh Adam at 0.001, and gradient '
              'clipping at 5. The six fits receive 10,560 optimizer updates in total.'),
             ('All 5,589 TRAIN prefixes from 192 teacher episodes are used with uniform row loss. The 1,109 VALID prefixes '
              'from 48 episodes are descriptive only. Monte Carlo targets are remaining teacher steps divided by 64. '
              'Bellman targets refresh before epochs 1, 6, ..., 36 from each fit\'s delayed checkpoint: 24 refreshes and '
              '134,136 state backups, each enumerating 16 action-hit branches. The baseline and final checkpoint rule '
              'are fixed; no checkpoint selection or new trajectory collection occurs during continuation.'),
             ('Evaluation has 48 paired searches per setting, eight blocks and 16 cases per initial-hit stratum. '
              'Lambda 3 and 4 are training-supported; lambda 5 supplies an unseen known kernel. Failures contribute '
              'the 2,188-move horizon. Means use the setting-specific initial-hit mixture, then equal fit-seed weighting. '
              'These are bounded local cases, not a population significance test.'),
             r'\subsection*{Complete family means}', r'{\small\centering',
             table(['Setting', 'Family', 'Success (%)', 'Moves', 'Controller s'], families, 'llrrr'), r'\par}',
             r'\subsection*{Observed outcome}', outcome(summary),
             r'\clearpage\begin{landscape}\section{All arms on common axes}',
             r'\begin{center}\includegraphics[width=\linewidth,height=0.82\textheight,keepaspectratio]{comparison.pdf}\end{center}',
             ('All 30 arm-setting cells are displayed. Each metric has one zero-based scale across settings. Raw found '
              'counts and accounting horizons are provided on the next page. Error bars are omitted because the protocol '
              'does not estimate sampling uncertainty.'), r'\end{landscape}',
             r'\begin{landscape}\section{All fits and cost horizons}', r'{\small\centering',
             table(['Setting', 'Arm', 'Raw found', 'Success (%)', 'Moves', 'Controller s', 'H=1 s', 'H=100 s', 'H=10000 s'], all_rows, 'llrrrrrrr'), r'\par}',
             ('Raw found is an unweighted count out of 48; success is mixture weighted. Controller cost includes '
              'initialization, public updates, branch/features, scalar prediction, choice and allocated deployment setup. '
              'Recorded journal I/O is excluded. H columns are accounting scenarios, not further searches.'),
             r'\end{landscape}\section{Training, accounting and evidence limits}',
             (f'Preparation took {training["preparation_seconds"]:.3f} s; the original worker interval was '
              f'{summary["costs"]["worker_seconds"]:.3f} s. Paid fit time includes target generation. Separate parity, '
              'final scalar diagnostics and independent target auditing are excluded from fit time and retained in the '
              'complete execution ledger. The design matches optimizer updates, not total training compute.'),
             r'{\small\centering', table(['Fit', 'Fit s', 'TRAIN MSE', 'TRAIN MAE', 'VALID MSE', 'VALID MAE'], fit_rows, 'lrrrrr'), r'\par}',
             ('Final diagnostic MSE uses normalized remaining-return labels; MAE is in physical moves. Both continuation '
              'families are evaluated against the same original Monte Carlo labels here. These scalar errors are not '
              'policy competence, and optimization losses on different training targets are not interchangeable.'),
             (r'The horizon accounting is $U+(L+P/6+D)/H$: $U$ is per-search controller work without deployment '
              r'allocation, $L$ is continuation time, $P$ preparation, and $D$ head loading plus one ninth of shared module '
              r'setup. Unchanged references have $L=P=0$; original training is a common prior investment. Analytic '
              'initialization is paid per search. These timings describe one local execution. Audit, presentation, and '
              'historical model-development costs are not hidden inside fit time or treated as deployment savings.'),
             ('Monte Carlo returns estimate the analytic teacher policy, not optimal costs. Backups use only teacher-visited '
              'states and signed values, without new exploration or output clipping. Coverage error, target noise and '
              'undiscounted bootstrapping remain possible. A passing pilot would justify further evaluation, not a novelty '
              'claim. Earlier unsuccessful studies and their gates remain unchanged.'),
             (f'The saved-output audit made {summary["comparisons"]:,} comparisons and '
              f'{summary["saved_checkpoint_readout_calls"]:,} local checkpoint-readout calls. It independently reconstructs '
              'public filtering, targets, scalar predictions, action choices, costs and aggregates without retraining or '
              'simulator calls. Optimizer trajectories, actual Torch execution, native RNG and timing truth remain '
              'authenticated producer evidence. This paper builder only formats those closed results.'),
             r'\subsection*{Evidence and restoration}',
             (r'The repository protocol is \path{research/otto-bellman-control-protocol.md}. Machine-readable family/stratum/block '
              'results, all checkpoints, original work journals, and independent readback are retained under '
              r'\path{output/otto-bellman-control-v1/}. The accompanying \path{paper-values.json} retains every plotted value '
              r'and condition. \path{receipt.json} binds this source, the authenticated inputs, and all staged artifacts. '
              'Compilation and visual review require a separate build record.'),
             r'{\footnotesize\begin{description}']
    for name, entry in inputs.items():
        lines.append(r'\item[' + tex(name) + r'] SHA256 \path{' + entry['sha256'] + '}')
    lines += [r'\end{description}}\clearpage\appendix\section{All frozen conditions}',
              (r'Every Bellman fit must achieve at least 95\% weighted success and at most 105\% of analytic moves in each '
               'setting. Against both learned controls, the Bellman family must preserve success, reduce moves by at least '
               r'5\%, improve at least six of eight paired blocks, and not increase controller cost. All 42 checks are required.'),
              ('The decisions below are copied from the authenticated independent audit. Display rounding never changes '
               'a gate. Success and block conditions use inclusive lower bounds; moves and controller-cost conditions '
               'use inclusive upper bounds.'),
              r'{\small\begin{longtable}{p{0.58\linewidth}rrl}\toprule Condition & Value & Bound & Result \\ \midrule\endhead']
    lines += [tex(c['name']) + f' & {c["value"]:.7g} & {c["threshold"]:.7g} & ' + ('PASS' if c['passes'] else 'FAIL') + r' \\' for c in checks]
    lines += [r'\bottomrule\end{longtable}}\end{document}', '']
    return '\n'.join(line + (r'\par' if line and not line.startswith('\\') and not line.endswith(r'\\') else '') for line in lines)


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = records(summary)
    metrics = [('found', 'Weighted success (%)', 100), ('steps', 'Capped moves', 1), ('controller_seconds', 'Controller seconds / search', 1)]
    with plt.rc_context({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.5), layout='constrained', sharex='col')
        for ri, regime in enumerate(R.REGIMES):
            group = [r for r in rows if r['regime'] == regime]
            for mi, (key, title, scale) in enumerate(metrics):
                ax = axes[ri, mi]
                ax.barh(range(10), [r[key]*scale for r in group], color=[R.COLORS[r['arm'].split('@')[0]] for r in group])
                ax.set_yticks(range(10), [label(r['arm']) for r in group] if mi == 0 else ['']*10)
                ax.invert_yaxis()
                ax.set_xlim(0, 100 if key == 'found' else max(1e-9, max(r[key] for r in rows)*1.08))
                ax.grid(axis='x', alpha=.2)
                ax.set_axisbelow(True)
                if ri == 0:
                    ax.set_title(title, fontsize=11)
                if mi == 0:
                    ax.set_ylabel(regime + ('\nunseen supplied kernel' if ri == 2 else '\ntraining-supported'))
        fig.suptitle(f'All ten arms in each setting | {R.rules(summary)}/42 checks | ' + ('PASS' if summary['pilot_continuation'] else 'FAIL'), fontsize=15)
        for suffix in ('pdf', 'png'):
            fig.savefig(out / f'comparison.{suffix}', dpi=160)
        plt.close(fig)
    return matplotlib.__version__


def execute(args):
    r = R.Report(args)
    r.receipt['version'] = VERSION
    require(args.output.is_absolute() and not args.output.exists() and not any(p.is_symlink() for p in args.output.parents), 'exclusive staging output')
    args.output.mkdir(parents=True, exist_ok=False)
    old = signal.getsignal(signal.SIGALRM)
    try:
        require(digest(R.regular(ROOT / R.H.CLOCK))['sha256'] == R.H.CLOCK_PIN, 'qualified native clock')
        r.clock = R.H.load(ROOT / R.H.CLOCK, '_bellman_paper_clock').SuspendClock()
        r.start = r.clock.now_ns()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('paper staging deadline')))
        signal.setitimer(signal.ITIMER_REAL, R.LIMITS['seconds'])
        write(args.output / 'started.json', {'version': VERSION, 'request': {k: str(v) for k, v in vars(args).items()}, 'limits': R.LIMITS})
        summary = r.authenticate()
        content = document(summary, r.receipt['inputs'])
        matplotlib_version = plot(summary, args.output)
        (args.output / 'paper.tex').write_text(content)
        write(args.output / 'paper-values.json', {'rows': records(summary), 'regimes': summary['regimes'],
              'amortization': summary['amortization'], 'costs': summary['costs'],
              'diagnostics': summary['final_prediction_diagnostics'], 'conditions': summary['competence_checks'] + summary['improvement_checks']})
        write(args.output / 'build-instructions.json', {'docker_image': IMAGE, 'network': 'none',
              'working_directory': '/work', 'command': ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'],
              'copy_to_fresh_exclusive_build_directory': ['paper.tex', 'comparison.pdf', 'comparison.png'],
              'compiler_invoked': False, 'visual_review_required': True})
        r.check()
        r.receipt.update(status='completed', artifact_stage='tex_and_figure_staged', paper_compiled=False, visual_qa_complete=False,
                         source={'path': str(Path(__file__).resolve()), **digest(Path(__file__))}, helper_sha256=HELPER_PIN,
                         matplotlib_version=matplotlib_version, limits=R.LIMITS,
                         wall_seconds=(r.clock.now_ns()-r.start)/1e9,
                         files={p.name: digest(p, r.check) for p in args.output.iterdir() if p.is_file()})
        write(args.output / 'receipt.json', r.receipt)
        r.check()
        return r.receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.output / 'receipt.json').exists():
                (args.output / 'receipt.json').rename(args.output / 'invalid-completed-receipt.json')
            write(args.output / 'failed.json', {**r.receipt, 'status': 'failed', 'error': repr(error)})
        except BaseException as secondary:  # noqa: BLE001 - Preserve the primary failure if writing evidence fails.
            error.add_note(f'Failure receipt: {secondary!r}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'run', 'terminal', 'audit', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('plan-sha256', 'receipt-sha256', 'terminal-sha256', 'audit-receipt-sha256'):
        parser.add_argument('--'+flag, required=True)
    execute(parser.parse_args())
