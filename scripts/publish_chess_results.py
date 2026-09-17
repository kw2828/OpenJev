"""Publish a completed frozen chess pilot without rerunning any model."""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location('chess_publisher_runner', ROOT/'scripts/chess_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_results(execution, plan_path):
    execution, plan_path = Path(execution), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    runner = load_runner()
    if runner.canonical_hash({k: v for k, v in plan.items() if k != 'plan_id'}) != plan['plan_id']:
        raise ValueError('Canonical plan identity mismatch')
    if (execution/'failed.json').exists():
        raise ValueError('Execution has a failure receipt')
    completed = json.loads((execution/'completed.json').read_text())
    summary = json.loads((execution/'summary.json').read_text())
    started = json.loads((execution/'started.json').read_text())
    if completed['status'] != 'selected_schedule_executed':
        raise ValueError('Execution must have a completion receipt')
    if completed['plan_id'] != plan['plan_id'] or summary['plan_id'] != plan['plan_id']:
        raise ValueError('Plan identity mismatch')
    if not summary['entire_prepared_panel']:
        raise ValueError('Cannot publish a subset as the full panel')
    expected_selection = {'game_ids': [g['id'] for g in plan['config']['games']],
                          'puzzle_policies': plan['config']['puzzle_policies']}
    if summary['selection'] != expected_selection or completed['selection'] != expected_selection:
        raise ValueError('Incomplete or different schedule')
    if (started['selection'] != expected_selection or started['plan_id'] != plan['plan_id']
            or started['plan_file_sha256'] != sha(plan_path)):
        raise ValueError('Start receipt does not bind the frozen plan and selection')
    required = {'plan.json', 'started.json', 'loads.json', 'warmups.json', 'puzzles.jsonl', 'summary.json'}
    required |= {f'game-{g["id"]}.{ext}' for g in plan['config']['games'] for ext in ('json', 'pgn')}
    if not required.issubset(completed['outputs']):
        raise ValueError('Completion receipt omits required output hashes')
    for name, expected in completed['outputs'].items():
        if Path(name).name != name or sha(execution/name) != expected:
            raise ValueError(f'Output hash mismatch: {name}')
    if sha(execution/'plan.json') != sha(plan_path):
        raise ValueError('Execution plan differs from published plan')
    puzzles = [json.loads(line) for line in (execution/'puzzles.jsonl').read_text().splitlines()]
    expected_pairs = {(p['puzzle_id'], name) for p in plan['puzzles']['positions']
                      for name in plan['config']['puzzle_policies']}
    actual_pairs = [(r['puzzle_id'], r['policy_name']) for r in puzzles]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError('Puzzle panel is incomplete or duplicated')
    games = []
    for declared in plan['config']['games']:
        game = json.loads((execution/f'game-{declared["id"]}.json').read_text())
        if any(game[key] != declared[key] for key in ('white', 'black')) or game['game_id'] != declared['id']:
            raise ValueError('Game identity differs from schedule')
        games.append(game)
    recomputed = runner.summarize(puzzles, games)
    if any(summary[key] != value for key, value in recomputed.items()):
        raise ValueError('Summary does not match raw attempts')
    if completed['puzzle_attempts'] != len(puzzles) or completed['games_executed'] != len(games):
        raise ValueError('Completion counts differ')
    return plan, summary, games


def figure(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = summary['puzzles']
    names = list(rows)
    labels = [n.replace(' direct', '').replace(' via Codex', '\n(Codex route)') for n in names]
    counts = [rows[n]['strict_gold_matches'] for n in names]
    times = [rows[n]['latency_all_attempts']['median_ms'] for n in names]
    colors = ['#e59532' if 'Astra' in n else '#14786d' if 'OpenJev' in n else '#6c7f91' for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.9), gridspec_kw={'width_ratios': [1.35, 1]})
    axes[0].barh(labels, counts, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 26)
    axes[0].set_xlabel('First-move matches out of 24 public puzzles')
    for i, n in enumerate(names):
        axes[0].text(counts[i]+.3, i, f'{counts[i]}/24', va='center', fontsize=9)
    axes[1].barh(labels, times, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Median policy-call wall time (ms, log scale)')
    axes[1].tick_params(axis='y', labelleft=False)
    for i, t in enumerate(times):
        axes[1].text(t*1.15, i, f'{t:.2f}', va='center', fontsize=9)
    axes[1].set_xlim(max(min(times)/2, .0001), max(times)*5)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Chess development pilot: move quality and route latency', fontsize=17, x=.08, ha='left')
    fig.text(.08, .015, 'Public convenience sample, possible training overlap. First move only, no Elo estimate.\n'
             'Local backends differ; Astra time includes Codex orchestration. Model load and warmups excluded.\n'
             'Astra and Qwen receive SAN check/mate hints; inputs, training data and model sizes differ.',
             fontsize=9, color='#4d5862')
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)


def document(plan, summary, games):
    lines = [
        '# Chess pilot', '',
        ('OpenJev now includes a trained 182,289-parameter recurrent circuit, a legal-move arena, '
        'and recorded games against ChessFly, ChessLFM, Qwen and GPT-6 Astra through Codex. '
        'This is a development experiment, not an Elo rating or a demonstrated architecture advantage.'), '',
        ('[Play the first scheduled game](https://kw2828.github.io/OpenJev/chess-replay.html) · '
        '[Our trained models](chess-student.md) · [Research plan](../research/chess-research-plan.md) · '
        '[Paper draft](../paper/chess-study.tex)'), '',
        ('[![First scheduled OpenJev circuit versus ChessFly game](assets/chess-game-01.gif)]'
        '(https://kw2828.github.io/OpenJev/chess-replay.html)'), '',
        ('The replay is game-01, selected before gameplay. It includes every accepted move; '
        'playback timing is illustrative. No engine chooses either player\'s game moves.'), '',
        '## Public puzzle panel', '',
        ('We fixed the first eight valid Lichess puzzles in each of three rating bands: at most 1200, '
        '1201-1800 and above 1800. The opponent setup move is applied before the solver chooses. '
        'The reported score is the first solver move matching the published solution, not a full puzzle solve. '
        'This 24-position convenience sample may overlap published models\' training data.'), '',
        '![Puzzle matches and route latency](assets/chess-benchmark.png)', '',
        '| Policy | First-move matches | Immediate-mate alternative included | Failed calls | Median wall ms |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name, row in summary['puzzles'].items():
        lines.append(f'| {name} | {row["strict_gold_matches"]}/{row["positions"]} | '
                     f'{row["gold_or_immediate_mate"]}/{row["positions"]} | {row["errors"]} | '
                     f'{row["latency_all_attempts"]["median_ms"]:.2f} |')
    lines += ['', '## Recorded games', '',
              ('Both colors are played from the standard starting position. Each side has 300 seconds '
              'with no increment. A game reaching 120 plies remains unfinished; it is not scored as a draw. '
              'Timeouts and model errors remain visible. These are individual exhibitions, '
              'not independent repeated samples.'), '',
              '| Game | White | Black | Result | Termination | Plies |',
              '| --- | --- | --- | --- | --- | ---: |']
    for g in games:
        lines.append(f'| [{g["game_id"]}](../evidence/chess-v1/results/game-{g["game_id"]}.pgn) | '
                     f'{g["white"]} | {g["black"]} | {g["result"]} | {g["termination"]} | {len(g["moves"])} |')
    lines += ['', (f'{summary["scored_game_count"]} games reached a scored outcome; '
              f'{summary["unfinished_game_count"]} were unfinished and {summary["failed_game_count"]} failed.'), '',
              '## What each policy uses', '',
              ('- **Our circuit, rewired control and GRU:** exactly the first predeclared fit, seed 17. '
              'All use the same 4,096 engine-labeled training positions and three epochs. Hidden state resets '
              'at every board; four recurrent updates refine the current board. This is not cross-move memory or a world model.'),
              ('- **ChessFly:** independently implemented CPU adapter for the pinned public weights and FlyWire graph. '
              'Five recurrent updates from zero per board. The browser demo\'s depth-three search is excluded.'),
              ('- **ChessLFM:** pinned public hybrid convolution/attention model, two forward passes, all legal moves scored. '
              'Its demo search is excluded. This does not reproduce the author\'s search-assisted rating.'),
              '- **OpenJev Qwen:** one forward pass scores single-token candidate labels for every legal move; no text generation or search.',
              ('- **Astra:** an explicitly dispatched `gpt-6-astra` Codex agent receives FEN, board, recent moves and legal candidates. '
              'It is instructed to use no engine, browsing or workspace data beyond the board helper. '
              'One persistent agent serves the panel, so its conversation also contains earlier packets; this is not an independently reset API call per move. '
              'The packets do not supply remaining game clocks. '
              'This is instruction-limited isolation, not a technical sandbox or provider-attested Responses API benchmark. '
              'No probabilities are invented for Astra.'),
              ('- **Stockfish:** puzzle reference with 1,000 nodes per position, one thread and cleared hash. '
              'It is not consulted by any game player. Greedy material and seeded uniform random are additional puzzle controls.'), '',
              '## Timing and claim limits', '',
              ('Timing is the complete policy-call wall time. Initial loading and two declared local warmups are excluded '
              'and recorded separately. Astra includes Codex reasoning, scheduling and helper calls. '
              'Qwen runs on MLX; ChessFly, ChessLFM and students use CPU PyTorch with two threads. '
              'The LFM convolution uses the reference implementation. The timings are route measurements, '
              'not equal-hardware model speed or equal-compute architecture comparisons. '
              'Background local work can affect timing. A policy call is adjudicated after return; '
              'the Astra transport has a separate 120-second per-request timeout.'), '',
              ('Astra and Qwen receive SAN check/checkmate markers in their legal candidate descriptions. '
              'Three of the 24 puzzle positions contain an immediate mate revealed by this notation; numerical policies receive legal masks without SAN. '
              'This is another representation confound, so the panel does not measure unaided chess reasoning. '
              'Move probabilities are conditional on the legal candidates and are uncalibrated. '
              'Neither confidence nor checkmate strength follows from a large candidate probability. '
              'External models have different pretraining data and scale. The controlled architectural comparison '
              'is [our nine-fit student study](chess-student.md), whose circuit continuation gate failed.'), '',
              '## Reproduction and evidence', '',
              ('[Frozen plan](../evidence/chess-v1/plan.json) · '
              '[Pre-results schedule](../evidence/chess-v1/design.json) · '
              '[Raw attempts and game traces](../evidence/chess-v1/results/) · '
              '[Summary](../evidence/chess-v1/results/summary.json) · '
              '[Publication receipt](../evidence/chess-v1/publication.json)'), '',
              ('Install the project\'s language/research dependencies and `research/requirements-chess.txt`. '
              'Download the pinned external assets according to their model cards. For a new run, copy the config, '
              'set local artifact paths and a fresh Astra bridge directory, then prepare a new plan. '
              'Do not overwrite this frozen run or substitute checkpoints under its identity.'), '',
              '```sh',
              'python scripts/chess_study.py prepare --config YOUR_CONFIG.json --puzzles YOUR_LICHESS.csv --out NEW_PLAN.json',
              'python scripts/chess_study.py run --plan NEW_PLAN.json --out NEW_EXECUTION',
              '```', '',
              ('The Astra policy requires a separately dispatched Codex worker. Without one, remove it from '
              'the new schedule before preparation; the runner will never substitute a different model.'), '',
              '## Sources and licenses', '',
              ('[Lichess puzzles](https://database.lichess.org/) are CC0. '
              '[Stockfish](https://stockfishchess.org/) is the local GPL engine teacher/reference; its binary is not redistributed. '
              '[ChessFly](https://huggingface.co/mlabonne/chessfly) uses externally licensed FlyWire assets with noncommercial terms. '
              '[ChessLFM](https://huggingface.co/mlabonne/LFM2.5-230M-Chess) has its own model license. '
              'External model weights, graph assets and Space code are not redistributed here. '
              'Our independently written adapters and original student weights use the project MIT license.'), '']
    return '\n'.join(lines)


def publish(execution, plan_path, out, docs):
    plan, summary, games = verified_results(execution, plan_path)
    execution, out, docs = Path(execution), Path(out), Path(docs)
    out.mkdir(parents=True, exist_ok=False)
    for path in sorted(execution.iterdir()):
        if path.is_file():
            shutil.copy2(path, out/path.name)
    figure(summary, docs/'assets/chess-benchmark.png')
    (docs/'chess.md').write_text(document(plan, summary, games))
    receipt = {'plan_id': plan['plan_id'], 'source_execution': str(execution.resolve()),
               'selection': 'entire frozen panel; no omitted failed calls or games',
               'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
               'figure_sha256': sha(docs/'assets/chess-benchmark.png'),
               'publisher_sha256': sha(__file__)}
    (out.parent/'publication.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--docs', type=Path, default=ROOT/'docs')
    args = parser.parse_args()
    publish(args.execution, args.plan, args.out, args.docs)
