"""Saved-only scientific figure and fixed-case public-card replay.

Reads externally hash-bound completed summary/evaluation artifacts. No model,
environment, hidden-board access, simulation or seed allocation is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

MODES = ('delta', 'gated_delta', 'kalman', 'innovation_local', 'innovation_matched', 'gru')
ROWS = (*MODES, 'exact', 'last32')
FIXED = ('innovation_local-pair0', 'innovation_matched-pair0')
LABELS = ('Delta', 'Gated delta', 'Diagonal KDN', 'Local innovation', 'Gain-matched diffuse', 'GRU512',
          'Exact public map', 'Last 32 reveals')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def authenticate(summary, summary_sha256, evaluation):
    require(sha(summary) == summary_sha256, 'Summary differs from external hash')
    data = read(summary)
    require(data['status'] == 'complete', 'Summary is not complete')
    require(sha(evaluation / 'completed.json') == data['evaluation_receipt_sha256'], 'Evaluation receipt changed')
    completed = read(evaluation / 'completed.json')
    require(completed['status'] == 'complete' and completed['controllers'] == 20 and completed['episodes'] == 1280,
            'Evaluation is not a complete 20-controller/64-case study')
    require(not (evaluation / 'failed.json').exists(), 'Failed evaluation is not renderable as completed')
    actual = {str(p.relative_to(evaluation)) for p in evaluation.rglob('*') if p.is_file()}
    require(actual == set(completed['files']) | {'completed.json'}, 'Evaluation membership changed')
    for name, item in completed['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe artifact member')
        path = evaluation / path
        require(sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], 'Evaluation artifact changed')
    require(len(data['rows']) == 8 and {row['mode'] for row in data['rows']} == set(ROWS), 'All eight rows required')
    expected_fits = {f'{mode}-pair{i}' for mode in MODES for i in range(3)} | {'exact', 'last32'}
    require(set(data['per_fit']) == expected_fits, 'All eighteen fits and both references required')
    rows = {row['mode']: row for row in data['rows']}
    for mode in ROWS:
        row = rows[mode]
        names = [f'{mode}-pair{i}' for i in range(3)] if mode in MODES else [mode]
        means = []
        for name in names:
            fit = data['per_fit'][name]
            values = fit['per_seed_returns']
            require(len(values) == 64 and all(math.isfinite(v) and -1 - 1e-12 <= v <= 1 + 1e-12 for v in values),
                    'Exactly 64 finite native returns per controller required')
            mean = math.fsum(values) / 64
            require(math.isclose(mean, fit['mean_return'], rel_tol=0, abs_tol=1e-12), 'Fit mean mismatch')
            means.append(mean)
        require(row['episodes'] == 64 * len(names) and math.isclose(math.fsum(means) / len(means),
                row['mean_return'], rel_tol=0, abs_tol=1e-12), 'Family mean/denominator mismatch')
        require(math.isfinite(row['whole_controller_ms_per_action']) and row['whole_controller_ms_per_action'] > 0,
                'Positive whole-controller timing required')
        require(math.isclose(1000 * row['whole_controller_seconds'] / row['native_steps'],
                row['whole_controller_ms_per_action'], rel_tol=1e-12), 'Timing denominator mismatch')
        if mode in MODES:
            accuracy = row['development_age_gt32_accuracy']
            require(math.isfinite(accuracy) and 0 <= accuracy <= 1, 'Finite development hidden-card accuracy required')
    episodes, receipts = [], []
    for name in FIXED:
        stem = evaluation / 'controllers' / name / 'episodes' / '000'
        receipt = read(stem.with_suffix('.json'))
        require(receipt['status'] == 'complete' and receipt['index'] == 0 and receipt['controller'] == name,
                'Fixed first case/pair identity changed')
        require(sha(stem.with_suffix('.npz')) == receipt['npz_sha256'], 'Fixed replay bytes changed')
        with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ('observations', 'actions', 'rewards', 'terminated', 'truncated')}
        n = receipt['native_steps']
        require(1 <= n <= 104 and arrays['observations'].shape == (n + 1, 52)
                and arrays['observations'].dtype == np.int64
                and np.all((arrays['observations'] >= 0) & (arrays['observations'] <= 13)), 'Invalid public frames')
        require(all(arrays[k].shape == (n,) for k in ('actions', 'rewards', 'terminated', 'truncated'))
                and np.all((arrays['actions'] >= 0) & (arrays['actions'] < 52))
                and np.isfinite(arrays['rewards']).all(), 'Invalid public action/reward trajectory')
        require(math.isclose(math.fsum(arrays['rewards']), receipt['return'], rel_tol=0, abs_tol=1e-12), 'Replay return mismatch')
        episodes.append(arrays)
        receipts.append(receipt)
    require(receipts[0]['seed'] == receipts[1]['seed'] and receipts[0]['inputs_sha256'] == receipts[1]['inputs_sha256']
            and receipts[0]['protocol_sha256'] == receipts[1]['protocol_sha256'] == completed['protocol_sha256'],
            'Fixed replay cases are not paired')
    return data, episodes, receipts, completed


def plot(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = {row['mode']: row for row in data['rows']}
    colors = ['#9ca3af'] * 8
    colors[3], colors[4], colors[6], colors[7] = '#2474ad', '#cf7333', '#58966b', '#9eb68c'
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.3, 1, 1]})
    y = np.arange(8)
    axes[0].barh(y, [rows[m]['mean_return'] for m in ROWS], color=colors, alpha=.75)
    for pair, marker in enumerate(('o', 's', '^')):
        values = [data['per_fit'][f'{m}-pair{pair}']['mean_return'] for m in MODES]
        axes[0].scatter(values, np.arange(6) + (pair - 1) * .15, s=23, marker=marker,
                        color='#222222', label=f'Fit pair {pair}', zorder=3)
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].axvline(0, color='#777777', linewidth=.7)
    axes[0].set_yticks(y, LABELS)
    axes[0].set_xlabel('Mean native return, higher is better')
    axes[0].set_title('All final policies, 64 paired cases each', fontsize=10)
    axes[0].legend(loc='lower left', fontsize=8)
    axes[1].barh(np.arange(6), [100 * rows[m]['development_age_gt32_accuracy'] for m in MODES], color=colors[:6])
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Development hidden-rank accuracy (%)')
    axes[1].set_title('More than 32 steps since public visibility', fontsize=10)
    for i in (6, 7):
        axes[1].text(3, i, 'Not measured', va='center', fontsize=9, color='#555555')
    axes[2].scatter([rows[m]['whole_controller_ms_per_action'] for m in ROWS], y, color=colors, s=55)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Whole-controller ms / native action (log)')
    axes[2].set_title('Includes restore, public logic, game and I/O', fontsize=10)
    for index, axis in enumerate(axes):
        axis.set_ylim(7.6, -.6)
        axis.grid(axis='x', alpha=.18)
        if index:
            axis.set_yticks(y, [''] * 8)
        axis.spines[['top', 'right']].set_visible(False)
    passed = sum(data['criteria'].values())
    fig.suptitle(f"ConcentrationHard development pilot | continuation {'PASS' if data['continuation_passed'] else 'FAIL'} ({passed}/{len(data['criteria'])})",
                 fontsize=14)
    fig.text(.5, .025, 'Three fits per learned family; dots are fits, not confidence intervals. State size, parameters and compute are not matched.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def public_replay(episodes, receipts, path):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    frames = []
    lengths = [len(episode['actions']) for episode in episodes]
    for step in range(max(lengths) + 1):
        canvas = Image.new('RGB', (1100, 370), '#f8f9fb')
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 10), f'PUBLIC OBSERVATIONS ONLY | fixed case index 0, seed {receipts[0]["seed"]} | pair 0', fill='#1e293b', font=font)
        draw.text((20, 28), 'Entire saved episode; ended panels hold final frame. Illustrative first case, not selected for an outcome.', fill='#475569', font=font)
        for side, (episode, length) in enumerate(zip(episodes, lengths, strict=True)):
            t = min(step, length)
            x0 = 20 + side * 550
            label = 'Local innovation' if side == 0 else 'Gain-matched diffuse'
            draw.text((x0, 55), label, fill='#1e293b', font=font)
            cumulative = math.fsum(episode['rewards'][:t])
            last = f'action {int(episode["actions"][t-1])}, reward {episode["rewards"][t-1]:+.4f}' if t else 'reset, no action'
            draw.text((x0, 74), f'{t}/{length} actions | {last}', fill='#334155', font=font)
            draw.text((x0, 91), f'Native return so far {cumulative:+.4f}' + (' | episode ended' if t == length else ''), fill='#334155', font=font)
            for pos, rank in enumerate(episode['observations'][t]):
                x, y = x0 + (pos % 13) * 39, 116 + (pos // 13) * 51
                active = bool(t and pos == episode['actions'][t - 1])
                draw.rounded_rectangle((x, y, x + 34, y + 44), radius=3,
                                       fill='#e5e7eb' if rank == 13 else '#fff7df',
                                       outline='#2474ad' if active else '#9ca3af', width=3 if active else 1)
                draw.text((x + 3, y + 4), str(pos), fill='#64748b', font=font)
                draw.text((x + 6, y + 23), '?' if rank == 13 else f'R{rank}', fill='#1e293b', font=font)
        draw.text((20, 337), 'Blue outline: last actual selection. Hidden ranks stay hidden; native mismatch frames remain visible until the next return.',
                  fill='#475569', font=font)
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=150, optimize=False, disposal=2)
    with Image.open(path) as saved:
        require(saved.n_frames == len(frames), 'GIF lost full-episode frames')
        for index in range(saved.n_frames):
            saved.seek(index)
            require(saved.size == (1100, 370) and saved.info['duration'] == 150, 'GIF frame metadata changed')
    return {'frames': len(frames), 'duration_ms': 150, 'case_index': 0, 'pair': 0, 'controllers': list(FIXED),
            'seed': receipts[0]['seed'], 'native_actions': lengths, 'public_only': True, 'outcome_selection': False}


def render(summary, summary_sha256, evaluation, out):
    begin = time.monotonic()
    summary, evaluation, out = Path(summary), Path(evaluation), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        data, episodes, receipts, completed = authenticate(summary, summary_sha256, evaluation)
        plot(data, out / 'benchmark.png')
        replay = public_replay(episodes, receipts, out / 'cards.gif')
        require(sha(summary) == summary_sha256 and sha(evaluation / 'completed.json') == data['evaluation_receipt_sha256'],
                'Inputs changed during rendering')
        result = {'status': 'complete', 'source_sha256': sha(__file__), 'summary_sha256': summary_sha256,
                  'evaluation_receipt_sha256': data['evaluation_receipt_sha256'],
                  'protocol_sha256': completed['protocol_sha256'], 'replay': replay,
                  'continuation_passed': data['continuation_passed'], 'new_native_calls': 0, 'new_model_calls': 0,
                  'wall_seconds': time.monotonic() - begin,
                  'files': {name: {'sha256': sha(out / name), 'bytes': (out / name).stat().st_size}
                            for name in ('benchmark.png', 'cards.gif')}}
        write(out / 'receipt.json', result)
        return result
    except BaseException as error:
        try:
            write(out / 'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic() - begin})
        except BaseException as secondary:  # noqa: BLE001 - preserve original render/authentication error.
            note = getattr(error, 'add_note', None)
            if callable(note):
                note(f'Failure receipt also failed: {secondary!r}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    render(**vars(parser.parse_args()))
