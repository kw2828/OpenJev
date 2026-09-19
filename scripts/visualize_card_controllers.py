"""Authenticated saved-only card-controller figures and fixed public-frame replay.

No model/environment import, native calls, private-deck reads or seed allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import types
from pathlib import Path

import numpy as np

POLICIES = ('A', 'B', 'C')
MODES = ('delta', 'gated_delta', 'kalman', 'innovation_local', 'innovation_matched', 'gru')
FAMILIES = (*MODES, 'exact', 'last32')
LABELS = ('Delta', 'Gated delta', 'Diagonal KDN', 'Local innovation', 'Matched diffuse', 'GRU512',
          'Exact public map', 'Last 32 reveals')
FIXED_CONTROLLER = 'gated_delta-pair0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def reporter(expected_sha256):
    """Execute exactly the report source bytes authenticated by its receipt."""
    path = Path(__file__).with_name('report_card_controllers.py')
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_sha256, 'Reporter source changed')
    module = types.ModuleType('authenticated_card_controller_report')
    module.__file__ = str(path)
    exec(compile(payload, str(path), 'exec'), module.__dict__)  # noqa: S102 - execute the exact hash-authenticated local source bytes.
    return module


def authenticate(report_dir, expected_receipt_sha256, evaluation):
    require(sha(report_dir / 'receipt.json') == expected_receipt_sha256, 'External audit receipt differs')
    receipt = read(report_dir / 'receipt.json')
    require(receipt['status'] == 'complete' and receipt['new_native_calls'] == receipt['new_model_calls'] == 0,
            'Audit not complete')
    require({p.name for p in report_dir.iterdir() if p.is_file()} == {'receipt.json', 'report.md', 'summary.json'},
            'Audit output membership differs or failure marker exists')
    require(set(receipt['files']) == {'report.md', 'summary.json'}, 'Audit member contract differs')
    for name, identity in receipt['files'].items():
        require(sha(report_dir / name) == identity['sha256'] and (report_dir / name).stat().st_size == identity['bytes'],
                'Audit output changed')
    audit = reporter(receipt['source_sha256'])
    data = read(report_dir / 'summary.json')
    require(data['status'] == 'complete' and data['bindings']['source_sha256'] == receipt['source_sha256'],
            'Summary source/status differs')
    complete = audit.authenticate_evaluation(evaluation, receipt['evaluation_completed_sha256'])
    require(data['bindings']['evaluation_completed_sha256'] == receipt['evaluation_completed_sha256']
            and data['bindings']['protocol_sha256'] == receipt['protocol_sha256'] == complete['protocol_sha256'],
            'Audit/execution lineage differs')
    families, contrasts, gate = audit.aggregate(data['per_fit'])
    require(families == data['families'] and contrasts == data['contrasts'] and gate == data['continuation_gate']
            and gate['passed'] == receipt['continuation_passed'], 'Summary numerical/gate arithmetic differs')
    require(data['original_architecture_gate'] == {'passed': False, 'checks_passed': 1, 'total_checks': 6, 'unchanged': True},
            'Original failed architecture gate must remain unchanged')
    episodes, receipts = [], []
    for policy in POLICIES:
        stem = evaluation / 'controllers' / FIXED_CONTROLLER / policy / 'episodes' / '000'
        item = read(stem.with_suffix('.json'))
        require(item['status'] == 'complete' and item['controller'] == FIXED_CONTROLLER and item['policy'] == policy
                and item['index'] == 0 and sha(stem.with_suffix('.npz')) == item['npz_sha256'],
                'Fixed first-case identity differs')
        with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ('observations', 'actions', 'rewards', 'terminated', 'truncated')}
        n = item['native_steps']
        require(1 <= n <= 104 and arrays['observations'].shape == (n + 1, 52)
                and arrays['observations'].dtype == np.int64 and np.all((arrays['observations'] >= 0) & (arrays['observations'] <= 13)),
                'Invalid fixed public observations')
        require(all(arrays[k].shape == (n,) for k in ('actions', 'rewards', 'terminated', 'truncated'))
                and np.isfinite(arrays['rewards']).all() and np.all((arrays['actions'] >= 0) & (arrays['actions'] < 52)),
                'Invalid fixed action/reward arrays')
        require(math.isclose(math.fsum(arrays['rewards']), item['return'], rel_tol=0, abs_tol=1e-12)
                and item['return'] == data['per_fit'][policy][FIXED_CONTROLLER]['per_seed_returns'][0],
                'Fixed replay return differs')
        episodes.append(arrays)
        receipts.append(item)
    require(len({r['seed'] for r in receipts}) == len({r['layout_sha256'] for r in receipts}) == 1,
            'Fixed replay cases are not paired')
    return data, episodes, receipts, receipt


def plot(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ('#577590', '#1f77b4', '#8c564b', '#d6602d', '#9467bd', '#c44e52', '#29864a', '#707070')
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.9))
    x = np.arange(3)
    for i, (family, label, color) in enumerate(zip(FAMILIES, LABELS, colors, strict=True)):
        rows = [data['families'][policy][family] for policy in POLICIES]
        style = '--' if family in ('exact', 'last32') else '-'
        for ax, key, scale in ((axes[0], 'mean_return', 1), (axes[1], 'success_rate', 100),
                               (axes[2], 'whole_controller_ms_per_action', 1)):
            ax.plot(x, [r[key] * scale for r in rows], style, marker='o', markersize=4,
                    linewidth=1.7, color=color, label=label)
        if family in MODES:
            for pair, marker in enumerate(('o', 's', '^')):
                axes[0].scatter(x + (pair - 1) * .045,
                                [r['fit_mean_returns'][pair] for r in rows], color=color,
                                marker=marker, s=16, alpha=.65, zorder=3)
    axes[0].set_ylabel('Mean native return'); axes[0].set_ylim(-1.05, 1.05)
    axes[0].axhline(0, color='#999999', linewidth=.6)
    axes[0].set_title('All six learned families and both references')
    axes[1].set_ylabel('Solved episodes (%)'); axes[1].set_ylim(-3, 103)
    axes[1].set_title('192 cases/family; 64 per reference')
    axes[2].set_ylabel('Instrumented episode ms / action (log)'); axes[2].set_yscale('log')
    axes[2].set_title('Recorded system work, not intrinsic latency')
    for axis in axes:
        axis.set_xticks(x, ['A\nOriginal', 'B\nStable', 'C\nExplore'])
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.title.set_fontsize(10)
    gate = data['continuation_gate']
    fig.suptitle(f"Fixed memories, fresh controller comparison | C-B rule {'PASS' if gate['passed'] else 'FAIL'} ({gate['checks_passed']}/13 check rows)", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, .04), fontsize=9, frameon=False)
    fig.text(.5, .015, '64 shared deck draws; fit markers are not intervals. A includes an extra diagnostic choice pass. Original architecture rule remains FAIL 1/6.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .16, 1, .93))
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(out / ('controllers.' + extension), dpi=180)
    plt.close(fig)


def replay_frame(episodes, receipts, step):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=13)
    small = ImageFont.load_default(size=11)
    canvas = Image.new('RGB', (1280, 390), '#f8f9fb')
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 9), f'PUBLIC FRAMES ONLY | Gated delta, pair 0 | first fresh case, seed {receipts[0]["seed"]}', fill='#1e293b', font=font)
    draw.text((18, 29), 'Model fixed before these outcomes: strongest conventional family in the original pilot. No winner selection from this follow-up.', fill='#475569', font=small)
    for index, (policy, episode) in enumerate(zip(POLICIES, episodes, strict=True)):
        length = len(episode['actions']); t = min(step, length); x0 = 18 + index * 425
        title = {'A': 'A: Original', 'B': 'B: Stable ties', 'C': 'C: Explore tied pairs'}[policy]
        draw.text((x0, 59), title, fill='#1e293b', font=font)
        draw.text((x0, 81), f'Actions {t}/{length} | return {math.fsum(episode["rewards"][:t]):+.4f}', fill='#334155', font=font)
        last = 'Reset: no action' if not t else f'Last actual position {int(episode["actions"][t-1])}'
        draw.text((x0, 101), last + (' | ended' if t == length else ''), fill='#475569', font=small)
        for pos, rank in enumerate(episode['observations'][t]):
            x, y = x0 + pos % 13 * 30, 127 + pos // 13 * 50
            active = bool(t and pos == int(episode['actions'][t - 1]))
            draw.rounded_rectangle((x, y, x + 26, y + 43), radius=3,
                                   fill='#e5e7eb' if rank == 13 else '#fff7df',
                                   outline='#2474ad' if active else '#9ca3af', width=3 if active else 1)
            draw.text((x + 3, y + 3), str(pos), fill='#64748b', font=small)
            draw.text((x + 3, y + 23), '?' if rank == 13 else str(rank), fill='#1e293b', font=font)
    draw.text((18, 346), 'Blue outline: last actual selection. Hidden ranks remain hidden. All saved actions shown; ended panels hold the final frame.', fill='#475569', font=small)
    draw.text((18, 365), 'Illustrative fixed case only. Different public histories follow each policy; no simulated counterfactual frames.', fill='#475569', font=small)
    return canvas


def public_replay(episodes, receipts, path):
    from PIL import Image

    length = max(len(e['actions']) for e in episodes)
    frames = [replay_frame(episodes, receipts, step) for step in range(length + 1)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=False, disposal=2)
    with Image.open(path) as output:
        require(output.n_frames == length + 1, 'GIF dropped public frames')
        for step in range(output.n_frames):
            output.seek(step)
            require(output.size == (1280, 390) and output.info['duration'] == 160, 'GIF dimensions/timing changed')
    return {'controller': FIXED_CONTROLLER, 'policies': list(POLICIES), 'case_index': 0,
            'seed': receipts[0]['seed'], 'frames': length + 1, 'duration_ms': 160,
            'native_actions': [len(e['actions']) for e in episodes], 'public_only': True, 'outcome_selected': False}


def render(report_dir, expected_receipt_sha256, evaluation, out):
    begin = time.monotonic()
    report_dir, evaluation, out = (Path(p).resolve() for p in (report_dir, evaluation, out))
    require(not out.is_relative_to(evaluation) and not out.is_relative_to(report_dir), 'Render output must be separate')
    out.mkdir(parents=True, exist_ok=False)
    try:
        data, episodes, receipts, audit = authenticate(report_dir, expected_receipt_sha256, evaluation)
        plot(data, out)
        replay = public_replay(episodes, receipts, out / 'fixed-public-replay.gif')
        write(out / 'figure-data.json', data)
        authenticate(report_dir, expected_receipt_sha256, evaluation)
        members = ('controllers.png', 'controllers.svg', 'controllers.pdf', 'fixed-public-replay.gif', 'figure-data.json')
        result = {'status': 'complete', 'source_sha256': sha(__file__), 'report_receipt_sha256': expected_receipt_sha256,
                  'evaluation_completed_sha256': audit['evaluation_completed_sha256'],
                  'continuation_passed': data['continuation_gate']['passed'], 'original_architecture_gate_unchanged': True,
                  'replay': replay, 'new_native_calls': 0, 'new_model_calls': 0,
                  'wall_seconds': time.monotonic() - begin,
                  'files': {name: {'sha256': sha(out / name), 'bytes': (out / name).stat().st_size} for name in members}}
        write(out / 'receipt.json', result)
        return result
    except BaseException as error:
        try:
            write(out / 'failed.json', {'status': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic() - begin})
        except BaseException as secondary:  # noqa: BLE001 - preserve original render failure.
            if callable(getattr(error, 'add_note', None)):
                error.add_note(f'Failure receipt also failed: {secondary!r}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(render(**vars(parser.parse_args())), indent=2))
