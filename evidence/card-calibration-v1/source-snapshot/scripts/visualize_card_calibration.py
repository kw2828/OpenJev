"""Authenticated saved-only calibration/control figures and fixed public-frame replay.

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

POLICIES = ('baseline', 'temperature', 'hard')
MODES = ('delta', 'gated_delta', 'kalman', 'innovation_local', 'innovation_matched', 'gru')
FAMILIES = (*MODES, 'exact', 'last32')
LABELS = ('Delta', 'Gated delta', 'Diagonal KDN', 'Local innovation', 'Matched diffuse', 'GRU512',
          'Exact public map', 'Last 32 reveals')
FIXED_CONTROLLER = 'kalman-pair0'


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
    path = Path(__file__).with_name('report_card_calibration.py')
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_sha256, 'Reporter source changed')
    module = types.ModuleType('authenticated_card_calibration_report')
    module.__file__ = str(path)
    exec(compile(payload, str(path), 'exec'), module.__dict__)  # noqa: S102 - execute the exact hash-authenticated local source bytes.
    return module


def authenticate(root, protocol_path, report_dir, expected_receipt_sha256, evaluation):
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
    require(Path(protocol_path).resolve().is_relative_to(root) and sha(protocol_path) == receipt['protocol_sha256'],
            'Prospective protocol identity differs')
    protocol = read(protocol_path)
    preset = protocol['preset_visualization']
    require(preset['controller'] == FIXED_CONTROLLER and preset['case_index'] == 0,
            'Prospectively fixed replay case differs')
    binding_path = root / protocol['bindings_file']
    require(binding_path.resolve().is_relative_to(root) and sha(binding_path) == receipt['bindings_sha256'],
            'Frozen source-binding identity differs')
    binding = read(binding_path)
    for relative, digest in binding['files'].items():
        path = (root / relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(root) and sha(path) == digest,
                'Frozen source/input bytes changed')
    require(binding['files'].get(str(Path(__file__).resolve().relative_to(root))) == sha(__file__),
            'Visualizer must be source-bound before use')
    audit = reporter(receipt['source_sha256'])
    data = read(report_dir / 'summary.json')
    require(data['status'] == 'complete' and data['bindings']['source_sha256'] == receipt['source_sha256'],
            'Summary source/status differs')
    complete = audit.authenticate_tree(evaluation, receipt['evaluation_completed_sha256'], audit.expected_members())
    require(data['bindings']['evaluation_completed_sha256'] == receipt['evaluation_completed_sha256']
            and data['bindings']['protocol_sha256'] == receipt['protocol_sha256'] == complete['protocol_sha256'],
            'Audit/execution lineage differs')
    families, contrasts, transfer, gate = audit.aggregate(data['per_fit'], data['fresh_baseline_prefix_scores'])
    require(families == data['families'] and contrasts == data['contrasts'] and transfer == data['transfer']
            and gate == data['continuation_gate'] and gate['total_checks'] == 12
            and gate['passed'] == receipt['continuation_passed'], 'Summary numerical/gate arithmetic differs')
    require(data['original_architecture_gate'] == {'passed': False, 'checks_passed': 1, 'total_checks': 6, 'unchanged': True},
            'Original failed architecture gate must remain unchanged')
    require(data['references_evaluated_once'] is True
            and data['references'] == {name: data['per_fit']['baseline'][name] for name in ('exact', 'last32')}
            and data['coverage']['native_games'] == 3584 and data['coverage']['native_rows'] == 56,
            'Reference or total evaluation coverage differs')
    episodes, receipts = [], []
    for policy in POLICIES:
        stem = evaluation / 'controllers' / FIXED_CONTROLLER / policy / 'episodes' / '000'
        item = read(stem.with_suffix('.json'))
        require(item['status'] == 'complete' and item['controller'] == FIXED_CONTROLLER and item['policy'] == policy
                and item['index'] == 0 and item['tracker_policy'] == 'C'
                and item['calibration_completed_sha256'] == receipt['calibration_completed_sha256']
                and sha(stem.with_suffix('.npz')) == item['npz_sha256'],
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
    labels = ('Baseline', 'Temperature', 'Hard')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.9))
    x = np.arange(3)
    for family, label, color in zip(FAMILIES, LABELS, colors, strict=True):
        if family in MODES:
            rows = [data['families'][policy][family] for policy in POLICIES]
            axes[0].plot(x, [r['mean_return'] for r in rows], marker='o', markersize=4,
                         linewidth=1.7, color=color, label=label)
            axes[1].plot(x, [100 * r['successes'] / r['episodes'] for r in rows], marker='o',
                         markersize=4, linewidth=1.7, color=color)
            for pair, marker in enumerate(('o', 's', '^')):
                axes[0].scatter(x + (pair - 1) * .045, [r['fit_mean_returns'][pair] for r in rows],
                                color=color, marker=marker, s=18, alpha=.65, zorder=3)
        else:
            row = data['references'][family]
            axes[0].scatter([0], [row['mean_return']], marker='D', color=color, s=42,
                            label=label + ' (baseline only)', zorder=4)
            axes[1].scatter([0], [100 * sum(row['per_seed_successes']) / len(row['per_seed_successes'])],
                            marker='D', color=color, s=42, zorder=4)
    axes[0].set_ylabel('Mean native return'); axes[0].set_ylim(-1.05, 1.05)
    axes[0].axhline(0, color='#999999', linewidth=.6)
    axes[0].set_title('Same 18 memories; fixed picker C')
    axes[1].set_ylabel('Solved episodes (%)'); axes[1].set_ylim(-3, 103)
    axes[1].set_title('192 games/family; 64 per reference')
    scores = [data['transfer'][p] for p in POLICIES]
    finite = [r['nll'] for r in scores if not r['nll_is_infinite']]
    ceiling = max(finite, default=1.0) * 1.4 or 1.0
    for i, row in enumerate(scores):
        if row['nll_is_infinite']:
            axes[2].bar(i, .75 * ceiling, color='none', edgecolor='#b91c1c', hatch='///')
            axes[2].text(i, .8 * ceiling, 'Infinite NLL', ha='center', fontsize=9, color='#b91c1c')
        else:
            axes[2].bar(i, row['nll'], color=('#6b7280', '#287c8e', '#d07e23')[i])
            axes[2].text(i, row['nll'] + .025 * ceiling, f"{row['nll']:.4f}", ha='center', fontsize=9)
    axes[2].set_ylim(0, ceiling)
    axes[2].set_ylabel('Mean NLL (nats); lower is better')
    axes[2].set_title('Same fresh baseline histories; equal fits')
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.title.set_fontsize(10)
    gate = data['continuation_gate']
    fig.suptitle(f"Probability calibration | temperature rule {'PASS' if gate['passed'] else 'FAIL'} ({gate['checks_passed']}/12 checks)", fontsize=14)
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='lower center', ncol=4, bbox_to_anchor=(.5, .065), fontsize=9, frameon=False)
    fig.text(.5, .035, '64 shared fresh deck draws; fit markers are not intervals. References run once per deck. Original architecture gate remains FAIL 1/6.',
             ha='center', fontsize=8)
    fig.text(.5, .013, 'NLL weights fits, episodes, nonempty boundaries and queries hierarchically. Hatched infinite-NLL height is only a marker, not a finite value.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .18, 1, .93))
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(out / ('calibration.' + extension), dpi=180)
    plt.close(fig)


def replay_frame(episodes, receipts, step):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=13)
    small = ImageFont.load_default(size=11)
    canvas = Image.new('RGB', (1280, 390), '#f8f9fb')
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 9), f'PUBLIC FRAMES ONLY | Diagonal KDN, pair 0 | first fresh case, seed {receipts[0]["seed"]}', fill='#1e293b', font=font)
    draw.text((18, 29), 'Case and model fixed in the prospective protocol. All three belief modes shown with the same frozen model and C picker.', fill='#475569', font=small)
    for index, (policy, episode) in enumerate(zip(POLICIES, episodes, strict=True)):
        length = len(episode['actions']); t = min(step, length); x0 = 18 + index * 425
        title = {'baseline': 'Baseline beliefs', 'temperature': 'Temperature beliefs', 'hard': 'Hard top-rank beliefs'}[policy]
        draw.text((x0, 59), title + (f" (beta={receipts[index]['beta']:.3g})" if policy == 'temperature' else ''), fill='#1e293b', font=font)
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


def render(root, protocol_path, report_dir, expected_receipt_sha256, evaluation, out):
    begin = time.monotonic()
    root, protocol_path, report_dir, evaluation, out = (Path(p).resolve() for p in (root, protocol_path, report_dir, evaluation, out))
    require(not out.is_relative_to(evaluation) and not out.is_relative_to(report_dir), 'Render output must be separate')
    out.mkdir(parents=True, exist_ok=False)
    try:
        data, episodes, receipts, audit = authenticate(root, protocol_path, report_dir, expected_receipt_sha256, evaluation)
        plot(data, out)
        replay = public_replay(episodes, receipts, out / 'fixed-public-replay.gif')
        write(out / 'figure-data.json', data)
        authenticate(root, protocol_path, report_dir, expected_receipt_sha256, evaluation)
        members = ('calibration.png', 'calibration.svg', 'calibration.pdf', 'fixed-public-replay.gif', 'figure-data.json')
        result = {'status': 'complete', 'source_sha256': sha(__file__), 'report_receipt_sha256': expected_receipt_sha256,
                  'evaluation_completed_sha256': audit['evaluation_completed_sha256'],
                  'calibration_completed_sha256': audit['calibration_completed_sha256'],
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
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-path', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(render(**vars(parser.parse_args())), indent=2))
