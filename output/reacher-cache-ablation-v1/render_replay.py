"""Authenticate and draw a fixed saved schematic episode, without simulation.

Only ordinary/case0/pair0, all five arms, all50 executed steps are supported.
Scored rendering requires explicit completion authorization. Engineering mode
requires an authenticated engineering plan and displays a persistent warning.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'reacher-cache-ablation-v1'
EXPECTED_SCORED_PLAN = '7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa'
WIDTH, HEIGHT = 1500, 930
ARMS = ('residual_gru', 'encoded_current_gru', 'cached_gru', 'packet_mlp', 'cached_mlp')
LABELS = ('Persistent GRU', 'Encoded-current GRU', 'Cached GRU', 'Packet MLP', 'Cached MLP')
DESCRIPTIONS = ('Carries learned recurrent history', 'Resets and encodes the current packet',
                'Resets + last actual visible angles', 'Current public packet',
                'Last actual angles + current public flags')
COLORS = {'ink': '#213047', 'muted': '#596777', 'paper': '#f3f5f8', 'visible': '#ffffff',
          'missing': '#fff2d9', 'first': '#17649a', 'second': '#138b87', 'target': '#d7524e',
          'trail': '#a1b7bd', 'grid': '#dde3e9', 'warning': '#b05110'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    def check(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'Nonfinite JSON')
        elif isinstance(item, dict):
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
    check(value)
    return value


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def checked(path, expected):
    require(isinstance(expected, str) and len(expected) == 64 and set(expected) <= set('0123456789abcdef'),
            'External/member SHA-256 required')
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == expected, 'Changed member: ' + str(path))
    return path


def member(folder, name):
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() == name and '\\' not in name,
            'Safe canonical relative member')
    return Path(folder).joinpath(*rel.parts)


def authenticate(plan_path, expected_plan, audit, expected_audit, execution, engineering):
    require(engineering or expected_plan == EXPECTED_SCORED_PLAN, 'Exact preselected scored plan required')
    receipt = read(checked(audit / 'receipt.json', expected_audit))
    require(receipt['status'] == 'completed' and receipt['saved_output_only'] is True,
            'Completed independent audit required before any execution-data read')
    plan = read(checked(plan_path, expected_plan))
    require(plan['study'] == STUDY and plan['engineering'] is engineering
            and receipt['status'] == 'completed' and receipt['engineering'] is engineering
            and receipt['plan_sha256'] == expected_plan and receipt['saved_output_only'] is True,
            'Completed plan/audit scope')
    if engineering:
        require(plan['rng_namespace'] == 'reacher-cache-engineering-whole-tree-v1'
                and plan['fixture_scope'] == 'synthetic_engineering_only_not_production_lineage_or_efficacy',
                'Explicit completed rehearsal only')
    else:
        require(plan['rng_namespace'] == 'reacher-cache-ablation-v1-scored', 'Scored namespace')
    require(plan['arms'] == list(ARMS) and plan['pairs'] == ['pair0', 'pair1', 'pair2']
            and plan['panels'] == ['full', 'ordinary', 'shift'] and plan['steps'] == 50,
            'All five preselected arms and complete scope')
    require(receipt['runtime'] == plan['runtime'], 'Frozen runtime binding')
    require(receipt['source_sha256'] == plan['sources'] and len(plan['sources']) == 70,
            'Exact70 source bindings')
    for name, expected in plan['sources'].items():
        checked(member(ROOT, name), expected)
    complete_path = checked(execution / 'completed.json', receipt['execution_completed_sha256'])
    complete = read(complete_path)
    require(complete['status'] == 'completed' and complete['plan_sha256'] == expected_plan
            and complete['files'] == receipt['execution_members'] and complete['fits'] == 15
            and complete['control_rows'] == 60, 'Complete15-fit60-row execution')
    for folder, hashes, extra in ((execution, complete['files'], {'completed.json'}),
                                   (audit, receipt['files'], {'receipt.json'})):
        paths = list(folder.rglob('*'))
        require(not folder.is_symlink() and not any(path.is_symlink() for path in paths), 'No symlinks')
        actual = {path.relative_to(folder).as_posix() for path in paths if path.is_file()}
        require(actual == set(hashes) | extra, 'Complete exact member set')
        for name, expected in hashes.items():
            checked(member(folder, name), expected)
    summary = read(audit / 'summary.json')
    require(summary['status'] == 'completed' and summary['saved_output_only'] is True
            and summary['engineering'] is engineering and summary['plan_sha256'] == expected_plan
            and summary['execution_completed_sha256'] == receipt['execution_completed_sha256']
            and summary['new_model_calls'] == summary['new_policy_calls'] == summary['new_fits'] == 0
            and summary['coverage']['fits'] == 15 and summary['coverage']['control_rows'] == 60,
            'Summary authentication and audit scope')
    return plan, receipt, summary


def geometry(plan):
    path = Path(importlib.metadata.distribution('gymnasium').locate_file('gymnasium/envs/mujoco/assets/reacher.xml'))
    checked(path, plan['runtime']['native_xml_sha256'])
    xml = ET.parse(path).getroot()
    first = xml.find("./worldbody/body[@name='body0']")
    second = first.find("./body[@name='body1']")
    tip = second.find("./body[@name='fingertip']")
    target = xml.find("./worldbody/body[@name='target']")
    def vector(item, name):
        return np.fromstring(item.attrib[name], sep=' ', dtype=np.float64)
    origin, elbow_offset, tip_offset = vector(first, 'pos'), vector(second, 'pos'), vector(tip, 'pos')
    require(np.array_equal(origin[:2], [0, 0]) and np.array_equal(elbow_offset[1:], [0, 0])
            and np.array_equal(tip_offset[1:], [0, 0]), 'Supported planar XY chain only')
    for body, name in ((first, 'joint0'), (second, 'joint1')):
        joint = body.find(f"./joint[@name='{name}']")
        require(joint.attrib['type'] == 'hinge' and np.array_equal(vector(joint, 'axis'), [0, 0, 1])
                and np.array_equal(vector(joint, 'pos'), [0, 0, 0]), 'Planar zero-origin hinge')
    refs = np.asarray([float(target.find(f"./joint[@name='{name}']").attrib['ref'])
                       for name in ('target_x', 'target_y')])
    require(np.array_equal(refs, vector(target, 'pos')[:2]), 'Saved target qpos directly denotes world XY')
    return {'xml_path': str(path), 'xml_sha256': sha(path), 'first_link_m': float(elbow_offset[0]),
            'joint_to_fingertip_center_m': float(tip_offset[0]),
            'scope': 'Planar centerline schematic from hash-bound XML; not a simulator image.'}


def forward_kinematics(qpos, lengths):
    q = np.asarray(qpos, dtype=np.float64)
    first, second = lengths['first_link_m'], lengths['joint_to_fingertip_center_m']
    elbow = first * np.stack((np.cos(q[..., 0]), np.sin(q[..., 0])), axis=-1)
    tip = elbow + second * np.stack((np.cos(q[..., 0]+q[..., 1]), np.sin(q[..., 0]+q[..., 1])), axis=-1)
    return elbow, tip


def load_rows(plan, execution, summary, lengths):
    rows, bindings = [], {}
    for arm in ARMS:
        name = f'{arm}-pair0'
        folder = execution / 'control' / 'ordinary' / name
        meta = read(folder / 'episodes.json')
        require(len(meta) == plan['control_episodes'], 'All recorded cases remain present')
        with np.load(folder / 'episodes.npz', allow_pickle=False) as data:
            arrays = {key: data[key][0].copy() for key in ('audit__qpos', 'audit__raw_obs', 'audit__time',
                      'audit__rewards', 'policy__packets')}
        q, raw, times, rewards, public = (arrays[key] for key in ('audit__qpos', 'audit__raw_obs', 'audit__time',
                                                       'audit__rewards', 'policy__packets'))
        require(q.shape == (51, 4) and raw.shape == (51, 10) and times.shape == (51,)
                and rewards.shape == (50,) and public.shape == (51, 8)
                and all(np.isfinite(value).all() for value in arrays.values()), 'Complete50-step saved arrays')
        require(np.allclose(np.diff(times), .02, rtol=0, atol=1e-12) and plan['dt'] == .02
                and meta[0]['dt'] == .02 and meta[0]['horizon'] == 50, 'Recorded native timing')
        require(np.all(q[:, 2:4] == q[0, 2:4]) and np.allclose(public[:, 4:6], q[:, 2:4], atol=1e-8),
                'Static known target')
        valid = np.asarray(meta[0]['sensor_schedule'], dtype=bool)
        require(valid.shape == (51,) and np.array_equal(valid, public[:, 6].astype(bool))
                and not valid.all() and valid[0], 'Ordinary blackout visibility from actual public packets')
        elbow, tip = forward_kinematics(q, lengths)
        fk_error = float(np.max(np.abs((tip-q[:, 2:4])-raw[:, -2:])))
        require(np.allclose(raw[:, :2], np.cos(q[:, :2]), rtol=0, atol=1e-12)
                and np.allclose(raw[:, 2:4], np.sin(q[:, :2]), rtol=0, atol=1e-12),
                'Saved raw angle channels agree with qpos')
        # The displacement comparison is descriptive. Graphics use FK of saved
        # qpos; cached native body positions need not be bit-identical after RK4.
        cumulative = np.concatenate(([0.], -np.cumsum(rewards, dtype=np.float64)))
        recorded_cost = summary['control']['ordinary'][name]['episode_costs'][0]
        require(math.isclose(float(cumulative[-1]), recorded_cost, rel_tol=1e-12, abs_tol=1e-12),
                'Saved cumulative cost matches authenticated case0 audit')
        rows.append({'name': name, 'qpos': q, 'elbow': elbow, 'tip': tip, 'times': times,
                     'valid': valid, 'cost': cumulative, 'fk_max_abs_error': fk_error})
        bindings[name] = {'case_index': 0, 'pair': 'pair0', 'panel': 'ordinary',
                         'episodes_npz_sha256': sha(folder / 'episodes.npz'),
                         'episodes_json_sha256': sha(folder / 'episodes.json'),
                         'fit_completed_sha256': sha(execution / 'fits' / name / 'completed.json'),
                         'weights_sha256': sha(execution / 'fits' / name / 'weights.pt'),
                         'native_time_seconds': times.tolist(), 'decision_observation_valid': valid[:-1].tolist(),
                         'final_cumulative_native_cost': float(cumulative[-1]), 'fk_max_abs_error': fk_error}
    for row in rows[1:]:
        require(np.array_equal(row['qpos'][0], rows[0]['qpos'][0])
                and np.array_equal(row['valid'], rows[0]['valid'])
                and np.array_equal(row['times'], rows[0]['times']), 'Same first case, initial state and sensor schedule')
    return rows, bindings


def fonts():
    candidates = [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    path = next((value for value in candidates if value.is_file()), None)
    require(path is not None, 'A hashable Arial/DejaVuSans font is required')
    return {size: ImageFont.truetype(str(path), size=size) for size in (14, 16, 18, 20, 22, 28)}, {'path': str(path), 'sha256': sha(path)}


def frame(rows, step, font, engineering):
    require(len(rows) == 5 and type(step) is int and 1 <= step <= 50, 'Five panels and one executed step')
    image = Image.new('RGB', (WIDTH, HEIGHT), COLORS['paper'])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 84), fill=COLORS['ink'])
    draw.text((26, 12), 'OpenJev | saved schematic replay', font=font[28], fill='white')
    draw.text((27, 51), 'Ordinary blackout | first case (0), first fit pair (0) | all five models',
              font=font[18], fill='#dbe5ef')
    if engineering:
        draw.rectangle((0, 84, WIDTH, 119), fill=COLORS['warning'])
        draw.text((26, 91), 'ENGINEERING FIXTURE - NOT A SCIENTIFIC RESULT', font=font[18], fill='white')
    else:
        draw.rectangle((0, 84, WIDTH, 119), fill='#dceaf4')
        draw.text((26, 91), 'One fixed episode, not a performance summary. Full study: 15 fits / 60 rows.',
                  font=font[18], fill=COLORS['ink'])
    for index, (row, title, description) in enumerate(zip(rows, LABELS, DESCRIPTIONS, strict=True)):
        x, y = 25+(index % 3)*490, 142+(index // 3)*355
        missing = not row['valid'][step-1]
        draw.rounded_rectangle((x, y, x+470, y+330), radius=12,
                               fill=COLORS['missing'] if missing else COLORS['visible'], outline='#d2dce5', width=1)
        draw.text((x+16, y+11), title, font=font[22], fill=COLORS['ink'])
        draw.text((x+16, y+41), description, font=font[14], fill=COLORS['muted'])
        state = 'ANGLES MISSING' if missing else 'ANGLES OBSERVED'
        draw.text((x+16, y+65), state, font=font[16],
                  fill=COLORS['warning'] if missing else COLORS['muted'])
        cx, cy, scale = x+235, y+182, 450
        def point(value, cx=cx, cy=cy, scale=scale):
            return (round(cx+float(value[0])*scale), round(cy-float(value[1])*scale))
        for value in (-.2, -.1, 0, .1, .2):
            draw.line((point((value, -.22)), point((value, .22))), fill=COLORS['grid'], width=1)
            draw.line((point((-.22, value)), point((.22, value))), fill=COLORS['grid'], width=1)
        for value in (-.2, 0, .2):
            draw.text((point((value, 0))[0]-11, cy+101), f'{value:.1f}', font=font[14], fill=COLORS['muted'])
        trail = [point(p) for p in row['tip'][:step+1]]
        draw.line(trail, fill=COLORS['trail'], width=2)
        target = point(row['qpos'][0, 2:4])
        draw.ellipse((target[0]-7, target[1]-7, target[0]+7, target[1]+7), outline=COLORS['target'], width=3)
        draw.line((target[0]-11, target[1], target[0]+11, target[1]), fill=COLORS['target'], width=1)
        draw.line((target[0], target[1]-11, target[0], target[1]+11), fill=COLORS['target'], width=1)
        origin, elbow, tip = point((0, 0)), point(row['elbow'][step]), point(row['tip'][step])
        draw.line((origin, elbow), fill=COLORS['first'], width=10)
        draw.line((elbow, tip), fill=COLORS['second'], width=8)
        for dot, radius, color in ((origin, 6, COLORS['ink']), (elbow, 5, COLORS['first']), (tip, 5, COLORS['second'])):
            draw.ellipse((dot[0]-radius, dot[1]-radius, dot[0]+radius, dot[1]+radius), fill=color)
        draw.text((x+16, y+307), f'Saved cumulative native cost: {row["cost"][step]:.3f}',
                  font=font[16], fill=COLORS['ink'])
    x, y = 1005, 497
    draw.rounded_rectangle((x, y, x+470, y+330), radius=12, fill='#e8edf3', outline='#d2dce5')
    draw.text((x+16, y+14), 'How to read this replay', font=font[22], fill=COLORS['ink'])
    legend = (
        'Amber: decision angles missing. White: observed.',
        'State shown: after the saved action.',
        'Red target; faint trail shows the fingertip path.',
        'Motion uses saved native joint positions (qpos).',
        'Clean angles were not model inputs in blackouts.',
        'Cost = cumulative negative native reward.',
        'Lower cost is better within this fixed episode.',
        'Same initial state and blackout schedule.',
        'All 50 actions shown. No score-based selection.',
        'Schematic centerlines, not simulator screenshots.',
    )
    for line, text in enumerate(legend):
        draw.text((x+16, y+54+line*26), text, font=font[16], fill=COLORS['muted'])
    elapsed = rows[0]['times'][step]-rows[0]['times'][0]
    draw.text((27, 847), f'Step {step:02d}/50  |  simulation {elapsed:.2f} s  |  playback 5x slower  |  axes in meters',
              font=font[18], fill=COLORS['ink'])
    draw.text((27, 876), "Panel shading uses the actual observation at this action's decision; motion shows saved post-action state.",
              font=font[16], fill=COLORS['muted'])
    draw.text((27, 903), 'One second of recorded simulation, replayed over five seconds. This illustration makes no aggregate or superiority claim.',
              font=font[16], fill=COLORS['muted'])
    return image


def render(args):
    require(args.engineering or args.expected_plan_sha256 == EXPECTED_SCORED_PLAN,
            'Exact preselected scored plan required before reading any outputs')
    require(args.engineering or args.completed_authorized,
            'Scored render requires explicit completed-study authorization before reading outputs')
    begin = time.perf_counter()
    plan_path, audit, execution, out = (Path(value).resolve() for value in (args.plan, args.audit, args.execution, args.out))
    require(not out.exists(), 'Exclusive rendering output directory')
    plan, receipt, summary = authenticate(plan_path, args.expected_plan_sha256, audit,
                                         args.expected_audit_receipt_sha256, execution, args.engineering)
    lengths = geometry(plan)
    rows, bindings = load_rows(plan, execution, summary, lengths)
    font, font_metadata = fonts()
    frames = [frame(rows, step, font, args.engineering) for step in range(1, 51)]
    out.mkdir(parents=True, exist_ok=False)
    try:
        path = out / 'first-ordinary-case-pair0.gif'
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False, disposal=2)
        # The overview is a QA aid, not an extra episode or selected successful clip.
        preview = Image.new('RGB', (WIDTH*3, HEIGHT))
        first_missing = int(np.flatnonzero(~rows[0]['valid'][:-1])[0])
        for index, source in enumerate((frames[0], frames[first_missing], frames[-1])):
            preview.paste(source, (index*WIDTH, 0))
        preview.save(out / 'layout-preview.png')
        with Image.open(path) as saved:
            require(saved.n_frames == 50, 'Every executed step appears once')
            durations = []
            for index in range(saved.n_frames):
                saved.seek(index)
                durations.append(saved.info.get('duration'))
            require(durations == [100]*50, 'Exact10fps, 5-second animation')
        result = {'status': 'completed', 'scope': 'engineering_only_no_efficacy' if args.engineering else 'fixed_first_case_schematic_scored_replay',
                  'engineering': args.engineering, 'plan_sha256': args.expected_plan_sha256,
                  'audit_receipt_sha256': args.expected_audit_receipt_sha256,
                  'audit_summary_sha256': receipt['files']['summary.json'],
                  'execution_completed_sha256': receipt['execution_completed_sha256'],
                  'renderer_source_sha256': sha(Path(__file__)), 'source_sha256': plan['sources'],
                  'case_index': 0, 'pair': 'pair0', 'panel': 'ordinary', 'models': bindings,
                  'layout': {'columns': 3, 'rows': 2, 'model_panels': 5, 'legend_panels': 1,
                             'width': WIDTH, 'height': HEIGHT},
                  'frames': 50, 'executed_steps': list(range(1, 51)), 'frame_duration_ms': 100,
                  'playback_duration_seconds': 5., 'native_episode_duration_seconds': 1., 'slowdown_factor': 5,
                  'frame_native_timestamps_seconds': rows[0]['times'][1:].tolist(),
                  'initial_state_included_in_path_not_separate_frame': True,
                  'visibility_semantics': 'Packet at decision t controls blackout shading; frame displays recorded post-action state t+1. Clean audit angles never become model input.',
                  'geometry': lengths, 'font': font_metadata,
                  'fk_displacement_comparison': 'Descriptive difference from saved raw fingertip-target displacement. The centerlines use XML and saved qpos, not native camera pixels or bit-identical body-cache coordinates.',
                  'runtime': {'numpy': np.__version__, 'pillow': importlib.metadata.version('pillow')},
                  'new_model_calls': 0, 'new_policy_calls': 0, 'new_simulator_calls': 0,
                  'selection': 'Fixed ordinary case0/pair0 for all five arms; all50 steps; no score-based selection.',
                  'limits': 'A schematic single-case illustration, not native camera pixels, an aggregate result, or an architecture-superiority claim.',
                  'rendered_at_utc': datetime.datetime.now(datetime.UTC).isoformat(),
                  'wall_seconds': time.perf_counter()-begin,
                  'files': {name: sha(out/name) for name in ('first-ordinary-case-pair0.gif', 'layout-preview.png')}}
        write(out / 'receipt.json', result)
        return result
    except BaseException as error:
        write(out / 'failed.json', {'status': 'failed', 'error': repr(error), 'scope': 'rendering_only'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--expected-audit-receipt-sha256', required=True)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--completed-authorized', action='store_true')
    args = parser.parse_args()
    result = render(args)
    print(json.dumps({key: result[key] for key in ('status', 'scope', 'frames', 'wall_seconds', 'files')}, indent=2))


if __name__ == '__main__':
    main()
