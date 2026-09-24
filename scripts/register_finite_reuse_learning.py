"""Freeze a fresh equal-update comparison and its qualified engineering exposure."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import qualify_finite_joint_reuse_throughput_v2 as throughput
from finite_reuse_learning_worker import (
    ROOT,
    admit_qualification,
    descriptor,
    files,
    publish,
    require,
    runtime,
)

CONFIG = {'seed_namespace': 434260924, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': [434261001, 434261002, 434261003],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.}
EXPOSURE = {'config': {**CONFIG, 'seed_namespace': 944201, 'fit_seeds': [944301],
                       'dev_attempts': 8, 'prefix_updates': 32, 'joint_updates': 64,
                       'fit_cap_seconds': 30.},
            'target_counts': {'prefix_updates': 1024, 'joint_updates': 3072},
            'maximum_projected_seconds': 90.0}
SOURCES = tuple(dict.fromkeys((*throughput.SOURCES,
    'scripts/publish_finite_joint_reuse_throughput.py',
    'scripts/run_finite_reuse_learning.py', 'scripts/audit_finite_reuse_learning.py',
    'scripts/register_finite_reuse_learning.py', 'scripts/finite_reuse_learning_worker.py',
    'scripts/qualify_finite_reuse_exposure.py',
    'tests/test_finite_reuse_learning_runner.py',
    'tests/test_finite_reuse_learning_audit.py', 'tests/test_finite_reuse_learning_worker.py',
    'research/finite-reuse-learning-protocol.md',
)))
TESTS = ['tests/test_finite_rounded_transition.py', 'tests/test_finite_rounded_models.py',
         'tests/test_finite_update_allocation.py', 'tests/test_finite_joint_reuse.py',
         'tests/test_finite_joint_reuse_training.py', 'tests/test_finite_reuse_learning_runner.py',
         'tests/test_finite_reuse_learning_audit.py', 'tests/test_finite_reuse_learning_worker.py']
THROUGHPUT_PLAN_SHA256 = '2ebc7bda8b49b0c0ac9684a15bcdd614c17ed5b2f4e7f60d9b3329e92c6eebc6'
THROUGHPUT_PUBLICATION_SHA256 = 'b759f446559cb338b011b6c2f06e3b96385c7cd82fbfdb35a27bd4ddf9fb6304'


def phase_specs():
    folder = ROOT / 'output/finite-reuse-learning-v1'
    return {name: {'cap_seconds': cap, 'output': str(folder / output),
                   'supervision': str(folder / (prefix + '.launch.json'))}
            for name, cap, output, prefix in (
                ('qualify', 300, 'engineering-01', 'engineering-native-01'),
                ('fit', 1200, 'run-01', 'fit-native-01'),
                ('audit', 600, 'audit-01', 'audit-native-01'))}


def authenticate_throughput():
    """Read only original closed metadata and opaque registered evidence."""
    publication = ROOT / 'research/finite-joint-reuse-throughput-results'
    receipt_path = publication / 'receipt.json'
    require(descriptor(receipt_path)['sha256'] == THROUGHPUT_PUBLICATION_SHA256,
            'exact published throughput receipt')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['status'] == 'PASS' and receipt['inputs_unchanged'] is True
            and receipt['scientific_admission'] is False
            and receipt['model_calls'] == receipt['checkpoint_decodes'] == receipt['audit_reruns'] == 0,
            'metadata-only published throughput proof')
    require(descriptor(throughput.PLAN_PATH)['sha256'] == THROUGHPUT_PLAN_SHA256
            and receipt['registration'] == {'path': str(throughput.PLAN_PATH),
                                             **descriptor(throughput.PLAN_PATH)},
            'exact original throughput registration')
    plan = json.loads(throughput.PLAN_PATH.read_text())
    require(len(plan['sources']) == 92, 'exact92-source throughput closure')
    closures = {phase: throughput.closed_phase(plan, phase) for phase in ('qualify', 'run', 'audit')}
    require(files(throughput.FOLDER) == receipt['study_files']
            and files(throughput.FAILED) == receipt['failed_qualification_files']
            and files(publication) == {**receipt['files'], 'receipt.json': descriptor(receipt_path)},
            'complete unchanged throughput and original failure inventories')
    require(receipt['publisher'] == descriptor(ROOT / 'scripts/publish_finite_joint_reuse_throughput.py'),
            'original throughput publisher source')
    overview = ROOT / 'research/finite-joint-reuse-throughput-results.md'
    require(receipt['overview'] == {'path': str(overview), **descriptor(overview)}, 'published overview identity')
    audit = json.loads((closures['audit']['directory'] / 'audit.json').read_text())
    require(audit['agreement'] is True and audit['scientific_admission'] is False
            and audit['counts']['fits'] == 24 and audit['counts']['measured_pairs'] == 9
            and audit['sources'] == plan['sources'] and audit['throughput']['passed'] is True
            and len(audit['throughput']['conditions']) == 12
            and all(audit['throughput']['conditions'].values()), 'complete audited throughput result')
    return {'folder': str(throughput.FOLDER), 'files': receipt['study_files'],
            'registration': receipt['registration'], 'publication': str(publication),
            'publication_files': files(publication), 'overview': receipt['overview'],
            'sources': plan['sources'], 'prerequisite': plan['prerequisite'],
            'failed_attempt': plan['failed_attempt'],
            'closures': {phase: {key: descriptor(row[key]) for key in
                         ('receipt_path', 'launch_path', 'terminal_path')} for phase, row in closures.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('engineering', 'study'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'dedicated repository')
    folder = ROOT / 'output/finite-reuse-learning-v1'
    folder.mkdir(exist_ok=True)
    expected_plan = folder / ('engineering-registration-01.json' if args.mode == 'engineering'
                              else 'study-registration.json')
    require(args.plan.resolve() == expected_plan and not expected_plan.exists(), 'exclusive registered plan path')
    phases = phase_specs()
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    qualified = authenticate_throughput()
    plan = {'version': 'finite-reuse-learning-v1', 'mode': args.mode,
            'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'config': CONFIG, 'exposure_probe': EXPOSURE, 'phases': phases,
            'tests': TESTS, 'lint_sources': [name for name in SOURCES if name.endswith('.py')],
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2,
            'throughput_qualification': qualified}
    if args.mode == 'study':
        require(args.qualification is not None, 'completed engineering receipt required')
        terminal = admit_qualification(plan, args.qualification)
        plan['qualification'] = {'path': str(args.qualification.resolve()),
            'descriptor': descriptor(args.qualification), 'terminal': descriptor(terminal)}
        require(not Path(phases['fit']['output']).exists(), 'no prior scientific run')
    publish(args.plan, plan)
    snapshot = folder / ('source-snapshot-' + args.mode + '-01')
    snapshot.mkdir(exist_ok=False)
    for name, pin in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT / name).read_bytes())
        require(descriptor(target) == pin, 'unchanged snapshot source')
    publish(snapshot / 'manifest.json', {'registration': {'path': str(args.plan.resolve()),
            **descriptor(args.plan)}, 'sources': sources})
    print(json.dumps({'plan': str(args.plan.resolve()), **descriptor(args.plan)}))


if __name__ == '__main__':
    main()
