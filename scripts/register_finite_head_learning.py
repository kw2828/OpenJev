"""Freeze the separate task-independent readout initialization diagnostic."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import register_finite_reuse_replication as parent
from finite_head_learning_worker import (
    ROOT,
    admit_qualification,
    descriptor,
    files,
    publish,
    require,
    runtime,
)

CONFIG = {'seed_namespace': 436260924, 'train_attempts': 512, 'dev_attempts': 512,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': [436261001, 436261002, 436261003, 436261004, 436261005],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'prefix_updates': 1024, 'joint_updates': 3072, 'fit_cap_seconds': 120.}
EXPOSURE = {'config': {**CONFIG, 'seed_namespace': 946201, 'fit_seeds': [946301],
                       'dev_attempts': 8, 'prefix_updates': 32, 'joint_updates': 64,
                       'fit_cap_seconds': 30.},
            'target_counts': {'prefix_updates': 1024, 'joint_updates': 3072},
            'maximum_projected_seconds': 90.0}
SOURCES = tuple(dict.fromkeys((*parent.SOURCES,
    'scripts/publish_finite_reuse_replication.py',
    'src/openjev/research/finite_head_initialization.py',
    'scripts/run_finite_head_training.py', 'scripts/run_finite_head_learning.py',
    'scripts/audit_finite_head_learning.py', 'scripts/register_finite_head_learning.py',
    'scripts/finite_head_learning_worker.py', 'scripts/qualify_finite_head_exposure.py',
    'tests/test_finite_head_initialization.py', 'tests/test_finite_head_training.py',
    'tests/test_finite_head_learning_runner.py', 'tests/test_finite_head_learning_audit.py',
    'tests/test_finite_head_learning_worker.py',
    'research/finite-head-learning-protocol.md',
    'research/finite-recurrent-next-direction.md',
)))
TESTS = ['tests/test_finite_observation_world.py', 'tests/test_finite_prefix_learning.py',
         'tests/test_finite_rounded_transition.py', 'tests/test_finite_rounded_models.py',
         'tests/test_finite_update_allocation.py', 'tests/test_finite_joint_reuse.py',
         'tests/test_finite_joint_reuse_training.py', 'tests/test_finite_regime_reference.py',
         'tests/test_finite_head_initialization.py', 'tests/test_finite_head_training.py',
         'tests/test_finite_head_learning_runner.py', 'tests/test_finite_head_learning_audit.py',
         'tests/test_finite_head_learning_worker.py']
PARENT_PLAN_SHA256 = '2b456de9caf180c6f7505a584d91fa9c58fb4222b8848b24f73363994e426f21'
PARENT_PUBLICATION_SHA256 = '01996f92dd52ef9f1013475b775dff65879b168c3f68ecd2c890a5ad4dd067f0'
PARENT_PUBLISHER_SHA256 = '49755507c3f162ab153d1fce0071cee6e714604818cc34f9d5dce7577b02509f'


def phase_specs():
    folder = ROOT / 'output/finite-head-learning-v1'
    return {name: {'cap_seconds': cap, 'output': str(folder / output),
                   'supervision': str(folder / (prefix + '.launch.json'))}
            for name, cap, output, prefix in (
                ('qualify', 300, 'engineering-01', 'engineering-native-01'),
                ('fit', 1200, 'run-01', 'fit-native-01'),
                ('audit', 600, 'audit-01', 'audit-native-01'))}


def authenticate_parent():
    """Preserve the closed replication failure before admitting a new diagnostic."""
    publication = ROOT / 'research/finite-reuse-replication-results'
    receipt_path = publication / 'receipt.json'
    require(descriptor(receipt_path)['sha256'] == PARENT_PUBLICATION_SHA256,
            'exact published replication receipt')
    publisher_path = ROOT / 'scripts/publish_finite_reuse_replication.py'
    require(descriptor(publisher_path)['sha256'] == PARENT_PUBLISHER_SHA256,
            'exact metadata-only parent admission helper')
    import publish_finite_reuse_replication as published
    require(descriptor(published.PLAN)['sha256'] == PARENT_PLAN_SHA256,
            'exact original replication registration')
    authenticated = published.authenticate()
    receipt = json.loads(receipt_path.read_text())
    require(receipt['status'] == 'PASS' and receipt['inputs_unchanged'] is True
            and receipt['publisher'] == descriptor(publisher_path)
            and all(value == 0 for value in receipt['publication_counts'].values()),
            'parent publication completed without numerical replay')
    advance = authenticated['audit']['advance']
    require(advance['passed'] is False and len(advance['conditions']) == 54
            and sum(advance['conditions'].values()) == 45
            and advance['status'] == 'REPLICATION_SHIFT_ADVANCE_FAIL',
            'preserve failed replication; no candidate promotion')
    require(authenticated['study_files'] == receipt['study_files']
            and authenticated['plan']['sources'] == receipt['sources']
            and files(publication) == {**receipt['files'], 'receipt.json': descriptor(receipt_path)}
            and descriptor(published.OVERVIEW) == {key: receipt['overview'][key] for key in ('sha256', 'bytes')},
            'complete unchanged parent study and publication')
    return {'folder': str(published.STUDY), 'files': receipt['study_files'],
            'registration': receipt['registration'], 'publication': str(publication),
            'publication_files': files(publication), 'overview': receipt['overview'],
            'publisher': descriptor(publisher_path), 'sources': receipt['sources'],
            'phase_closures': receipt['phase_closures'],
            'replication_prerequisite': receipt['replication_prerequisite'],
            'parent_advance': advance}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('engineering', 'study'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'dedicated repository')
    folder = ROOT / 'output/finite-head-learning-v1'
    folder.mkdir(exist_ok=True)
    expected_plan = folder / ('engineering-registration-01.json' if args.mode == 'engineering'
                              else 'study-registration.json')
    require(args.plan.resolve() == expected_plan and not expected_plan.exists(), 'exclusive registered plan path')
    phases = phase_specs()
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    qualified = authenticate_parent()
    plan = {'version': 'finite-head-learning-v1', 'mode': args.mode,
            'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'config': CONFIG, 'exposure_probe': EXPOSURE, 'phases': phases,
            'tests': TESTS, 'lint_sources': [name for name in SOURCES if name.endswith('.py')],
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2,
            'head_prerequisite': qualified}
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
