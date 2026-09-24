"""Create a new immutable engineering or study registration, without model calls."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finite_rounded_learning_worker import (
    ROOT,
    admit_qualification,
    closed_producer,
    descriptor,
    files,
    publish,
    require,
    runtime,
)
from register_finite_training_allocation import SOURCES as PREVIOUS_SOURCES

KERNEL_SHA256 = '0922b1ecc92016b42410ef652d87f2a534be0b64b3750bff977947322e8ae375'
KERNEL_PUBLICATION_SHA256 = '1970b5920d5cb3b239bc2d2ddf79c99dc9280361195c5a9c403bb8bf9a9b78b7'


def authenticate_kernel(folder):
    """Admit the original closed numerical qualification without executing it."""
    plan_path = folder / 'registration-01.json'
    require(descriptor(plan_path)['sha256'] == KERNEL_SHA256, 'exact rounded primitive registration')
    plan = json.loads(plan_path.read_text())
    receipt = json.loads((folder / 'engineering-01.receipt.json').read_text())
    require(plan['version'] == 'finite-rounded-transition-qualification-v1'
            and plan['root'] == str(ROOT) and receipt['status'] == 'PASS'
            and receipt['plan'] == str(plan_path) and receipt['plan_sha256'] == KERNEL_SHA256,
            'original numerical qualification')
    require(receipt['sources_before'] == receipt['sources_after'] == plan['sources'],
            'numerical qualification source closure')
    for name, pin in plan['sources'].items():
        require(descriptor(ROOT / name) == pin == descriptor(folder / 'source-snapshot-01' / name),
                'unchanged rounded kernel source and snapshot')
    require(receipt['supervision'] == str(folder / 'native-01.launch.json')
            and json.loads(Path(receipt['supervision']).read_text()) == receipt['launch']
            and receipt['output'] == str(folder / 'engineering-01'), 'original numerical phase paths')
    closed_producer(receipt)
    expected = [plan['runtime']['executable'], str(ROOT / 'scripts/qualify_finite_rounded_transition.py'),
                'worker', '--plan', str(plan_path), '--plan-sha256', KERNEL_SHA256,
                '--supervision', receipt['supervision'], '--output', receipt['output']]
    require(receipt['launch']['command'] == expected and receipt['launch']['cap_seconds'] == 90
            and receipt['launch']['cwd'] == str(ROOT), 'original rounded qualifier invocation')
    require([row['command'] for row in receipt['commands']] == plan['commands']
            and len(receipt['commands']) == 2 and all(row['returncode'] == 0 for row in receipt['commands'])
            and files(Path(receipt['output'])) == receipt['files'], 'successful exact primitive commands and logs')
    publication = ROOT / 'research/finite-rounded-transition-qualification-results'
    publication_receipt = publication / 'receipt.json'
    require(descriptor(publication_receipt)['sha256'] == KERNEL_PUBLICATION_SHA256,
            'original independently checked qualification publication')
    published = json.loads(publication_receipt.read_text())
    require(published['status'] == 'PASS' and published['archive_roundtrip'] is True
            and published['numerical_calls'] == published['array_decodes'] == 0,
            'original opaque numerical-qualification publication')
    require(files(folder) == published['inputs'], 'every original numerical source, receipt and log')
    require(files(publication) == {**published['outputs'], 'receipt.json': descriptor(publication_receipt)},
            'unchanged numerical qualification publication')
    return {'plan': plan, 'publication': publication}

SOURCES = tuple(dict.fromkeys((*PREVIOUS_SOURCES,
    'src/openjev/research/finite_rounded_transition.py',
    'src/openjev/research/finite_rounded_models.py',
    'scripts/qualify_finite_rounded_transition.py',
    'scripts/publish_finite_rounded_transition_qualification.py',
    'scripts/register_finite_rounded_learning.py',
    'scripts/finite_rounded_learning_worker.py',
    'scripts/run_finite_rounded_learning.py',
    'scripts/audit_finite_rounded_learning.py',
    'tests/test_finite_rounded_transition.py',
    'tests/test_finite_rounded_models.py',
    'tests/test_finite_rounded_learning_runner.py',
    'tests/test_finite_rounded_learning_audit.py',
    'tests/test_finite_rounded_learning_worker.py',
    'research/finite-rounded-transition-qualification-protocol.md',
    'research/finite-rounded-learning-protocol.md',
    'research/finite-rounded-learning-requirements.txt',
    'src/openjev/__init__.py', 'src/openjev/research/__init__.py',
    'pyproject.toml', 'uv.lock',
)))
TESTS = ['tests/test_finite_rounded_transition.py', 'tests/test_finite_rounded_models.py',
         'tests/test_finite_training_allocation.py', 'tests/test_finite_rounded_learning_runner.py',
         'tests/test_finite_rounded_learning_audit.py', 'tests/test_finite_rounded_learning_worker.py']
LINT = [name for name in SOURCES if name.endswith('.py')]
CONFIG = {'seed_namespace': 432260924, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': [432261001, 432261002, 432261003],
          'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8,
          'stage1_seconds': 10., 'total_seconds': 40.}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('engineering', 'study'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--engineering-attempt', type=int, default=1)
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'dedicated repository only')
    require(args.engineering_attempt > 0, 'positive engineering attempt number')
    engineering_suffix = f'{args.engineering_attempt:02d}'
    folder = ROOT / 'output/finite-rounded-learning-v1'
    folder.mkdir(exist_ok=True)
    phases = {name: {'cap_seconds': cap, 'output': str(folder / output),
                     'supervision': str(folder / (prefix + '.launch.json'))}
              for name, cap, output, prefix in (
                  ('qualify', 300, 'engineering-' + engineering_suffix, 'engineering-native-' + engineering_suffix),
                  ('fit', 1200, 'run-01', 'fit-native-01'),
                  ('audit', 600, 'audit-01', 'audit-native-01'))}
    sources = {name: descriptor(ROOT / name) for name in SOURCES}
    kernel = ROOT / 'output/finite-rounded-transition-qualification-v1'
    qualified = authenticate_kernel(kernel)
    kernel_plan = qualified['plan']
    require({name: sources[name] for name in kernel_plan['sources']} == kernel_plan['sources'],
            'unchanged qualified balancing sources')
    plan = {'version': 'finite-rounded-learning-v1', 'mode': args.mode,
            'kernel_qualification': {'folder': str(kernel), 'files': files(kernel),
                                     'publication': str(qualified['publication']),
                                     'publication_files': files(qualified['publication'])},
            'root': str(ROOT), 'runtime': runtime(), 'sources': sources,
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'config': CONFIG, 'phases': phases, 'tests': TESTS, 'lint_sources': LINT,
            'rss_limit_bytes': 4 * 1024**3, 'output_limit_bytes': 512 * 1024**2}
    if args.mode == 'study':
        require(args.qualification is not None, 'completed engineering receipt required')
        qualification = json.loads(args.qualification.read_text())
        require(qualification['status'] == 'PASS' and qualification['sources_after'] == sources,
                'qualification must pass on identical sources')
        terminal = admit_qualification(plan, args.qualification)
        plan['qualification'] = {'path': str(args.qualification.resolve()),
                                 'descriptor': descriptor(args.qualification),
                                 'terminal': descriptor(terminal)}
        require(not Path(phases['fit']['output']).exists(), 'no prior study outputs')
    publish(args.plan, plan)
    snapshot = folder / ('source-snapshot-' + args.mode + '-' + engineering_suffix)
    snapshot.mkdir(exist_ok=False)
    for name, pin in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (ROOT / name).read_bytes()
        with target.open('xb') as stream:
            stream.write(payload)
        require(descriptor(target) == pin, 'unchanged source while snapshotting')
    publish(snapshot / 'manifest.json', {'registration': {'path': str(args.plan.resolve()),
            **descriptor(args.plan)}, 'sources': sources})
    print(json.dumps({'plan': str(args.plan.resolve()), **descriptor(args.plan)}))


if __name__ == '__main__':
    main()
