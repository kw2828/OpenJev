"""Freeze the conditioning study using authenticated metadata only."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_conditioning_freezer', ROOT/'scripts/study_otto_conditioning.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
BASE = ROOT/'output/otto-capacity-v1/plan-01.json'
BASE_PIN = 'f00b138d01c8687d600a04fd37eac793387e026a1926e4094a2c0c079601e4de'
INPUTS = {
    'capacity_plan': ('output/otto-capacity-v1/plan-01.json', BASE_PIN),
    'capacity_receipt': ('output/otto-capacity-v1/run-01/receipt.json', 'ec7f9551c5a590f8548b14ec2a42a4456b0c65a87756185c76b67cc81986e8a9'),
    'capacity_terminal': ('output/otto-capacity-v1/run-process-01.terminal.json', '8d7ef2cc90ab986e2dc53e47bbe9dbdee7e6b7044c360ce03c27b2417e4c00d8'),
    'capacity_audit': ('output/otto-capacity-v1/audit-01/receipt.json', 'ac33b263d27a45dfd3f92f9486c4e1c6afe8da746c4bf6142851f2dc4adcc447'),
}


def freeze(args):
    output = args.output
    S.require(output.is_absolute() and not output.exists() and not any(p.is_symlink() for p in output.parents), 'exclusive plan')
    base = S.C.authenticate(SimpleNamespace(plan=BASE, plan_sha256=BASE_PIN))
    sources = dict(base['sources'])
    for name in S.NEW:
        sources[name] = S.sha(S.regular(name))
    inputs = {}
    for role, (name, pin) in INPUTS.items():
        path = S.regular(name)
        S.require(S.sha(path) == pin, 'externally bound historical evidence')
        inputs[role] = {'path': name, **S.descriptor(path)}
    for role in ('train_data', 'train_rows', 'valid_data', 'valid_rows') if args.mode == 'study' else ('train_data', 'train_rows'):
        inputs[role] = base['inputs'][role]
    qualification = {}
    for role in ('plan', 'receipt', 'terminal'):
        path, pin = getattr(args, 'qualification_'+role), getattr(args, 'qualification_'+role+'_sha256')
        if args.mode == 'qualify':
            S.require(path is None and pin is None, 'no qualification inputs in disposable mode')
        else:
            S.require(path is not None and path.is_absolute() and pin is not None and S.sha(path) == pin,
                      'external qualification completion pins')
            qualification[role] = {'path': str(path.relative_to(ROOT)), **S.descriptor(path)}
    plan = {'version': S.VERSION, 'status': 'frozen_before_execution', 'mode': args.mode,
            'configuration': S.configuration(args.mode), 'limits': S.limits(args.mode),
            'independent_audit_limits': {'seconds': 600, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2},
            'sources': sources, 'inputs': inputs, 'qualification': qualification,
            'python_executable': base['python_executable'], 'python_version': base['python_version'],
            'all_distributions': base['all_distributions']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path': str(output), 'sha256': S.sha(output), 'sources': len(sources), 'arrays_decoded': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('qualify', 'study'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    for role in ('plan', 'receipt', 'terminal'):
        parser.add_argument('--qualification-'+role, type=Path)
        parser.add_argument('--qualification-'+role+'-sha256')
    freeze(parser.parse_args())
