"""Freeze metadata for the matched autonomous conditioning comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_freeze_conditioning_control',ROOT/'scripts/study_otto_conditioning_control.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
INPUTS = {
    'conditioning_plan': ('output/otto-conditioning-v1/plan-01.json','10072bb7227cfe1e106a6519e5822a8fc9d3cf0a66dfaec4cbba0b60801aedb2'),
    'conditioning_receipt': ('output/otto-conditioning-v1/run-01/receipt.json','0c879a9c8f0f418080c01f1a98c7ef79255909a92a48ec7e6dd61289de7a9d90'),
    'conditioning_terminal': ('output/otto-conditioning-v1/run-process-01.terminal.json','182617cf0b6814f0ac86d3a6caa50f6872568f121cb6bd49bda7a70d57a6c201'),
    'conditioning_audit': ('output/otto-conditioning-v1/audit-01/receipt.json','bcee27c5083acfbe0d7611b9fce62328a1282cb3cd2ab9569f30dd917e752daf'),
    'scalar_receipt': ('output/otto-return-value-v1/run-01/receipt.json','965cc3eee7c77ea64700137bda2961af1a259f1a98d15b54041f9530a7f66ac3'),
}


def freeze(args):
    C.require(args.output.is_absolute() and not args.output.exists()
              and not any(p.is_symlink() for p in args.output.parents), 'exclusive plan')
    name,pin = INPUTS['conditioning_plan']
    prior = C.S.authenticate(SimpleNamespace(plan=ROOT/name,plan_sha256=pin))
    sources = dict(prior['sources'])
    sources.update({name:C.sha(C.regular(name)) for name in C.NEW})
    inputs = {}
    for role,(name,pin) in INPUTS.items():
        path = C.regular(name)
        C.require(C.sha(path) == pin, 'externally bound completed evidence')
        inputs[role] = {'path':name,**C.descriptor(path)}
    for regime in C.REGIMES:
        name = f'output/otto-return-value-v1/run-01/kernel-{regime}.npz'
        inputs['kernel_'+regime] = {'path':name,**C.descriptor(C.regular(name))}
    for arm in C.ARMS[:-1]:
        name = 'output/otto-conditioning-v1/run-01/final-'+arm.replace('@','-')+'.npz'
        inputs['head_'+arm.replace('@','_')] = {'path':name,**C.descriptor(C.regular(name))}
    if args.mode == 'qualify':
        inputs.update({role:prior['inputs'][role] for role in ('train_data','valid_data','train_rows','valid_rows')})
    C.require(args.seed_review.is_absolute() and C.sha(args.seed_review) == args.seed_review_sha256, 'external seed review binding')
    inputs['seed_review'] = {'path':str(args.seed_review.relative_to(ROOT)),**C.descriptor(args.seed_review)}
    qualification = {}
    for role in ('plan','receipt','terminal','audit'):
        path,pin = getattr(args,'qualification_'+role),getattr(args,'qualification_'+role+'_sha256')
        if args.mode == 'qualify':
            C.require(path is None and pin is None, 'fresh qualification')
        else:
            C.require(path is not None and path.is_absolute() and pin is not None and C.sha(path) == pin, 'external qualification completion')
            qualification[role] = {'path':str(path.relative_to(ROOT)),**C.descriptor(path)}
    plan = {'version':C.VERSION,'status':'frozen_before_execution','mode':args.mode,
            'configuration':C.configuration(args.mode),'limits':C.limits(args.mode),
            'independent_audit_limits':C.audit_limits(args.mode),'sources':sources,'inputs':inputs,'qualification':qualification,
            **{k:prior[k] for k in ('python_executable','python_version','all_distributions')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    C.write(args.output,plan)
    print(json.dumps({'path':str(args.output),'sha256':C.sha(args.output),'sources':len(sources),'arrays_decoded':0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('qualify','study'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-review',type=Path,required=True)
    parser.add_argument('--seed-review-sha256',required=True)
    for role in ('plan','receipt','terminal','audit'):
        parser.add_argument('--qualification-'+role,type=Path)
        parser.add_argument('--qualification-'+role+'-sha256')
    freeze(parser.parse_args())
