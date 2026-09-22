"""Freeze only metadata for all-head deployment qualification or autonomous study."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_freeze_spatial_control', ROOT/'scripts/study_otto_spatial_control.py')
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def freeze(args):
    C.require(args.output.is_absolute() and not args.output.exists()
              and not any(p.is_symlink() for p in args.output.parents), 'exclusive frozen plan')
    name,pin = C.INPUTS['spatial_plan']
    prior = C.S.authenticate(SimpleNamespace(plan=ROOT/name,plan_sha256=pin))
    sources = dict(prior['sources'])
    for name,pin in C.AUDIT_DEPENDENCIES.items():
        C.require(C.sha(C.regular(name)) == pin and (name not in sources or sources[name] == pin),
                  'unchanged inherited audit dependency')
        sources[name] = pin
    for name in C.NEW:
        pin = C.sha(C.regular(name))
        C.require(name not in sources or sources[name] == pin, 'no replacement of frozen source')
        sources[name] = pin
    inputs = {}
    for role,(name,pin) in C.INPUTS.items():
        path = C.regular(name)
        C.require(C.sha(path) == pin, 'external completed evidence before decode')
        inputs[role] = {'path':name,**C.descriptor(path)}
    for regime in C.REGIMES:
        name = f'output/otto-return-value-v1/run-01/kernel-{regime}.npz'
        inputs['kernel_'+regime] = {'path':name,**C.descriptor(C.regular(name))}
    for arm in C.ARMS[:-1]:
        name = 'output/otto-spatial-study-v1/run-01/final-'+arm.replace('@','-')+'.npz'
        inputs['head_'+arm.replace('@','_')] = {'path':name,**C.descriptor(C.regular(name))}
    if args.mode == 'qualify':
        inputs.update({role:prior['inputs'][role] for role in ('train_data','valid_data','train_rows','valid_rows')})
    C.require(args.seed_review.is_absolute() and C.sha(args.seed_review) == args.seed_review_sha256, 'external seed review')
    inputs['seed_review'] = {'path':str(args.seed_review.relative_to(ROOT)),**C.descriptor(args.seed_review)}
    qualification = {}
    for role in ('plan','receipt','terminal','audit'):
        path,pin = getattr(args,'qualification_'+role),getattr(args,'qualification_'+role+'_sha256')
        if args.mode == 'qualify':
            C.require(path is None and pin is None, 'fresh qualification only')
        else:
            C.require(path is not None and path.is_absolute() and pin is not None and C.sha(path) == pin,
                      'externally pinned complete qualification')
            qualification[role] = {'path':str(path.relative_to(ROOT)),**C.descriptor(path)}
    projection = C.serialization_projection()
    C.require(projection['within_worker_cap'], 'fabricated actual serialization projection fits allocation')
    plan = {'version':C.VERSION,'status':'frozen_before_execution','mode':args.mode,
            'configuration':C.configuration(args.mode),'limits':C.limits(args.mode),
            'expected_calls':C.expected_calls(args.mode),'independent_audit_limits':C.audit_limits(args.mode),
            'serialization_projection':projection,'audit_serialization_projection':C.audit_projection(args.mode,sources),
            'audit_study_serialization_projection':C.audit_projection('study',sources),
            'sources':sources,'inputs':inputs,'qualification':qualification,
            **{k:prior[k] for k in ('python_executable','python_version','all_distributions')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    C.write(args.output,plan)
    C.require(C.authenticate(SimpleNamespace(plan=args.output,plan_sha256=C.sha(args.output))) == plan,
              'complete prospective metadata authentication without array decoding')
    print(json.dumps({'path':str(args.output),'sha256':C.sha(args.output),'sources':len(sources),
                      'projected_output_bytes':projection['projected_bytes'],'arrays_decoded':0}),flush=True)


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
