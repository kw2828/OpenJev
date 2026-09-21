"""Assemble an exclusive capacity protocol from authenticated metadata; no array reads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_capacity_plan_source', ROOT/'scripts/study_otto_capacity.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
BASE = ROOT/'output/otto-bellman-control-v1/plan-01.json'
BASE_PIN = 'f8e62494bc67d7859824900f4599ea4c1f890d0ec13d2a43ca1ae679e41aac38'
ADDED = ('scripts/freeze_otto_capacity.py', 'scripts/audit_otto_capacity.py', 'tests/test_audit_otto_capacity.py')


def freeze(args):
    S.require(args.output.is_absolute() and not args.output.exists()
              and not any(p.is_symlink() for p in args.output.parents), 'exclusive plan path')
    S.require(S.sha(BASE)==BASE_PIN, 'historical main plan')
    base = S.read(BASE)
    sources = dict(base['sources'])
    for name,pin in sources.items():
        S.require(S.sha(S.regular(name))==pin, 'unchanged historical source')
    for name in (*S.NEW,*ADDED):
        sources[name] = S.sha(S.regular(name))
    inputs = dict(base['inputs'])
    prior = S.read(S.regular(inputs['prior_receipt']['path']))
    parent = Path(inputs['prior_receipt']['path']).parent
    for split in ('train','valid') if args.mode=='study' else ('train',):
        for suffix,role in (('data.npz','data'),('rows.jsonl','rows')):
            name = split+'-'+suffix
            path = S.regular(str(parent/name))
            S.require(S.descriptor(path)==prior['files'][name], 'closed input bytes')
            inputs[split+'_'+role] = {'path':str(parent/name),**prior['files'][name]}
    qualification = {}
    if args.mode=='study':
        for role in ('plan','receipt','terminal'):
            path = getattr(args,'qualification_'+role)
            S.require(path is not None and path.is_absolute(), 'qualification identity required')
            qualification[role] = {'path':str(path.relative_to(ROOT)),**S.descriptor(path)}
    else:
        S.require(all(getattr(args,'qualification_'+k) is None for k in ('plan','receipt','terminal')), 'no qualification for qualification')
    plan = {'version':S.VERSION,'mode':args.mode,'status':'frozen_before_execution',
            'configuration':S.configuration(args.mode),'limits':S.limits(args.mode),
            'sources':sources,'inputs':inputs,'qualification':qualification,
            'python_executable':base['python_executable'],'python_version':base['python_version'],
            'all_distributions':base['all_distributions'],
            'independent_audit_limits':{'seconds':600,'rss_bytes':4*1024**3,'output_bytes':128*1024**2}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(plan,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path':str(args.output),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'sources':len(sources),'mode':args.mode,'arrays_decoded':0}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('qualify','study'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    for kind in ('plan','receipt','terminal'):
        parser.add_argument('--qualification-'+kind,type=Path)
    freeze(parser.parse_args())
