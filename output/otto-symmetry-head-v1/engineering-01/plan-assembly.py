"""Assemble this one reviewed pilot plan; no numerical or native execution."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUTS = {
    'prior_plan': ('output/otto-boundary-control-v1/plan-01.json', 'fad1e7add6d1ad763dc26343f9be30cdf48094f075869daa2420e3dbd18726e5'),
    'prior_receipt': ('output/otto-boundary-control-v1/run-01/receipt.json', '835852e7881316510539b3b041bb817c60e05ab08aec71e8c865fa8a9bee0196'),
    'prior_terminal': ('output/otto-boundary-control-v1/run-process-01.terminal.json', 'be28dbf58f429f48f662be2642bb8950db84f5aaf12541f8d32c87e04bc03ee6'),
    'prior_audit_receipt': ('output/otto-boundary-control-v1/audit-01/receipt.json', 'f525c41ec8d9a9f333de2427194306d8ca4bbcdc7b2cdad28a572f34e116fb78'),
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()

def regular(name):
    rel=Path(name)
    require(not rel.is_absolute() and '..' not in rel.parts, 'relative contained source')
    path=ROOT/rel
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'regular contained source')
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and not args.output.exists()
            and not any(p.is_symlink() for p in (args.output,*args.output.parents)), 'exclusive absolute plan path')
    engineering=Path(__file__).parent
    reviewed=json.loads((engineering/'final-source-review.json').read_text())
    require(reviewed['status']=='completed' and reviewed['source_reviews_clear'] is True, 'final actual source reviews')
    for name,pin in reviewed['sources'].items():
        require(sha(regular(name))==pin, 'held source changed: '+name)
    inputs={}
    for role,(name,pin) in INPUTS.items():
        path=regular(name)
        require(sha(path)==pin, 'completed input pin: '+role)
        inputs[role]={'path':name,'sha256':pin,'bytes':path.stat().st_size}
    prior=json.loads(regular(INPUTS['prior_plan'][0]).read_text())
    worker=json.loads(regular(INPUTS['prior_receipt'][0]).read_text())
    audit=json.loads(regular(INPUTS['prior_audit_receipt'][0]).read_text())
    require(len(prior['sources'])==95 and worker['status']=='completed' and worker['completed_episodes']==768
            and worker['competent_reference'] is True and worker['restriction_benefit'] is False
            and worker['stronger_value_teacher'] is False and worker['utility_compute_advantage'] is False,
            'completed prior evidence and unchanged scientific decisions')
    require(audit['status']=='completed' and audit['agreement'] is True, 'completed prior audit')
    spec=importlib.util.spec_from_file_location('_symmetry_plan_constants',ROOT/'scripts/study_otto_symmetry_head.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    require(not {'numpy','torch','scipy','tensorflow'}.intersection(sys.modules), 'source-only plan assembly')
    require(set(inputs)==module.ROLES and module.REQUIRED<=reviewed['sources'].keys()|prior['sources'].keys(), 'complete reviewed new sources')
    sources=dict(prior['sources'])
    def merge(name,pin):
        require(name not in sources or sources[name]==pin, 'source conflict: '+name)
        require(sha(regular(name))==pin, 'current source identity: '+name)
        sources[name]=pin
    for name,pin in reviewed['sources'].items():
        merge(name,pin)
    files=[]
    for path in sorted(engineering.rglob('*')):
        require(not path.is_symlink(), 'no engineering symlink')
        if path.is_file():
            require('__pycache__' not in path.parts, 'do not bind volatile bytecode')
            name=path.relative_to(ROOT).as_posix();merge(name,sha(path));files.append(name)
    for name,pin in sources.items():
        require(sha(regular(name))==pin, 'unchanged complete source closure: '+name)
    plan={'version':module.VERSION,'status':'frozen_before_native_run','configuration':module.CONFIGURATION,
          'limits':module.LIMITS,'sources':dict(sorted(sources.items())),'inputs':inputs,
          'python_version':sys.version.split()[0],'python_executable':sys.executable,
          'runtime_versions':{n:importlib.metadata.version(n) for n in ('numpy','scipy','torch')},
          'all_distributions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
          'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'assembly':{'helper_sha256':sha(Path(__file__)),'inherited_source_count':95,'engineering_files':files,
                      'scope':'Source and identity assembly only. Final authentication and supervised execution remain separate.'}}
    with args.output.open('x') as f:
        json.dump(plan,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'plan':str(args.output),'sha256':sha(args.output),'sources':len(sources),
                      'runtime_distributions':len(plan['all_distributions']),'new_native_or_training_calls':0}))

if __name__=='__main__':
    main()
