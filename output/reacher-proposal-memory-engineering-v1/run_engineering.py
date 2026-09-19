"""Fixed nine-row engineering comparison; original case and innovations reused."""
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research.reacher_proposal_memory_audit import audit_row
from openjev.research.reacher_proposal_memory_rollout import run_row
from openjev.research.reacher_tracking_dynamics import nominal_model
from openjev.research.reacher_tracking_rollout import check_deadline, sha, write

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE/'attempt-01'
OLD = ROOT/'output/persistent-dynamics-qualification-v1/control-engineering-01'
PROTOCOL = BASE/'protocol.json'
OLD_COMPLETE = '064a0e7157088a6625bfc53afd1022d6d16232dc85dbf6cbde3238c1314bc050'
NEW = ['src/openjev/research/'+name+'.py' for name in ('reacher_cem_proposal_memory',
    'reacher_proposal_memory_rollout','reacher_proposal_memory_audit')]+[Path(__file__).relative_to(ROOT).as_posix(),PROTOCOL.relative_to(ROOT).as_posix(),(BASE/'report_results.py').relative_to(ROOT).as_posix()]


def protected_sources():
    bound=[]
    for name,key in [('evidence/reacher-two-observation-study-v1/protocol/plan.json','sources'),
                     ('evidence/reacher-innovation-pilot-v1/protocol/plan.json','source_sha256')]:
        mapping=json.loads((ROOT/name).read_text())[key]
        assert all(sha(ROOT/path)==digest for path,digest in mapping.items())
        bound.append({'plan':name,'count':len(mapping),'plan_sha256':sha(ROOT/name)})
    return bound


def cold_parity(folder,arm,deadline):
    old=OLD/'rows'/arm
    old_receipt=json.loads((old/'completed.json').read_text())
    checked=0
    for t in range(200):
        check_deadline(deadline)
        name=f'decisions/{t:03d}.npz'
        assert sha(old/name)==old_receipt['files'][name]
        with np.load(old/name,allow_pickle=False) as left,np.load(folder/name,allow_pickle=False) as right:
            assert set(left.files)==set(right.files)
            for key in left.files:
                assert left[key].dtype==right[key].dtype and np.array_equal(left[key],right[key]), (arm,t,key)
                checked+=1
    assert sha(old/'episode.json')==old_receipt['files']['episode.json']
    left=json.loads((old/'episode.json').read_text());right=json.loads((folder/'episode.json').read_text())
    assert left==right, 'cold native episode must reproduce bit-exactly'
    return {'status':'identical','arm':arm,'decision_arrays':checked,'episode_equal':True,
            'old_completed_sha256':sha(old/'completed.json')}


def main():
    torch.set_num_threads(1)
    protected=protected_sources()
    assert sha(OLD/'completed.json')==OLD_COMPLETE and not (OLD/'failed.json').exists()
    old=json.loads((OLD/'completed.json').read_text())
    assert old['status']=='completed'
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest
    for row in old['rows']:assert sha(OLD/'rows'/row['arm']/'completed.json')==row['completed_sha256']
    for audit in old['audits']:assert sha(OLD/f"audit-{audit['arm']}.json")==audit['sha256']
    protocol=json.loads(PROTOCOL.read_text())
    assert protocol['source_run_completed_sha256']==OLD_COMPLETE
    sources={**old['source_sha256'],**{name:sha(ROOT/name) for name in NEW}}
    OUT.mkdir(exist_ok=False)
    begin=time.monotonic();stage='preparation';rows=[];audits=[];parities=[];execution_valid=False
    try:
        write(OUT/'started.json',{'status':'started','engineering':True,'protocol':protocol,
            'source_sha256':sources,'protected_sources':protected,'old_completed_sha256':OLD_COMPLETE,
            'runtime':{'python':platform.python_version(),'platform':platform.platform(),
                'packages':{n:importlib.metadata.version(n) for n in ('numpy','torch','mujoco','gymnasium')},
                'torch_threads':torch.get_num_threads()},'automatic_retry':False})
        for name,digest in sources.items():
            dest=OUT/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write((ROOT/name).read_bytes())
            assert sha(dest)==digest
        with np.load(OLD/'case.npz',allow_pickle=False) as case:
            targets,gains,noise=(case[k].copy() for k in ('targets','gains','noise'))
        old_episode=json.loads((OLD/'rows/nominal/episode.json').read_text())
        old_row=json.loads((OLD/'rows/nominal/completed.json').read_text())
        assert sha(OLD/'rows/nominal/episode.json')==old_row['files']['episode.json']
        assert np.array_equal(targets,old_episode['metadata']['target_path'])
        assert np.array_equal(gains,old_episode['metadata']['gear_multiplier'])
        assert np.array_equal(noise,old_episode['metadata']['noise'])
        stems=[OLD/'inputs'/f'{t:03d}' for t in range(200)]
        old_cfg=json.loads((OLD/'rows/nominal/started.json').read_text())
        assert sha(OLD/'rows/nominal/started.json')==old_row['files']['started.json']
        for stem,binding in zip(stems,old_cfg['inputs_by_step'],strict=True):
            for suffix in ('npz','json'):assert sha(stem.with_suffix('.'+suffix))==binding[suffix+'_sha256']
        write(OUT/'case-binding.json',{'case_npz_sha256':sha(OLD/'case.npz'),'source':str(OLD),
                                      'inputs_by_step':old_cfg['inputs_by_step']})
        stage='execution'
        for spec in protocol['matrix']:
            name=spec['arm']+'--'+spec['proposal_mode']
            value=run_row(**spec,target_path=targets,gain_schedule=gains,noise=noise,reset_seed=410,
                inputs_by_step=stems,gain_grid=np.array(old_cfg['gain_grid']),window=20,freeze_after=40,
                noise_std=.05,planning_horizon=12,action_block=3,out=OUT/'rows'/name,
                deadline=begin+protocol['execution_cap_seconds'],engineering=True)
            rows.append({**spec,'name':name,'completed_sha256':sha(OUT/'rows'/name/'completed.json'),
                         'wall_seconds':value['wall_seconds']})
            print(json.dumps({'phase':'row_complete','name':name,'wall_seconds':value['wall_seconds']}),flush=True)
        check_deadline(begin+protocol['execution_cap_seconds'])
        write(OUT/'execution-completed.json',{'status':'completed','engineering':True,'rows':rows,
            'wall_seconds':time.monotonic()-begin,'work':protocol['planned_work']})
        check_deadline(begin+protocol['execution_cap_seconds']);execution_valid=True
        stage='audit';audit_start=time.monotonic();deadline=audit_start+protocol['audit_cap_seconds']
        for row in rows:
            folder=OUT/'rows'/row['name']
            value=audit_row(folder,nominal_model=nominal_model(),inputs_by_step=stems,deadline=deadline)
            write(OUT/('audit-'+row['name']+'.json'),value)
            audits.append({'name':row['name'],'sha256':sha(OUT/('audit-'+row['name']+'.json'))})
            if row['proposal_mode']=='cold':parities.append(cold_parity(folder,row['arm'],deadline))
            print(json.dumps({'phase':'audit_complete','name':row['name']}),flush=True)
        protected_sources()
        assert all(sha(ROOT/name)==digest for name,digest in sources.items())
        check_deadline(deadline)
        write(OUT/'completed.json',{'status':'completed','engineering':True,'rows':rows,'audits':audits,
            'cold_parity':parities,'scientific_gate':None,'scientific_data_allocated':False,
            'execution_completed_sha256':sha(OUT/'execution-completed.json'),'source_sha256':sources,
            'audit_wall_seconds':time.monotonic()-audit_start,'total_wall_seconds':time.monotonic()-begin})
        check_deadline(deadline)
    except BaseException as error:
        for name in ('completed.json','execution-completed.json'):
            if name=='execution-completed.json' and execution_valid:continue
            if (OUT/name).exists():
                try:(OUT/name).rename(OUT/('partial-'+name))
                except BaseException as secondary:  # noqa: BLE001 - preserve original
                    error.add_note('Completion demotion failed: '+repr(secondary))
        try:
            write(OUT/'failed.json',{'status':'failed','stage':stage,'error':repr(error),'rows':rows,'audits':audits,
                                    'wall_seconds':time.monotonic()-begin,'automatic_retry':False})
        except BaseException as secondary:  # noqa: BLE001 - preserve original
            error.add_note('Failure receipt failed: '+repr(secondary))
        raise


if __name__=='__main__':
    main()
