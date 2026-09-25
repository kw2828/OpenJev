"""Fabricated scalar/metadata/array witnesses; no measured data or model calls."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_robot_joint_observer as audit


def fixture():
    old=audit.parent_audit.parent_audit
    rates={'local_affine':.003,'temporal_affine':.001,**dict.fromkeys(old.FIXED),**old.CACHED_RATES,
           'observer_position':.003,'observer_position_fixed':None}
    cfg={'inherited_rates':rates,'selection_scope':'fixed prospective rates, no selection'}
    rows=[]
    for name in audit.EXPOSED:
        for recipe in (*old.identities(),*audit.parent_audit.identities(),*audit.identities()):
            for horizon in (64,128):
                error=.75 if recipe['arm']==audit.CANDIDATE else 1.
                failed=recipe['arm']=='observer_learned'; count=22*horizon*6
                rows.append({'recording':name,'fit_key':recipe['key'],'arm':recipe['arm'],'seed':recipe['seed'],
                             'learning_rate':recipe['learning_rate'],'horizon':horizon,'status':'FAILED' if failed else 'PASS',
                             'error':{'type':'RetainedParentFailure'} if failed else None,
                             'metrics':None if failed else {'standardized_rmse':error,'standardized_sse':error**2*count,
                             'physical_rmse_deg':error,'per_joint_rmse_deg':[error]*6,'scalars':count,'windows':22,'horizon':horizon}})
    resources=[]
    for identity in audit.resource_identities(rates):
        duration=1.5 if identity['arm']==audit.CANDIDATE else 1.
        resources.append({**identity,**audit.storage(identity['arm']),'status':'PASS','error':None,'timing':{'median_seconds':duration}})
    return cfg,rows,resources


def change(rows,arm,value,rate=None,recording=None,horizon=None):
    for row in rows:
        if row['arm']==arm and (rate is None or row['learning_rate']==rate) and (recording is None or row['recording']==recording) and (horizon is None or row['horizon']==horizon):
            row['metrics']['standardized_rmse']=value
            row['metrics']['standardized_sse']=value**2*row['metrics']['scalars']


def flags(cfg,rows,resources):
    return [r['passed'] for r in audit.decisions(rows,resources,cfg)['result']['conditions']]


def test_independent584_rows61_costs_and_all_five_pass():
    cfg,rows,resources=fixture(); value=audit.decisions(rows,resources,cfg)
    assert len(rows)==584 and len(resources)==61 and len(audit.ELIGIBLE)==23 and len(audit.DISPLAY)==24
    assert value['result']['passed']==value['result']['total']==5
    assert value['selection']['fixed_rates']==dict.fromkeys(audit.ARMS,.001)
    assert not any(r['arm']==audit.ALIAS for r in rows)
    assert value['result']['equal_file_means'][audit.ALIAS]==1.
    assert 'observer_learned' not in value['result']['equal_file_means']
    assert [(r['arm'],r['seed']) for r in audit.identities()]==[(a,s) for a in audit.ARMS for s in (8101,8102,8103)]


@pytest.mark.parametrize('value,passed',[(.95,True),(.950001,False)])
def test_five_percent_boundary_and_zero_baseline_not_improvement(value,passed):
    cfg,rows,resources=fixture(); change(rows,audit.CANDIDATE,value)
    assert flags(cfg,rows,resources)[1] is passed
    change(rows,audit.CANDIDATE,0.);change(rows,audit.LAST,0.)
    assert flags(cfg,rows,resources)[1] is False


def test_same_rate_frozen_alias_uses001_without_reselecting003_parent():
    cfg,rows,resources=fixture()
    change(rows,'observer_position',.5,rate=.001);change(rows,'observer_position',2.,rate=.003)
    result=audit.decisions(rows,resources,cfg)
    assert result['result']['equal_file_means'][audit.ALIAS]==.5
    assert result['result']['equal_file_means']['observer_position']==2.
    assert result['selection']['inherited_rates']['observer_position']==.003
    assert result['result']['conditions'][1]['passed'] is False


@pytest.mark.parametrize('value,passed',[(.51,True),(.510001,False)])
def test_per_file_two_percent_uses_fresh_simple_control(value,passed):
    cfg,rows,resources=fixture();change(rows,audit.CANDIDATE,value)
    change(rows,audit.SHORT,.5,recording=audit.EXPOSED[0])
    assert flags(cfg,rows,resources)[2] is passed


def test_historical_simple_does_not_replace_fresh_per_file_comparator():
    cfg,rows,resources=fixture();change(rows,'last_two',.1,recording=audit.EXPOSED[0])
    assert flags(cfg,rows,resources)[1] is False
    assert flags(cfg,rows,resources)[2] is True


@pytest.mark.parametrize('duration,passed',[(1.5,True),(1.500001,False)])
def test_latency_is_relative_to_fresh_last_two(duration,passed):
    cfg,rows,resources=fixture()
    for row in resources:
        if row['arm']==audit.CANDIDATE:row['timing']['median_seconds']=duration
        if row['arm']=='last_two':row['timing']['median_seconds']=.01
    assert flags(cfg,rows,resources)[3] is passed


def test_complete_frontier_ties_are_not_strict_domination():
    cfg,rows,resources=fixture();change(rows,audit.SHORT,.75)
    for row in resources:
        if row['arm']==audit.SHORT:row['timing']['median_seconds']=1.5
    assert flags(cfg,rows,resources)[4] is True
    for row in resources:
        if row['arm']==audit.SHORT:row['timing']['median_seconds']=1.499
    assert flags(cfg,rows,resources)[4] is False


def test_selected_h64_failure_and_unavailable_training_cost_remain_explicit():
    cfg,rows,resources=fixture()
    row=next(r for r in rows if r['arm']==audit.CANDIDATE and r['horizon']==64)
    row.update(status='FAILED',metrics=None,error={'type':'FailedTrainingAttempt'})
    assert flags(cfg,rows,resources)[0] is False
    for row in resources:
        if row['arm']==audit.CANDIDATE:
            row.update(status='UNAVAILABLE',error={'type':'FailedTrainingAttempt','effective_status':'FAILED'},timing=None)
    result=flags(cfg,rows,resources)
    assert result[0] is result[3] is result[4] is False


@pytest.mark.parametrize('damage',['row','duplicate','alias_row','wrong_rate','fit_key','cost','cost_alias','unknown_status','nan_cost'])
def test_full_identity_and_alias_rosters_reject_corruption(damage):
    cfg,rows,resources=fixture()
    if damage=='row':rows.pop()
    elif damage=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    elif damage=='alias_row':rows[-1]['arm']=audit.ALIAS
    elif damage=='wrong_rate':next(r for r in rows if r['arm']==audit.CANDIDATE)['learning_rate']=.003
    elif damage=='fit_key':rows[0]['fit_key']='unrelated'
    elif damage=='cost':resources.pop()
    elif damage=='cost_alias':resources[-1]['model_key']='observer_position-8103-lr1'
    elif damage=='unknown_status':resources[0]['status']='MAYBE'
    else:resources[0]['timing']['median_seconds']=float('nan')
    with pytest.raises(ValueError):audit.decisions(rows,resources,cfg)


def joint_fixture(arm=audit.CANDIDATE):
    backbone={'cell.weight':np.zeros(590,np.float32)}
    constructed={k:v.copy() for k,v in backbone.items()}
    if arm in (audit.CANDIDATE,audit.SHORT):
        constructed={'gain':np.concatenate((np.eye(6,dtype=np.float32),np.zeros((6,6),np.float32))),**constructed}
    elif arm==audit.TEMPORAL:
        constructed.update({'head.weight':np.zeros((12,30),np.float32),'head.bias':np.zeros(12,np.float32)})
    initial={k:v.copy() for k,v in constructed.items()}; final={k:v.copy() for k,v in initial.items()}
    final['cell.weight'][:]=.25
    optimizer={}
    for name,value in initial.items():
        optimizer[name+'/step']=np.array(2.,np.float32)
        optimizer[name+'/exp_avg']=np.zeros_like(value);optimizer[name+'/exp_avg_sq']=np.zeros_like(value)
    receipt={'status':'PASS','completed_updates':2}
    return initial,final,optimizer,constructed,backbone,arm,receipt


@pytest.mark.parametrize('arm',audit.ARMS)
def test_joint_cell_is_allowed_to_learn_and_all_parameters_own_adam(arm):
    args=joint_fixture(arm)
    audit.validate_joint_evidence(*args,np,updates=2)
    assert not np.array_equal(args[0]['cell.weight'],args[1]['cell.weight'])
    assert sum(v.size for v in args[0].values())==audit.storage(arm)['parameters']


@pytest.mark.parametrize('damage',['initial_cell','initial_gain','missing_cell_adam','extra_adam','wrong_step','final_nan','moment_nan'])
def test_initialization_and_optimizer_ownership_corruptions(damage):
    args=list(joint_fixture());initial,final,optimizer=args[:3]
    if damage=='initial_cell':initial['cell.weight'][0]=1.
    elif damage=='initial_gain':initial['gain'][6,0]=1.
    elif damage=='missing_cell_adam':optimizer.pop('cell.weight/exp_avg')
    elif damage=='extra_adam':optimizer['unregistered/step']=np.array(2.,np.float32)
    elif damage=='wrong_step':optimizer['cell.weight/step']=np.array(1.,np.float32)
    elif damage=='final_nan':final['gain'][0,0]=np.nan
    else:optimizer['cell.weight/exp_avg'][0]=np.inf
    with pytest.raises(ValueError):audit.validate_joint_evidence(*args,np,updates=2)


def test_failed_attempt_preserves_updated_nonfinite_values_without_promoting():
    args=list(joint_fixture());args[1]['cell.weight'][0]=np.nan
    args[6]={'status':'FAILED','completed_updates':1}
    audit.validate_joint_evidence(*args,np,updates=2)


def test_generic_gradient_norm_includes_cell_and_initializer_and_reports_nonfinite_entries():
    gradients={'cell.weight':np.zeros(590,np.float32),'gain':np.zeros((12,6),np.float32)}
    gradients['cell.weight'][0]=3e20;gradients['gain'][0,0]=4e20
    summary=audit.gradient_summary(gradients,np)
    assert summary['numel']==662 and summary['all_finite']
    assert summary['norm64']==pytest.approx(5e20,rel=1e-7)
    gradients['cell.weight'][1]=np.nan;gradients['gain'][1,0]=np.inf
    summary=audit.gradient_summary(gradients,np)
    assert summary['nonfinite_count']==2 and summary['max_abs'] is summary['norm64'] is None


def test_closed_form_metrics_and_u31_alignment():
    target=np.zeros((2,128,6),np.float64);prediction=np.ones_like(target)
    prediction[:,64:]=3.
    metrics64=audit.scored(prediction,target,np.arange(1.,7.),64,np)
    metrics128=audit.scored(prediction,target,np.arange(1.,7.),128,np)
    assert metrics64['standardized_sse']==768. and metrics128['standardized_sse']==7680.
    assert metrics64['physical_rmse_deg']==pytest.approx(math.sqrt(91/6))
    index=np.arange(3636,dtype=np.float64)[:,None]
    record={'q':np.repeat(index,6,axis=1),'u':np.repeat(index+10000.,6,axis=1),
            'raw_indices':np.arange(0,90881,25,dtype=np.int64)}
    norm={'q_mean':np.zeros(6),'q_std':np.ones(6),'u_mean':np.zeros(6),'u_std':np.ones(6)}
    starts,batch,truth=audit.windows(record,norm,np)
    np.testing.assert_array_equal(starts,64+160*np.arange(22))
    assert batch['future_u'][0,0,0]==10095 and truth[0,0,0]==96


def write(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

def admission_fixture(root, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', root)
    study, eng = root/audit.STUDY, root/audit.ENGINEERING
    sources = {}
    for name in audit.SOURCES:
        payload = ('opaque source '+name).encode()
        write(root/name, payload); write(study/'sources'/name, payload)
        sources[name] = audit.descriptor(root/name)
    write(root/audit.LAUNCHER, b'opaque launcher')
    qpath = eng/'qualification-01/receipt.json'
    commands = [['.venv/bin/ruff','check',*[n for n in audit.QUALIFICATION_SOURCES if n.endswith('.py')]],
                ['.venv/bin/python','-m','pytest','--noconftest','-q','tests/test_robot_joint_observer.py',
                 'tests/test_robot_joint_observer_study.py','tests/test_audit_robot_joint_observer.py']]
    command_rows = []
    for i, command in enumerate(commands):
        path = qpath.parent/f'command-{i+1:02d}.log'; write(path, b'original fabricated qualification log')
        command_rows.append({'command':command,'returncode':0,'seconds':1.,'log':str(path),
                             'sha256':audit.descriptor(path)['sha256']})
    qualification = {'status':'PASS','sources_unchanged':True,'sources':sources,'launcher':audit.descriptor(root/audit.LAUNCHER),
                     'thread_env':dict.fromkeys(audit.THREADS,'1'),'commands':command_rows,'created_utc':'2026-01-01T00:00:00+00:00'}
    write(qpath, qualification)
    plan = {'version':audit.STUDY_VERSION,'sources':sources,'launcher':qualification['launcher'],
            'created_utc':'2026-01-02T00:00:00+00:00','qualification':audit.pin(qpath)}
    write(root/audit.REGISTRATION, plan); write(study/'registration.json',plan)
    launch = {'command':audit.COMMAND.copy(),'prefit_commit':'a'*40,'started_utc':'2026-01-03T00:00:00+00:00',
              'registration_sha256':audit.descriptor(root/audit.REGISTRATION)['sha256'],'launcher':plan['launcher'],
              'thread_env':dict.fromkeys(audit.THREADS,'1'),'scope':'fabricated admission'}
    write(eng/'run-process-01.log', b'original fabricated terminal log')
    process = {**launch,'returncode':0,'elapsed_seconds':2.,'external_timeout':False,'log':audit.descriptor(eng/'run-process-01.log')}
    write(eng/'run-launch-01.json',launch); write(eng/'run-process-01.json',process)
    runtime = {'python':audit.sys.version,'platform':audit.platform.platform(),'machine':audit.platform.machine(),
               'torch_threads':1,'thread_env':dict.fromkeys(audit.THREADS,'1'),'numpy':'test-numpy','torch':'test-torch',
               'clock':'mach_continuous_time' if audit.sys.platform=='darwin' else 'CLOCK_BOOTTIME'}
    write(study/'runtime.json',runtime); write(study/'receipt.json',{'opaque':'no model execution'})
    write(study/'not-decoded.npz', b'not an NPZ archive')
    write(study/'manifest.json',{'files':{str(p.relative_to(study)):audit.descriptor(p)
          for p in study.rglob('*') if p.is_file() and p.name!='receipt.json'}})
    monkeypatch.setattr(audit.importlib.metadata,'version',lambda package:'test-'+package)
    monkeypatch.setattr(audit.subprocess,'check_output',lambda argv,**kwargs:(root/argv[-1].split(':',1)[1]).read_bytes())
    for key in audit.THREADS: monkeypatch.setenv(key,'1')
    def no_decode(*args,**kwargs): raise AssertionError('arrays decoded before admission')
    monkeypatch.setattr(np,'load',no_decode)
    return study,eng,plan,launch,process,qualification

def test_original_metadata_admits_opaque_arrays_without_decoding(tmp_path,monkeypatch):
    study,eng,plan,_,_,_ = admission_fixture(tmp_path,monkeypatch)
    actual,inputs = audit.original_admission(study,eng/'run-process-01.json')
    assert actual == plan and len(plan['sources']) == 63
    assert inputs['registration']['sha256'] == audit.descriptor(tmp_path/audit.REGISTRATION)['sha256']

@pytest.mark.parametrize('damage',('source','snapshot','launcher','registration','command','launch_join','code','timeout','elapsed',
                                    'log','commit','committed_source','qualification_pin','manifest','runtime'))
def test_original_admission_tampering_fails_before_decode(tmp_path,monkeypatch,damage):
    study,eng,plan,launch,process,_ = admission_fixture(tmp_path,monkeypatch)
    if damage=='source': write(tmp_path/audit.SOURCES[0],b'changed')
    elif damage=='snapshot': write(study/'sources'/audit.SOURCES[0],b'changed')
    elif damage=='launcher': write(tmp_path/audit.LAUNCHER,b'changed')
    elif damage=='registration': write(study/'registration.json',b'changed')
    elif damage=='command': launch['command']=process['command']=['python','unrelated.py']
    elif damage=='launch_join': launch['started_utc']='2026-01-04T00:00:00+00:00'
    elif damage=='code': process['returncode']=1
    elif damage=='timeout': process['external_timeout']=True
    elif damage=='elapsed': process['elapsed_seconds']=7261.
    elif damage=='log': write(eng/'run-process-01.log',b'changed')
    elif damage=='commit': launch['prefit_commit']=process['prefit_commit']='bad'
    elif damage=='committed_source': monkeypatch.setattr(audit.subprocess,'check_output',lambda *a,**k:b'changed')
    elif damage=='qualification_pin': write(Path(plan['qualification']['path']),b'changed')
    elif damage=='manifest': write(study/'not-decoded.npz',b'changed')
    else: write(study/'runtime.json',{'python':'different'})
    write(eng/'run-launch-01.json',launch); write(eng/'run-process-01.json',process)
    with pytest.raises(ValueError): audit.original_admission(study,eng/'run-process-01.json')


def diagnostic_fixture(arm=audit.CANDIDATE):
    initial=joint_fixture(arm)[0]; shapes={k:v.shape for k,v in initial.items()}
    raw={k:np.zeros_like(v) for k,v in initial.items()}
    raw['cell.weight'][0]=3.;raw['cell.weight'][1]=4.;raw['native_norm']=np.array(5.,np.float32)
    depth={audit.CANDIDATE:30,audit.SHORT:1,audit.LAST:0,audit.TEMPORAL:0}[arm]
    prefix={'prefix_steps':depth,'max_state_abs':2.,'max_state_norm64':3.,
            'max_innovation_abs':0. if depth==0 else 4.,'max_innovation_norm64':0. if depth==0 else 5.}
    gradient={'numel':sum(v.size for v in initial.values()),'nonfinite_count':0,'max_abs':4.,'norm64':5.,'all_finite':True}
    records=[{'update':1,'prefix':prefix,'gradient':gradient,'native_norm':{'kind':'finite','value':5.},'status':'PASS','error':None}]
    summary={'attempts':1,'backward_calls':1,'clip_calls':1,'nonfinite_gradient_attempts':0,'native_norm_nonfinite_attempts':0,
             'max_gradient_abs':4.,'max_gradient_norm64':5.,**prefix}
    receipt={'status':'FAILED','completed_updates':1,'error':{'type':'FitFailure','message':'preservation cap'}}
    trace=[{'gradient_norm_before_clip':5.}]
    return records,summary,raw,receipt,trace,shapes,depth


def check_diagnostics(values):
    records,summary,raw,receipt,trace,shapes,depth=values
    audit.validate_diagnostics(records,summary,raw,receipt,trace,shapes,np,depth)


@pytest.mark.parametrize('arm',audit.ARMS)
def test_complete_cell_plus_initializer_diagnostics_and_preservation_failure(arm):
    values=diagnostic_fixture(arm);check_diagnostics(values)
    assert values[0][0]['gradient']['numel']==audit.storage(arm)['parameters']


def test_finite_preclip_gradients_and_native_overflow_remain_distinct():
    values=list(diagnostic_fixture());records,summary,raw,receipt,trace,_,_=values
    receipt.update(completed_updates=0,error={'type':'FitFailure','message':'nonfinite training gradient norm'})
    records[0].update(status='FAILED',error=receipt['error'],native_norm={'kind':'positive_infinity','value':None})
    raw['native_norm']=np.array(np.inf,np.float32);trace.clear();summary['native_norm_nonfinite_attempts']=1
    check_diagnostics(values)
    assert records[0]['gradient']['all_finite'] is True and summary['nonfinite_gradient_attempts']==0


@pytest.mark.parametrize('damage',['missing_cell','gain_only_count','empty_prefix','incomplete_prefix','native_kind','summary_max','raw_shape'])
def test_generic_diagnostic_corruption_rejected(damage):
    values=list(diagnostic_fixture());record=values[0][0]
    if damage=='missing_cell':values[2].pop('cell.weight')
    elif damage=='gain_only_count':record['gradient']['numel']=72
    elif damage=='empty_prefix':record['prefix']={}
    elif damage=='incomplete_prefix':record['prefix']['prefix_steps']=29
    elif damage=='native_kind':record['native_norm']['kind']='repaired'
    elif damage=='summary_max':values[1]['max_gradient_norm64']=4.
    else:values[2]['gain']=np.zeros(72,np.float32)
    with pytest.raises(ValueError):check_diagnostics(values)


def test_failure_before_backward_has_empty_raw_bank_and_no_native_clip():
    error={'type':'FitFailure','message':'single-fit wall cap exceeded'}
    records=[{'update':1,'prefix':{},'gradient':None,'native_norm':None,'status':'FAILED','error':error}]
    summary={'attempts':1,'backward_calls':0,'clip_calls':0,'nonfinite_gradient_attempts':0,'native_norm_nonfinite_attempts':0,
             'max_gradient_abs':None,'max_gradient_norm64':None,'prefix_steps':0,
             'max_state_abs':0.,'max_state_norm64':0.,'max_innovation_abs':0.,'max_innovation_norm64':0.}
    audit.validate_diagnostics(records,summary,{}, {'status':'FAILED','completed_updates':0,'error':error},[],
                               {'cell.weight':(590,),'gain':(12,6)},np,30)


def resources_fixture():
    cfg,_,resources=fixture()
    for row in resources:
        row['timing']={'seconds':[1.+.01*i for i in range(20)],'median_seconds':1.095,'p95_seconds':1.1805,
                      'scope':audit.parent_audit.parent_audit.prior_audit.TIMING_SCOPE}
    fits=[{**r,'effective_status':'PASS'} for r in audit.identities()]
    return cfg,resources,fits


def test_all61_current_resource_costs_and_alias_storage_are_reconciled():
    cfg,resources,fits=resources_fixture()
    audit.validate_resources(resources,cfg['inherited_rates'],fits,np)
    assert len(resources)==61 and resources[-1]['model_key']=='observer_position-8103-lr0'
    assert resources[-1]['parameter_bytes']==4*662
    fits[0]['effective_status']='FAILED'
    resources[0].update(status='UNAVAILABLE',error={'type':'FailedTrainingAttempt','effective_status':'FAILED'},timing=None)
    audit.validate_resources(resources,cfg['inherited_rates'],fits,np)


@pytest.mark.parametrize('damage',['storage','duration','timing_scope','pass_error','failed_fit_timed','missing_slot','alias_key'])
def test_cost_tampering_and_false_success_rejected(damage):
    cfg,resources,fits=resources_fixture()
    if damage=='storage':resources[0]['parameter_bytes']-=4
    elif damage=='duration':resources[0]['timing']['seconds'][10]+=.1
    elif damage=='timing_scope':resources[0]['timing']['scope']='rollout only'
    elif damage=='pass_error':resources[0]['error']={'type':'Ignored'}
    elif damage=='failed_fit_timed':fits[0]['effective_status']='FAILED'
    elif damage=='missing_slot':resources.pop()
    else:resources[-1]['model_key']='observer_position-8103-lr1'
    with pytest.raises(ValueError):audit.validate_resources(resources,cfg['inherited_rates'],fits,np)


def terminal_fixture():
    return {'status':'PASS','registration_sha256':'a'*64,'seconds':3.,'monotonic_seconds':2.,'clock':'native-test',
            'fresh_fits':12,'inherited_models':45,'rows':584,'prediction_attempts':292,'timing_attempts':61,
            'raw_mat_decodes':0,'saved_fit_loads':7,'saved_exposed_loads':4,'inherited_forecast_regenerations':0,
            'all_evaluation_data_exposed':True,'official_test_access':False,'scientific_result':'JOINT_OBSERVER_DEVELOPMENT_FAIL'}


def test_terminal_quality_failure_is_valid_but_successful_execution_counts_are_exact():
    audit.validate_terminal(terminal_fixture(),'a'*64,{'clock':'native-test'},{'elapsed_seconds':4.})


@pytest.mark.parametrize('field',['fresh_fits','inherited_models','rows','prediction_attempts','timing_attempts',
                                  'raw_mat_decodes','saved_fit_loads','saved_exposed_loads','inherited_forecast_regenerations'])
def test_terminal_cannot_misstate_independently_reconstructed_counts(field):
    value=terminal_fixture();value[field]+=1
    with pytest.raises(ValueError):audit.validate_terminal(value,'a'*64,{'clock':'native-test'},{'elapsed_seconds':4.})


@pytest.mark.parametrize('damage',['extra','missing','boolean_count','test_access','seconds_nan','monotonic_overrun','status'])
def test_terminal_strict_schema_and_time_access_guards(damage):
    value=terminal_fixture()
    if damage=='extra':value['unregistered']=1
    elif damage=='missing':value.pop('rows')
    elif damage=='boolean_count':value['raw_mat_decodes']=False
    elif damage=='test_access':value['official_test_access']=True
    elif damage=='seconds_nan':value['seconds']=float('nan')
    elif damage=='monotonic_overrun':value['monotonic_seconds']=5.
    else:value['scientific_result']='OBSERVER_DEVELOPMENT_PASS'
    with pytest.raises(ValueError):audit.validate_terminal(value,'a'*64,{'clock':'native-test'},{'elapsed_seconds':4.})


def test_parent_role_resolution_cannot_shadow_current_ledger_with_ancestor(tmp_path,monkeypatch):
    import publish_robot_position_observer as publisher
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    cfg,_,_=fixture();rates={k:v for k,v in cfg['inherited_rates'].items() if k in audit.parent_audit.PARENT_ARMS}
    inputs={}
    for i in range(11):
        path=tmp_path/'ancestor'/f'record-{i}.npz';write(path,b'opaque inherited recording')
        inputs[f'data/record-{i}.npz']=audit.pin(path)
    common=('normalizers.npz','linear.npz','causal_ridge_1.npz','causal_ridge_1.json','causal_ridge_100.npz',
            'causal_ridge_100.json',*(f'batches-{s}.npz' for s in audit.SEEDS))
    for name in common:
        path=tmp_path/'ancestor'/name;write(path,b'opaque inherited common input');inputs['parent/'+name]=audit.pin(path)
    ancestor=tmp_path/'ancestor/fits.json';write(ancestor,{'wrong':'ancestor ledger'});inputs['parent/fits.json']=audit.pin(ancestor)
    folder=tmp_path/publisher.STUDY;inventory={}
    names=['results.json','fits.json','resources.json','prediction-attempts.json',
           *(r['key']+'/final.npz' for r in audit.inherited_identities(rates))]
    for name in names:
        write(folder/name,('immediate parent '+name).encode());inventory[name]=audit.descriptor(folder/name)
    prior={'plan':{'inputs':inputs,'config':{'inherited_rates':rates}},'inventory':inventory,
           'audit':{'results':{'selection':{'selected_rates':{'observer_position':.003}}}}}
    def forbidden(*a,**k):raise AssertionError('numeric decode during parent role admission')
    monkeypatch.setattr(np,'load',forbidden)
    actual,selected=audit.parent_inputs(prior)
    assert len(actual)==69 and len(names)==49
    assert actual['parent/fits.json']==audit.pin(folder/'fits.json')!=inputs['parent/fits.json']
    assert actual['parent/normalizers.npz']==inputs['parent/normalizers.npz']
    assert selected['observer_position']==.003 and selected['observer_position_fixed'] is None
