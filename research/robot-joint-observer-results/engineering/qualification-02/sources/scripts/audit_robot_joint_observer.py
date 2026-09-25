"""Independent joint-observer saved-output audit; no fitting or parent forecast replay."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import audit_robot_position_observer as parent_audit

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'robot-joint-observer-audit-v1'
STUDY_VERSION = 'robot-joint-observer-study-v1'
ARMS = ('joint_observer_long','joint_observer_short','continued_last_two','continued_temporal')
CANDIDATE, SHORT, LAST, TEMPORAL = ARMS
ALIAS = 'frozen_position_lr001'
SEEDS, REFS, DEV, EXPOSED, THREADS = parent_audit.SEEDS, parent_audit.REFS, parent_audit.DEV, parent_audit.EXPOSED, parent_audit.THREADS
ELIGIBLE = (*ARMS,*parent_audit.ELIGIBLE,ALIAS)
DISPLAY = (*ARMS,*parent_audit.DISPLAY,ALIAS)
CONDITIONS = parent_audit.CONDITIONS
SOURCES = (*parent_audit.SOURCES,'scripts/publish_robot_position_observer.py',
           'tests/test_publish_robot_position_observer.py','src/openjev/research/robot_joint_observer.py',
           'tests/test_robot_joint_observer.py','scripts/robot_joint_observer_study.py',
           'tests/test_robot_joint_observer_study.py','research/robot-joint-observer-protocol.md',
           'scripts/audit_robot_joint_observer.py','tests/test_audit_robot_joint_observer.py')
LAUNCHER = 'scripts/launch_robot_joint_observer_study.py'
QUALIFICATION_SOURCES = (*SOURCES[-7:],LAUNCHER)
STUDY = 'output/robot-joint-observer-study-v1'
ENGINEERING = 'output/robot-joint-observer-engineering-v1'
REGISTRATION = 'research/robot-joint-observer-registration.json'
COMMAND = ['.venv/bin/python','-u','scripts/robot_joint_observer_study.py','--registration',REGISTRATION,'--output',STUDY]
NUMERIC_ERRORS = parent_audit.NUMERIC_ERRORS
require,read,descriptor,pin = parent_audit.require,parent_audit.read,parent_audit.descriptor,parent_audit.pin
checked_pin,close = parent_audit.checked_pin,parent_audit.close
scored,metric_rows,windows,state_hash = parent_audit.scored,parent_audit.metric_rows,parent_audit.windows,parent_audit.state_hash


def identities():
    return [{'key':f'{arm}-{seed}-lr0','arm':arm,'seed':seed,'learning_rate':.001,'origin':'fresh'}
            for arm in ARMS for seed in SEEDS]


def inherited_identities(rates):
    old_rates = {a:rates[a] for a in parent_audit.PARENT_ARMS}
    return [*parent_audit.identities(),*[r for r in parent_audit.inherited_identities(old_rates) if r['arm'] not in REFS]]


def storage(arm):
    if arm == ALIAS:
        return parent_audit.storage(parent_audit.CANDIDATE)
    if arm not in ARMS:
        return parent_audit.storage(arm)
    count = {CANDIDATE:662,SHORT:662,LAST:590,TEMPORAL:962}[arm]
    return {'parameters':count,'parameter_bytes':4*count,'buffer_bytes':0,'state_scalars':12,'state_bytes':48,
            'normalizer_bytes':192,'inactive_parameters':0,'dtype':'float32','input_bytes':9216,'output_bytes':6144,
            'temporary_workspace':'not measured'}


def original_admission(study, run_receipt):
    """Fixed source, launch, qualification and opaque output joins before arrays."""
    study, run_receipt = Path(study).resolve(), Path(run_receipt).resolve()
    engineering, registration = ROOT/ENGINEERING, ROOT/REGISTRATION
    require(study == ROOT/STUDY and run_receipt == engineering/'run-process-01.json', 'fixed original study/process')
    plan, sha = read(registration), descriptor(registration)['sha256']
    require(plan['version'] == STUDY_VERSION and set(plan['sources']) == set(SOURCES), 'exact63 frozen sources')
    for name, expected in plan['sources'].items():
        require(descriptor(ROOT/name) == expected == descriptor(study/'sources'/name), 'source/snapshot: '+name)
    require(descriptor(study/'registration.json') == descriptor(registration)
            and descriptor(ROOT/LAUNCHER) == plan['launcher'], 'registration/launcher identity')
    require(all(os.environ.get(k) == '1' for k in THREADS), 'single-thread environment')
    process, launch = read(run_receipt), read(engineering/'run-launch-01.json')
    keys = {'command', 'prefit_commit', 'started_utc', 'registration_sha256', 'launcher', 'thread_env', 'scope'}
    require(set(launch) == keys and set(process) == keys | {'returncode', 'elapsed_seconds', 'external_timeout', 'log'}
            and all(process[k] == v for k, v in launch.items()), 'original launch/terminal schema and join')
    require(process['command'] == COMMAND and process['registration_sha256'] == sha and process['launcher'] == plan['launcher']
            and process['thread_env'] == dict.fromkeys(THREADS, '1') and type(process['returncode']) is int
            and process['returncode'] == 0 and process['external_timeout'] is False
            and type(process['elapsed_seconds']) in (int, float) and math.isfinite(process['elapsed_seconds'])
            and 0 < process['elapsed_seconds'] <= 7260 and descriptor(run_receipt.with_suffix('.log')) == process['log'],
            'successful bounded original process')
    commit = process['prefit_commit']
    require(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'prefit commit identity')
    for name in (*SOURCES, LAUNCHER, REGISTRATION):
        require(subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) == (ROOT/name).read_bytes(), 'prefit committed bytes: '+name)
    q = read(checked_pin(plan['qualification'], 'qualification')['path'])
    commands = [['.venv/bin/ruff', 'check', *[p for p in QUALIFICATION_SOURCES if p.endswith('.py')]],
                ['.venv/bin/python', '-m', 'pytest', '--noconftest', '-q', 'tests/test_robot_joint_observer.py',
                 'tests/test_robot_joint_observer_study.py', 'tests/test_audit_robot_joint_observer.py']]
    require(q['status'] == 'PASS' and q['sources_unchanged'] is True and q['sources'] == plan['sources']
            and q['launcher'] == plan['launcher'] and q['thread_env'] == dict.fromkeys(THREADS, '1')
            and [r['command'] for r in q['commands']] == commands, 'exact original qualification')
    require(datetime.fromisoformat(q['created_utc']) <= datetime.fromisoformat(plan['created_utc'])
            <= datetime.fromisoformat(launch['started_utc']), 'qualification and registration precede launch')
    for row in q['commands']:
        require(type(row['returncode']) is int and row['returncode'] == 0
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and 0 < row['seconds'] <= 180
                and descriptor(row['log'])['sha256'] == row['sha256'], 'original qualification command/log')
    manifest, paths = read(study/'manifest.json'), list(study.rglob('*'))
    require(set(manifest) == {'files'} and not any(p.is_symlink() for p in paths), 'regular complete output inventory')
    require({str(p.relative_to(study)) for p in paths if p.is_file()} == set(manifest['files']) | {'manifest.json', 'receipt.json'},
            'no omitted output files')
    for name, expected in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and str(Path(name)) == name
                and descriptor(study/name) == expected, 'opaque output pin: '+name)
    runtime = read(study/'runtime.json')
    require(runtime['python'] == sys.version and runtime['platform'] == platform.platform()
            and runtime['machine'] == platform.machine() and runtime['torch_threads'] == 1
            and runtime['thread_env'] == dict.fromkeys(THREADS, '1'), 'same qualified runtime/platform')
    require(runtime['clock'] == ('mach_continuous_time' if sys.platform == 'darwin' else 'CLOCK_BOOTTIME'), 'native suspend clock')
    for package in ('numpy', 'torch'):
        require(importlib.metadata.version(package) == runtime[package], 'same numeric package: '+package)
    inputs = {key: pin(path) for key, path in {'manifest': study/'manifest.json', 'producer_receipt': study/'receipt.json',
              'registration': registration, 'run_receipt': run_receipt, 'run_log': run_receipt.with_suffix('.log'),
              'run_launch': engineering/'run-launch-01.json', 'runtime': study/'runtime.json', 'launcher': ROOT/LAUNCHER}.items()}
    inputs.update(qualification=plan['qualification'], qualification_logs=[pin(r['log']) for r in q['commands']])
    return plan, inputs



def parent_inputs(prior):
    """Resolve immediate-parent ledgers explicitly; ancestor keys cannot shadow them."""
    import publish_robot_position_observer as publisher
    folder,inventory = ROOT/publisher.STUDY,prior['inventory']
    common = ('normalizers.npz','linear.npz','causal_ridge_1.npz','causal_ridge_1.json',
              'causal_ridge_100.npz','causal_ridge_100.json',*(f'batches-{s}.npz' for s in SEEDS))
    expected = {k:v for k,v in prior['plan']['inputs'].items() if k.startswith('data/')}
    for name in common:
        expected['parent/'+name] = prior['plan']['inputs']['parent/'+name]
    selected = prior['audit']['results']['selection']['selected_rates']
    require(selected == {parent_audit.CANDIDATE:.003},'frozen selected parent position rate')
    rates = dict(prior['plan']['config']['inherited_rates'])
    current = ['results.json','fits.json','resources.json','prediction-attempts.json']
    current += [r['key']+'/final.npz' for r in inherited_identities(rates)]
    for name in current:
        item = pin(folder/name)
        require({k:item[k] for k in ('sha256','bytes')} == inventory[name],'immediate parent payload: '+name)
        expected['parent/'+name] = item
    require(len(expected) == 69 and len(current) == 49,'11data9common4JSON45states')
    return expected,{**rates,parent_audit.CANDIDATE:.003,parent_audit.FIXED:None}


def gradient_summary(gradients,np):
    require(isinstance(gradients,dict) and bool(gradients) and all(v.dtype == np.float32 for v in gradients.values()),
            'all original float32 trainable gradients')
    values = list(gradients.values())
    count = sum(v.size for v in values)
    bad = sum(int(np.count_nonzero(~np.isfinite(v))) for v in values)
    maximum,norm = None,None
    if not bad:
        maximum = max(float(np.max(np.abs(v))) for v in values)
        norm = 0. if maximum == 0 else maximum*math.sqrt(math.fsum(float(np.sum((v.astype(np.float64)/maximum)**2)) for v in values))
    return {'numel':count,'nonfinite_count':bad,'max_abs':maximum,'norm64':norm,'all_finite':bad == 0}


def validate_joint_evidence(initial,final,optimizer,constructed,backbone,arm,receipt,np,updates=4096):
    require(arm in ARMS and set(initial) == set(final) == set(constructed),'exact joint checkpoint keys')
    require(set(backbone) == {k for k in constructed if k.startswith('cell.')} and sum(v.size for v in backbone.values()) == 590,
            'same590 common initial transition')
    for name,value in constructed.items():
        require(value.dtype == initial[name].dtype == final[name].dtype == np.float32
                and value.shape == initial[name].shape == final[name].shape and np.isfinite(initial[name]).all()
                and initial[name].tobytes() == value.tobytes(),'exact warm-start/initializer state: '+name)
        if name in backbone:
            require(initial[name].tobytes() == backbone[name].tobytes(),'paired inherited cell at initialization')
    shapes = {k:v.shape for k,v in initial.items()}
    require(sum(math.prod(s) for s in shapes.values()) == storage(arm)['parameters'],'all declared trainable parameters')
    permitted = {k+'/'+suffix for k in shapes for suffix in ('step','exp_avg','exp_avg_sq')}
    require(set(optimizer) in (set(),permitted),'complete all-parameter Adam ownership')
    steps = receipt['completed_updates']
    require(type(steps) is int and 0<=steps<=updates and receipt['status'] in ('PASS','FAILED')
            and (not steps or optimizer),'completed updates retain Adam')
    for name,shape in shapes.items():
        if not optimizer: continue
        for suffix in ('exp_avg','exp_avg_sq'):
            value = optimizer[name+'/'+suffix]
            require(value.dtype == np.float32 and value.shape == shape,'Adam moment shape and dtype')
        step = optimizer[name+'/step']
        require(step.dtype == np.float32 and step.shape == () and np.isfinite(step)
                and float(step) in (steps,steps+1),'Adam attempted-step ledger')
    if receipt['status'] == 'PASS':
        require(steps == updates and set(optimizer) == permitted
                and all(np.isfinite(v).all() for v in (*final.values(),*optimizer.values()))
                and all(float(optimizer[n+'/step']) == updates for n in shapes),'finite complete joint fit')


def validate_diagnostics(records,summary,raw,receipt,trace,shapes,np,prefix_steps):
    steps = receipt['completed_updates']; count = sum(math.prod(s) for s in shapes.values())
    require(isinstance(records,list) and 1 <= len(records) <= 4096 and len(records) in (steps,steps+1),'one diagnostic per attempted joint update')
    for i,row in enumerate(records,1):
        require(set(row) == {'update','prefix','gradient','native_norm','status','error'} and row['update'] == i
                and row['status'] in ('PASS','FAILED'),'ordered joint diagnostic attempts')
        prefix = row['prefix']
        parent_audit.validate_prefix(prefix)
        require(not prefix or prefix['prefix_steps'] <= prefix_steps,'declared prefix depth')
        gradient,native = row['gradient'],row['native_norm']
        if gradient is not None:
            require(set(gradient)=={'numel','nonfinite_count','max_abs','norm64','all_finite'}
                    and type(gradient['numel']) is int and gradient['numel']==count
                    and type(gradient['nonfinite_count']) is int and 0<=gradient['nonfinite_count']<=count
                    and type(gradient['all_finite']) is bool
                    and gradient['all_finite']==(gradient['nonfinite_count']==0),'complete trainable gradient summary')
            if gradient['all_finite']:
                require(all(type(gradient[k]) in (int,float) and math.isfinite(gradient[k]) and gradient[k]>=0
                            for k in ('max_abs','norm64')),'finite float64 trainable gradient magnitudes')
            else:
                require(gradient['max_abs'] is None and gradient['norm64'] is None,'no substitute for nonfinite entries')
            require(isinstance(native,dict) and set(native)=={'kind','value'}
                    and native['kind'] in ('finite','nan','positive_infinity','negative_infinity'),'native norm classification')
            require((type(native['value']) in (int,float) and math.isfinite(native['value']) and native['value']>=0)
                    if native['kind']=='finite' else native['value'] is None,'original JSON-safe native norm')
            require(bool(prefix) and prefix['prefix_steps']==prefix_steps,'completed prescribed prefix before backward')
        else:
            require(native is None,'no norm without original backward evidence')
        if row['status'] == 'FAILED':
            require(i == len(records) and receipt['status'] == 'FAILED' and row['error'] == receipt['error'],'last failed attempt retained')
        else:
            require(row['error'] is None and gradient is not None and gradient['all_finite'] and native['kind'] == 'finite',
                    'finite successful original attempt')
        if i<=steps:
            require(gradient is not None and gradient['all_finite'] and native['kind'] == 'finite','completed update had finite gradient norm')
            close(native['value'],trace[i-1]['gradient_norm_before_clip'],'trace/native norm join')
    if receipt['status']=='PASS':
        require(len(records)==steps==4096 and all(r['status']=='PASS' for r in records),'full4096 original joint updates')
    gradient,native = records[-1]['gradient'],records[-1]['native_norm']
    require(set(raw) == (set() if gradient is None else set(shapes)|{'native_norm'}),'raw all-parameter gradient roster')
    if gradient is not None:
        values = {n:raw[n] for n in shapes}
        require(all(v.dtype==np.float32 and v.shape==shapes[n] for n,v in values.items()),'raw gradient shapes/dtypes')
        close(gradient,gradient_summary(values,np),'independent full preclip gradient summary')
        require(native == parent_audit.native_norm_summary(raw['native_norm'],np),'raw original native norm')
    close(summary,parent_audit.aggregate_diagnostics(records),'twelve complete joint diagnostic aggregate fields')


def alias_identities():
    return [{'key':f'{ALIAS}-{seed}-lr0','arm':ALIAS,'seed':seed,'learning_rate':.001,'origin':'inherited_alias',
             'model_key':f'{parent_audit.CANDIDATE}-{seed}-lr0'} for seed in SEEDS]


def resource_identities(rates):
    old_rates = {a:rates[a] for a in parent_audit.PARENT_ARMS}
    return [*identities(),*parent_audit.resource_identities({'selected_rates':{parent_audit.CANDIDATE:.003}},old_rates),*alias_identities()]


def decisions(rows,resources,cfg):
    rates = cfg['inherited_rates']
    require(set(rates) == set(parent_audit.PARENT_ARMS)|{parent_audit.CANDIDATE,parent_audit.FIXED}
            and rates[parent_audit.CANDIDATE] == .003 and rates[parent_audit.FIXED] is None,'complete frozen inherited rates')
    roster = [*parent_audit.parent_audit.identities(),*parent_audit.identities(),*identities()]
    lookup = {(r['recording'],r['fit_key'],r['horizon']):r for r in rows}
    expected = {(name,r['key'],h) for name in EXPOSED for r in roster for h in (64,128)}
    require(len(rows) == len(lookup) == 584 and set(lookup) == expected,'all584 original metric rows, no duplicated alias')
    recipes = {r['key']:r for r in roster}
    for row in rows:
        require(all(row[k] == recipes[row['fit_key']][k] for k in ('arm','seed','learning_rate')),'score/checkpoint identity')
        require(row['status'] in ('PASS','FAILED'),'declared metric status')
        if row['status']=='FAILED':
            require(row['metrics'] is None and row['error'] is not None,'failed metric retained')
        else:
            m=row['metrics']; h=row['horizon']
            require(row['error'] is None and isinstance(m,dict) and m['windows']==22 and m['horizon']==h
                    and type(m['scalars']) is int and m['scalars']==22*h*6 and len(m['per_joint_rmse_deg'])==6,
                    'complete finite forecast geometry')
            require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in
                        [m['standardized_rmse'],m['standardized_sse'],m['physical_rmse_deg'],*m['per_joint_rmse_deg']]),'finite metrics')
    selection = {'fixed_rates':dict.fromkeys(ARMS,.001),'inherited_rates':rates,
                 'alias':{'arm':ALIAS,'source_arm':parent_audit.CANDIDATE,'learning_rate':.001},'scope':cfg['selection_scope']}
    selected = {}
    for arm in ELIGIBLE:
        source = parent_audit.CANDIDATE if arm==ALIAS else arm
        rate = .001 if arm in (*ARMS,ALIAS) else None if arm in REFS else rates[arm]
        selected[arm] = [r for r in roster if r['arm']==source and r['learning_rate']==rate]
        require(len(selected[arm]) == (1 if arm in REFS else 3),'one complete fixed recipe per family')
    details,means = {},{}
    complete = True
    for name in EXPOSED:
        values = {}
        for arm,recipe in selected.items():
            group = [lookup[name,r['key'],128] for r in recipe]
            complete &= all(lookup[name,r['key'],h]['status']=='PASS' for r in recipe for h in (64,128))
            if all(r['status']=='PASS' for r in group):
                values[arm]=sum(r['metrics']['standardized_rmse'] for r in group)/len(group)
        details[name]={'means':values}
    for arm in ELIGIBLE:
        if all(arm in details[n]['means'] for n in EXPOSED):
            means[arm]=sum(details[n]['means'][arm] for n in EXPOSED)/4
    require(len(resources)==61,'all61 current costs')
    for row,identity in zip(resources,resource_identities(rates),strict=True):
        require(all(row[k]==v for k,v in identity.items()),'ordered current cost identity and alias mapping')
        require(row['status'] in ('PASS','FAILED','UNAVAILABLE'),'declared cost status')
        require((row['error'] is None and isinstance(row['timing'],dict)) if row['status']=='PASS'
                else row['error'] is not None and row['timing'] is None,'cost status/evidence join')
    costs={}
    for arm in ELIGIBLE:
        subset=[r for r in resources if r['arm']==arm]
        require(len(subset)==(1 if arm in REFS else 3),'every fixed cost recipe')
        costs[arm]=None
        if any(r['status']!='PASS' for r in subset): continue
        latencies=[r['timing']['median_seconds'] for r in subset]
        sizes=[[r[k] for k in ('parameter_bytes','buffer_bytes','state_bytes','normalizer_bytes')] for r in subset]
        require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in latencies)
                and all(type(v) is int and v>=0 for parts in sizes for v in parts),'valid request costs')
        costs[arm]={'latency':sorted(latencies)[len(latencies)//2],'bytes':max(sum(p) for p in sizes)}
    frontier=all(a in means and costs[a] is not None for a in ELIGIBLE)
    controls=[a for a in ELIGIBLE if a!=CANDIDATE]
    best=min(means[a] for a in controls) if all(a in means for a in controls) else None
    gain=CANDIDATE in means and best is not None and best>0 and means[CANDIDATE]<=.95*best
    no_harm=all(all(a in details[n]['means'] for a in ARMS)
                and details[n]['means'][CANDIDATE]<=1.02*min(details[n]['means'][a] for a in ARMS[1:]) for n in EXPOSED)
    left,right=costs[CANDIDATE],costs[LAST]
    latency=left is not None and right is not None and left['latency']<=1.5*right['latency']
    dominators=[]
    if frontier:
        candidate=(means[CANDIDATE],left['latency'],left['bytes'])
        for arm in controls:
            point=(means[arm],costs[arm]['latency'],costs[arm]['bytes'])
            if all(x<=y for x,y in zip(point,candidate,strict=True)) and any(x<y for x,y in zip(point,candidate,strict=True)):
                dominators.append(arm)
    conditions=[{'name':name,'passed':bool(flag)} for name,flag in zip(CONDITIONS,
                (complete and frontier,gain,no_harm,latency,frontier and not dominators),strict=True)]
    passed=sum(r['passed'] for r in conditions)
    return {'selection':selection,'result':{'status':'JOINT_OBSERVER_DEVELOPMENT_PASS' if passed==5 else 'JOINT_OBSERVER_DEVELOPMENT_FAIL',
            'passed':passed,'total':5,'conditions':conditions,'details':details,'equal_file_means':means,'costs':costs,
            'best_control_mean':best,'frontier_complete':frontier,'dominators':dominators,
            'scope':'All four files exposed development; no confirmation, novelty or control claim'}}


def expected_config(parent,rates):
    cfg=dict(parent)
    for key in ('learned','fixed','diagnostic_probes','new_fixed_models'): cfg.pop(key,None)
    cfg.update(version=STUDY_VERSION,arms=list(ARMS),learning_rates=[.001],families=list(DISPLAY),eligible_families=list(ELIGIBLE),
               inherited_rates=rates,fixed_rates=dict.fromkeys(ARMS,.001),alias=ALIAS,alias_source_arm=parent_audit.CANDIDATE,
               alias_learning_rate=.001,new_fits=12,inherited_models=45,final_models=57,inherited_rows=488,
               inherited_prediction_attempts=244,input_count=69,wall_cap_seconds=7200.,
               selection_scope='No new rate selection; four fresh rates fixed .001 and inherited recipes frozen',
               gradient_clipping='unchanged native float32 norm; float64 diagnostic only',
               optimizer_initialization='fresh empty Adam for every fresh fit; no inherited optimizer')
    return cfg


def authenticate(study,run_receipt):
    plan,inputs=original_admission(study,run_receipt); study=Path(study).resolve()
    import publish_robot_position_observer as publisher
    prior=publisher.authenticate(ROOT/publisher.STUDY,ROOT/publisher.AUDIT,ROOT/publisher.ENGINEERING)
    require(prior['audit']['results']['result']['status']=='POSITION_OBSERVER_DEVELOPMENT_FAIL'
            and prior['audit']['results']['result']['passed']==3,'preserve closed parent quality failure')
    expected,rates=parent_inputs(prior)
    require(plan['inputs']==expected and plan['config']==expected_config(prior['plan']['config'],rates),'exact69 roles and joint config')
    for name,item in expected.items():
        checked_pin(item,name)
        if not name.startswith('data/'):
            require(descriptor(study/name)=={k:item[k] for k in ('sha256','bytes')},'unchanged inherited copy: '+name)
    require(plan['parent_publication_manifest']==pin(ROOT/publisher.OUTPUT/'manifest.json')
            and plan['parent_audit']==pin(ROOT/publisher.AUDIT),'closed parent audit/publication joins')
    receipt,runtime,process=read(study/'receipt.json'),read(study/'runtime.json'),read(run_receipt)
    validate_terminal(receipt,inputs['registration']['sha256'],runtime,process)
    inputs.update(parent_admission=prior['inputs'],parent_publication_inputs=prior['publication_inputs'],parent_inputs=plan['inputs'],
                  parent_audit=plan['parent_audit'],parent_publication_manifest=plan['parent_publication_manifest'])
    return plan,inputs


def validate_terminal(receipt,registration_sha256,runtime,process):
    counts={'fresh_fits':12,'inherited_models':45,'rows':584,'prediction_attempts':292,'timing_attempts':61,
            'raw_mat_decodes':0,'saved_fit_loads':7,'saved_exposed_loads':4,'inherited_forecast_regenerations':0}
    require(set(receipt)==set(counts)|{'status','registration_sha256','seconds','monotonic_seconds','clock',
            'all_evaluation_data_exposed','official_test_access','scientific_result'},'exact original terminal receipt schema')
    require(all(type(receipt[k]) is int and receipt[k]==value for k,value in counts.items()),'original terminal count claims')
    require(receipt['status']=='PASS' and receipt['registration_sha256']==registration_sha256
            and receipt['all_evaluation_data_exposed'] is True and receipt['official_test_access'] is False
            and receipt['clock']==runtime['clock']
            and type(receipt['seconds']) in (int,float) and math.isfinite(receipt['seconds']) and 0<receipt['seconds']<=7200
            and type(receipt['monotonic_seconds']) in (int,float) and math.isfinite(receipt['monotonic_seconds'])
            and 0<receipt['monotonic_seconds']<=process['elapsed_seconds']
            and receipt['scientific_result'] in ('JOINT_OBSERVER_DEVELOPMENT_PASS','JOINT_OBSERVER_DEVELOPMENT_FAIL'),
            'bounded complete child terminal')


def validate_resources(resources,rates,fits,np):
    require(len(resources)==61,'all61 cost slots')
    ledger={r['key']:r for r in fits}
    for row,identity in zip(resources,resource_identities(rates),strict=True):
        spec=storage(identity['arm'])
        require(set(row)==set(identity)|set(spec)|{'status','error','timing'} and all(row[k]==v for k,v in identity.items()),
                'exact cost schema/alias identity')
        close({k:row[k] for k in spec},spec,'all persistent/request storage')
        failed=identity['arm'] in ARMS and ledger[identity['key']]['effective_status']!='PASS'
        if failed:
            require(row['status']=='UNAVAILABLE' and row['timing'] is None and row['error']==
                    {'type':'FailedTrainingAttempt','effective_status':ledger[identity['key']]['effective_status']},
                    'failed fresh fit cannot acquire successful timing')
        elif row['status']=='FAILED':
            require(row['timing'] is None and row['error'] in [{'type':'NonfiniteTiming','message':m} for m in
                    (*NUMERIC_ERRORS,'finite complete timed request','nonfinite ridge prediction')],'known retained numerical timing failure')
        else:
            require(row['status']=='PASS' and row['error'] is None,'successful current cost')
            timing=row['timing']
            require(set(timing)=={'seconds','median_seconds','p95_seconds','scope'}
                    and timing['scope']==parent_audit.parent_audit.prior_audit.TIMING_SCOPE
                    and len(timing['seconds'])==20 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in timing['seconds']),
                    'twenty original complete request durations')
            close(timing['median_seconds'],float(np.median(timing['seconds'])),'original timing median')
            close(timing['p95_seconds'],float(np.percentile(timing['seconds'],95)),'original timing p95')


def audit(study,run_receipt):
    started=time.monotonic(); study=Path(study).resolve()
    plan,inputs=authenticate(study,run_receipt)
    import numpy as np
    import robot_joint_observer_study as replay
    import torch
    torch.set_num_threads(1)
    cfg,roster=plan['config'],identities()
    counts={'npz_decodes':0,'array_decodes':0,'qualified_model_replays':0,'parent_forecast_replays':0,
            'raw_mat_decodes':0,'official_test_decodes':0,'optimizer_updates':0,'backward_replays':0,'native_clip_replays':0,'timing_replays':0}
    expected={'registration.json','runtime.json','fits.json','initialization-checks.json','checkpoint-barrier.json',
              'parameter-checks.json','prediction-attempts.json','resources.json','results.json'}
    expected|={'sources/'+name for name in SOURCES}
    expected|={name for name in plan['inputs'] if not name.startswith('data/')}

    def load(name,external=False):
        if external:
            require(name in plan['inputs'] and name.startswith('data/'),'only11 inherited saved recordings')
            path=checked_pin(plan['inputs'][name],name)['path']
        else:
            expected.add(name);path=study/name
        with np.load(path,allow_pickle=False) as bank:
            values={name:bank[name].copy(order='K') for name in bank.files}
        counts['npz_decodes']+=1;counts['array_decodes']+=len(values)
        return values

    def equal(actual,wanted,label):
        require(actual.dtype==wanted.dtype and actual.shape==wanted.shape and np.array_equal(actual,wanted,equal_nan=True),label)

    norm=load('parent/normalizers.npz')
    require(set(norm)=={'q_mean','q_std','u_mean','u_std'} and all(v.dtype==np.float64 and v.shape==(6,) and np.isfinite(v).all()
            for v in norm.values()) and all((norm[k]>0).all() for k in ('q_std','u_std')),'finite original normalization')
    records=[load('data/'+name,external=True) for name in cfg['partitions']['fit']]
    require(len(records)==7,'seven complete FIT recordings')
    for record in records:windows(record,norm,np)
    for field in ('q','u'):
        values=np.concatenate([r[field][64:] for r in records])
        equal(norm[field+'_mean'],values.mean(0),'original FIT mean');equal(norm[field+'_std'],values.std(0),'original FIT scale')
    for seed in SEEDS:
        bank=load(f'parent/batches-{seed}.npz');require(set(bank)=={'record','start'},'original batch schema')
        rng=np.random.Generator(np.random.PCG64(seed+520000));choice=rng.integers(0,7,size=(4096,16),dtype=np.int64)
        starts=np.empty_like(choice)
        for index in np.ndindex(choice.shape):starts[index]=rng.integers(64,3636-32-128+1)
        equal(bank['record'],choice,'same paired record draws');equal(bank['start'],starts,'same paired window draws')
    linear_bank=load('parent/linear.npz');require(set(linear_bank)=={'coefficient'},'linear bank schema')
    linear=linear_bank['coefficient']
    require(linear.dtype==np.float64 and linear.shape==(6,25) and np.isfinite(linear).all(),'finite inherited linear coefficients')
    # Ridge banks stay opaque: no parent forecasts or reference fitting/replay.
    backbones={seed:load(f'parent/last_two-{seed}-fixed/final.npz') for seed in SEEDS}
    parent_rows=read(study/'parent/results.json')['rows'];parent_attempts=read(study/'parent/prediction-attempts.json')
    inherited=inherited_identities(cfg['inherited_rates']);parent_fits=read(study/'parent/fits.json')
    require(len(parent_rows)==488 and len(parent_attempts)==244 and len(parent_fits)==45
            and {r['key'] for r in parent_fits}=={r['key'] for r in inherited},'complete immediate-parent evidence')
    parent_ledger={r['key']:r for r in parent_fits};fits=read(study/'fits.json')
    require(len(fits)==57,'all12 fresh and45 inherited model records')
    states,models,initialization,passed={},{},{},set()
    for number,(fit,identity) in enumerate(zip(fits,[*roster,*inherited],strict=True),1):
        key,arm,seed=identity['key'],identity['arm'],identity['seed']
        require(all(fit[k]==v for k,v in identity.items()),'ordered fresh/inherited model identity')
        final=load(key+'/final.npz')
        if arm in ARMS:
            model=replay.model_for(arm,seed,linear);template=model.state_dict();backbone=backbones[seed]
            require(set(final)==set(template) and all(final[n].shape==tuple(v.shape) and final[n].dtype==np.float32
                    for n,v in template.items()),'qualified final state shape/keys/dtype')
            require(set(backbone)=={'cell.'+n for n in model.cell.state_dict()},'same complete590 initial cell')
            model.cell.load_state_dict({n:torch.from_numpy(backbone['cell.'+n]) for n in model.cell.state_dict()},strict=True)
            constructed={n:v.detach().numpy().copy(order='K') for n,v in model.state_dict().items()}
            require(all(p.requires_grad and p.grad is None for p in model.parameters()),'qualified initially trainable parameter roster')
            initialization[key]={'cell_sha256':state_hash({n.removeprefix('cell.'):v for n,v in backbone.items()}),
                                 'parameter_sha256':state_hash(constructed),'optimizer_state_entries':0,
                                 'scope':'Common parent cell weights only; fresh empty Adam; no inherited optimizer'}
            require(fit['initialization']==initialization[key],'exact common initial weight/empty optimizer receipt')
            expected|={f'completed-fit-{number:02d}.json',key+'/trace.json',key+'/fit-receipt.json',
                       key+'/diagnostics.json',key+'/diagnostic-summary.json'}
            require(read(study/f'completed-fit-{number:02d}.json')==fit,'every fresh completed attempt')
            initial,optimizer=load(key+'/initial.npz'),load(key+'/optimizer.npz')
            receipt,trace=read(study/key/'fit-receipt.json'),read(study/key/'trace.json')
            require(fit['fit']==receipt,'fit ledger joins original helper receipt')
            parent_audit.validate_fit_receipt(receipt,trace,.001,cfg['fit_cap_seconds'])
            for name,item in receipt['files'].items():require(descriptor(study/key/name)==item,'all seven fit evidence pins')
            validate_joint_evidence(initial,final,optimizer,constructed,backbone,arm,receipt,np)
            validate_diagnostics(read(study/key/'diagnostics.json'),read(study/key/'diagnostic-summary.json'),
                                 load(key+'/last-gradient.npz'),receipt,trace,{n:v.shape for n,v in initial.items()},np,
                                 {CANDIDATE:30,SHORT:1,LAST:0,TEMPORAL:0}[arm])
            elapsed=fit['native_fit_seconds']
            require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0
                    and fit['effective_status']==('FAILED' if elapsed>=1800 else receipt['status'])
                    and fit['native_fit_error']==(None if elapsed<1800 else {'type':'FitFailure','message':'native fit cap includes preservation'}),
                    'native preservation deadline classification')
            require(set(fit)==set(identity)|{'fit','effective_status','native_fit_seconds','native_fit_error','initialization','resources'},
                    'complete fresh fit metadata schema')
            if fit['effective_status']=='PASS':
                require(all(np.isfinite(v).all() for v in final.values()),'finite successful model')
                model.load_state_dict({n:torch.from_numpy(v) for n,v in final.items()},strict=True);model.eval()
                require(all(p.grad is None for p in model.parameters()),'no retained inference gradients')
                models[key]=model;passed.add(key)
        else:
            require(fit==parent_ledger[key] and fit['effective_status']=='PASS','unchanged parent model record')
            require(descriptor(study/key/'final.npz')==descriptor(study/'parent'/key/'final.npz'),'opaque identical inherited final bytes')
            require(all(v.dtype==np.float32 and np.isfinite(v).all() for v in final.values()),'finite inherited checkpoint')
        close(fit['resources'],storage(arm),'actual model parameter/buffer/state/request bytes')
        states[key]=state_hash(final)
    require(read(study/'initialization-checks.json')==initialization,'all12 initial pairing records')
    for seed in SEEDS:
        require(len({initialization[f'{arm}-{seed}-lr0']['cell_sha256'] for arm in ARMS})==1,'same per-seed cell acrossfour arms')
    barrier={'fresh_fit_attempts':12,'inherited_models':45,'exposed_loads_this_run':0,'all_evaluation_data_exposed':True,
             'state_sha256':states,'checkpoints':{r['key']:descriptor(study/r['key']/'final.npz') for r in fits}}
    require(read(study/'checkpoint-barrier.json')==barrier,'all57 states closed before exposed decoder')
    require(read(study/'parameter-checks.json')=={'before':states,'after':states,'unchanged':True},'all inference state immutable')
    rows,attempts=parent_rows.copy(),parent_attempts.copy();ledger={r['key']:r for r in fits}
    for name in EXPOSED:
        starts,batch,target=windows(load('data/'+name,external=True),norm,np)
        bank=load('dev-windows-'+name+'.npz');require(set(bank)=={'starts','target'},'target window bank')
        equal(bank['starts'],starts,'all22 exact start indices');equal(bank['target'],target,'independent causal target windows')
        for identity in roster:
            key,arm=identity['key'],identity['arm']
            common={'recording':name,'arm':arm,'seed':identity['seed'],'learning_rate':.001,'fit_key':key}
            prediction,error=None,None;filename='prediction-'+name+'-'+key+'.npz'
            if key not in passed:
                error={'type':'FailedTrainingAttempt','effective_status':ledger[key]['effective_status']}
            else:
                try:
                    counts['qualified_model_replays']+=1
                    with torch.no_grad():generated=replay.old.infer(models[key],batch).numpy().astype(np.float64)
                except replay.old.FitFailure as exc:
                    require(str(exc) in NUMERIC_ERRORS,'known qualified numerical forecast failure')
                    error={'type':'NonfiniteEvaluation','message':str(exc)}
                else:
                    bank=load(filename);require(set(bank)=={'prediction'},'single saved forecast field')
                    prediction=bank['prediction'];equal(prediction,generated,'exact qualified joint forecast replay')
            if prediction is None:require(not (study/filename).exists(),'no bank for unavailable failed forecast')
            group=metric_rows(common,prediction,target,norm['q_std'],error,np);rows.extend(group)
            attempt={**common,'status':'PASS' if all(r['status']=='PASS' for r in group) else 'FAILED',
                     'errors':[r['error'] for r in group],'prediction_file':filename if prediction is not None else None}
            attempts.append(attempt);completed=f'completed-prediction-{len(attempts)-244:03d}.json';expected.add(completed)
            close(read(study/completed),attempt,'every scheduled new forecast outcome')
    saved=read(study/'results.json');saved_attempts=read(study/'prediction-attempts.json')
    require(saved['rows'][:488]==parent_rows and saved_attempts[:244]==parent_attempts,'exact unchanged parent rows and attempts')
    close(saved_attempts,attempts,'all292 original attempts')
    resources=read(study/'resources.json');decision=decisions(rows,resources,cfg)
    validate_resources(resources,cfg['inherited_rates'],fits,np)
    for i,row in enumerate(resources,1):
        name=f'completed-timing-{i:02d}.json';expected.add(name);require(read(study/name)==row,'all61 current timing receipts')
    results={'version':cfg['version'],'config':cfg,'selection':decision['selection'],'rows':rows,'result':decision['result']}
    close(saved,results,'independent96 new metrics and five fixed-recipe rules')
    require(read(study/'receipt.json')['scientific_result']==decision['result']['status'],'original terminal scientific decision')
    require(set(read(study/'manifest.json')['files'])==expected,'exact full dynamic evidence inventory')
    require(all(state_hash({n:v.detach().numpy() for n,v in model.state_dict().items()})==states[key] for key,model in models.items()),
            'qualified replay did not change weights')
    require(authenticate(study,run_receipt)==(plan,inputs),'all admitted evidence unchanged after replay')
    counts.update(fresh_fits=12,inherited_models=45,final_checkpoints=57,initial_checkpoints=12,optimizer_checkpoints=12,
                  raw_gradient_banks=12,metric_rows=584,inherited_metric_rows=488,new_metric_rows=96,
                  prediction_attempts=292,inherited_prediction_attempts=244,new_prediction_attempts=48,
                  new_prediction_files=sum(r['prediction_file'] is not None for r in attempts[244:]),resource_rows=61,
                  requested_fresh_updates=49152,completed_fresh_updates=sum(r['fit']['completed_updates'] for r in fits[:12]),
                  failed_fresh_fits=sum(r['effective_status']!='PASS' for r in fits[:12]),failed_metric_rows=sum(r['status']!='PASS' for r in rows),
                  manifest_files=len(expected),initial_cell_pairings=12,saved_fit_recordings=7,saved_exposed_recordings=4,condition_rows=5)
    return {'version':VERSION,'status':'PASS','agreement':True,'study':str(study),'registration_sha256':inputs['registration']['sha256'],
            'source_pins':plan['sources'],'auditor':pin(__file__),'inputs':inputs,'results':results,'resources':resources,
            'prediction_attempts':attempts,'fit_diagnostics':{r['key']:read(study/r['key']/'diagnostic-summary.json') for r in fits[:12]},
            'counts':counts,'seconds':time.monotonic()-started,
            'checks':{'original_closed_process':True,'independent_new_metrics_and_rules':True,'parent_rows_retained_exactly':True,
                      'no_new_rate_selection':True,'same590_initial_cell':True,'all_parameter_adam_state':True,
                      'exact_qualified_new_model_replay':True,'raw_preclip_gradient_reconciliation':True,'all_failed_attempts_retained':True},
            'scope':['No raw MAT numerical decoding, new fitting, backward, clipping, optimizer or timing replay.',
                     'Parent488 scores and244 attempts remain unchanged; no parent forecast bank is decoded or replayed.',
                     'Recurrence uses qualified model code; new scoring, fixed-recipe rules and saved-gradient summaries are independent.',
                     'Only the last raw preclip gradients per fit are numerically reconciled; earlier histories/prefix maxima are execution attestations.',
                     'No numerical diagnostic probes. All four recordings are exposed development; historical media are hashed opaquely in parent admission.']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--run-receipt',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists() and not args.output.parent.exists(),'exclusive audit output directory')
    result=audit(args.study,args.run_receipt);args.output.parent.mkdir(parents=True,exist_ok=False)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    with (args.output.parent/'manifest.json').open('x') as handle:
        json.dump({'files':{args.output.name:descriptor(args.output)}},handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps({'status':result['status'],'agreement':result['agreement'],'counts':result['counts'],
                      'scientific_status':result['results']['result']['status']}),flush=True)


if __name__=='__main__':main()
