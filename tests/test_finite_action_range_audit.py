"""Fabricated scalar/receipt fixtures only; no learner or generator execution."""
from __future__ import annotations

import builtins
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import audit_finite_action_range_learning as a
import numpy as np
import pytest


def rule_fixture(cohorts=5):
    rows, baselines, populations = [], [], {}
    for c in range(cohorts):
        populations[c] = {'train':300,'base':100+c,'shift':120+c}
        for regime in ('base','shift'):
            for horizon in (1,2,4,8):
                baselines.append({'cohort_index':c,'regime':regime,'horizon':horizon,
                    'blind_regret':1.,'blind_cost_mse':1.})
                for arm in a.ARMS:
                    rows.append({'cohort_index':c,'seed_namespace':948001+c,'seed':948101+c,
                        'arm':arm,'regime':regime,'horizon':horizon,'cases':populations[c][regime],
                        'blind_regret':.1 if arm == 'rounded_range' else .2,'blind_cost_mse':.1,
                        'observed_kl':.01,'blind_survival_mae':.01})
    return rows,baselines,populations


def test_rule_has_all_54_conditions_and_equal_cohort_means():
    rows,baselines,populations = rule_fixture()
    out = a.evaluate_rule(rows,baselines,populations,rule=json.loads(json.dumps(dict(a.RULE))))
    assert out['advance']['passed']
    assert len(out['advance']['conditions']) == 54
    assert len(out['paired_loss_contrasts']) == 40
    assert len(out['equal_cohort_means']) == 8
    assert len(out['architecture_contrasts']) == 120
    assert all(row['reduction_fraction'] == .5 for row in out['equal_cohort_means'])
    assert out['architecture_contrasts_descriptive_only']


@pytest.mark.parametrize('bad', ['tie','worse','short','observed','survival','support','reference'])
def test_one_cohort_or_one_regime_cannot_be_rescued(bad):
    rows,baselines,populations = rule_fixture()
    selected = next(x for x in rows if x['cohort_index'] == 4 and x['arm'] == 'rounded_range'
                    and x['regime'] == 'shift' and x['horizon'] == (1 if bad == 'short' else 8))
    if bad in ('tie','worse'):
        selected['blind_regret'] = .2 if bad == 'tie' else .200001
    elif bad == 'short':
        selected['blind_cost_mse'] = .51
    elif bad == 'observed':
        selected['observed_kl'] = .100001
    elif bad == 'survival':
        selected['blind_survival_mae'] = .050001
    elif bad == 'support':
        populations[4]['shift'] = 63
    else:
        next(x for x in baselines if x['cohort_index'] == 4 and x['regime'] == 'shift' and x['horizon'] == 8)['blind_regret'] = 0.
    result = a.evaluate_rule(rows,baselines,populations)
    assert not result['advance']['passed']
    assert any(not value for value in result['advance']['conditions'].values())


def test_means_are_equal_cohort_and_free_arm_cannot_rescue():
    rows,baselines,populations = rule_fixture()
    for row in rows:
        if row['arm'].startswith('free_'):
            row['blind_regret'] = 1000.
        elif row['arm'] == 'rounded_range':
            row['blind_regret'] = .181 if row['cohort_index'] < 4 else .17
        row['cases'] = 1000000 if row['cohort_index'] == 4 else 1
    result = a.evaluate_rule(rows,baselines,populations)
    # (.181*4 + .17)/5=.1788. Case-weighting would give nearly .17.
    assert all(x['candidate_regret'] == pytest.approx(.1788, abs=1e-15) for x in result['equal_cohort_means'])
    assert result['advance']['passed']
    for row in rows:
        if row['arm'] == 'rounded_range':
            row['blind_regret'] = .181
    assert not a.evaluate_rule(rows,baselines,populations)['advance']['passed']


@pytest.mark.parametrize('bad', ['nan','missing','duplicate','changed_rule','bool_rule'])
def test_rule_rejects_incomplete_or_changed_evidence(bad):
    rows,baselines,populations = rule_fixture()
    rule = dict(a.RULE)
    if bad == 'nan':
        rows[0]['blind_regret'] = float('nan')
    elif bad == 'missing':
        rows.pop()
    elif bad == 'duplicate':
        rows[-1] = rows[0]
    elif bad == 'changed_rule':
        rule['mean_reduction'] = .09
    else:
        rule['minimum_train'] = True
    with pytest.raises(ValueError):
        a.evaluate_rule(rows,baselines,populations,rule=rule)


def test_smoke_keeps_scientific_support_and_22_conditions():
    rows,baselines,populations = rule_fixture(1)
    populations[0] = {'train':60,'base':30,'shift':31}
    result = a.evaluate_rule(rows,baselines,populations)
    assert len(result['advance']['conditions']) == 22
    assert not result['advance']['passed']
    assert all(not result['gates'][regime]['rounded_range'][criterion]['passed']
               for regime in a.REGIMES for criterion in result['gates'][regime]['rounded_range'])


@pytest.mark.parametrize('kind', ['mse','double','range'])
def test_loss_work_manual_nonempty_and_empty_oracle(kind):
    expected = dict.fromkeys(a.LOSS_WORK_KEYS,0)
    expected['wrapper_calls'] = 1
    if kind != 'mse':
        expected.update(blind_error_entries=24,mse_square_entries=24,mse_mean_calls=1,
            weight_scalar_divisions=2,weight_scalar_multiplications=1,
            loss_adjustment_multiplications=1,loss_adjustment_additions=1)
        if kind == 'range':
            expected.update(range_max_rows=6,range_min_rows=6,range_subtract_entries=6,
                range_square_entries=6,range_scale_entries=6,range_mean_calls=1,replacement_subtractions=1)
    assert a.loss_work(3,2,kind) == expected
    empty = dict.fromkeys(a.LOSS_WORK_KEYS,0)
    empty.update(wrapper_calls=1,zero_endpoint_batches=1)
    assert a.loss_work(0,2,kind) == empty


def test_exact_payload_roster_and_profiles():
    local = a.local_files(948101)
    assert len(local) == 75
    assert sum(name.endswith('.npz') for name in local) == 44
    assert sum('-optimizer-' in name for name in local) == 18
    assert 2+5*len(local) == 377
    assert len(a.SCIENCE) == 5
    assert a.profile_config(a.PROFILES['smoke'])['cohorts'] == [{'seed_namespace':948001,'fit_seed':948101}]
    assert a.PROFILES['smoke'][0].train_attempts == 64
    assert a.EXPOSURE.config['dev_attempts'] == 8


def test_wrong_profile_rejected_before_numerical_import(tmp_path,monkeypatch):
    (tmp_path/'summary.json').write_text(json.dumps({'config':a.profile_config(a.PROFILES['smoke'])}))
    original = builtins.__import__
    def forbid(name,*args,**kwargs):
        if name == 'numpy':
            raise AssertionError('numerical import before profile admission')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',forbid)
    with pytest.raises(ValueError,match='immutable cohort profile'):
        a.audit(tmp_path.resolve())
    with pytest.raises(ValueError,match='immutable audit profile'):
        a.audit(tmp_path.resolve(),profile='anything')


def state_hash(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        header = json.dumps([name,value.dtype.str,list(value.shape)],separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'little'))
        digest.update(header)
        digest.update(value.tobytes())
    return digest.hexdigest()


def adam(kind,steps):
    shapes = [(4,8,8),(4,8),(4,8)] + ([(4,8)] if kind == 'joint' else [])
    group = {'params':list(range(len(shapes))),'lr':.003,'betas':[.9,.999],'eps':1e-8,
        'weight_decay':0,'amsgrad':False,'maximize':False,'capturable':False,'differentiable':False,
        'foreach':None,'fused':None}
    state = {}
    for i,shape in enumerate(shapes):
        if steps:
            def tensor(values,dtype,shape):
                return {'kind':'tensor','dtype':dtype,'shape':list(shape),'values':values}
            state[str(i)] = {'step':tensor(float(steps),'torch.float32',()),
                'exp_avg':tensor(np.zeros(shape).tolist(),'torch.float64',shape),
                'exp_avg_sq':tensor(np.zeros(shape).tolist(),'torch.float64',shape)}
    return {'state':state,'param_groups':[group]}


def optimizer_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def boundary_fixture():
    spec = a.PROFILES['smoke'][0]
    seed, = spec.seeds
    head = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,436,1]))).standard_normal((4,8))
    arrays,optimizers,allocations = {},{},{}
    for arm in a.ARMS:
        points = []
        for label in ('initial','boundary','final'):
            values = {'transition_logits':np.zeros((4,8,8)), 'emission_logits':np.zeros((4,8)),
                'hazard_logits':np.zeros((4,8)), 'cost_logits':head.copy()}
            if arm.startswith('free_'):
                values['transition_logits'].fill(-np.log(8.))
            if label != 'initial':
                values['emission_logits'].fill(.1)
            if label == 'final':
                values['cost_logits'] += .2
            kind = 'joint' if label == 'final' else 'prefix'
            steps = spec.joint_updates if label == 'final' else spec.prefix_updates if label == 'boundary' else 0
            opt = adam(kind,steps)
            arrays[label,arm],optimizers[label,arm] = values,opt
            meta = {'model_state_sha256':state_hash(values),
                'dynamics_state_sha256':state_hash({k:v for k,v in values.items() if k != 'cost_logits'}),
                'head_state_sha256':state_hash({'cost_logits':values['cost_logits']}),
                'optimizer_state_sha256':optimizer_hash(opt)}
            points.append({'label':label,'metadata':meta,'model_sha256':meta['model_state_sha256'],
                'optimizer_sha256':meta['optimizer_state_sha256']})
        allocations[arm,seed] = {'checkpoints':points,'boundary':{'optimizer_after_sha256':optimizer_hash(adam('joint',0))}}
    groups = []
    for transport,arms in [('rounded',list(a.ARMS[:3])),('matched_free',list(a.ARMS[3:]))]:
        points = allocations[arms[0],seed]['checkpoints']
        groups.append({'transport_arm':transport,'arms':arms,'initial_model_sha256':points[0]['model_sha256'],
            'boundary_model_sha256':points[1]['model_sha256'],'initial_optimizer_sha256':points[0]['optimizer_sha256'],
            'boundary_optimizer_sha256':points[1]['optimizer_sha256'],'same_initial_and_prefix_model':True,'same_prefix_optimizer_states':True})
    paired = {'seed':seed,'groups':groups,'head_initial_boundary_sha256':{arm:state_hash({'cost_logits':head}) for arm in a.ARMS},
        'all_initial_heads_paired':True,'all_heads_unchanged_after_prefix':True,
        'scope':'Producer metadata joins before any DEV; saved bytes and initial transition functions independently audited.'}
    return spec,allocations,arrays,optimizers,paired


def test_independent_boundary_hashes_head_stream_and_architecture_pairs():
    spec,allocations,arrays,optimizers,paired = boundary_fixture()
    result = a.verify_boundaries(allocations,arrays,optimizers,paired,np,spec=spec)
    assert len(result['checks']) == len(result['transition_diagnostics']) == 18
    assert result['initializer_reconstructions'] == 1
    assert sum(x['correction'] is not None for x in result['transition_diagnostics']) == 9
    assert result['initial_transition_max_abs'] <= 1e-12
    assert result['optimizer_updates_replayed'] is False


@pytest.mark.parametrize('bad', ['head','prefix_model','adam','nan','shape','missing','paired_metadata'])
def test_actual_boundary_corruption_fails(bad):
    spec,allocations,arrays,optimizers,paired = boundary_fixture()
    if bad == 'head':
        arrays['initial','rounded_range']['cost_logits'][0,0] += .01
    elif bad == 'prefix_model':
        arrays['boundary','rounded_double']['hazard_logits'][0,0] += .01
    elif bad == 'adam':
        optimizers['boundary','rounded_range']['state']['0']['exp_avg']['values'][0][0][0] = .01
    elif bad == 'nan':
        arrays['final','free_mse']['hazard_logits'][0,0] = float('nan')
    elif bad == 'shape':
        arrays['initial','free_range']['emission_logits'] = np.zeros((8,4))
    elif bad == 'missing':
        arrays.pop(('final','rounded_range'))
    else:
        paired['all_initial_heads_paired'] = False
    with pytest.raises(ValueError):
        a.verify_boundaries(allocations,arrays,optimizers,paired,np,spec=spec)


def test_auditor_source_has_no_model_generator_or_torch_import():
    import ast
    tree = ast.parse(Path(a.__file__).read_text())
    names = [node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    names += [item.name for node in ast.walk(tree) if isinstance(node,ast.Import) for item in node.names]
    assert not any(name.startswith(('torch','openjev','run_finite','finite_prefix_learning')) for name in names)


def execution_fixture():
    """Two public prefixes, one terminal; one full-prefix and one joint update."""
    spec = a.r.StudyProfile('fabricated',948001,(948101,),2,8,2,1,1,30.)
    seed = 948101
    p = np.zeros((2,9,31),np.float32)
    p[:,0,4] = p[:,0,9] = 1
    p[0,1:,0] = p[0,1:,4] = 1
    p[1,1,0] = p[1,1,8] = 1
    prefixes = {'prefix':p,'lengths':np.array([9,2],np.int64),
        'event_mask':np.arange(9)[None,:] < np.array([9,2])[:,None],
        'endpoint_rows':np.array([0,-1],np.int64)}
    data = {'case_ids':np.array(['fabricated'],dtype='U64'),'observations':np.array([[0,4]],np.int64)}
    order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,0,818]))).permutation(2)
    order_hash = hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
    prefix_work = dict.fromkeys(a.UPDATE_WORK,0)
    prefix_work.update(prefix_updates=1,prefix_event_exposures=11,prefix_full_rollouts=1,backward_passes=1,adam_steps=1)
    joint_work = dict.fromkeys(a.UPDATE_WORK,0)
    joint_work.update(joint_updates=1,joint_attempt_exposures=2,joint_case_exposures=1,joint_event_exposures=11,
        joint_prefix_rollouts=1,joint_blind_rollouts=1,joint_observed_rollouts=1,backward_passes=1,adam_steps=1)
    work = {arm:{route:{} for route in a.ROUTES} for arm in a.ARMS}
    allocations,orders,fits = {},[],[]
    for arm in a.ARMS:
        extra = dict.fromkeys(a.LOSS_WORK_KEYS,0)
        extra['wrapper_calls'] = 1
        if not arm.endswith('_mse'):
            extra.update(blind_error_entries=8,mse_square_entries=8,mse_mean_calls=1,
                weight_scalar_divisions=2,weight_scalar_multiplications=1,
                loss_adjustment_multiplications=1,loss_adjustment_additions=1)
            if arm.endswith('_range'):
                extra.update(range_max_rows=2,range_min_rows=2,range_subtract_entries=2,
                    range_square_entries=2,range_scale_entries=2,range_mean_calls=1,replacement_subtractions=1)
        trace = [
            {'kind':'prefix','accepted':True,'result':{'work':dict(prefix_work),'diagnostics':{
                'kind':'prefix','attempts':2,'valid_events':11,'pseudocount':.001,'cost_head_updated':False}}},
            {'kind':'joint','accepted':True,'cursor_before':0,'result':{'work':dict(joint_work),'diagnostics':{
                'kind':'joint','cursor':0,'epoch':0,'batch_offset':0,'indices':order.tolist(),
                'eligible':1,'valid_events':11,'batch_size':2,'order_sha256':order_hash,
                'loss_kind':arm.rsplit('_',1)[1],'loss_work':dict(extra)}}}]
        allocations[arm,seed] = {'trace':trace,'attempted_updates':2,'final_summary':{'loss_work':dict(extra)}}
        fits.append({'arm':arm,'loss_work':dict(extra)})
        orders.append({'arm':arm,'seed':seed,'epoch':0,'indices':order.tolist(),'batch_size':2})
        # Qualified independent primitive geometry supplies unchanged model work;
        # the new scalar intervention and full exposure totals above are hand-derived.
        work[arm]['training_prefix'] = a.h.forward_work(a.HEAD[arm],a.h.prefix_work(prefixes,np.array([0,1]),np),prefix=True,prior=True)
        work[arm].update(a.reuse_work(a.TRANSPORT[arm],prefixes,order,data['observations'],np))
    counts = {'model_constructions':6,'checkpoint_writes':18,'optimizer_checkpoint_writes':18,
        'optimizer_attempts':12,'optimizer_steps':12,'epoch_order_generations':6,
        'training_attempt_exposures':12,'training_case_exposures':6,'training_prefix_event_exposures':132,
        'training_blind_rollouts':6,'training_observed_rollouts':6,'training_prefix_rollouts':12,
        'zero_endpoint_batches':0,'accepted_optimizer_steps':12,'accepted_joint_steps':6,
        'accepted_prefix_steps':6,'fit_count':6}
    digest = hashlib.sha256(json.dumps([order.tolist()],separators=(',',':'),allow_nan=False).encode()).hexdigest()
    summary = {'structural_work':work,'fits':fits,'counts':counts,'paired_batch_sha256':digest}
    return spec,summary,allocations,{'train':data},{'train':prefixes},orders


def test_training_only_geometry_is_disjoint_and_loss_specific():
    spec,summary,allocations,datasets,prefixes,orders = execution_fixture()
    out = a.validate_execution(summary,allocations,datasets,prefixes,orders,np,spec=spec,training_only=True)
    assert out['attempted_update_work']['joint_attempt_exposures'] == 12
    assert out['attempted_update_work']['joint_case_exposures'] == 6
    assert out['attempted_update_work']['joint_event_exposures'] == 66
    assert out['attempted_update_work']['prefix_event_exposures'] == 66
    assert out['counts']['dev_generation_count'] == out['counts']['oracle_model_constructions'] == 0
    assert out['counts']['evaluation_prefix_rollouts'] == 0
    assert out['loss_work']['rounded_range']['range_max_rows'] == 2
    assert out['loss_work']['free_double']['range_max_rows'] == 0
    assert not out['model_updates_replayed']


@pytest.mark.parametrize('bad', ['loss','paired_digest','duplicate_forward','missing_shared','exposure','extra_order'])
def test_execution_tampering_is_not_only_a_reported_counter_check(bad):
    spec,summary,allocations,datasets,prefixes,orders = execution_fixture()
    if bad == 'loss':
        allocations['rounded_range',948101]['trace'][1]['result']['diagnostics']['loss_work']['range_max_rows'] += 1
    elif bad == 'paired_digest':
        summary['paired_batch_sha256'] = '0'*64
    elif bad == 'duplicate_forward':
        summary['structural_work']['rounded_range']['training_blind'] = dict(summary['structural_work']['rounded_range']['joint_reuse_blind'])
    elif bad == 'missing_shared':
        summary['structural_work']['rounded_range']['joint_reuse_shared'] = {}
    elif bad == 'exposure':
        summary['counts']['training_case_exposures'] += 1
    else:
        orders.append(dict(orders[0]))
    with pytest.raises(ValueError):
        a.validate_execution(summary,allocations,datasets,prefixes,orders,np,spec=spec,training_only=True)
