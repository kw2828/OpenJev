"""Synthetic-only independent feature/coordinate/readout arithmetic fixtures."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def audit():
    return load('synthetic_symmetry_audit', 'scripts/audit_otto_symmetry_head.py')


@pytest.fixture
def model():
    return load('synthetic_symmetry_producer', 'src/openjev/research/otto_symmetry_head.py')


def kernel():
    result = np.full((4, 107, 107), .25, np.float64)
    result[:, 53, 53] = 0
    return result


def test_public_features_match_independent_closed_form_and_source(audit, model):
    probability = np.zeros((53, 53), np.float64)
    probability[1, 0], probability[0, 1] = .25, .75
    public = audit.B.packet([0, 0], 0, False, 8)
    x = audit.feature_vector(probability, public, kernel(), 5., np)
    assert x.shape == (2836,) and x.dtype == np.float32
    np.testing.assert_array_equal(x[2809:2816], [-1, -1, 0, 1, 0, 1, 1])
    np.testing.assert_array_equal(x[-20:].reshape(4, 5), [[0., .25, .25, .25, .25],
                                  [.25, .1875, .1875, .1875, .1875], [0., .25, .25, .25, .25],
                                  [.75, .0625, .0625, .0625, .0625]])
    np.testing.assert_array_equal(x, model.PublicFeatureMap(kernel(), 5).features(probability, public))
    tiny = probability * 1e-20
    np.testing.assert_array_equal(audit.feature_vector(tiny, public, kernel(), 3., np),
                                  model.PublicFeatureMap(kernel(), 3).features(tiny, public))
    public['source'] = [1, 0]
    with pytest.raises(ValueError, match='public'):
        audit.feature_vector(probability, public, kernel(), 5., np)


def test_all_eight_permutations_transport_grid_position_and_local_actions(audit, model):
    x = np.arange(2836, dtype=np.float32)[None]
    for g in range(8):
        perm = audit.action_permutation(g)
        assert sorted(perm) == list(range(4)) and tuple(perm) == model.action_permutation(g)
        transformed = audit.transformed_features(x, g, np)
        np.testing.assert_array_equal(transformed, model.transform_features(x, g))
        assert transformed[0, 2815] == x[0, 2815]
        for action in range(4):
            assert transformed[0, 2811+perm[action]] == x[0, 2811+action]
            np.testing.assert_array_equal(transformed[0, 2816+5*perm[action]:2821+5*perm[action]],
                                          x[0, 2816+5*action:2821+5*action])
    assert audit.action_permutation(1) == [2, 3, 1, 0]
    assert audit.action_permutation(4) == [0, 1, 3, 2]


def weights(kind):
    rng = np.random.default_rng(814)
    outputs = 1 if kind == 'd4_shared' else 4
    shapes = [(32, 2836), (32,), (16, 32), (16,), (outputs, 16), (outputs,)]
    return {key: rng.uniform(-.05, .05, shape).astype(np.float32)
            for key, shape in zip(('weight0', 'bias0', 'weight1', 'bias1', 'weight2', 'bias2'), shapes, strict=True)}


@pytest.mark.parametrize('mode', ['d4_shared', 'dense_augmented', 'dense_ensemble'])
@pytest.mark.parametrize('batch', [None, 3])
def test_saved_numpy_readouts_match_export_contract_without_torch(audit, model, mode, batch):
    kind = 'd4_shared' if mode == 'd4_shared' else 'dense_augmented'
    tensors = weights(kind)
    audit.validate_weights(tensors, kind, np)
    frozen = model.FrozenHead({'version': model.VERSION, 'input_dim': 2836, 'kind': kind, **tensors})
    x = np.random.default_rng(22).uniform(-1, 1, (2836,) if batch is None else (batch, 2836)).astype(np.float32)
    expected = frozen.scores(x, kind=mode)
    actual = audit.readout(x, tensors, mode, np)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
    assert actual.shape == (4,) if batch is None else actual.shape == (batch, 4)


def test_episode_balancing_and_teacher_mask_are_not_row_micro(audit):
    weighted = audit.balanced_weights(['a', 'a', 'a', 'b'], np)
    np.testing.assert_allclose(weighted, [2/3, 2/3, 2/3, 2], rtol=1e-7)
    assert np.isclose(weighted[:3].sum(), weighted[3])
    target = audit.teacher_target([None, 2., 4., None], [False, True, True, False], np)
    np.testing.assert_allclose(target, [0, 1/(1+np.exp(-4)), 1/(1+np.exp(4)), 0], rtol=1e-7)
    assert target.dtype == np.float32 and target.sum() == 1
    np.testing.assert_array_equal(audit.teacher_target([None, 2., 2., None], [False, True, True, False], np), [0, .5, .5, 0])


def test_wrong_checkpoint_dtype_nonfinite_and_extra_tensors_rejected(audit):
    for fault in ('dtype', 'finite', 'extra'):
        tensor = weights('d4_shared')
        if fault == 'dtype':
            tensor['bias0'] = tensor['bias0'].astype(np.float64)
        elif fault == 'finite':
            tensor['bias0'][0] = np.inf
        else:
            tensor['optimizer'] = np.zeros(1, np.float32)
        with pytest.raises(ValueError):
            audit.validate_weights(tensor, 'd4_shared', np)


def test_fixed_sampling_rounds_and_includes_all_endpoints(audit):
    assert audit.selected_prefixes(1) == [0]
    assert audit.selected_prefixes(12) == list(range(12))
    for steps in (65, 66, 100, 2188):
        chosen = audit.selected_prefixes(steps)
        assert len(chosen) == 64 and len(set(chosen)) == 64
        assert chosen[0] == 0 and chosen[-1] == steps - 1
        assert chosen == np.rint(np.linspace(0, steps-1, 64)).astype(int).tolist()
    assert audit.selected_prefixes(66)[16] == 17  # Floor would wrongly choose16.


def complete_rows(audit):
    rows=[]
    for regime,seed,hit,arm,block in audit.episode_order('eval'):
        steps={'shared':9,'dense':12,'dense_ensemble':11,'analytic_inbounds':10}[arm.split('@')[0]]+hit
        seconds={'shared':.5,'dense':1.,'dense_ensemble':2.,'analytic_inbounds':1.}[arm.split('@')[0]]
        rows.append({'regime':regime,'seed':seed,'initial_hit':hit,'arm':arm,'block':block,
                     'steps':steps,'updates':steps,'blocked_steps':0,'found':True,'init_seconds':0.,'choose_seconds':seconds,'update_seconds':0.,
                     'setup_allocation_seconds':0.,'controller_seconds':seconds,'environment_seconds':.1,'state_bytes':22472})
    return rows


def test_all_720_54_rules_and_unequal_mixture_against_hand_counts(audit):
    rows=complete_rows(audit)
    weights={r:{1:.6,2:.3,3:.1} for r in audit.REGIMES}
    result=audit.aggregate(rows,weights)
    assert result['pilot_continuation'] is True
    assert [len(result[k]) for k in ('competence_checks','compression_checks','architecture_checks')]==[18,12,24]
    for regime in audit.REGIMES:
        assert result['regimes'][regime]['means']['shared@9101']['steps']==10.5
        assert result['regimes'][regime]['family_means']['shared']['steps']==10.5
        assert result['regimes'][regime]['means']['analytic_inbounds']['steps']==11.5
        assert len(result['regimes'][regime]['blocks'])==8
    rows[0]['found']=False
    result=audit.aggregate(rows,weights)
    first=rows[0]
    expected=1-weights[first['regime']][first['initial_hit']]/8
    assert result['regimes'][first['regime']]['means'][first['arm']]['found']==pytest.approx(expected)
    assert result['pilot_continuation'] is False
    with pytest.raises(ValueError,match='identities'):
        audit.aggregate(rows[:-1],weights)


def test_independent_aggregate_matches_qualified_schema_and_all_strata(audit):
    producer=load('synthetic_symmetry_run','scripts/study_otto_symmetry_head.py')
    rows=complete_rows(audit)
    weights={r:{1:.7,2:.2,3:.1} for r in audit.REGIMES}
    import json
    expected=json.loads(json.dumps(producer.summary(rows,weights)))
    actual=audit.aggregate(rows,weights)
    for key,value in actual.items():
        audit.B.Comparisons().tree(expected[key],value,'synthetic full summary')


def test_strict_cost_and_exact_inclusive_move_boundary(audit):
    rows=complete_rows(audit)
    weights={r:{1:.6,2:.3,3:.1} for r in audit.REGIMES}
    for row in rows:
        row['steps']=21 if row['arm'].startswith('shared') else 20
        row['controller_seconds']=1.
    result=audit.aggregate(rows,weights)
    assert all(c['passes'] for c in result['competence_checks'])  # Exact1.05 boundary.
    assert not any(c['passes'] for c in result['compression_checks'] if c['name'].endswith('every_cost'))
    for row in rows:
        if row['arm'].startswith('shared'):
            row['steps']=22
    assert not any(c['passes'] for c in audit.aggregate(rows,weights)['competence_checks'] if c['name'].endswith('moves'))


def test_work_pairs_context_and_net_cost_corruptions(audit):
    events=[[1,0,'head_predict',0,3],[1,1,'head_predict',.8,1.,.2]]
    work=audit.Work(iter(events),[{'phase':'eval','episode':'fake'}],audit.B.Comparisons())
    assert work.call('head_predict',{'phase':'eval','episode':'fake','step':3})['seconds']==.8
    assert work.used_contexts=={0} and work.counts['head_predict']['returned']==1
    for event in ([2,1,'head_predict',.8,1.,.2],[1,1,'head_predict',.9,1.,.2],[1,1,'native_step',.8,1.,.2]):
        work=audit.Work(iter([events[0],event]),[{'phase':'eval','episode':'fake'}],audit.B.Comparisons())
        with pytest.raises(ValueError):
            work.call('head_predict',{'phase':'eval','episode':'fake','step':3})


def test_action_first_tie_restricted_mask_and_raw_float32(audit):
    assert audit.saved_choice([0.,1.,0.,-10.],[0,1,2],True,np)==0
    assert audit.saved_choice([None,2.,2.,None],[1,2],False,np)==1
    with pytest.raises(ValueError,match='float32'):
        audit.saved_choice([0.,1.0000000001,2.,3.],[0,1,2,3],True,np)
    with pytest.raises(ValueError):
        audit.saved_choice([0.,1.,2.,3.],[],True,np)


def test_metadata_pin_failure_precedes_any_array_or_reader_import(audit,tmp_path,monkeypatch):
    from types import SimpleNamespace
    plan=tmp_path/'plan.json'
    plan.write_text('{}')
    args=SimpleNamespace(plan=plan,run=tmp_path/'run',terminal=tmp_path/'terminal.json',output=tmp_path/'out',
                         plan_sha256='bad',receipt_sha256='bad',terminal_sha256='bad')
    checker=audit.Audit(args)
    monkeypatch.setattr(checker,'check',lambda:None)
    monkeypatch.setattr(audit.B,'load',lambda *a:pytest.fail('untrusted producer import'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('untrusted arrays'))
    with pytest.raises(ValueError,match='identity'):
        checker.authenticate()


def tiny_episode(audit,tmp_path):
    from types import SimpleNamespace
    checker=audit.Audit(SimpleNamespace(output=tmp_path))
    checker.np=np
    checker.check=lambda:None
    checker.sources,checker.uniforms={},{}
    checker.offsets={'train':0}
    checker.episode_counts={'train':0}
    checker.kernels={'lambda3':kernel()}
    public=audit.B.packet([26,26],1,False,0)
    before=audit.B.posterior(np.ones((53,53),np.float64)/2808,public,kernel(),np)
    after=audit.B.packet([25,26],-2,True,1)
    final=audit.B.posterior(before,after,kernel(),np)
    identity={'episode_id':'train:lambda3:910001:teacher','stage':'train','regime':'lambda3','seed':910001,'initial_hit':1,'arm':'teacher','block':None}
    costs=[0.,1.,2.,3.]
    record={'row_index':0,'episode_id':identity['episode_id'],'stage':'train','regime':'lambda3','seed':910001,
            'initial_hit':1,'arm':'teacher','prefix_index':0,'public':public,'posterior':audit.posterior_record(before),'teacher_costs':costs}
    x=audit.feature_vector(before,public,kernel(),3.,np)
    checker.datasets={'train':{'rows':[record],'features':x[None],'valid':np.ones((1,4),bool),
                              'target':audit.teacher_target(costs,[True]*4,np)[None]}}
    events=[]
    for i,(channel,step) in enumerate((('native_reset',0),('teacher_label',1),('native_step',1)),1):
        events.extend(([i,0,channel,0,step],[i,1,channel,.1,.1,0.]))
    checker.work=audit.Work(iter(events),[{'phase':'train','episode':identity['episode_id']}],checker.c)
    row={**identity,'steps':1,'found':True,'updates':1,'blocked_steps':0,'init_seconds':.1,'choose_seconds':.1,
         'update_seconds':.1,'setup_allocation_seconds':0.,'controller_seconds':.3,'choose_instrumented_seconds':.1,
         'choose_excluded_io_seconds':0.,'environment_seconds':.1,'state_bytes':22472,
         'storage':{'public_actor':{'immutable_array_bytes':457960,'mutable_array_bytes':22472,
                    'immutable_arrays':{'observation_kernel':366368,'manhattan_distance_table':91592},'scope':'synthetic'},
                    'feature_kernel_bytes':366368,'head':None},
         'source_evaluation_only':[25,26],'draws_evaluation_only':[{'channel':'source','index':0,'uniform':.4,
                  'selected_index':25*53+26,'cdf_mass':1.}], 'final_public':after,'final_update_assimilated':True}
    transitions=[{'kind':'reset',**identity,'public':public,'posterior_after':audit.posterior_record(before),'source_evaluation_only':[25,26]},
                 {'kind':'step','episode_id':identity['episode_id'],'step':1,'action':0,'costs':costs,
                  'allowed_actions':[0,1,2,3],'public':after,'posterior_before':audit.posterior_record(before),
                  'posterior_after':audit.posterior_record(final),'choose_seconds':.1,'choose_instrumented_seconds':.1,
                  'choose_excluded_io_seconds':0.,'update_seconds':.1,'environment_seconds':.1,'native_p_end':1.}]
    return checker,row,transitions


def test_full_saved_episode_reconstructs_filter_features_targets_and_terminal(audit,tmp_path):
    checker,row,transitions=tiny_episode(audit,tmp_path)
    actual=checker.episode(row,'train',next(audit.episode_order('train')),iter(transitions))
    assert actual['found'] and checker.offsets['train']==1 and checker.episode_counts['train']==1


@pytest.mark.parametrize('fault',['action','posterior','target','last_update','cost'])
def test_saved_episode_rejects_causal_probability_membership_and_cost_corruption(audit,tmp_path,fault):
    checker,row,transitions=tiny_episode(audit,tmp_path)
    if fault=='action':
        transitions[1]['action']=1
    elif fault=='posterior':
        transitions[1]['posterior_after']['sha256']='wrong'
    elif fault=='target':
        checker.datasets['train']['target'][0]=.25
    elif fault=='last_update':
        row['final_update_assimilated']=False
    else:
        row['controller_seconds']=.2
    with pytest.raises(ValueError):
        checker.episode(row,'train',next(audit.episode_order('train')),iter(transitions))


@pytest.mark.parametrize('late',[False,True])
def test_failed_auth_or_late_publication_preserves_failure(audit,tmp_path,monkeypatch,late):
    from types import SimpleNamespace
    args=SimpleNamespace(output=tmp_path/'exclusive')
    checker=audit.Audit(args)
    def auth():
        if not late:
            raise ValueError('synthetic missing complete run')
        return {},{}
    monkeypatch.setattr(checker,'authenticate',auth)
    monkeypatch.setattr(checker,'compute',lambda *a:{'comparisons':0,'agreement':True})
    if late:
        original=checker.check
        def check():
            original()
            if (args.output/'receipt.json').exists():
                raise TimeoutError('synthetic postpublication timeout')
        monkeypatch.setattr(checker,'check',check)
    with pytest.raises((ValueError,TimeoutError)):
        checker.execute()
    assert (args.output/'failed.json').exists()
    assert not (args.output/'receipt.json').exists()
    if late:
        assert (args.output/'invalid-completed-receipt.json').exists()
    with pytest.raises((ValueError,FileExistsError)):
        checker.execute()


def fit_fixture(audit,tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    checker=audit.Audit(SimpleNamespace(output=tmp_path/'out',run=tmp_path))
    checker.np=np
    checker.check=lambda:None
    checker.optimizer_steps={}
    checker.fit_costs={'initial':0.,'final':0.}
    checker.stage_updates={}
    checker.datasets={}
    for split,n in (('train',192),('valid',48),('dagger',144)):
        records=[{'row_index':i,'episode_id':f'{split}:{i}'} for i in range(n)]
        checker.datasets[split]={'rows':records,'features':np.zeros((n,2836),np.float32),
                 'target':np.full((n,4),.25,np.float32),'valid':np.ones((n,4),bool),'weights':np.ones(n,np.float32)}
        np.savez(tmp_path/f'{split}-data.npz',**{k:v for k,v in checker.datasets[split].items() if k!='rows'})
    events,contexts=[],[]
    stages={}
    sequence=0
    for stage,parts in (('initial',('train',)),('final',('train','dagger'))):
        arrays={k:np.concatenate([checker.datasets[p][k] for p in parts]) for k in ('features','target','valid','weights')}
        records=[{k:v for k,v in r.items() if k!='row_index'} for p in parts for r in checker.datasets[p]['rows']]
        n=len(records)
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
        rowhash=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        fits,curves,orders=[],[],[]
        for seed in audit.FIT_SEEDS:
            for family in ('shared','dense'):
                fid=f'{family}@{seed}'
                contextid=len(contexts)
                contexts.append({'phase':'fit','stage':stage,'fit_id':fid})
                rng=np.random.default_rng(seed+(0 if stage=='initial' else 100000))
                for epoch in range(1,41):
                    order=rng.permutation(n)
                    orders.append({'stage':stage,'fit_id':fid,'epoch':epoch,'rows':n,'sha256':hashlib.sha256(order.tobytes()).hexdigest()})
                    for _ in range((n+127)//128):
                        sequence+=1
                        events.extend(([sequence,0,'optimizer_update',contextid,None],[sequence,1,'optimizer_update',.001,.001,0.]))
                curve=[{'stage':stage,'fit_id':fid,'epoch':e,'training_weighted_ce':1.,'weighted_ce':1.,'argmax_agreement':.5} for e in range(5,41,5)]
                curves.extend(curve)
                checkpoint=tmp_path/f'{stage}-{family}-{seed}.npz'
                checkpoint.write_bytes(b'synthetic opaque already-authenticated checkpoint')
                fits.append({'stage':stage,'fit_id':fid,'family':family,'seed':seed,'epochs':40,
                    'training_rows':n,'training_episodes':n,'row_order_sha256':rowhash,'row_weights_sha256':hashes['weights'],
                    'training_array_sha256':hashes,'optimizer_continued':stage=='final','optimizer_steps_before':[] if stage=='initial' else [80]*6,
                    'optimizer_steps_after':[80 if stage=='initial' else 200]*6,'curve':curve,'checkpoint':checkpoint.name,
                    'checkpoint_sha256':audit.B.digest(checkpoint)['sha256'],'fit_seconds':1.,'export_max_abs_error':0.,
                    'export_validation_rows':list(range(16)),'export_atol':2e-5,'export_rtol':2e-5,'export_action_identity_asserted':False})
        stages[stage]=(fits,curves,orders)
        if stage=='final':
            np.savez(tmp_path/'pooled-weights.npz',weights=arrays['weights'])
            audit.B.write(tmp_path/'pooling.json',{'order':['train','dagger'],'rows':n,'episodes':n,'shared_for_all_six_final_fits':True,
                'datasets':{p:audit.B.digest(tmp_path/f'{p}-data.npz') for p in parts},'weights_sha256':hashes['weights']})
    checker.work=audit.Work(iter(events),contexts,checker.c)
    return checker,stages


def test_all_twelve_fits_480_orders_and_continued_adam_steps(audit,tmp_path):
    checker,stages=fit_fixture(audit,tmp_path)
    for stage in ('initial','final'):
        checker.fit_stage(stage,*(iter(v) for v in stages[stage]))
    assert checker.stage_updates=={'initial':480,'final':720}
    assert len(checker.optimizer_steps)==6 and all(v==[200]*6 for v in checker.optimizer_steps.values())
    audit.B.exhausted(checker.work.iterator,'synthetic optimizer work')


@pytest.mark.parametrize('fault',['order','dataset','optimizer','missing_fit'])
def test_fit_exposure_order_and_optimizer_witness_corruptions_rejected(audit,tmp_path,fault):
    checker,stages=fit_fixture(audit,tmp_path)
    fits,curves,orders=stages['initial']
    if fault=='order':
        orders[0]['sha256']='wrong'
    elif fault=='dataset':
        fits[0]['training_array_sha256']['features']='wrong'
    elif fault=='optimizer':
        fits[0]['optimizer_steps_before']=[80]*6
    else:
        fits.pop()
    with pytest.raises((ValueError,StopIteration)):
        checker.fit_stage('initial',iter(fits),iter(curves),iter(orders))
