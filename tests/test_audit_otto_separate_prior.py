"""Independent fabricated sampled-window reconstruction, no scientific calls."""
from __future__ import annotations

import copy
import importlib.util
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_separate_prior_audit", ROOT / "scripts/audit_otto_separate_prior.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(tmp_path):
    obj = audit.Audit(SimpleNamespace(output=tmp_path))
    obj.np = np
    obj.collection = tmp_path
    obj.run = tmp_path
    obj.check = lambda: None
    lengths = [37] + [1] * 53
    offsets = np.asarray([0, *np.cumsum(lengths)], np.int64)
    n = int(offsets[-1])
    x = np.zeros((n, 31), np.float32)
    correction = np.zeros(n, np.bool_)
    records, episodes = [], []
    for index, (identity, length) in enumerate(zip(audit.cohort()[:54], lengths, strict=True)):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        steps = np.arange(length)
        x[lo:hi, 15] = steps / 2188
        x[lo:hi, 16] = steps % 4 / 2188
        x[lo:hi, 17] = 1
        correction[lo:hi] = steps % 4 == 0
        w, seed = (length + 3) // 4, audit.SELECTION_START + index
        k = min(w, 8)
        starts = sorted(int(v) * 4 for v in np.random.Generator(np.random.PCG64(seed)).choice(w, k, replace=False))
        records.append({"episode_id": identity["episode_id"], "version": "otto-sampled-forecast-data-v1",
            "length": length, "seed": seed, "period": 4, "population_windows": w, "selected_windows": k,
            "start_offsets": starts, "inclusion_numerator": k, "inclusion_denominator": w,
            "inclusion_probability": k / w, "algorithm": "numpy.Generator(PCG64(seed)).choice(W,size=k,replace=False); sorted"})
        episodes.append({**identity, "start_row": lo, "end_row": hi, "steps": length, "teacher_calls": length})
    flat = {"features": x, "raw_q": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            "legal": np.ones((n, 4), np.bool_), "actions": np.zeros(n, np.int64),
            "correction": correction, "episode_offsets": offsets, "label_mask": np.ones(n, np.bool_)}
    obj.episodes = episodes
    obj.arrays = lambda path: flat
    obj.read = lambda path: records
    return obj, flat, records


def test_all_query_inputs_but_only_selected_nonquery_targets(tmp_path):
    obj, flat, records = fixture(tmp_path)
    history, meta, _ = obj.reconstruct('train')
    assert meta['counts'] == {'episodes': 54, 'rows': 90, 'query_rows': 63,
        'selected_nonquery_rows': sum(min(3, 36-s) for s in records[0]['start_offsets']),
        'selected_windows': 61, 'zero_support_episodes': 53, 'prior_rows':9,
        'prior_supported_episodes':1, 'prior_zero_support_episodes':53,
        'prior_weight_mass':float(history['prior_weights'].sum())}
    assert history['query_scores'][::1][history['query_mask']].tobytes() == flat['raw_q'][flat['correction']].tobytes()
    selected = {step for start in records[0]['start_offsets'] for step in range(start+1, min(start+4,37))}
    for step in range(37):
        assert (history['weights'][step] > 0) == (step in selected)
        if step in selected:
            assert history['weights'][step] == (10/8)/(54*27)
            assert history['targets'][step].tobytes() == flat['raw_q'][step].tobytes()
        else:
            assert history['targets'][step].tobytes() == np.zeros(4,np.float32).tobytes()


def test_query_outside_sample_requires_positive_provenance(tmp_path):
    obj, flat, records = fixture(tmp_path)
    step = next(i for i in range(0,37,4) if i not in records[0]['start_offsets'])
    flat['label_mask'][step] = False
    flat['raw_q'][step] = 0
    obj.episodes[0]['teacher_calls'] -= 1
    with pytest.raises(ValueError, match='chronological queries'):
        obj.reconstruct('train')


def test_incidental_nonquery_labels_do_not_enter_history(tmp_path):
    obj, flat, records = fixture(tmp_path)
    before, _, _ = obj.reconstruct('train')
    selected = {i for s in records[0]['start_offsets'] for i in range(s,min(s+4,37))}
    step = next(i for i in range(37) if i%4 and i not in selected)
    flat['raw_q'][step] = np.nan  # authenticated producer would reject; audit must not use incidental scores.
    after, _, _ = obj.reconstruct('train')
    for key in before:
        assert before[key].tobytes() == after[key].tobytes()


def metric_fixture():
    # One two-window path and a query-only sibling case, with no postcorrection support.
    windows = {'lengths': np.array([4,4,1]), 'episode_index': np.array([0,0,1]),
        'step_offsets': np.array([0,4,0]), 'targets': np.tile(np.array([0,2,8,9],np.float32),(3,4,1)),
        'legal': np.ones((3,4,4),np.bool_)}
    predictions = windows['targets'].copy()
    predictions[1,1:,0] = 3 # choose action1, teacher gap2 at absolute steps5..7
    ids = [{'episode_id':'a','regime':'lambda3','case':0,'arm':'analytic'},
           {'episode_id':'b','regime':'lambda3','case':1,'arm':'analytic'}]
    meta = {'episode_ids':['a','b'],'episode_regimes':['lambda3','lambda3']}
    return predictions, windows, meta, ids


def test_scalar_postcorrection_domain_and_zero_support_denominator():
    result = audit.scalar_metrics(*metric_fixture())
    full, primary = result['full']['overall'], result['postcorrection']['overall']
    assert full['episode_weighted_agreement'] == .25
    assert primary['episode_weighted_agreement'] == 0
    assert primary['episode_weighted_raw_gap'] == 1
    assert primary['supported_episode_raw_gap'] == 2
    assert primary['weight_mass'] == .5 and primary['zero_support_episode_ids'] == ['b']
    assert primary['declared_case_count'] == 2 and primary['supported_case_count'] == 1
    assert primary['by_age']['3']['episode_weighted_raw_gap'] == 1
    assert result['postcorrection']['by_case'][1]['supported_episode_agreement'] is None


def test_float32_ties_and_legal_centered_common_offset():
    predictions, windows, meta, ids = metric_fixture()
    windows['targets'][:] = np.array([0,5e-11,999,999],np.float32)
    windows['legal'][:,:,2:] = False
    predictions[:] = np.array([2e-10,0,-999,-999],np.float32)
    result = audit.scalar_metrics(predictions,windows,meta,ids)
    assert result['postcorrection']['overall']['episode_weighted_agreement'] == .5
    assert result['postcorrection']['overall']['episode_weighted_first_argmin_match'] == 0
    assert result['postcorrection']['overall']['episode_weighted_raw_gap'] == float(np.float32(5e-11))/2
    windows['targets'][:] = np.array([0,2,999,999],np.float32)
    predictions[:] = np.array([100,102,-999,-999],np.float32)
    assert audit.scalar_metrics(predictions,windows,meta,ids)['full']['overall']['episode_weighted_centered_mse'] == 0


def test_renormalized_persisted_history_weights_rejected(tmp_path):
    obj,_,_=fixture(tmp_path)
    values,meta,_=obj.reconstruct('train')
    saved={k:v.copy() for k,v in values.items()}
    saved['weights'] /= saved['weights'].sum()
    obj.arrays=lambda path:saved
    obj.read=lambda path:meta
    with pytest.raises(ValueError,match='window bytes weights'):
        obj.saved_windows('training-history',values,meta)


def test_complete_batch_work_includes_zero_target_lanes_and_full_tails(tmp_path):
    obj = audit.Audit(SimpleNamespace(output=tmp_path)); obj.np = np
    lengths = [1,32,33,64,65,2]+[1]*48
    offsets = np.asarray([0,*np.cumsum(lengths)],np.int64)
    weights = np.zeros(int(offsets[-1]),np.float64)
    weights[int(offsets[4])+33] = 1
    history={'episode_offsets':offsets,'weights':weights,'prior_weights':np.zeros_like(weights)}
    rows = obj.expected_batches(history,[list(range(54))]*80,'mse')
    assert len(rows) == 720 and rows[-1]['optimizer_step'] == 720
    assert rows[0] == {'epoch':1,'batch':0,'episode_indices':list(range(6)),
        'forward_chunks':3,'backward_chunks':1,'no_grad_chunks':2,'forward_rows':197,'optimizer_step':1,'objective':'mse','prior_rows':46}
    assert rows[1]['forward_chunks'] == 1 and rows[1]['backward_chunks'] == 0
    assert sum(r['forward_rows'] for r in rows) == 80*sum(lengths)


def test_successful_worker_cannot_replace_failed_original_parent():
    root=str(audit.ROOT)
    cap = audit.LIMITS['seconds']
    assert cap == 240 and audit.base.LIMITS == {
        'seconds':240,'rss_bytes':2*1024**3,'output_bytes':256*1024**2}
    launch={'cwd':root,'cap_seconds':cap,'clock_source_sha256':audit.CLOCK_PIN,
        'watchdog_sha256':audit.SUPERVISOR_PIN,'started_ns':0,'deadline_ns':cap*10**9}
    worker={'status':'completed','complete':True,'started_ns':1_000_000_000,
        'finished_ns':2_000_000_000,'wall_seconds':1.}
    terminal={**launch,'status':'completed','returncode':0,'timed_out':False,'error':None,
        'clock_error':None,'group_absent':True,'cleanup':{'reaped':True,'errors':[]},
        'finished_ns':3_000_000_000,'elapsed_ns':3_000_000_000,'wall_seconds':3.}
    audit.base.closed_parent(worker,terminal,launch,cap)
    terminal['returncode']=1
    with pytest.raises(ValueError,match='successful fully closed'):
        audit.base.closed_parent(worker,terminal,launch,cap)


def test_extra_json_null_is_not_treated_as_journal_eof():
    assert audit.journal_exhausted(iter([]))
    assert not audit.journal_exhausted(iter([None]))
    assert not audit.journal_exhausted(iter([{}]))


def prior_fixture():
    offsets = np.array([0,9,14,18],np.int64)
    query = np.full((18,4),np.nan,np.float32)
    mask = np.zeros(18,np.bool_)
    for low,high in pairwise(offsets):
        mask[low:high:4]=True
        query[low+4:high:4]=0
    prior = np.full_like(query,np.nan)
    prior[4]=[0,0,0,4]; prior[8]=[0,0,0,8]; prior[13]=[0,0,0,4]
    identities=[{'episode_id':'a','regime':'lambda3','case':0,'arm':'analytic'},
                {'episode_id':'b','regime':'lambda3','case':0,'arm':'neural'},
                {'episode_id':'c','regime':'lambda4','case':0,'arm':'analytic'}]
    history={'episode_offsets':offsets,'query_scores':query,'query_mask':mask,
             'episode_ids':['a','b','c'],'episode_regimes':['lambda3','lambda3','lambda4']}
    return history,prior,identities


def test_independent_prior_scalar_all_four_and_complete_episode_case_denominators():
    history,prior,ids=prior_fixture()
    report=audit.scalar_prior_metrics(history,prior,ids)
    assert report['overall']['episode_weighted_centered_mse']==3.5
    assert report['overall']['supported_episode_centered_mse']==5.25
    assert report['overall']['supported_case_count']==1
    assert report['overall']['declared_case_count']==2
    assert report['by_case'][0]['episodes']==2
    assert report['overall']['zero_support_episode_ids']==['c']
    assert report['by_regime']['lambda4']['supported_episode_centered_mse'] is None
    assert [row['episode_weighted_centered_mse'] for row in report['by_episode']]==[7.5,3.,0.]
    changed={**history,'query_scores':history['query_scores']+np.float32(256)}
    assert audit.scalar_prior_metrics(changed,prior-np.float32(128),ids)==report


def test_prior_targets_weights_masks_keep_query_only_episodes_and_ignore_legal_mask(tmp_path):
    obj,flat,_=fixture(tmp_path)
    flat['legal'][:,3]=False
    history,_,_=obj.reconstruct('train')
    rows=np.arange(4,37,4)
    assert np.flatnonzero(history['prior_mask']).tolist()==rows.tolist()
    assert history['prior_targets'][rows].tobytes()==flat['raw_q'][rows].tobytes()
    assert np.all(history['prior_weights'][rows]==1/(54*9))
    assert np.all(history['prior_weights'][~history['prior_mask']]==0)
    assert history['prior_targets'][~history['prior_mask']].tobytes()==np.zeros((81,4),np.float32).tobytes()
    bad={k:v.copy() for k,v in history.items()}
    bad['prior_weights']*=54
    obj.arrays=lambda path:bad
    with pytest.raises(ValueError,match='prior_weights'):
        obj.saved_windows('training-history',history,{})


@pytest.mark.parametrize('fault',['drop','first','nonquery','active_nan','inactive_nonzero'])
def test_saved_prior_mask_cannot_omit_query_or_use_returned_query_as_free_calibration(tmp_path,fault):
    obj,_,_=fixture(tmp_path)
    history,_,_=obj.reconstruct('train')
    saved={'prior':np.zeros_like(history['query_scores']),'prior_mask':history['prior_mask'].copy()}
    obj.checked_prior(saved,history)
    if fault=='drop': saved['prior_mask'][4]=False
    elif fault=='first': saved['prior_mask'][0]=True
    elif fault=='nonquery': saved['prior_mask'][5]=True
    elif fault=='active_nan': saved['prior'][4,0]=np.nan
    else: saved['prior'][0,0]=1
    with pytest.raises(ValueError,match='later-query|unexposed prior'):
        obj.checked_prior(saved,history)


def test_auxiliary_only_chunk_changes_backward_count_not_forward_or_optimizer_work(tmp_path):
    obj=audit.Audit(SimpleNamespace(output=tmp_path));obj.np=np
    offsets=np.array([0,33,*range(34,87)],np.int64)
    weights=np.zeros(86,np.float64);prior=np.zeros_like(weights);prior[32]=1/(54*8)
    history={'episode_offsets':offsets,'weights':weights,'prior_weights':prior}
    orders=[list(range(54))]*80
    mse=obj.expected_batches(history,orders,'mse')
    aux=obj.expected_batches(history,orders,'query_aux')
    assert len(mse)==len(aux)==720
    assert mse[0]['forward_chunks']==aux[0]['forward_chunks']==2
    assert mse[0]['backward_chunks']==0 and aux[0]['backward_chunks']==1
    assert mse[0]['prior_rows']==aux[0]['prior_rows']==8
    assert mse[-1]['optimizer_step']==aux[-1]['optimizer_step']==720


def test_scalar_training_loss_hand_units_mask_weights_and_f32_bound():
    p=np.array([[64,192,999,-999],[0,0,0,128]],np.float32)
    t=np.array([[0,64,999,-999],[0,0,0,0]],np.float32)
    legal=np.array([[True,True,False,False],[True,True,True,True]])
    weights=np.array([1/54,0],np.float64)
    loss,bound=audit.scalar_training_loss(p,t,weights,legal)
    assert loss==.25*float(np.float32(1/54)) and 0<bound<1e-5
    prior,prior_bound=audit.scalar_training_loss(p[1:],t[1:],np.array([1/54]))
    assert prior==.75*float(np.float32(1/54)) and 0<prior_bound<1e-5
    p[1]=np.nan
    assert audit.scalar_training_loss(p,t,weights,legal)==(loss,bound)


def reports():
    p, w, m, ids = metric_fixture()
    one = audit.scalar_metrics(p,w,m,ids)
    for scope in ('full','postcorrection','initial'):
        group = one[scope]['by_regime']['lambda3']
        group['supported_case_count'] = 4
        for age in group['by_age'].values():
            age['supported_case_count'] = 4
        one[scope]['by_regime']['lambda4'] = copy.deepcopy(group)
    models = []
    for family in audit.FAMILIES:
        for seed in audit.SEEDS:
            metric = copy.deepcopy(one)
            is_candidate = family.endswith('separate_aux')
            for scope in ('full','postcorrection','initial'):
                for regime in audit.REGIMES:
                    group = metric[scope]['by_regime'][regime]
                    group['episode_weighted_agreement'] = .48 if family.endswith('shared_aux') else .5
                    group['episode_weighted_raw_gap'] = (7. if family.startswith('innovation_') else 8.) if is_candidate else 8. if family.endswith('shared_aux') else 10.
            prior = {'by_regime':{r:{'episode_weighted_centered_mse':6. if is_candidate else 8. if family.endswith('shared_aux') else 10.} for r in audit.REGIMES}}
            models.append({'family':family,'seed':seed,'metrics':metric,'prior_metrics':prior})
    return models, one


def test_independent_all39_records_have_three_distinct_memberships():
    models, hold = reports()
    rows = audit.continuation_rules(models,hold,technical_complete=True)
    assert len(rows) == 39 and all(r['passes'] for r in rows)
    gates = audit.gate_decisions(rows)
    assert {key:value['total'] for key,value in gates.items()} == {
        'innovation_mechanism':19,'gru_mechanism':19,'architecture':29}
    for row in rows:
        changed = [{**r,'passes':False} if r['name'] == row['name'] else r for r in rows]
        outcomes = audit.gate_decisions(changed)
        for key, value in outcomes.items():
            assert value['passes'] is (row['name'] not in gates[key]['conditions'])
    pending = audit.gate_decisions(audit.continuation_rules(models,hold,technical_complete=False))
    assert all(not v['passes'] for v in pending.values())
    with pytest.raises(ValueError,match='all final'):
        audit.continuation_rules(models[:-1],hold,technical_complete=True)


def test_initial_scope_is_not_full_or_primary_and_preserves_zero_support():
    result = audit.scalar_metrics(*metric_fixture())
    initial = result['initial']['overall']
    assert initial['episode_weighted_agreement'] == .5
    assert initial['episode_weighted_raw_gap'] == 0
    assert initial['nonquery_rows'] == 3 and initial['weight_mass'] == .5
    assert initial['zero_support_episode_ids'] == ['b']
    assert result['full']['overall']['episode_weighted_agreement'] == .25
    assert result['postcorrection']['overall']['episode_weighted_agreement'] == 0


def test_independent_recovery_strict_zero_and_interaction_arithmetic():
    models, hold = reports()
    row = next(r for r in audit.continuation_rules(models,hold,technical_complete=True)
               if r['name'] == 'mechanism.innovation.lambda3.initial.agreement_recovery')
    assert row['value'] == row['threshold'] == .5
    contrasts = audit.factorial_interactions(models)
    assert len(contrasts['per_seed']) == 84 and len(contrasts['means']) == 28
    one = next(r for r in contrasts['means'] if r['architecture'] == 'innovation'
               and r['regime'] == 'lambda3' and r['scope'] == 'postcorrection'
               and r['metric'] == 'episode_weighted_raw_gap')
    assert (one['shared_aux_minus_mse'],one['separate_aux_minus_mse'],one['interaction']) == (-2.,-3.,-1.)
    for model in models:
        for regime in audit.REGIMES:
            model['metrics']['postcorrection']['by_regime'][regime]['episode_weighted_raw_gap'] = 0.
            model['prior_metrics']['by_regime'][regime]['episode_weighted_centered_mse'] = 0.
    failed = [r for r in audit.continuation_rules(models,hold,technical_complete=True) if not r['passes']]
    assert len(failed) == 10 and all(r['relation'] == '<= and <' for r in failed)


@pytest.mark.parametrize('family',audit.FAMILIES)
def test_each_checkpoint_exact_shared_or_separate_tensor_contract(tmp_path,family):
    obj = audit.Audit(SimpleNamespace(output=tmp_path));obj.np = np;obj.run = tmp_path
    architecture,readout,_ = audit.CELLS[family]
    width,inputs = (29,35) if architecture == 'innovation' else (28,40)
    shapes = {'recurrent.weight_ih_l0':(3*width,inputs),'recurrent.weight_hh_l0':(3*width,width),
              'recurrent.bias_ih_l0':(3*width,),'recurrent.bias_hh_l0':(3*width,),
              'output.weight':(4,width),'output.bias':(4,)}
    if architecture == 'innovation':
        shapes['correction.weight'] = (29,4)
    if readout == 'separate':
        shapes.update({'prior_output.weight':(4,width),'prior_output.bias':(4,)})
    saved = {k:np.zeros(shape,np.float32) for k,shape in shapes.items()}
    obj.arrays = lambda path:saved
    obj.checkpoint(family,'fabricated.npz')
    assert sum(v.size for v in saved.values()) == audit.PARAMETERS[family]
    saved['unexpected'] = np.zeros(1,np.float32)
    with pytest.raises(ValueError,match='tensor names'):
        obj.checkpoint(family,'fabricated.npz')


def test_exact_new_cohort_seed_and_recipe_cardinality():
    rows = audit.cohort()
    assert len(rows) == 90 and len({(r['stage'],r['regime'],r['case']) for r in rows}) == 30
    assert sorted({r['seed'] for r in rows if r['stage'] == 'train' and r['regime'] == 'lambda3'}) == list(range(281000001,281000010))
    assert audit.SEEDS == (285000001,285000002,285000003) and audit.SELECTION_START == 286000001
    assert len(audit.FAMILIES)*len(audit.SEEDS)*80*9 == 17280
    assert 13+3*len(audit.FAMILIES)*len(audit.SEEDS) == 85
    assert audit.CONFIG['required_conditions'] == 39
