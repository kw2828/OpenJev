"""Fabricated integration and decision-rule checks, no measured data."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from openjev.research import fsm_data, fsm_linear
from openjev.research import fsm_study as study


def config():
    return {'context': 100, 'horizon': 128, 'latent': 6, 'aux_width': 4, 'batch': 2,
            'updates': 2, 'learning_rate': .001, 'gradient_clip': 1.,
            'auxiliary_weight': .1, 'fit_timeout_seconds': 30,
            'evaluation_stride': 256, 'linear_orders': [8, 16, 32], 'linear_alphas': [1e-6, .001, .1]}


def fabricated(samples=240):
    n, c, r, p = np.indices((samples, 3, 6, 2))
    return fsm_data.records_from_fixture({
        f'{key}_{amplitude}_train': np.asarray(
            np.sin(n*.071+c*.13+r*.79+p*.33)+(0.2 if key=='y' else 0), dtype=np.float64)
        for amplitude in fsm_data.AMPLITUDES for key in ('u', 'y')})


def test_schedule_deterministic_local_rng_inclusive_endpoint_and_matched_families():
    np.random.seed(77)
    before = np.random.get_state()
    kwargs = {'updates': 20, 'batch': 8, 'records': 12, 'samples': 228, 'context': 100, 'horizon': 128}
    a, b = study.sampling_schedule(9101, **kwargs), study.sampling_schedule(9101, **kwargs)
    np.testing.assert_array_equal(a['record'], b['record'])
    assert np.all(a['start']==0) and a['record'].min()>=0 and a['record'].max()<12
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    assert before[2:]==np.random.get_state()[2:]
    with pytest.raises(ValueError):
        study.sampling_schedule(1, **{**kwargs, 'samples': 227})


def test_training_batch_matches_hand_index_contract_and_cannot_cross_record():
    y = torch.arange(3*300*3, dtype=torch.float32).reshape(3,300,3)
    u = y+10000
    yc, uc, fu, target = study.batch_from_arrays(y,u,[0,2],[0,72],100,128)
    torch.testing.assert_close(yc[0],y[0,:100]); torch.testing.assert_close(yc[1],y[2,72:172])
    torch.testing.assert_close(uc[1],u[2,73:172])
    torch.testing.assert_close(fu[1],u[2,172:300]); torch.testing.assert_close(target[1],y[2,172:300])


@pytest.mark.parametrize('family',study.FAMILIES)
def test_registered_family_one_update_and_causal_full_request(family,tmp_path):
    cfg = config(); data=fabricated(); norm=fsm_data.fit_normalizer(data.partition('fit'))
    train=study.normalized_records(data.partition('fit'),norm)
    y,u=(torch.tensor(np.stack([getattr(r,k) for r in train]),dtype=torch.float32) for k in ('y','u'))
    schedule=study.sampling_schedule(9101,updates=2,batch=2,records=12,samples=240,context=100,horizon=128)
    model,receipt=study.fit_neural(family,9101,y,u,schedule,cfg,tmp_path/family)
    assert receipt['status']=='complete' and receipt['accepted_updates']==2
    assert all(p.grad is None for p in model.parameters())
    record=data.partition('dev')[0]
    pred=study.full_request(model,record,0,cfg,norm)
    altered=record.y.copy(); altered[100:]=1e8
    other=study.full_request(model,replace(record,y=altered),0,cfg,norm)
    np.testing.assert_array_equal(pred,other)
    assert pred.shape==(1,128,3) and np.isfinite(pred).all()
    if family=='selective':
        inputs,_=study.normalized_request(record,[0],100,128,norm)
        assert study.routing_diagnostic(model,inputs).sum()==99


def test_linear_evaluation_integration_and_raw_units(tmp_path):
    cfg=config();data=fabricated();norm=fsm_data.fit_normalizer(data.partition('fit'))
    train=study.normalized_records(data.partition('fit'),norm)
    model=fsm_linear.fit_varx(train,order=8,alpha=.001)
    receipt=study.evaluate(model,{'family':'varx8-ridge0.001'},data.partition('dev'),cfg,norm,tmp_path/'evaluation',linear=True)
    assert len(receipt['rows'])==12 and all(r['status']=='complete' for r in receipt['rows'])
    assert len(receipt['request_ms'])==24 and receipt['timing_error'] is None
    assert receipt['persistent_numeric_bytes']==model.coefficients.nbytes+24+48*8+96


def test_independent_metric_equal_channel_horizon_normalization_and_overflow():
    target=np.zeros((2,4,3)); pred=np.broadcast_to([1.,2.,3.],target.shape)
    result=study.record_metric(pred,target)
    assert result['rmse']==pytest.approx(np.sqrt(14/3))
    assert result['per_channel_rmse']==[1.,2.,3.]
    with pytest.raises(FloatingPointError):
        study.record_metric(np.full_like(target,1e250),target)


def fake_results():
    cfg=config(); ids=[f'{a}-realization-{r}-period-{p}' for a in fsm_data.AMPLITUDES for r in (3,4,5) for p in (0,1)]
    roster=[(f,s) for f in study.FAMILIES for s in study.SEEDS]
    roster += [(f'varx{o}-ridge{a:g}',None) for o in cfg['linear_orders'] for a in cfg['linear_alphas']]
    fits=[];evaluations=[]
    for family,seed in roster:
        fits.append({'family':family,'seed':seed,'status':'complete','accepted_updates':2})
        evaluations.append({'family':family,'seed':seed,
                            'rows':[{'record_id':rid,'status':'complete','rmse':.9 if family=='selective' else 1.} for rid in ids],
                            'timing_error':None,'median_request_ms':1.,'persistent_numeric_bytes':100,
                            'routing_nonzero_counts':[40,30,30] if family=='selective' else None})
    return cfg,fits,evaluations


def test_continuation_requires_strongest_control_full_roster_and_adaptive_routing():
    cfg,fits,ev=fake_results()
    assert study.summarize(ev,fits,cfg)['status']=='DEVELOPMENT_PASS'
    ev[-1]['rows'][0]['rmse']=.01
    assert study.summarize(ev,fits,cfg)['status']=='DEVELOPMENT_FAIL'
    cfg,fits,ev=fake_results()
    ev[0]=ev[1]
    assert not study.summarize(ev,fits,cfg)['conditions']['all_declared_evaluations_complete']
    cfg,fits,ev=fake_results()
    next(e for e in ev if e['family']=='selective')['routing_nonzero_counts']=[100,0,0]
    assert study.summarize(ev,fits,cfg)['status']=='DEVELOPMENT_FAIL'
    cfg,fits,ev=fake_results()
    fits[0]['accepted_updates']=1
    assert not study.summarize(ev,fits,cfg)['conditions']['all_declared_fits_complete']
    cfg,fits,ev=fake_results()
    ev[0]['rows'][0]=ev[0]['rows'][1]
    assert study.summarize(ev,fits,cfg)['status']=='DEVELOPMENT_FAIL'


def test_nonfinite_weight_update_is_retained_as_failed_fit(tmp_path,monkeypatch):
    cfg=config();data=fabricated();norm=fsm_data.fit_normalizer(data.partition('fit'))
    train=study.normalized_records(data.partition('fit'),norm)
    y,u=(torch.tensor(np.stack([getattr(r,k) for r in train]),dtype=torch.float32) for k in ('y','u'))
    schedule=study.sampling_schedule(9101,updates=2,batch=2,records=12,samples=240,context=100,horizon=128)

    def broken_step(optimizer,*args,**kwargs):
        with torch.no_grad():
            optimizer.param_groups[0]['params'][0].flatten()[0]=float('nan')

    monkeypatch.setattr(torch.optim.Adam,'step',broken_step)
    model,receipt=study.fit_neural('dense',9101,y,u,schedule,cfg,tmp_path/'failed')
    assert receipt['status']=='failed' and receipt['accepted_updates']==0
    assert receipt['error']['type']=='FloatingPointError'
    assert (tmp_path/'failed'/'final.pt').exists() and (tmp_path/'failed'/'receipt.json').exists()
    assert all(p.grad is None for p in model.parameters())


def test_diagnostic_failure_retains_exactly_one_failed_row_per_record(tmp_path,monkeypatch):
    cfg=config();data=fabricated();norm=fsm_data.fit_normalizer(data.partition('fit'))
    model=study.model_for('selective',9101,cfg)

    def failed(*args,**kwargs):
        raise RuntimeError('deliberate diagnostic failure')

    monkeypatch.setattr(study,'routing_diagnostic',failed)
    receipt=study.evaluate(model,{'family':'selective','seed':9101},data.partition('dev'),cfg,norm,tmp_path/'evaluation')
    assert len(receipt['rows'])==12 and len({r['record_id'] for r in receipt['rows']})==12
    assert all(r['status']=='failed' for r in receipt['rows'])
