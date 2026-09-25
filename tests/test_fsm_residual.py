"""Fabricated frozen-VARX equations, propagation, causal boundaries and precision."""
import copy

import numpy as np
import pytest
import torch

from openjev.research.fsm_linear import FIT_IDS, VARXModel
from openjev.research.fsm_linear import predict as varx_predict
from openjev.research.fsm_residual import MODES, RESIDUAL_KINDS, FSMResidual


def coefficients(p):
    value=np.zeros((3,6*p+4),dtype=np.float64)
    value[:,3*(p-1):3*p]=np.array([[.35,.02,0],[-.01,.3,.02],[0,.01,.25]])
    value[:,3*p:3*p+3]=np.eye(3)*.2
    value[:,6*p:6*p+3]=np.eye(3)*.05
    value[:,-1]=[.01,-.02,.03]
    return value


def wave(shape,offset=0.):
    return torch.sin(torch.arange(int(np.prod(shape)),dtype=torch.float64).reshape(shape)*.17+offset)*.3


def model(p=2,mode='output_only',kind='tanh',seed=17):
    return FSMResidual(coefficients(p),order=p,mode=mode,residual_kind=kind,seed=seed)


def activate(cell):
    with torch.no_grad():
        if cell.residual_kind=='tanh':
            cell.residual[2].weight.copy_(wave(tuple(cell.residual[2].weight.shape),.8)*.2)
            cell.residual[2].bias.copy_(torch.tensor([.02,-.03,.04],dtype=torch.float64))
        else:
            cell.residual.weight.copy_(wave(tuple(cell.residual.weight.shape),.8)*.2)
            cell.residual.bias.copy_(torch.tensor([.02,-.03,.04],dtype=torch.float64))
    return cell


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('p',[1,8,32])
def test_zero_head_matches_original_float64_varx_for_full_horizon(kind,mode,p):
    co=np.asfortranarray(coefficients(p))
    cell=FSMResidual(co,order=p,mode=mode,residual_kind=kind)
    y,u,future=wave((2,p+4,3)),wave((2,p+3,3),.3),wave((2,128,3),.7)
    reference=VARXModel(co,p,.001,12,FIT_IDS)
    expected=varx_predict(reference,y.numpy(),u.numpy(),future.numpy())
    actual=cell.predict(y,u,future)
    np.testing.assert_allclose(actual.detach().numpy(),expected,rtol=2e-14,atol=2e-14)
    assert actual.dtype==torch.float64
    assert all(p.dtype==torch.float64 for p in cell.parameters())


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
@pytest.mark.parametrize('mode',MODES)
def test_independent_nonzero_residual_and_lag_recursion_oracle(kind,mode):
    cell=activate(model(mode=mode,kind=kind));p=cell.order
    y,u,future=wave((2,6,3)),wave((2,5,3),.3),wave((2,7,3),.7)
    w={name:value.detach().numpy().copy() for name,value in cell.state_dict().items()}
    past_y,past_u=y.numpy()[:,-p:].copy(),u.numpy()[:,-p:].copy();expected=[]
    for t in range(7):
        x=np.concatenate((past_y.reshape(2,-1),future.numpy()[:,t],past_u.reshape(2,-1)),axis=1)
        base=x@w['coefficients'][:,:-1].T+w['coefficients'][:,-1]
        if kind=='tanh':
            hidden=np.tanh(x@w['residual.0.weight'].T+w['residual.0.bias'])
            correction=hidden@w['residual.2.weight'].T+w['residual.2.bias']
        else:correction=x@w['residual.weight'].T+w['residual.bias']
        prediction=base+correction;expected.append(prediction)
        inserted=base if mode=='output_only' else prediction
        past_y=np.concatenate((past_y[:,1:],inserted[:,None]),axis=1)
        past_u=np.concatenate((past_u[:,1:],future.numpy()[:,t:t+1]),axis=1)
    actual,state=cell.rollout(future,cell.condition(y,u))
    np.testing.assert_allclose(actual.detach().numpy(),np.stack(expected,axis=1),rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(state.detach().numpy(),np.concatenate((past_y.reshape(2,-1),past_u.reshape(2,-1)),axis=1),rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
def test_only_corrected_output_propagation_distinguishes_modes(kind):
    co=np.zeros((3,10),dtype=np.float64);co[:,:3]=np.eye(3)*.5
    output=FSMResidual(co,order=1,residual_kind=kind)
    feedback=FSMResidual(co,order=1,residual_kind=kind,mode='feedback')
    for cell in (output,feedback):
        with torch.no_grad():
            head=cell.residual[2] if kind=='tanh' else cell.residual
            head.bias.fill_(1.)
    state=torch.zeros((1,6),dtype=torch.float64);future=torch.zeros((1,3,3),dtype=torch.float64)
    a,sa=output.rollout(future,state);b,sb=feedback.rollout(future,state)
    torch.testing.assert_close(a,torch.ones_like(a),rtol=0,atol=0)
    torch.testing.assert_close(b,torch.tensor([[[1.]*3,[1.5]*3,[1.75]*3]],dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(sa[:,:3],torch.zeros((1,3),dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(sb[:,:3],b[:,-1],rtol=0,atol=0)
    assert set(output.state_dict())==set(feedback.state_dict())
    for key in output.state_dict():assert torch.equal(output.state_dict()[key],feedback.state_dict()[key])


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
def test_feature_alignment_uses_current_input_and_chronological_lags(kind,monkeypatch):
    cell=model(p=2,kind=kind)
    y=torch.arange(18,dtype=torch.float64).reshape(1,6,3)
    u=100+torch.arange(15,dtype=torch.float64).reshape(1,5,3)
    future=200+torch.arange(9,dtype=torch.float64).reshape(1,3,3)
    calls=[];original=cell.residual.forward
    def spy(x):calls.append(x.detach().clone());return original(x)
    monkeypatch.setattr(cell.residual,'forward',spy)
    state=cell.condition(y,u);prediction,_=cell.rollout(future,state)
    torch.testing.assert_close(state,torch.cat((y[:,-2:].reshape(1,-1),u[:,-2:].reshape(1,-1)),dim=1),rtol=0,atol=0)
    for t in range(3):
        past_y=torch.cat((y[:,-2:],prediction[:,:t]),dim=1)[:,-2:]
        past_u=torch.cat((u[:,-2:],future[:,:t]),dim=1)[:,-2:]
        expected=torch.cat((past_y.reshape(1,-1),future[:,t],past_u.reshape(1,-1)),dim=1)
        torch.testing.assert_close(calls[t],expected,rtol=0,atol=0)
    old_y,old_u=y.clone(),u.clone();old_y[:,:-2]+=999;old_u[:,:-2]-=999
    assert torch.equal(cell.condition(old_y,old_u),state)
    with pytest.raises(TypeError):cell.rollout(future,state,y)


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
@pytest.mark.parametrize('mode',MODES)
def test_causal_suffix_chunk_batch_and_owned_state_without_input_mutation(kind,mode):
    cell=activate(model(mode=mode,kind=kind))
    y,u,future=wave((2,6,3)),wave((2,5,3),.4),wave((2,9,3),.8)
    originals=[x.clone() for x in (y,u,future)]
    state=cell.condition(y,u);saved=state.clone();weights={k:v.clone() for k,v in cell.state_dict().items()}
    whole,end=cell.rollout(future,state)
    first,middle=cell.rollout(future[:,:4],state);rest,last=cell(future[:,4:],middle)
    torch.testing.assert_close(torch.cat((first,rest),dim=1),whole,rtol=0,atol=0)
    torch.testing.assert_close(last,end,rtol=0,atol=0)
    for i in range(2):
        single,single_end=cell.rollout(future[i:i+1],state[i:i+1])
        torch.testing.assert_close(single,whole[i:i+1],rtol=2e-14,atol=2e-14)
        torch.testing.assert_close(single_end,end[i:i+1],rtol=2e-14,atol=2e-14)
    altered=future.clone();altered[:,5:]+=7
    changed,_=cell.rollout(altered,state)
    torch.testing.assert_close(changed[:,:5],whole[:,:5],rtol=0,atol=0)
    one,one_end=cell.step(state,future[:,0]);prefix,prefix_end=cell.rollout(future[:,:1],state)
    torch.testing.assert_close(one,prefix[:,0],rtol=0,atol=0)
    torch.testing.assert_close(one_end,prefix_end,rtol=0,atol=0)
    empty,empty_end=cell.rollout(future[:,:0],state)
    assert empty.shape==(2,0,3) and empty_end.data_ptr()!=state.data_ptr()
    for a,b in zip((y,u,future,state),(*originals,saved),strict=True):assert torch.equal(a,b)
    for key,value in cell.state_dict().items():assert torch.equal(value,weights[key])
    with torch.no_grad():state.add_(11)
    for a,b in zip((y,u,future),originals,strict=True):assert torch.equal(a,b)


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
def test_seed_pairing_local_rng_and_coefficient_ownership(kind):
    before=torch.get_rng_state().clone();numpy_before=np.random.get_state()
    a=model(kind=kind,seed=31);b=model(kind=kind,seed=31,mode='feedback');c=model(kind=kind,seed=32)
    assert torch.equal(before,torch.get_rng_state())
    after=np.random.get_state()
    assert all(np.array_equal(x,y) for x,y in zip(numpy_before,after,strict=True))
    for key in a.state_dict():assert torch.equal(a.state_dict()[key],b.state_dict()[key])
    if kind=='tanh':assert not torch.equal(a.residual[0].weight,c.residual[0].weight)
    else:assert all(torch.count_nonzero(p)==0 for p in a.parameters())
    co=coefficients(2);owned=FSMResidual(co,order=2,residual_kind=kind);co[:]=123
    assert torch.equal(owned.coefficients,torch.from_numpy(coefficients(2)))
    tensor=torch.from_numpy(coefficients(2)).requires_grad_();owned=FSMResidual(tensor,order=2,residual_kind=kind)
    with torch.no_grad():tensor.fill_(99)
    assert owned.coefficients.grad_fn is None and not owned.coefficients.requires_grad
    assert torch.equal(owned.coefficients,torch.from_numpy(coefficients(2)))


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
@pytest.mark.parametrize('mode',MODES)
def test_frozen_coefficients_zero_head_and_residual_only_gradient_ownership(kind,mode):
    cell=model(mode=mode,kind=kind);saved=cell.coefficients.clone()
    y,u,future=wave((2,6,3)),wave((2,5,3),.4),wave((2,7,3),.8)
    optimizer=torch.optim.Adam(cell.parameters(),lr=.001)
    names=set(dict(cell.named_parameters()))
    expected={'residual.0.weight','residual.0.bias','residual.2.weight','residual.2.bias'} if kind=='tanh' else {'residual.weight','residual.bias'}
    assert names==expected and set(dict(cell.named_buffers()))=={'coefficients'}
    assert not optimizer.state
    prediction=cell.predict(y,u,future);(prediction-.4).square().mean().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in cell.parameters())
    if kind=='tanh':
        assert torch.count_nonzero(cell.residual[0].weight.grad)==0
        assert torch.count_nonzero(cell.residual[0].bias.grad)==0
        assert torch.count_nonzero(cell.residual[2].weight.grad)>0
        assert torch.count_nonzero(cell.residual[2].bias.grad)>0
    else:assert all(torch.count_nonzero(p.grad)>0 for p in cell.parameters())
    assert cell.coefficients.grad is None
    optimizer.step();assert torch.equal(cell.coefficients,saved)
    assert set(optimizer.state)==set(cell.parameters())
    optimizer.zero_grad(set_to_none=True)
    cell.predict(y,u,future).square().sum().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in cell.parameters())
    if kind=='tanh':assert torch.count_nonzero(cell.residual[0].weight.grad)>0


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
@pytest.mark.parametrize('mode',MODES)
def test_later_output_gradients_match_finite_difference(kind,mode):
    cell=activate(model(mode=mode,kind=kind));y,u,future=wave((1,5,3)),wave((1,4,3),.4),wave((1,6,3),.8)
    parameter=cell.residual[2].weight if kind=='tanh' else cell.residual.weight
    value=cell.predict(y,u,future)[0,-1,0];value.backward();actual=parameter.grad[0,0].item()
    baseline=parameter[0,0].item();epsilon=1e-6
    with torch.no_grad():
        parameter[0,0]=baseline+epsilon;plus=cell.predict(y,u,future)[0,-1,0].item()
        parameter[0,0]=baseline-epsilon;minus=cell.predict(y,u,future)[0,-1,0].item()
        parameter[0,0]=baseline
    assert actual==pytest.approx((plus-minus)/(2*epsilon),rel=2e-6,abs=2e-8)


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
@pytest.mark.parametrize('p',[1,8,32])
def test_exact_parameter_buffer_and_state_counts(kind,p):
    cell=model(p=p,kind=kind);spec=cell.model_spec()
    expected=24*(6*p+7)+3 if kind=='tanh' else 18*p+12
    assert cell.parameter_count()==expected==spec['trainable_parameter_count']==spec['parameter_count']
    assert spec['frozen_parameter_count']==0 and spec['frozen_coefficient_count']==3*(6*p+4)
    assert spec['parameter_bytes']==8*expected and spec['buffer_bytes']==8*3*(6*p+4)
    assert spec['state_scalars']==6*p and spec['state_bytes_per_stream']==48*p
    meta=24 if kind=='tanh' else 16
    assert spec['numeric_metadata_bytes']==meta
    assert spec['retained_numeric_bytes']==8*(expected+3*(6*p+4))+meta
    assert spec['persistent_numeric_bytes_per_stream']==spec['retained_numeric_bytes']+48*p
    assert spec['residual_kind']==kind and not spec['persistent_cache'] and not spec['retained_trajectory']
    assert set(cell.__dict__) >= {'order','mode','residual_kind','initialization_seed'}
    assert set(cell._buffers)=={'coefficients'}
    if p==32:assert expected==(4779 if kind=='tanh' else 588)


@pytest.mark.parametrize('change',[
    {'order':0},{'order':100},{'order':True},{'mode':'other'},{'residual_kind':'identity'},
    {'hidden_width':0},{'hidden_width':True},{'seed':-1},{'seed':True},
])
def test_constructor_rejects_invalid_configuration(change):
    arguments={'order':2};arguments.update(change)
    with pytest.raises(ValueError):FSMResidual(coefficients(2),**arguments)


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
@pytest.mark.parametrize('case',['float32','shape','nan','list','tensor32'])
def test_constructor_rejects_invalid_coefficients(kind,case):
    co=coefficients(2)
    if case=='float32':co=co.astype(np.float32)
    elif case=='shape':co=co[:,:-1]
    elif case=='nan':co[0,0]=np.nan
    elif case=='list':co=co.tolist()
    else:co=torch.from_numpy(co).float()
    with pytest.raises(ValueError):FSMResidual(co,order=2,residual_kind=kind)


@pytest.mark.parametrize('case',['y32','u32','u_count','short','batch0','ynan','unonfinite','yshape','u_list'])
def test_condition_rejects_malformed_or_nonfinite_public_inputs(case):
    cell=model();y,u=wave((2,6,3)),wave((2,5,3))
    if case=='y32':y=y.float()
    elif case=='u32':u=u.float()
    elif case=='u_count':u=u[:,:-1]
    elif case=='short':y,u=y[:,:2],u[:,:1]
    elif case=='batch0':y,u=y[:0],u[:0]
    elif case=='ynan':y[0,0,0]=float('nan')
    elif case=='unonfinite':u[0,0,0]=float('inf')
    elif case=='yshape':y=y[:,:,0]
    else:u=u.tolist()
    with pytest.raises(ValueError):cell.condition(y,u)


@pytest.mark.parametrize('case',['state32','state_width','state0','statenan','future32','future_batch','future_channels','future_rank','futureinf'])
def test_rollout_rejects_malformed_or_nonfinite_inputs(case):
    cell=model();state=wave((2,12));future=wave((2,3,3))
    if case=='state32':state=state.float()
    elif case=='state_width':state=state[:,:-1]
    elif case=='state0':state,future=state[:0],future[:0]
    elif case=='statenan':state[0,0]=float('nan')
    elif case=='future32':future=future.float()
    elif case=='future_batch':future=future[:1]
    elif case=='future_channels':future=future[:,:,:2]
    elif case=='future_rank':future=future[:,0]
    else:future[0,0,0]=float('inf')
    with pytest.raises(ValueError):cell.rollout(future,state)


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
@pytest.mark.parametrize('case',['parameter_nan','coefficient_nan','buffer_grad','float32','extra_parameter','extra_buffer'])
def test_mutated_model_contract_fails_before_forecast(kind,case):
    cell=model(kind=kind)
    with torch.no_grad():
        if case=='parameter_nan':next(cell.parameters()).flatten()[0]=float('nan')
        elif case=='coefficient_nan':cell.coefficients[0,0]=float('nan')
        elif case=='buffer_grad':cell.coefficients.requires_grad_(True)
        elif case=='float32':cell.float()
        elif case=='extra_parameter':cell.register_parameter('hidden',torch.nn.Parameter(torch.ones(1,dtype=torch.float64)))
        else:cell.register_buffer('hidden',torch.ones(1,dtype=torch.float64))
    with pytest.raises(ValueError):cell.rollout(wave((1,2,3)),wave((1,12)))


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
def test_finite_input_overflow_raises_without_fallback_or_clipping(kind):
    co=np.zeros((3,10),dtype=np.float64);co[:,3:6]=np.eye(3)*np.finfo(np.float64).max
    cell=FSMResidual(co,order=1,residual_kind=kind)
    state=torch.zeros((1,6),dtype=torch.float64)
    with pytest.raises(ValueError,match='nonfinite residual forecast; no repair'):
        cell.rollout(torch.full((1,2,3),2.,dtype=torch.float64),state)
    with pytest.raises(ValueError,match='nonfinite residual forecast; no repair'):
        cell.step(state,torch.full((1,3),2.,dtype=torch.float64))
    assert torch.count_nonzero(state)==0


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
def test_all_forecasts_checked_even_if_a_later_state_recovers(kind,monkeypatch):
    cell=model(kind=kind);state=wave((1,12));future=wave((1,2,3));calls=[]
    def fake_step(current,u):
        prediction=torch.zeros((1,3),dtype=torch.float64)
        if not calls:prediction[0,0]=float('nan')
        calls.append(1)
        return prediction,current.clone()
    monkeypatch.setattr(cell,'_step',fake_step)
    with pytest.raises(ValueError,match='nonfinite residual forecast; no repair'):
        cell.rollout(future,state)
    assert len(calls)==2


@pytest.mark.parametrize('kind',RESIDUAL_KINDS)
def test_outer_boundaries_validate_once_not_per_forecast_step(kind,monkeypatch):
    cell=model(kind=kind);original=cell._validate_parameters;calls=[]
    def spy():calls.append(1);return original()
    monkeypatch.setattr(cell,'_validate_parameters',spy)
    output_calls=[];finite=cell._finite_result
    def output_spy(prediction,state):output_calls.append(1);return finite(prediction,state)
    monkeypatch.setattr(cell,'_finite_result',output_spy)
    state=cell.condition(wave((1,5,3)),wave((1,4,3)))
    assert len(calls)==1
    cell.rollout(wave((1,17,3)),state);assert len(calls)==2
    cell.step(state,wave((1,3)));assert len(calls)==3
    assert len(output_calls)==2
    clone=copy.deepcopy(cell)
    assert set(clone.state_dict())==set(cell.state_dict())
