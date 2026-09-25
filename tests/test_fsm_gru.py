"""Fabricated-only autoregressive GRU equations, causal boundaries and storage."""
import copy

import numpy as np
import pytest
import torch

from openjev.research.fsm_gru import FSMGRU, HORIZONS


def small(dtype=torch.float64,seed=17):
    return FSMGRU(2,3,latent_dim=6,aux_width=5,seed=seed,dtype=dtype)


def wave(shape,offset=0.,dtype=torch.float64):
    return (torch.arange(np.prod(shape),dtype=dtype).reshape(shape)*.19+offset).sin()*.3


def weights(cell):
    return {name:p.detach().numpy().copy() for name,p in cell.state_dict().items()}


def gru_oracle(z,x,w):
    n=z.shape[-1]
    a=x@w['transition.weight_ih'].T+w['transition.bias_ih']
    b=z@w['transition.weight_hh'].T+w['transition.bias_hh']
    r=1/(1+np.exp(-(a[:,:n]+b[:,:n])))
    update=1/(1+np.exp(-(a[:,n:2*n]+b[:,n:2*n])))
    candidate=np.tanh(a[:,2*n:]+r*b[:,2*n:])
    return (1-update)*candidate+update*z


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_teacher_forcing_and_autoregressive_equation_oracle(dtype):
    cell=small(dtype);w=weights(cell)
    y,u,future=wave((2,5,2),dtype=dtype),wave((2,4,3),.5,dtype),wave((2,9,3),1.2,dtype)
    yn,un,fn=y.numpy(),u.numpy(),future.numpy()
    z=np.tanh(yn[:,0]@w['initializer.weight'].T+w['initializer.bias'])
    for t in range(1,5):z=gru_oracle(z,np.concatenate((un[:,t-1],yn[:,t-1]),axis=-1),w)
    packed=np.concatenate((z,yn[:,-1]),axis=-1)
    actual=cell.condition(y,u)
    tolerance=2e-6 if dtype==torch.float32 else 2e-13
    np.testing.assert_allclose(actual.detach().numpy(),packed,rtol=tolerance,atol=tolerance)
    last=yn[:,-1];expected=[]
    for t in range(9):
        z=gru_oracle(z,np.concatenate((fn[:,t],last),axis=-1),w)
        last=z@w['observation.weight'].T+w['observation.bias'];expected.append(last)
    prediction,final=cell.rollout(future,actual)
    np.testing.assert_allclose(prediction.detach().numpy(),np.stack(expected,axis=1),rtol=tolerance,atol=tolerance)
    np.testing.assert_allclose(final.detach().numpy(),np.concatenate((z,last),axis=-1),rtol=tolerance,atol=tolerance)


def test_context_observation_and_input_alignment_then_prediction_feedback(monkeypatch):
    cell=small();y=wave((1,4,2));u=wave((1,3,3),.6);future=wave((1,3,3),1.4)
    calls=[];original=cell.transition.forward
    def spy(x,z):calls.append(x.detach().clone());return original(x,z)
    monkeypatch.setattr(cell.transition,'forward',spy)
    state=cell.condition(y,u)
    for t in range(3):torch.testing.assert_close(calls[t],torch.cat((u[:,t],y[:,t]),dim=-1),rtol=0,atol=0)
    torch.testing.assert_close(state[:,6:],y[:,-1],rtol=0,atol=0)
    prediction,_=cell.rollout(future,state)
    for t in range(3):
        previous=y[:,-1] if t==0 else prediction[:,t-1]
        torch.testing.assert_close(calls[3+t],torch.cat((future[:,t],previous),dim=-1),rtol=0,atol=0)


def test_last_arrived_output_changes_first_prediction_without_future_observation():
    cell=small();y=wave((2,4,2));u=wave((2,3,3),.4);future=wave((2,3,3),1.)
    changed=y.clone();changed[:,-1]+=3.
    original_state,changed_state=cell.condition(y,u),cell.condition(changed,u)
    torch.testing.assert_close(original_state[:,:6],changed_state[:,:6],rtol=0,atol=0)
    assert not torch.equal(original_state[:,6:],changed_state[:,6:])
    original,_=cell.rollout(future,original_state);altered,_=cell.rollout(future,changed_state)
    assert not torch.equal(original[:,0],altered[:,0])
    with pytest.raises(TypeError):cell.rollout(future,original_state,changed)


def test_chunk_state_ownership_and_future_suffix_causality():
    cell=small();y=wave((2,5,2));u=wave((2,4,3));future=wave((2,10,3),.5)
    ycopy,ucopy,fcopy=y.clone(),u.clone(),future.clone()
    state=cell.condition(y,u);saved=state.detach().clone()
    whole,end=cell.rollout(future,state)
    first,middle=cell.rollout(future[:,:3],state);rest,last=cell.rollout(future[:,3:],middle)
    torch.testing.assert_close(torch.cat((first,rest),dim=1),whole,rtol=0,atol=0)
    torch.testing.assert_close(last,end,rtol=0,atol=0)
    one,one_state=cell.step(state,future[:,0])
    torch.testing.assert_close(one,whole[:,0],rtol=0,atol=0)
    torch.testing.assert_close(one_state[:,6:],one,rtol=0,atol=0)
    mutated=future.clone();mutated[:,4:]+=7
    predicted,_=cell.rollout(mutated,state)
    torch.testing.assert_close(predicted[:,:4],whole[:,:4],rtol=0,atol=0)
    for a,b in ((state,saved),(y,ycopy),(u,ucopy),(future,fcopy)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    prior=cell.condition(y[:,:3],u[:,:2]);y[:,3:]+=20;u[:,2:]-=20
    torch.testing.assert_close(cell.condition(y[:,:3],u[:,:2]),prior,rtol=0,atol=0)


@pytest.mark.parametrize('horizon',HORIZONS)
def test_auxiliary_own_latent_block_and_input_prefix_oracle(horizon):
    cell=small();state=wave((2,8)).requires_grad_();future=wave((2,36,3),.7).requires_grad_()
    prediction=cell.aux_decode(state,future,horizon)
    changed=future.detach().clone();changed[:,horizon:]+=100
    changed_state=state.detach().clone();changed_state[:,6:]-=100
    torch.testing.assert_close(cell.aux_decode(changed_state,changed,horizon),prediction,rtol=0,atol=0)
    prediction.square().sum().backward()
    block=HORIZONS.index(horizon)
    for k in range(3):
        if k!=block:assert torch.count_nonzero(state.grad[:,k*2:(k+1)*2])==0
    assert torch.count_nonzero(state.grad[:,6:])==0 and torch.count_nonzero(future.grad[:,horizon:])==0
    w=weights(cell);prefix=f'auxiliary.{horizon}.'
    features=np.concatenate((state.detach().numpy()[:,block*2:(block+1)*2],future.detach().numpy()[:,:horizon].reshape(2,-1)),axis=-1)
    expected=np.tanh(features@w[prefix+'0.weight'].T+w[prefix+'0.bias'])@w[prefix+'2.weight'].T+w[prefix+'2.bias']
    np.testing.assert_allclose(prediction.detach().numpy(),expected,rtol=2e-13,atol=2e-13)


def test_all_parameters_and_caller_inputs_receive_finite_gradients():
    cell=small();y=wave((2,5,2)).requires_grad_();u=wave((2,4,3),.5).requires_grad_()
    future=wave((2,32,3),1.).requires_grad_();state=cell.condition(y,u)
    prediction,_=cell.rollout(future,state)
    loss=(prediction-.7).square().mean()
    for h in HORIZONS:loss=loss+(cell.aux_decode(state,future,h)+.3).square().mean()
    loss.backward()
    for name,p in cell.named_parameters():
        assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all(),name
        assert torch.count_nonzero(p.grad)>0,name
    for value in (y,u,future):assert value.grad is not None and torch.isfinite(value.grad).all()
    assert torch.count_nonzero(y.grad[:,-1])>0


def test_later_autoregressive_decoder_gradient_matches_finite_difference():
    cell=small();y=wave((1,4,2));u=wave((1,3,3),.7);future=wave((1,5,3),1.4)
    def loss():return cell.rollout(future,cell.condition(y,u))[0][:,-1].square().sum()
    loss().backward();actual=cell.observation.weight.grad[0,2].item()
    base=cell.observation.weight[0,2].item();epsilon=1e-6
    with torch.no_grad():cell.observation.weight[0,2]=base+epsilon
    plus=loss().item()
    with torch.no_grad():cell.observation.weight[0,2]=base-epsilon
    minus=loss().item()
    assert actual==pytest.approx((plus-minus)/(2*epsilon),rel=2e-6,abs=1e-9)


def test_repeated_seed_rng_isolation_no_cache_and_boundary_validation_once(monkeypatch):
    before=torch.random.get_rng_state().clone();cell=small();other=small();different=small(seed=18)
    torch.testing.assert_close(torch.random.get_rng_state(),before,rtol=0,atol=0)
    for name,p in cell.state_dict().items():torch.testing.assert_close(p,other.state_dict()[name],rtol=0,atol=0)
    assert not torch.equal(cell.transition.weight_ih,different.transition.weight_ih)
    original=copy.deepcopy(cell.state_dict());keys=set(vars(cell));calls=0;validate=cell._validate_parameters
    def counted():
        nonlocal calls
        calls+=1
        return validate()
    monkeypatch.setattr(cell,'_validate_parameters',counted)
    state=cell.condition(wave((2,5,2)),wave((2,4,3)));assert calls==1
    cell.rollout(wave((2,9,3)),state);assert calls==2
    assert set(vars(cell))==keys|{'_validate_parameters'} and not dict(cell.named_buffers())
    for name,p in cell.state_dict().items():torch.testing.assert_close(p,original[name],rtol=0,atol=0)


@pytest.mark.parametrize('o,i,l,w',[(3,3,72,96),(1,1,6,5),(3,1,12,7),(1,3,6,5)])
def test_exact_counts_packed_state_and_generic_shapes(o,i,l,w):
    cell=FSMGRU(o,i,latent_dim=l,aux_width=w)
    nonaux=3*l*(i+o)+3*l*l+6*l+o*l+o+l*o+l
    aux=sum(w*(l//3+h*i)+w+o*w+o for h in HORIZONS)
    assert cell.parameter_count()==nonaux+aux and cell.parameter_count(include_aux=False)==nonaux
    spec=cell.model_spec()
    assert spec['parameter_count']==nonaux+aux and spec['parameter_bytes']==4*(nonaux+aux)
    assert spec['state_scalars']==l+o and spec['state_bytes_per_stream']==4*(l+o)
    assert spec['buffer_bytes']==0 and spec['persistent_cache'] is False
    if (o,i,l,w)==(3,3,72,96):assert nonaux+aux==37668 and nonaux==17787
    state=cell.condition(wave((2,3,o),dtype=torch.float32),wave((2,2,i),dtype=torch.float32))
    prediction,final=cell.rollout(wave((2,4,i),dtype=torch.float32),state)
    assert prediction.shape==(2,4,o) and final.shape==(2,l+o)


def test_single_observation_empty_rollout_and_batch_invariance():
    cell=small();y=wave((2,1,2));state=cell.condition(y,torch.empty(2,0,3,dtype=torch.float64))
    torch.testing.assert_close(state,torch.cat((torch.tanh(cell.initializer(y[:,0])),y[:,0]),dim=-1),rtol=0,atol=0)
    empty,end=cell.rollout(torch.empty(2,0,3,dtype=torch.float64),state)
    assert empty.shape==(2,0,2) and end.data_ptr()!=state.data_ptr()
    torch.testing.assert_close(end,state,rtol=0,atol=0)
    future=wave((2,3,3));whole,last=cell.rollout(future,state)
    for b in range(2):
        subset,final=cell.rollout(future[b:b+1],state[b:b+1])
        torch.testing.assert_close(subset,whole[b:b+1],rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(final,last[b:b+1],rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('kwargs',[{'obs_dim':0},{'input_dim':True},{'latent_dim':7},{'aux_width':0},
                                   {'seed':-1},{'seed':True},{'dtype':torch.int64}])
def test_constructor_guards(kwargs):
    with pytest.raises(ValueError):FSMGRU(**kwargs)


@pytest.mark.parametrize('case',['extra_u','missing_u','bad_y','empty_batch','empty_context','dtype','nan_y','inf_u',
    'state_shape','state_nan','future_shape','future_inf','short_aux','aux_suffix_nan','horizon','parameter_nan','missing_parameter'])
def test_bad_shapes_dtypes_and_nonfinite_fail_without_repair(case):
    cell=small();y=wave((2,4,2));u=wave((2,3,3));state=wave((2,8));future=wave((2,32,3))
    call=lambda:cell.condition(y,u)
    if case=='extra_u':u=wave((2,4,3))
    elif case=='missing_u':u=u[:,:2]
    elif case=='bad_y':y=y[:,:,:1]
    elif case=='empty_batch':y=y[:0];u=u[:0]
    elif case=='empty_context':y=y[:,:0];u=u[:,:0]
    elif case=='dtype':y=y.float()
    elif case=='nan_y':y[0,0,0]=float('nan')
    elif case=='inf_u':u[0,0,0]=float('inf')
    elif case=='state_shape':call=lambda:cell.step(state[:,:6],future[:,0])
    elif case=='state_nan':state[0,0]=float('nan');call=lambda:cell.rollout(future,state)
    elif case=='future_shape':call=lambda:cell.rollout(future[:,:,:2],state)
    elif case=='future_inf':future[0,0,0]=float('inf');call=lambda:cell.rollout(future,state)
    elif case=='short_aux':call=lambda:cell.aux_decode(state,future[:,:4],8)
    elif case=='aux_suffix_nan':future[0,-1,0]=float('nan');call=lambda:cell.aux_decode(state,future,1)
    elif case=='horizon':call=lambda:cell.aux_decode(state,future,True)
    elif case=='parameter_nan':
        with torch.no_grad():cell.observation.weight[0,0]=float('nan')
    else:cell.observation.register_parameter('bias',None)
    with pytest.raises(ValueError):call()
