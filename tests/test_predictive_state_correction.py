"""Fabricated mathematical/causal qualification, with no measured inputs."""
import copy

import numpy as np
import pytest
import torch

from openjev.research.predictive_state_correction import HORIZONS, MODES, PredictiveStateCorrection


def model(mode='dense', dtype=torch.float64):
    return PredictiveStateCorrection(2, 2, mode=mode, latent_dim=6, aux_width=5, seed=91, dtype=dtype)


def wave(shape, dtype=torch.float64, offset=0.):
    return (torch.arange(np.prod(shape), dtype=dtype).reshape(shape)*.17+offset).sin()*.3


def weights(cell):
    return {k: v.detach().numpy().copy() for k,v in cell.state_dict().items()}


def gru_oracle(state, inputs, w):
    n = state.shape[-1]
    a = inputs@w['transition.weight_ih'].T+w['transition.bias_ih']
    b = state@w['transition.weight_hh'].T+w['transition.bias_hh']
    r = 1/(1+np.exp(-(a[:, :n]+b[:, :n])))
    z = 1/(1+np.exp(-(a[:, n:2*n]+b[:, n:2*n])))
    candidate = np.tanh(a[:, 2*n:]+r*b[:, 2*n:])
    return (1-z)*candidate+z*state


def correction_oracle(prior, observed, matrix, bias, mode):
    error = observed-prior@matrix.T-bias
    gradient = error@matrix
    selected = []
    for row in range(len(prior)):
        if mode == 'dense':
            selected.append([0,1,2])
        elif mode in ('fixed0','fixed1','fixed2'):
            selected.append([int(mode[-1])])
        else:
            blocks = np.split(gradient[row], 3)
            k = max(range(3), key=lambda i: float(sum(v*v for v in blocks[i])))
            selected.append([(k+1)%3 if mode == 'rewired' else k])
    result = prior.copy()
    denominator = sum(float(x*x) for x in matrix.ravel())+1e-6
    width = prior.shape[-1]//3
    for b in range(len(prior)):
        for block in selected[b]:
            for d in range(block*width, (block+1)*width):
                result[b,d] += gradient[b,d]/denominator
    return result


@pytest.mark.parametrize('mode', MODES)
def test_independent_gru_correction_and_eight_step_rollout_oracle(mode):
    cell = model(mode); w = weights(cell)
    y, u, future = wave((2,5,2)), wave((2,4,2),offset=.7), wave((2,8,2),offset=1.3)
    yn,un,fn = y.numpy(),u.numpy(),future.numpy()
    state = np.tanh(yn[:,0]@w['initializer.weight'].T+w['initializer.bias'])
    for t in range(1,5):
        prior = gru_oracle(state,un[:,t-1],w)
        state = correction_oracle(prior,yn[:,t],w['observation.weight'],w['observation.bias'],mode)
    actual = cell.condition(y,u)
    np.testing.assert_allclose(actual.detach().numpy(),state,rtol=2e-13,atol=2e-13)
    expected = []
    for t in range(8):
        state = gru_oracle(state,fn[:,t],w)
        expected.append(state@w['observation.weight'].T+w['observation.bias'])
    prediction,final = cell.rollout(future,actual)
    np.testing.assert_allclose(prediction.detach().numpy(),np.stack(expected,axis=1),rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(final.detach().numpy(),state,rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_fixed_linear_observation_residual_cannot_increase(mode,dtype):
    cell = model(mode,dtype)
    with torch.no_grad():
        cell.observation.weight.copy_(torch.tensor([[1,2,0,3,0,4],[0,-1,1,0,2,1]],dtype=dtype))
        cell.observation.bias.copy_(torch.tensor([.2,-.1],dtype=dtype))
    prior,observed = wave((13,6),dtype,offset=.9),wave((13,2),dtype,offset=2.1)
    before = (observed-cell.observation(prior)).square().sum(-1)
    corrected = cell.correct(prior,observed)
    after = (observed-cell.observation(corrected)).square().sum(-1)
    assert bool((after<=before+1e-5 if dtype==torch.float32 else after<=before+1e-12).all())
    np.testing.assert_allclose(corrected.detach().numpy(),correction_oracle(prior.numpy(),observed.numpy(),
        cell.observation.weight.detach().numpy(),cell.observation.bias.detach().numpy(),mode),rtol=2e-6,atol=2e-7)
    assert bool((after<before).any())


@pytest.mark.parametrize('mode,expected_block', [('dense',None),('selective',0),('rewired',1),
                                                   ('fixed0',0),('fixed1',1),('fixed2',2)])
def test_lowest_tie_and_dense_mask(mode,expected_block):
    cell = PredictiveStateCorrection(1,1,mode=mode,latent_dim=6,aux_width=3,dtype=torch.float64)
    with torch.no_grad():
        cell.observation.weight.copy_(torch.tensor([[1.,0.,1.,0.,1.,0.]],dtype=torch.float64))
        cell.observation.bias.zero_()
    result = cell.correct(torch.zeros(1,6,dtype=torch.float64),torch.ones(1,1,dtype=torch.float64))
    active = [i for i in range(3) if bool((result[0,2*i:2*i+2]!=0).any())]
    assert active == ([0,1,2] if expected_block is None else [expected_block])


def test_rewire_wrap_uses_destination_own_gradient_and_is_per_example():
    cell = model('rewired')
    with torch.no_grad():
        cell.observation.weight.copy_(torch.tensor([[1.,0.,2.,0.,4.,0.],[0.,5.,0.,1.,0.,2.]],dtype=torch.float64))
        cell.observation.bias.zero_()
    prior = torch.zeros(2,6,dtype=torch.float64)
    observed = torch.eye(2,dtype=torch.float64)
    got = cell.correct(prior,observed)
    denominator = 51.+1e-6
    expected = torch.zeros_like(got); expected[0,0] = 1/denominator; expected[1,3] = 1/denominator
    torch.testing.assert_close(got,expected,rtol=1e-14,atol=1e-14)
    assert got[0,0] != 4/denominator  # Moving the winning block's gradient would be wrong.


@pytest.mark.parametrize('mode,expected_block', [('selective',1),('rewired',2)])
@pytest.mark.parametrize('obs_dim',[1,2])
def test_scalar_and_rank_one_decoders_have_fixed_nonzero_gradient_route(mode,expected_block,obs_dim):
    cell=PredictiveStateCorrection(obs_dim,1,mode=mode,latent_dim=6,aux_width=3,dtype=torch.float64)
    vector=torch.tensor([1.,0.,3.,0.,2.,0.],dtype=torch.float64)
    matrix=vector[None] if obs_dim==1 else torch.stack((vector,2*vector))
    errors=(torch.tensor([[1.],[-3.],[.25]],dtype=torch.float64) if obs_dim==1 else
            torch.tensor([[1.,0.],[-3.,0.],[0.,.5],[2.,-.5]],dtype=torch.float64))
    with torch.no_grad():
        cell.observation.weight.copy_(matrix);cell.observation.bias.zero_()
    result=cell.correct(torch.zeros(len(errors),6,dtype=torch.float64),errors)
    active=result.reshape(len(errors),3,2).ne(0).any(-1)
    expected=torch.zeros_like(active);expected[:,expected_block]=True
    torch.testing.assert_close(active,expected,rtol=0,atol=0)
    if obs_dim==2:
        # A nonzero residual orthogonal to the decoder column space gives no update.
        null_error=torch.tensor([[2.,-1.]],dtype=torch.float64)
        assert torch.count_nonzero(cell.correct(torch.zeros(1,6,dtype=torch.float64),null_error))==0
    assert 'rank-one' in cell.model_spec()['routing_limitation']


def test_non_rank_one_decoder_can_route_distinct_residual_directions_differently():
    cell=model('selective')
    with torch.no_grad():
        cell.observation.weight.copy_(torch.tensor([[1.,0.,2.,0.,4.,0.],[0.,5.,0.,1.,0.,2.]],dtype=torch.float64))
        cell.observation.bias.zero_()
    result=cell.correct(torch.zeros(2,6,dtype=torch.float64),torch.eye(2,dtype=torch.float64))
    active=result.reshape(2,3,2).ne(0).any(-1)
    torch.testing.assert_close(active,torch.tensor([[False,False,True],[True,False,False]]),rtol=0,atol=0)


@pytest.mark.parametrize('mode,block',[('fixed0',0),('fixed1',1),('fixed2',2)])
def test_fixed_modes_ignore_argmax_and_zero_own_gradient_despite_other_blocks(monkeypatch,mode,block):
    cell=model(mode)
    def forbidden_argmax(*args,**kwargs):
        raise AssertionError('fixed correction must not score/choose another block')
    monkeypatch.setattr(torch.Tensor,'argmax',forbidden_argmax)
    with torch.no_grad():
        cell.observation.weight.fill_(2.)
        cell.observation.weight[:,block*2:(block+1)*2].zero_()
        cell.observation.bias.zero_()
    prior=torch.zeros(2,6,dtype=torch.float64)
    result=cell.correct(prior,torch.tensor([[1.,2.],[-3.,1.]],dtype=torch.float64))
    torch.testing.assert_close(result,prior,rtol=0,atol=0)
    spec=cell.model_spec()
    assert spec['fixed_block']==block and spec['block_energy_scoring'] is False
    assert spec['buffer_bytes']==0 and not dict(cell.named_buffers())


@pytest.mark.parametrize('mode,block',[('fixed0',0),('fixed1',1),('fixed2',2)])
def test_fixed_modes_do_not_compute_overflowing_gradient_energies(mode,block):
    cell=PredictiveStateCorrection(1,1,mode=mode,latent_dim=6,aux_width=3)
    with torch.no_grad():
        cell.observation.weight.fill_(1.);cell.observation.bias.zero_()
    prior=torch.zeros(1,6);observed=torch.full((1,1),1e20)
    result=cell.correct(prior,observed)
    expected=torch.zeros_like(result);expected[:,block*2:(block+1)*2]=observed/(6.+1e-6)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
    assert torch.isfinite(result).all()
    # Scored controls still reject their genuinely overflowing energies.
    cell.mode='selective'
    with pytest.raises(ValueError,match='block energy'):
        cell.correct(prior,observed)


@pytest.mark.parametrize('mode',MODES)
def test_zero_decoder_is_identity_with_finite_gradient(mode):
    cell = model(mode)
    with torch.no_grad(): cell.observation.weight.zero_()
    prior = wave((2,6)).requires_grad_(); observed = wave((2,2),offset=1.)
    result = cell.correct(prior,observed)
    torch.testing.assert_close(result,prior,rtol=0,atol=0)
    result.square().sum().backward()
    assert torch.isfinite(cell.observation.weight.grad).all()


@pytest.mark.parametrize('mode',MODES)
def test_denominator_and_decoder_gradients_match_independent_finite_difference(mode):
    cell = model(mode)
    prior,observed = wave((2,6),offset=.6),wave((2,2),offset=1.9)
    result = cell.correct(prior,observed)
    coefficient = torch.arange(1,13,dtype=torch.float64).reshape(2,6)
    (result.square()*coefficient).sum().backward()
    matrix = cell.observation.weight.detach().numpy().copy(); bias = cell.observation.bias.detach().numpy()
    h = 1e-6
    for index in ((0,0),(1,3),(0,5)):
        plus,minus = matrix.copy(),matrix.copy();plus[index]+=h;minus[index]-=h
        def loss(c):
            z=correction_oracle(prior.numpy(),observed.numpy(),c,bias,mode)
            return float(np.sum(z*z*coefficient.numpy()))
        expected=(loss(plus)-loss(minus))/(2*h)
        assert float(cell.observation.weight.grad[index]) == pytest.approx(expected,rel=2e-6,abs=2e-8)


@pytest.mark.parametrize('mode',MODES)
def test_all_modules_receive_finite_training_gradients(mode):
    cell=model(mode)
    observed=wave((2,5,2)).requires_grad_(); context=wave((2,4,2),offset=.4).requires_grad_()
    future=wave((2,32,2),offset=.8).requires_grad_()
    state=cell.condition(observed,context)
    prediction,_=cell.rollout(future,state)
    loss=(prediction-.37).square().mean()
    for h in HORIZONS: loss=loss+(cell.aux_decode(state,future,h)+.11).square().mean()
    loss.backward()
    for name,p in cell.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()),name
    for name in ('transition.weight_hh','transition.weight_ih','observation.weight','initializer.weight'):
        assert bool((dict(cell.named_parameters())[name].grad!=0).any()),name
    assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in (observed,context,future))


@pytest.mark.parametrize('mode',MODES)
def test_chunks_state_carry_and_forecast_suffix_causality(mode):
    cell=model(mode);state=cell.condition(wave((2,4,2)),wave((2,3,2)))
    original_state=state.detach().clone();future=wave((2,9,2),offset=.8)
    all_predictions,final=cell.rollout(future,state)
    first,carried=cell.rollout(future[:,:3],state);rest,last=cell.rollout(future[:,3:],carried)
    torch.testing.assert_close(torch.cat((first,rest),1),all_predictions,rtol=0,atol=0)
    torch.testing.assert_close(last,final,rtol=0,atol=0)
    altered=future.clone();altered[:,4:]+=2
    changed,_=cell.rollout(altered,state)
    torch.testing.assert_close(changed[:,:4],all_predictions[:,:4],rtol=0,atol=0)
    torch.testing.assert_close(state,original_state,rtol=0,atol=0)
    single,next_state=cell.step(state,future[:,0])
    torch.testing.assert_close(single,all_predictions[:,0],rtol=0,atol=0)
    torch.testing.assert_close(cell.rollout(future[:,:1],state)[1],next_state,rtol=0,atol=0)


def test_condition_alignment_and_prefix_causality_with_recording_spy(monkeypatch):
    cell=model('selective');observed=wave((1,6,2));inputs=wave((1,5,2),offset=1.1)
    calls=[];forward=cell.transition.forward
    def spy(u,state): calls.append(u.detach().clone());return forward(u,state)
    monkeypatch.setattr(cell.transition,'forward',spy)
    prefix=cell.condition(observed[:,:4],inputs[:,:3])
    assert len(calls)==3
    for t,u in enumerate(calls): torch.testing.assert_close(u,inputs[:,t],rtol=0,atol=0)
    observed[:,4:]+=10;inputs[:,3:]-=20
    torch.testing.assert_close(cell.condition(observed[:,:4],inputs[:,:3]),prefix,rtol=0,atol=0)
    calls.clear();state=cell.condition(observed,inputs)
    assert len(calls)==5
    future=wave((1,1,2),offset=4.)
    cell.rollout(future,state)
    torch.testing.assert_close(calls[-1],future[:,0],rtol=0,atol=0)


@pytest.mark.parametrize('horizon',HORIZONS)
def test_auxiliary_own_block_and_horizon_input_isolation(horizon):
    cell=model();state=wave((2,6)).requires_grad_();future=wave((2,36,2),offset=.8).requires_grad_()
    output=cell.aux_decode(state,future,horizon)
    changed=future.detach().clone();changed[:,horizon:]+=100
    torch.testing.assert_close(cell.aux_decode(state,changed,horizon),output,rtol=0,atol=0)
    output.square().sum().backward()
    assert torch.count_nonzero(future.grad[:,horizon:])==0
    block=HORIZONS.index(horizon)
    for index in range(3):
        if index!=block: assert torch.count_nonzero(state.grad[:,index*2:index*2+2])==0
    w=weights(cell);prefix=f'auxiliary.{horizon}.'
    features=np.concatenate((state.detach().numpy()[:,block*2:block*2+2],
                             future.detach().numpy()[:,:horizon].reshape(2,-1)),axis=-1)
    expected=np.tanh(features@w[prefix+'0.weight'].T+w[prefix+'0.bias'])@w[prefix+'2.weight'].T+w[prefix+'2.bias']
    np.testing.assert_allclose(output.detach().numpy(),expected,rtol=2e-13,atol=2e-13)


def test_preparation_once_and_fresh_no_hidden_state_or_inputs(monkeypatch):
    cell=model();keys=set(vars(cell));original=copy.deepcopy(cell.state_dict())
    count=0;prepare=cell._prepare_correction
    def counted():
        nonlocal count
        count+=1;return prepare()
    monkeypatch.setattr(cell,'_prepare_correction',counted)
    y,u=wave((2,5,2)),wave((2,4,2));yc,uc=y.clone(),u.clone()
    first=cell.condition(y,u);assert count==1
    cell.rollout(wave((2,3,2)),first);assert count==1
    with torch.no_grad(): cell.observation.weight.mul_(1.2)
    second=cell.condition(y,u);assert count==2 and not torch.equal(first,second)
    assert set(vars(cell))==keys|{'_prepare_correction'} and not dict(cell.named_buffers())
    torch.testing.assert_close(y,yc,rtol=0,atol=0);torch.testing.assert_close(u,uc,rtol=0,atol=0)
    for key,value in cell.state_dict().items():
        if key!='observation.weight': torch.testing.assert_close(value,original[key],rtol=0,atol=0)


def test_seed_and_global_rng_isolation_and_exact_shared_initialization():
    before=torch.random.get_rng_state().clone()
    cells=[model(mode) for mode in MODES]
    torch.testing.assert_close(torch.random.get_rng_state(),before,rtol=0,atol=0)
    for other in cells[1:]:
        assert cells[0].state_dict().keys()==other.state_dict().keys()
        for key,p in cells[0].state_dict().items():torch.testing.assert_close(p,other.state_dict()[key],rtol=0,atol=0)
    different=PredictiveStateCorrection(2,2,latent_dim=6,aux_width=5,seed=92,dtype=torch.float64)
    assert not torch.equal(cells[0].transition.weight_ih,different.transition.weight_ih)


@pytest.mark.parametrize('obs,inputs,latent,width,total',[(1,1,60,64,18372),(3,3,60,64,24612),
    (1,1,120,128,66020),(3,3,120,128,78492),(1,1,72,96,27844),(1,1,132,128,76772),(3,3,72,96,37020),(3,3,132,128,89364)])
def test_exact_size_and_storage_counts(obs,inputs,latent,width,total):
    cell=PredictiveStateCorrection(obs,inputs,latent_dim=latent,aux_width=width)
    nonaux=3*latent*inputs+3*latent*latent+6*latent+obs*latent+obs+latent*obs+latent
    auxiliary=sum(width*(latent//3+h*inputs)+width+obs*width+obs for h in (1,8,32))
    assert nonaux+auxiliary==total==cell.parameter_count()
    assert cell.parameter_count(include_aux=False)==nonaux
    spec=cell.model_spec()
    assert spec['parameter_count']==total and spec['auxiliary_parameter_count']==auxiliary
    assert spec['parameter_bytes']==4*total and spec['non_auxiliary_parameter_bytes']==4*nonaux
    assert spec['state_bytes_per_stream']==4*latent and spec['state_scalars']==latent
    assert spec['buffer_bytes']==0 and spec['persistent_cache'] is False and spec['retained_trajectory'] is False


@pytest.mark.parametrize('mode',MODES)
def test_minimum_context_empty_forecast_batch_invariance_and_ownership(mode):
    cell=model(mode);y=wave((2,1,2));u=y.new_empty(2,0,2)
    state=cell.condition(y,u)
    torch.testing.assert_close(state,torch.tanh(cell.initializer(y[:,0])),rtol=0,atol=0)
    predictions,final=cell.rollout(y.new_empty(2,0,2),state)
    assert predictions.shape==(2,0,2) and final.data_ptr()!=state.data_ptr()
    torch.testing.assert_close(final,state,rtol=0,atol=0)
    future=wave((2,5,2),offset=1.)
    whole,last=cell.rollout(future,state)
    for b in range(2):
        partial,end=cell.rollout(future[b:b+1],state[b:b+1])
        torch.testing.assert_close(partial,whole[b:b+1],rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(end,last[b:b+1],rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('kwargs',[{'obs_dim':0},{'input_dim':True},{'latent_dim':7},{'latent_dim':0},
    {'aux_width':0},{'mode':'unknown'},{'seed':-1},{'seed':True},{'dtype':torch.int64}])
def test_invalid_constructor(kwargs):
    args={'obs_dim':2,'input_dim':2,**kwargs}
    with pytest.raises(ValueError):PredictiveStateCorrection(**args)


@pytest.mark.parametrize('case',['extra_context_input','short_context_input','wrong_observation_channels','empty_batch',
    'empty_context','mixed_dtype','nonfinite_context','nonfinite_input','wrong_state','wrong_future_channels',
    'nonfinite_state','nonfinite_future','short_aux','bad_horizon','nonfinite_aux_suffix','nonfinite_parameter','missing_parameter'])
def test_invalid_inputs_and_parameters_fail_without_repair(case):
    cell=model();y=wave((2,4,2));u=wave((2,3,2));state=wave((2,6));future=wave((2,32,2))
    call=lambda:cell.condition(y,u)
    if case=='extra_context_input':u=wave((2,4,2))
    elif case=='short_context_input':u=u[:,:2]
    elif case=='wrong_observation_channels':y=y[:,:,:1]
    elif case=='empty_batch':y=y[:0];u=u[:0]
    elif case=='empty_context':y=y[:,:0];u=u[:,:0]
    elif case=='mixed_dtype':y=y.float()
    elif case=='nonfinite_context':y[0,0,0]=float('nan')
    elif case=='nonfinite_input':u[0,0,0]=float('inf')
    elif case=='wrong_state':call=lambda:cell.step(state[:,:5],future[:,0])
    elif case=='wrong_future_channels':call=lambda:cell.rollout(future[:,:,:1],state)
    elif case=='nonfinite_state':state[0,0]=float('nan');call=lambda:cell.rollout(future,state)
    elif case=='nonfinite_future':future[0,0,0]=float('inf');call=lambda:cell.rollout(future,state)
    elif case=='short_aux':call=lambda:cell.aux_decode(state,future[:,:3],8)
    elif case=='bad_horizon':call=lambda:cell.aux_decode(state,future,True)
    elif case=='nonfinite_aux_suffix':future[0,-1,0]=float('nan');call=lambda:cell.aux_decode(state,future,1)
    elif case=='nonfinite_parameter':
        with torch.no_grad():cell.observation.weight[0,0]=float('inf')
    else:cell.observation.register_parameter('bias',None)
    with pytest.raises(ValueError):call()


def test_finite_overflow_is_rejected_without_denominator_or_energy_repair():
    cell=model(dtype=torch.float32)
    with torch.no_grad():cell.observation.weight.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError,match='denominator'):
        cell.correct(torch.zeros(1,6),torch.zeros(1,2))


@pytest.mark.parametrize('obs_dim,input_dim', [(1,1),(1,3),(3,1),(3,3)])
def test_generic_scalar_and_three_channel_interfaces(obs_dim,input_dim):
    cell=PredictiveStateCorrection(obs_dim,input_dim,mode='selective',latent_dim=6,aux_width=5)
    y=wave((2,3,obs_dim),torch.float32);u=wave((2,2,input_dim),torch.float32)
    state=cell.condition(y,u)
    prediction,final=cell.rollout(wave((2,4,input_dim),torch.float32),state)
    assert prediction.shape==(2,4,obs_dim) and final.shape==(2,6)
    assert prediction.dtype==torch.float32 and bool(torch.isfinite(prediction).all())
