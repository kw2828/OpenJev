"""Fabricated joint-observer qualification; no measured inputs or checkpoints."""
import copy

import numpy as np
import pytest
import torch

from openjev.research import robot_joint_observer as core
from openjev.research.robot_history_initializer import HistoryInitializedDense
from openjev.research.robot_position_observer import FrozenPositionObserver
from openjev.research.structured_robot_transition import RADIUS


def inputs(dtype=torch.float64, batch=2, horizon=5):
    q = (.2*torch.sin(torch.arange(batch*32*6, dtype=dtype)*.13)).reshape(batch, 32, 6)
    u = (.1*torch.cos(torch.arange(batch*32*6, dtype=dtype)*.17)).reshape(batch, 32, 6)
    future = (.15*torch.sin(torch.arange(batch*horizon*6, dtype=dtype)*.21)).reshape(batch, horizon, 6)
    return q, u, future


def active(model):
    """Nondegenerate, small deterministic weights to expose gradient paths."""
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.copy_(.025*torch.sin(torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape)*.31+i))
        for e in range(2):
            model.cell.raw_matrix[e].add_(torch.diag(torch.linspace(.62+.05*e, .72+.05*e, 12, dtype=model.cell.raw_matrix.dtype)))
        if model.mode in ('long', 'short'):
            model.gain[:6].add_(.7*torch.eye(6, dtype=model.gain.dtype))
    return model


def linearize(model):
    """Both experts are exactly the same .5I affine state map."""
    dtype = model.cell.raw_matrix.dtype
    forcing = torch.cat((.04*torch.eye(6, dtype=dtype), .02*torch.eye(6, dtype=dtype)))
    bias = torch.arange(12, dtype=dtype)*.001
    with torch.no_grad():
        for p in model.cell.parameters(): p.zero_()
        model.cell.raw_matrix.copy_((.5/RADIUS)*torch.eye(12, dtype=dtype).repeat(2, 1, 1))
        model.cell.input_matrix.copy_(forcing.repeat(2, 1, 1))
        model.cell.expert_bias.copy_(bias.repeat(2, 1))
        if model.mode in ('long', 'short'):
            model.gain.copy_(torch.cat((.7*torch.eye(6, dtype=dtype), .2*torch.eye(6, dtype=dtype))))
    return forcing.numpy(), bias.numpy()


@pytest.mark.parametrize('mode,count', [('long',662),('short',662),('last_two',590),('temporal',962)])
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_exact_roles_trainability_storage_and_initialization(mode, count, dtype):
    model = core.JointObserver(17, mode, dtype=dtype)
    spec = model.model_spec()
    assert model.parameter_count == model.trainable_parameter_count == count
    assert spec['trainable_parameter_count'] == count and spec['frozen_parameter_count'] == 0
    assert model.state_scalars == 12 and not list(model.buffers())
    assert spec['parameter_bytes'] == count*torch.empty((),dtype=dtype).element_size()
    assert spec['buffer_bytes'] == 0 and spec['state_bytes_per_stream'] == 12*torch.empty((),dtype=dtype).element_size()
    expected = {'cell.'+n for n in model.cell.state_dict()}
    if mode in ('long','short'):
        expected.add('gain')
        assert torch.equal(model.gain,torch.cat((torch.eye(6,dtype=dtype),torch.zeros(6,6,dtype=dtype))))
    if mode == 'temporal':
        expected |= {'head.weight','head.bias'}
        assert torch.count_nonzero(model.head.weight) == torch.count_nonzero(model.head.bias) == 0
    assert set(model.state_dict()) == expected and sum(p.numel() for p in model.cell.parameters()) == 590
    assert spec['prefix_transition_count'] == {'long':30,'short':1,'last_two':0,'temporal':0}[mode]
    model.requires_grad_(False)
    assert model.model_spec()['trainable_parameter_count'] == 0


@pytest.mark.parametrize('mode,oldmode', [('last_two','last_two'),('temporal','temporal_affine')])
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_qualified_control_forward_and_all_gradient_parity(mode, oldmode, dtype):
    model = active(core.JointObserver(9,mode,dtype=dtype))
    original = HistoryInitializedDense(9,oldmode,dtype=dtype)
    original.load_state_dict(model.state_dict(),strict=True)
    originals = inputs(dtype)
    inputs_a = [v.clone().requires_grad_() for v in originals]
    inputs_b = [v.clone().requires_grad_() for v in originals]
    q,u,f = inputs_a; qb,ub,fb = inputs_b
    state = model.condition(q,u); refstate = original.condition(qb,ub)
    prediction,final = model(f,state); ref,refend = original(fb,refstate)
    assert torch.equal(state,refstate) and torch.equal(prediction,ref) and torch.equal(final,refend)
    (prediction.square().sum()+final.square().sum()).backward()
    (ref.square().sum()+refend.square().sum()).backward()
    for p,r in zip(model.parameters(),original.parameters(),strict=True): assert torch.equal(p.grad,r.grad)
    for p,r in zip(inputs_a,inputs_b,strict=True):
        assert (p.grad is None and r.grad is None) or torch.equal(p.grad,r.grad)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_long_identical_weights_match_qualified_frozen_position_condition_and_gain_grad(dtype):
    model = active(core.JointObserver(21,'long',dtype=dtype))
    frozen = FrozenPositionObserver(21,'learned',dtype=dtype)
    frozen.load_state_dict(model.state_dict(),strict=True)
    qa,ua,fa = [v.requires_grad_() for v in inputs(dtype)]
    qb,ub,fb = [v.detach().clone().requires_grad_() for v in (qa,ua,fa)]
    leftdiag,rightdiag = {},{}
    a = model.condition(qa,ua,leftdiag); b = frozen.condition(qb,ub,rightdiag)
    pa,sa = model(fa,a); pb,sb = frozen(fb,b)
    assert torch.equal(a,b) and torch.equal(pa,pb) and torch.equal(sa,sb) and leftdiag == rightdiag
    pa.square().sum().backward(); pb.square().sum().backward()
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(model.gain.grad,frozen.gain.grad,rtol=tol,atol=tol)
    for x,y in zip((qa,ua,fa),(qb,ub,fb),strict=True): torch.testing.assert_close(x.grad,y.grad,rtol=tol,atol=tol)
    assert all(p.grad is None for p in frozen.cell.parameters())


@pytest.mark.parametrize('mode,start', [('long',1),('short',30)])
def test_independent_affine_prefix_oracle_and_aggregated_diagnostics(mode, start):
    model = core.JointObserver(0,mode,dtype=torch.float64)
    forcing,bias = linearize(model); q,u,future = inputs()
    qn,un = q.numpy(),u.numpy(); gain = model.gain.detach().numpy()
    state = np.concatenate((qn[:,start],qn[:,start]-qn[:,start-1]),axis=-1)
    states,innovations = [state.copy()],[]
    for t in range(start+1,32):
        prior = .5*state+un[:,t-1]@forcing.T+bias
        innovation = qn[:,t]-prior[:,:6]
        state = prior+innovation@gain.T
        states += [prior.copy(),state.copy()]; innovations.append(innovation.copy())
    diag = {}; actual = model.condition(q,u,diag)
    np.testing.assert_allclose(actual.detach().numpy(),state,rtol=1e-12,atol=1e-12)
    assert diag['prefix_steps'] == 31-start
    for label,values in (('state',states),('innovation',innovations)):
        assert diag['max_'+label+'_abs'] == pytest.approx(max(float(np.abs(v).max()) for v in values),rel=1e-12,abs=1e-12)
        assert diag['max_'+label+'_norm64'] == pytest.approx(max(float(np.sqrt((v*v).sum(-1)).max()) for v in values),rel=1e-12,abs=1e-12)
    generated,_ = model(future,actual)
    expected = []
    for t in range(future.shape[1]):
        state = .5*state+future.numpy()[:,t]@forcing.T+bias; expected.append(state[:,:6].copy())
    np.testing.assert_allclose(generated.detach().numpy(),np.stack(expected,axis=1),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('mode', core.MODES)
def test_unused_u31_causal_forecast_chunking_empty_ownership_and_reset(mode):
    model = active(core.JointObserver(4,mode,dtype=torch.float64))
    q,u,f = inputs(); originals = [x.clone() for x in (q,u,f)]
    state = model.condition(q,u); changed = u.clone(); changed[:,31] += 900
    assert torch.equal(state,model.condition(q,changed))
    full,end = model(f,state); first,carry = model(f[:,:2],state); rest,last = model(f[:,2:],carry)
    assert torch.equal(full,torch.cat((first,rest),dim=1)) and torch.equal(end,last)
    prediction,stepstate = model.step(state,f[:,0])
    assert torch.equal(prediction,full[:,0])
    one,one_state = model(f[:,:1],state)
    assert torch.equal(prediction,one[:,0]) and torch.equal(stepstate,one_state)
    empty,same = model(f[:,:0],state)
    assert empty.shape == (2,0,6) and torch.equal(same,state) and same.data_ptr() != state.data_ptr()
    assert torch.equal(state,model.condition(q,u))
    for x,y in zip((q,u,f),originals,strict=True): assert torch.equal(x,y)
    changed_future = f.clone(); changed_future[:,2:] += 100
    assert torch.equal(model(changed_future,state)[0][:,:2],full[:,:2])


@pytest.mark.parametrize('mode,q_stop,u_stop', [('short',29,30),('last_two',30,32)])
def test_short_and_last_two_ignore_exact_older_numeric_inputs(mode,q_stop,u_stop):
    model = active(core.JointObserver(0,mode,dtype=torch.float64)); q,u,_ = inputs()
    q2,u2 = q.clone(),u.clone(); q2[:,:q_stop] += 17.; u2[:,:u_stop] -= 23.
    assert torch.equal(model.condition(q,u),model.condition(q2,u2))
    q.requires_grad_(); u.requires_grad_(); model.condition(q,u).square().sum().backward()
    assert torch.count_nonzero(q.grad[:,:q_stop]) == 0
    if mode == 'last_two': assert u.grad is None
    else: assert torch.count_nonzero(u.grad[:,:u_stop]) == 0 and torch.count_nonzero(u.grad[:,30]) > 0


def test_short_uses_three_positions_and_long_uses_older_observations():
    q,u,_ = inputs()
    for mode in ('long','short'):
        model = core.JointObserver(0,mode,dtype=torch.float64); linearize(model)
        old = model.condition(q,u)
        for index in ((29,30,31) if mode == 'short' else (0,1,8,31)):
            altered = q.clone(); altered[:,index] += .2
            assert not torch.equal(old,model.condition(altered,u))


@pytest.mark.parametrize('mode,steps', [('long',30),('short',1),('last_two',0),('temporal',0)])
def test_prepare_once_correct_torque_indices_and_diagnostic_noop(mode,steps,monkeypatch):
    model = active(core.JointObserver(0,mode,dtype=torch.float64)); q,u,f = inputs()
    calls = []; prepares = []; real_step = model.cell._step; real_prepare = model.cell._prepare
    def step(state,torque,prepared): calls.append(torque.detach().clone()); return real_step(state,torque,prepared)
    def prepare(): prepares.append(True); return real_prepare()
    monkeypatch.setattr(model.cell,'_step',step); monkeypatch.setattr(model.cell,'_prepare',prepare)
    before = set(model.__dict__); diag = {}; state = model.condition(q,u,diag)
    assert len(calls) == steps and len(prepares) == (1 if steps else 0)
    assert diag['prefix_steps'] == steps and set(diag) == set(core.DIAGNOSTIC_KEYS)
    for actual,index in zip(calls,range(1,31) if mode == 'long' else ([30] if mode == 'short' else []),strict=True):
        assert torch.equal(actual,u[:,index])
    p,final = model(f,state); (p.square().sum()+final.square().sum()).backward()
    gradients = {n:p.grad.clone() for n,p in model.named_parameters()}; model.zero_grad(set_to_none=True)
    def forbidden(*args): raise AssertionError('diagnostic work when disabled')
    monkeypatch.setattr(core,'_prefix_maxima',forbidden)
    second = model.condition(q,u); p2,final2 = model(f,second)
    assert torch.equal(state,second) and torch.equal(p,p2) and torch.equal(final,final2)
    (p2.square().sum()+final2.square().sum()).backward()
    for n,p in model.named_parameters(): assert torch.equal(p.grad,gradients[n])
    assert set(model.__dict__) == before and not any(isinstance(v,torch.Tensor) for v in diag.values())


@pytest.mark.parametrize('mode', core.MODES)
def test_every_cell_parameter_tensor_has_a_finite_nonzero_late_forecast_gradient(mode):
    model = active(core.JointObserver(3,mode,dtype=torch.float64)); q,u,f = inputs(horizon=7)
    q.requires_grad_(); u.requires_grad_(); f.requires_grad_()
    prediction,_ = model(f,model.condition(q,u))
    weights = torch.linspace(.1,1.2,prediction.numel(),dtype=torch.float64).reshape(prediction.shape)
    (prediction*weights).sum().backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert q.grad is not None and f.grad is not None
    if mode == 'last_two':
        assert u.grad is None
    else:
        assert u.grad is not None


@pytest.mark.parametrize('name,index', [('gain',(6,3)),('cell.raw_matrix',(0,7,2)),('cell.input_matrix',(1,2,4)),
                                       ('cell.gate.0.weight',(2,1))])
def test_joint_parameter_finite_difference_through_prefix_and_forecast(name,index):
    model = active(core.JointObserver(7,'long',dtype=torch.float64)); q,u,f = inputs(batch=1,horizon=4)
    def loss():
        pred,end = model(f,model.condition(q,u))
        return pred[:,-1].square().sum()+.1*end.square().sum()
    loss().backward(); p = dict(model.named_parameters())[name]; analytic = float(p.grad[index])
    original = float(p.detach()[index]); epsilon = 1e-6
    with torch.no_grad():
        p[index] = original+epsilon; plus = float(loss())
        p[index] = original-epsilon; minus = float(loss())
        p[index] = original
    assert analytic == pytest.approx((plus-minus)/(2*epsilon),rel=2e-5,abs=1e-9)
    assert abs(analytic) > 1e-12


@pytest.mark.parametrize('mode', core.MODES)
def test_local_rng_and_separate_model_storage(mode):
    with torch.random.fork_rng():
        torch.manual_seed(77); before = torch.random.get_rng_state().clone()
        model = core.JointObserver(12,mode); other = core.JointObserver(12,mode)
        assert torch.equal(before,torch.random.get_rng_state())
        for a,b in zip(model.parameters(),other.parameters(),strict=True):
            assert torch.equal(a,b) and a.data_ptr() != b.data_ptr()
        reference = HistoryInitializedDense(12,'last_two')
        for n,v in model.cell.state_dict().items(): assert torch.equal(v,reference.cell.state_dict()[n])


@pytest.mark.parametrize('damage', ['context_length','empty','wrong_dtype','nan_q','inf_u31','gain_nan',
                                  'extra_parameter','extra_buffer','mixed_trainability','frozen_grad','mode','initializer'])
def test_strict_schema_finiteness_and_trainability_no_repairs(damage):
    model = core.JointObserver(0,'long',dtype=torch.float64); q,u,_ = inputs()
    if damage == 'context_length': q=q[:,:31]; u=u[:,:31]
    elif damage == 'empty': q=q[:0]; u=u[:0]
    elif damage == 'wrong_dtype': u=u.float()
    elif damage == 'nan_q': q[0,0,0]=float('nan')
    elif damage == 'inf_u31': u[0,31,0]=float('inf')
    elif damage == 'gain_nan':
        with torch.no_grad(): model.gain[0,0]=float('nan')
    elif damage == 'extra_parameter': model.register_parameter('extra',torch.nn.Parameter(torch.zeros(1,dtype=torch.float64)))
    elif damage == 'extra_buffer': model.register_buffer('extra',torch.zeros(1,dtype=torch.float64))
    elif damage == 'mixed_trainability': model.gain.requires_grad_(False)
    elif damage == 'frozen_grad': model.requires_grad_(False); model.gain.grad=torch.ones_like(model.gain)
    elif damage == 'mode': model.mode='unknown'
    else: model.initializer='temporal_affine'
    with pytest.raises(ValueError): model.condition(q,u)


@pytest.mark.parametrize('diagnostics', [[],{'prefix_steps':0},True])
def test_diagnostic_collector_requires_empty_caller_dict(diagnostics):
    q,u,_=inputs(); model=core.JointObserver(dtype=torch.float64)
    with pytest.raises(ValueError): model.condition(q,u,diagnostics)


@pytest.mark.parametrize('kwargs', [{'mode':'local_affine'},{'seed':True},{'seed':-1},{'dtype':torch.float16}])
def test_constructor_guards(kwargs):
    with pytest.raises(ValueError): core.JointObserver(**kwargs)


def test_partial_diagnostics_preserve_completed_prefix_without_repair(monkeypatch):
    model=core.JointObserver(dtype=torch.float64); q,u,_=inputs(); real=model.cell._step; count=0
    def fail(state,torque,prepared):
        nonlocal count
        count+=1
        if count==4: return state[:,:6]*float('nan'),state*float('nan')
        return real(state,torque,prepared)
    monkeypatch.setattr(model.cell,'_step',fail); diag={}
    with pytest.raises(ValueError,match='no repair'): model.condition(q,u,diag)
    assert diag['prefix_steps']==3 and all(np.isfinite(v) for v in diag.values())


def test_continued_zero_temporal_head_matches_last_two_before_any_update():
    a=core.JointObserver(8,'last_two',dtype=torch.float64); b=core.JointObserver(8,'temporal',dtype=torch.float64)
    q,u,f=inputs(); sa=a.condition(q,u); sb=b.condition(q,u)
    assert torch.equal(sa,sb) and torch.equal(a(f,sa)[0],b(f,sb)[0])
    state=copy.deepcopy(a.state_dict())
    with torch.no_grad(): b.head.weight[6,18]=1.
    assert not torch.equal(b.condition(q,u),sa)
    assert all(torch.equal(v,a.state_dict()[n]) for n,v in state.items())
