"""Conventional input/output autoregressive GRU control, with explicit state.

Conditioning teacher-forces only arrived outputs. Forecasting feeds back its own
predictions. This is a standard baseline, not observation-error correction.
Input indices are positional; dataset adapters own physical time alignment.
"""
from __future__ import annotations

import torch
from torch import nn

HORIZONS = (1, 8, 32)
VERSION = 'fsm-gru-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class FSMGRU(nn.Module):
    """condition(y[B,C,O],u[B,C-1,I]) -> packed[B,L+O].

    Packed state is [latent, last_output]. The first supplied context input
    advances from observation 0 to observation 1. The first forecast input
    predicts the observation after the context. No future targets are accepted.
    """

    def __init__(self, obs_dim=3, input_dim=3, *, latent_dim=72, aux_width=96,
                 seed=0, dtype=torch.float32):
        super().__init__()
        _require(all(type(v) is int and v > 0 for v in (obs_dim,input_dim,latent_dim,aux_width)),
                 'positive integer dimensions required')
        _require(latent_dim % 3 == 0, 'latent dimension must divide into three blocks')
        _require(type(seed) is int and 0 <= seed < 2**63, 'nonnegative integer seed required')
        _require(dtype in (torch.float32,torch.float64), 'CPU float32 or float64 required')
        self.obs_dim,self.input_dim,self.latent_dim=obs_dim,input_dim,latent_dim
        self.aux_width,self.block_dim,self.initialization_seed=aux_width,latent_dim//3,seed
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            self.transition=nn.GRUCell(input_dim+obs_dim,latent_dim,dtype=dtype,device='cpu')
            self.observation=nn.Linear(latent_dim,obs_dim,dtype=dtype,device='cpu')
            self.initializer=nn.Linear(obs_dim,latent_dim,dtype=dtype,device='cpu')
            self.auxiliary=nn.ModuleDict({str(h):nn.Sequential(
                nn.Linear(self.block_dim+h*input_dim,aux_width,dtype=dtype,device='cpu'),
                nn.Tanh(),nn.Linear(aux_width,obs_dim,dtype=dtype,device='cpu')) for h in HORIZONS})

    def _shapes(self):
        n,o,i,w=self.latent_dim,self.obs_dim,self.input_dim,self.aux_width
        result={'transition.weight_ih':(3*n,i+o),'transition.weight_hh':(3*n,n),
                'transition.bias_ih':(3*n,),'transition.bias_hh':(3*n,),
                'observation.weight':(o,n),'observation.bias':(o,),
                'initializer.weight':(n,o),'initializer.bias':(n,)}
        for h in HORIZONS:
            result.update({f'auxiliary.{h}.0.weight':(w,self.block_dim+h*i),
                           f'auxiliary.{h}.0.bias':(w,),f'auxiliary.{h}.2.weight':(o,w),
                           f'auxiliary.{h}.2.bias':(o,)})
        return result

    @staticmethod
    def _tensor(value,shape,dtype,name):
        _require(isinstance(value,torch.Tensor) and value.layout==torch.strided
                 and value.device.type=='cpu' and value.dtype==dtype and tuple(value.shape)==shape
                 and bool(torch.isfinite(value).all()), 'finite CPU tensor with exact shape/dtype: '+name)

    def _validate_parameters(self):
        _require(self.latent_dim>0 and self.latent_dim%3==0 and self.block_dim==self.latent_dim//3,
                 'three-block latent geometry required')
        _require(type(self.transition) is nn.GRUCell and type(self.observation) is nn.Linear
                 and type(self.initializer) is nn.Linear and type(self.auxiliary) is nn.ModuleDict
                 and tuple(self.auxiliary)==tuple(map(str,HORIZONS)), 'declared module roster required')
        _require(all(type(head) is nn.Sequential and len(head)==3 and type(head[0]) is nn.Linear
                     and type(head[1]) is nn.Tanh and type(head[2]) is nn.Linear
                     for head in self.auxiliary.values()), 'declared auxiliary MLP heads required')
        parameters,shapes=dict(self.named_parameters()),self._shapes()
        _require(set(parameters)==set(shapes) and not dict(self.named_buffers()),
                 'exact parameter roster and no buffers required')
        dtype=self.observation.weight.dtype
        _require(dtype in (torch.float32,torch.float64), 'CPU float32 or float64 parameters required')
        for name,shape in shapes.items():self._tensor(parameters[name],shape,dtype,name)
        return dtype

    def _state(self,state,dtype):
        _require(isinstance(state,torch.Tensor) and state.ndim==2 and state.shape[0]>0,
                 'packed state shape [positive B,L+O] required')
        self._tensor(state,(len(state),self.state_scalars),dtype,'packed state')

    def condition(self,observations,inputs):
        dtype=self._validate_parameters()
        _require(isinstance(observations,torch.Tensor) and observations.ndim==3
                 and observations.shape[0]>0 and observations.shape[1]>=1,
                 'observations shape [positive B,C>=1,O] required')
        batch,context=observations.shape[:2]
        self._tensor(observations,(batch,context,self.obs_dim),dtype,'context observations')
        self._tensor(inputs,(batch,context-1,self.input_dim),dtype,'context transition inputs')
        latent=torch.tanh(self.initializer(observations[:,0]))
        _require(bool(torch.isfinite(latent).all()), 'nonfinite initialized latent state; no repair')
        for t in range(1,context):
            latent=self.transition(torch.cat((inputs[:,t-1],observations[:,t-1]),dim=-1),latent)
            _require(bool(torch.isfinite(latent).all()), 'nonfinite conditioned latent state; no repair')
        return torch.cat((latent,observations[:,-1]),dim=-1)

    def _step(self,state,inputs):
        latent,last_output=state[:,:self.latent_dim],state[:,self.latent_dim:]
        current=self.transition(torch.cat((inputs,last_output),dim=-1),latent)
        prediction=self.observation(current)
        _require(bool(torch.isfinite(current).all()) and bool(torch.isfinite(prediction).all()),
                 'nonfinite forecast state or prediction; no repair')
        return prediction,torch.cat((current,prediction),dim=-1)

    def step(self,state,inputs):
        dtype=self._validate_parameters()
        self._state(state,dtype)
        self._tensor(inputs,(len(state),self.input_dim),dtype,'next transition input')
        return self._step(state,inputs)

    def rollout(self,future_inputs,state):
        dtype=self._validate_parameters()
        self._state(state,dtype)
        _require(isinstance(future_inputs,torch.Tensor) and future_inputs.ndim==3,
                 'future inputs shape [B,H,I] required')
        self._tensor(future_inputs,(len(state),future_inputs.shape[1],self.input_dim),dtype,'future inputs')
        predictions,current=[],state
        for t in range(future_inputs.shape[1]):
            prediction,current=self._step(current,future_inputs[:,t])
            predictions.append(prediction)
        if not predictions:
            return future_inputs.new_empty(len(state),0,self.obs_dim),state.clone()
        return torch.stack(predictions,dim=1),current

    def forward(self,future_inputs,state):
        return self.rollout(future_inputs,state)

    def aux_decode(self,state,future_inputs,horizon):
        """Own latent block plus executed input prefix; packed last output excluded.

        Every supplied input must be finite; later finite input values do not
        affect this endpoint head. No auxiliary observation target is accepted.
        """
        dtype=self._validate_parameters()
        self._state(state,dtype)
        _require(type(horizon) is int and horizon in HORIZONS, 'auxiliary horizon must be1,8,32')
        _require(isinstance(future_inputs,torch.Tensor) and future_inputs.ndim==3
                 and future_inputs.shape[1]>=horizon, 'enough future inputs for auxiliary horizon')
        self._tensor(future_inputs,(len(state),future_inputs.shape[1],self.input_dim),dtype,'auxiliary inputs')
        block=HORIZONS.index(horizon)
        features=torch.cat((state[:,block*self.block_dim:(block+1)*self.block_dim],
                            future_inputs[:,:horizon].reshape(len(state),horizon*self.input_dim)),dim=-1)
        output=self.auxiliary[str(horizon)](features)
        _require(bool(torch.isfinite(output).all()), 'nonfinite auxiliary prediction; no repair')
        return output

    @property
    def state_scalars(self):
        return self.latent_dim+self.obs_dim

    def parameter_count(self,*,include_aux=True):
        _require(type(include_aux) is bool, 'include_aux must be bool')
        return sum(p.numel() for name,p in self.named_parameters()
                   if include_aux or not name.startswith('auxiliary.'))

    def model_spec(self):
        dtype=self._validate_parameters()
        total,nonaux=self.parameter_count(),self.parameter_count(include_aux=False)
        item_bytes=self.observation.weight.element_size()
        return {'version':VERSION,'obs_dim':self.obs_dim,'input_dim':self.input_dim,
                'latent_dim':self.latent_dim,'block_dim':self.block_dim,'auxiliary_horizons':list(HORIZONS),
                'auxiliary_width':self.aux_width,'parameter_count':total,
                'non_auxiliary_parameter_count':nonaux,'auxiliary_parameter_count':total-nonaux,
                'parameter_bytes':total*item_bytes,'non_auxiliary_parameter_bytes':nonaux*item_bytes,
                'buffer_bytes':0,'state_scalars':self.state_scalars,'state_bytes_per_stream':self.state_scalars*item_bytes,
                'dtype':str(dtype).removeprefix('torch.'),'initialization_seed':self.initialization_seed,
                'persistent_cache':False,'retained_trajectory':False,
                'context_inputs':'C-1 transition inputs; index t-1 advances to observation index t',
                'state_layout':'latent followed by last output',
                'conditioning':'teacher-forced preceding arrived output; final arrived output retained in caller state',
                'forecasting':'first input uses final arrived output; later steps use preceding prediction',
                'auxiliary_inputs':'own latent block plus horizon input prefix; no direct last-output feature',
                'limitations':'Conventional autoregressive GRU; no calibration, stability or novelty guarantee.',
                'storage_scope':'parameters and caller state; requests, normalization, optimizer and workspace excluded'}
