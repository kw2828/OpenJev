"""Frozen float64 VARX plus a conventional tanh or affine output residual.

The only mode difference is the y value written to the caller's lag state:
output_only writes the linear prediction; feedback writes the corrected output.
Both residual heads use the same VARX features, excluding its intercept.
Public step validates its outputs; rollout checks all forecast values and final
state after the loop, without per-step host synchronization or hidden repair. No
clipping, pole projection, fallback, hidden cache or stability repair is used.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('output_only','feedback')
RESIDUAL_KINDS = ('tanh','affine')
CHANNELS = 3
VERSION = 'fsm-residual-v1'


def _require(condition,message):
    if not condition:raise ValueError(message)


class FSMResidual(nn.Module):
    """condition(y[B,C,3],u[B,C-1,3]) -> state[B,6*p].

    State packs chronological last-p y followed by chronological last-p u.
    Future input h predicts future output h. Dataset adapters own physical
    timing; there is no future-output argument. predict returns forecasts,
    while rollout returns both forecasts and the owned final caller state.
    """

    def __init__(self,coefficients,*,order,mode='output_only',residual_kind='tanh',hidden_width=24,seed=0):
        super().__init__()
        _require(type(order) is int and 1<=order<=99,'integer VARX order in1..99 required')
        _require(mode in MODES,'declared residual mode required')
        _require(residual_kind in RESIDUAL_KINDS,'declared residual kind required')
        _require(type(hidden_width) is int and hidden_width>0,'positive integer hidden width required')
        _require(type(seed) is int and 0<=seed<2**63,'nonnegative integer seed required')
        shape=(CHANNELS,6*order+4)
        if isinstance(coefficients,np.ndarray):
            _require(coefficients.dtype==np.float64 and coefficients.shape==shape
                     and bool(np.isfinite(coefficients).all()),'finite float64 coefficient matrix required')
            owned=torch.from_numpy(coefficients.copy(order='C'))
        else:
            self._tensor(coefficients,shape,'VARX coefficients')
            owned=coefficients.detach().clone(memory_format=torch.contiguous_format)
        self.order,self.mode,self.initialization_seed=order,mode,seed
        self.residual_kind=residual_kind
        self.hidden_width=hidden_width if residual_kind=='tanh' else None
        self.register_buffer('coefficients',owned)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            if residual_kind=='tanh':
                self.residual=nn.Sequential(nn.Linear(6*order+3,hidden_width,dtype=torch.float64,device='cpu'),
                                            nn.Tanh(),nn.Linear(hidden_width,CHANNELS,dtype=torch.float64,device='cpu'))
                output=self.residual[2]
            else:
                self.residual=nn.Linear(6*order+3,CHANNELS,dtype=torch.float64,device='cpu')
                output=self.residual
            with torch.no_grad():
                output.weight.zero_();output.bias.zero_()

    @staticmethod
    def _tensor(value,shape,name):
        _require(isinstance(value,torch.Tensor) and value.layout==torch.strided
                 and value.device.type=='cpu' and value.dtype==torch.float64
                 and tuple(value.shape)==shape and bool(torch.isfinite(value).all()),
                 'finite CPU float64 tensor with exact shape: '+name)

    def _validate_parameters(self):
        _require(type(self.order) is int and 1<=self.order<=99 and self.mode in MODES
                 and self.residual_kind in RESIDUAL_KINDS,'declared architecture required')
        if self.residual_kind=='tanh':
            _require(type(self.hidden_width) is int and self.hidden_width>0
                     and type(self.residual) is nn.Sequential and len(self.residual)==3
                     and type(self.residual[0]) is nn.Linear and type(self.residual[1]) is nn.Tanh
                     and type(self.residual[2]) is nn.Linear,'declared tanh residual MLP required')
            shapes={'residual.0.weight':(self.hidden_width,6*self.order+3),
                    'residual.0.bias':(self.hidden_width,),
                    'residual.2.weight':(CHANNELS,self.hidden_width),'residual.2.bias':(CHANNELS,)}
        else:
            _require(self.hidden_width is None and type(self.residual) is nn.Linear,
                     'declared affine residual required')
            shapes={'residual.weight':(CHANNELS,6*self.order+3),'residual.bias':(CHANNELS,)}
        parameters=dict(self.named_parameters())
        _require(set(parameters)==set(shapes) and set(dict(self.named_buffers()))=={'coefficients'},
                 'exact residual parameter roster and frozen coefficient buffer required')
        for name,shape in shapes.items():self._tensor(parameters[name],shape,name)
        self._tensor(self.coefficients,(CHANNELS,6*self.order+4),'VARX coefficients')
        _require(not self.coefficients.requires_grad,'VARX coefficients must remain frozen')

    def _state(self,state):
        _require(isinstance(state,torch.Tensor) and state.ndim==2 and state.shape[0]>0,
                 'state shape [positive B,6*p] required')
        self._tensor(state,(len(state),self.state_scalars),'lag state')

    def condition(self,y_context,u_context):
        self._validate_parameters()
        _require(isinstance(y_context,torch.Tensor) and y_context.ndim==3
                 and y_context.shape[0]>0 and y_context.shape[1]>=self.order+1,
                 'context shape [positive B,C>=p+1,3] required')
        batch,context=y_context.shape[:2]
        self._tensor(y_context,(batch,context,CHANNELS),'observed outputs')
        self._tensor(u_context,(batch,context-1,CHANNELS),'context transition inputs')
        return torch.cat((y_context[:,-self.order:].reshape(batch,3*self.order),
                          u_context[:,-self.order:].reshape(batch,3*self.order)),dim=-1)

    def _step(self,state,current_u):
        split=CHANNELS*self.order
        features=torch.cat((state[:,:split],current_u,state[:,split:]),dim=-1)
        linear=F.linear(features,self.coefficients[:,:-1],self.coefficients[:,-1])
        residual=self.residual(features)
        prediction=linear+residual
        inserted=linear if self.mode=='output_only' else prediction
        new_state=torch.cat((state[:,CHANNELS:split],inserted,state[:,split+CHANNELS:],current_u),dim=-1)
        return prediction,new_state

    def step(self,state,current_u):
        self._validate_parameters();self._state(state)
        self._tensor(current_u,(len(state),CHANNELS),'current input')
        prediction,final=self._step(state,current_u)
        self._finite_result(prediction,final)
        return prediction,final

    @staticmethod
    def _finite_result(prediction,state):
        _require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(state).all()),
                 'nonfinite residual forecast; no repair')

    def rollout(self,future_u,state):
        self._validate_parameters();self._state(state)
        _require(isinstance(future_u,torch.Tensor) and future_u.ndim==3,
                 'future inputs shape [B,H,3] required')
        self._tensor(future_u,(len(state),future_u.shape[1],CHANNELS),'future inputs')
        predictions,current=[],state
        for t in range(future_u.shape[1]):
            value,current=self._step(current,future_u[:,t]);predictions.append(value)
        if not predictions:return future_u.new_empty(len(state),0,CHANNELS),state.clone()
        result=torch.stack(predictions,dim=1)
        self._finite_result(result,current)
        return result,current

    def predict(self,y_context,u_context,future_u):
        return self.rollout(future_u,self.condition(y_context,u_context))[0]

    def forward(self,future_u,state):
        return self.rollout(future_u,state)

    @property
    def state_scalars(self):return 6*self.order

    def parameter_count(self):return sum(p.numel() for p in self.parameters())

    def model_spec(self):
        self._validate_parameters()
        count=self.parameter_count();parameter_bytes=count*8
        buffer_bytes=self.coefficients.numel()*self.coefficients.element_size()
        metadata_bytes=24 if self.residual_kind=='tanh' else 16  # int64 order, seed and optional width
        retained=parameter_bytes+buffer_bytes+metadata_bytes
        return {'version':VERSION,'mode':self.mode,'residual_kind':self.residual_kind,
                'dtype':'float64','order':self.order,
                'hidden_width':self.hidden_width,'initialization_seed':self.initialization_seed,
                'input_channels':CHANNELS,'output_channels':CHANNELS,'parameter_count':count,
                'trainable_parameter_count':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'frozen_parameter_count':sum(p.numel() for p in self.parameters() if not p.requires_grad),
                'parameter_bytes':parameter_bytes,'buffer_bytes':buffer_bytes,
                'frozen_coefficient_count':self.coefficients.numel(),'numeric_metadata_bytes':metadata_bytes,
                'numeric_metadata_scope':'logical int64 order, initialization seed and tanh-only hidden width; mode/kind strings excluded',
                'retained_numeric_bytes':retained,'state_scalars':self.state_scalars,
                'state_bytes_per_stream':self.state_scalars*8,
                'persistent_numeric_bytes_per_stream':retained+self.state_scalars*8,
                'feature_order':'chronological last-p y, current u, chronological last-p u; no intercept in residual features',
                'lag_output':'linear prediction' if self.mode=='output_only' else 'corrected prediction',
                'zero_output_initialization':True,'persistent_cache':False,'retained_trajectory':False,
                'storage_scope':'residual parameters, frozen coefficients and logical numeric metadata; caller lag state separately counted; excludes normalization, requests, workspace and Python object overhead',
                'limitations':'Conventional frozen-model residual; no stability, calibration or novelty guarantee.'}
