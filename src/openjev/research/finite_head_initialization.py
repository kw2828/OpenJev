"""Task-independent head initialization around the unchanged rounded model.

The factory returns the exact qualified RoundedDynamicsModel type. Its parameter
roster, order, recurrence and losses are unchanged. Random arms overwrite only
cost_logits after the inherited constructor; that constructor's cost template
is still computed and is then discarded. No tensors or functions are cached or
monkeypatched. Use model_metadata() for the effective initialization policy:
the inherited method describes the original constructor, not this wrapper.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
import torch

from openjev.research.finite_rounded_models import RoundedDynamicsModel
from openjev.research.finite_rounded_models import make_model as parent_model
from openjev.research.otto_observation_operator_model import require

VERSION = 'finite-head-initialization-v1'
ARMS = ('rounded_anchor', 'rounded_random', 'matched_free_random')
TRANSPORT_ARMS = MappingProxyType({'rounded_anchor': 'rounded', 'rounded_random': 'rounded',
                                  'matched_free_random': 'matched_free'})
HEAD_WORK_KEYS = ('rng_constructions', 'standard_normal_calls', 'standard_normal_entries',
                  'parameter_copy_calls', 'parameter_copy_entries', 'parameter_copy_bytes')


def expected_head_work(arm):
    require(type(arm) is str and arm in ARMS, 'declared head initialization arm')
    return dict.fromkeys(HEAD_WORK_KEYS, 0) if arm == 'rounded_anchor' else dict(zip(
        HEAD_WORK_KEYS, (1, 1, 32, 1, 32, 256), strict=True))


def make_model(arm, seed, *, check=lambda: None):
    require(type(arm) is str and arm in ARMS, 'declared head initialization arm')
    require(type(seed) is int and 0 <= seed < 2**32, 'uint32 seed')
    require(callable(check), 'callable external check')
    work = dict.fromkeys(HEAD_WORK_KEYS, 0)
    try:
        model = parent_model(TRANSPORT_ARMS[arm], seed, check=check)
        if arm != 'rounded_anchor':
            generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 436, 1])))
            work['rng_constructions'] += 1
            values = generator.standard_normal((4, 8))
            work['standard_normal_calls'] += 1
            work['standard_normal_entries'] += values.size
            require(values.dtype == np.float64 and np.isfinite(values).all(), 'finite float64 random head')
            with torch.no_grad():
                model.cost_logits.copy_(torch.from_numpy(values))
            work['parameter_copy_calls'] += 1
            work['parameter_copy_entries'] += values.size
            work['parameter_copy_bytes'] += values.nbytes
        model.head_initialization_arm = arm
        model.head_initialization_work = dict(work)
        model._parameters_valid()
        require(work == expected_head_work(arm), 'complete head initialization work')
        return model
    except BaseException as error:
        error.head_initialization_work = dict(work)
        raise


def model_metadata(model, arm):
    """Owned, static policy metadata; no extra forward or head normalization."""
    require(type(model) is RoundedDynamicsModel and type(arm) is str and arm in ARMS,
            'exact qualified model and declared external arm')
    require(model.transport_arm == TRANSPORT_ARMS[arm] and model.head_initialization_arm == arm,
            'effective transport and head initialization identity')
    require(model.head_initialization_work == expected_head_work(arm), 'unchanged head work accounting')
    parent = model.parameter_metadata()
    random = arm != 'rounded_anchor'
    return {**parent, 'version': VERSION, 'transport_model_version': parent['version'], 'arm': arm,
            'transport_arm': TRANSPORT_ARMS[arm], 'privileged_readout_initialization': not random,
            'readout_delta': None if random else parent['readout_delta'],
            'head_initialization': {'policy': 'task_independent_standard_normal' if random else 'inherited_cost_template',
                'generator': 'numpy.random.Generator(PCG64)' if random else None,
                'seed_sequence_entropy': [model.seed, 436, 1] if random else None,
                'shape': [4, 8], 'dtype': 'float64', 'inherited_template_is_constructed': True},
            'head_initialization_work': dict(model.head_initialization_work),
            'head_initialization_scope': 'Additional local random draw and cost_logits copy only; inherited construction work is separate. Same parameter count and initial transitions do not imply equal gradients or compute.'}
