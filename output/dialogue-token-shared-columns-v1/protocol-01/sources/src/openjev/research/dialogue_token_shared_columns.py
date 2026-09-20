"""Share one differentiable state-column view within each complete forward.

The required-fill pooling and ordered normalized head are inherited. Only the
split feature container receives a private typed payload with the exogenous
tensor and a forward-local view of the existing final two weight columns.
Its argument zero remains the actual live belief/entropy pair checked by the
unchanged monitor. No parameter, buffer or tensor cache is introduced.

    class MonitoredSharedColumns(MonitoredCopyMemoryV2, DialogueTokenSharedColumnsMemory):
        pass

Logical view counts do not describe allocated/copied weights, autograd node
counts, memory traffic or speed. Shared gradient accumulation can change
floating addition order; numerical equivalence requires bounded checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_factored import _ExogenousTurn, _FactoredFeature
from openjev.research.dialogue_token_required_fill import VERSION as PARENT_VERSION
from openjev.research.dialogue_token_required_fill import DialogueTokenRequiredFillMemory

VERSION = "dialogue-token-shared-columns-v1"
REQUIRED_FILL_SOURCE = "src/openjev/research/dialogue_token_required_fill.py"
REQUIRED_FILL_SHA256 = "f52ad0ca268017e7903da0e076632391f38d1ec28a2b56c6a1721a2d098f54b4"


@dataclass(frozen=True)
class _SharedHeadTerms:
    """Forward-local references, not a promise of tensor immutability."""

    exogenous: torch.Tensor
    state_columns: torch.Tensor


@dataclass(frozen=True)
class _SharedStateTurn(_ExogenousTurn):
    """Explicit inherited dispatch marker with a typed feature-adapter payload.

    The pinned parent _advance only passes value to the feature container. It
    performs no tensor arithmetic on this field. Our adapter alone unwraps it.
    """

    value: _SharedHeadTerms


class _SharedColumnsFeature(_FactoredFeature):
    """Same Linear/Tanh objects; ordinary full and split calls still delegate."""

    def forward(self, features, exogenous=None):
        if isinstance(exogenous, _SharedHeadTerms):
            return self[1](exogenous.exogenous + F.linear(features, exogenous.state_columns))
        return super().forward(features, exogenous)


def _shared_column_work(work, time_steps, hidden_width):
    return {**work, "state_column_view_expressions": 1,
            "state_column_reference_expressions": time_steps,
            "state_column_view_scalars": 2*hidden_width,
            "state_column_linear_calls": time_steps}


class DialogueTokenSharedColumnsMemory(DialogueTokenRequiredFillMemory):
    """Same initializer, state_dict, pool, ordinary step and monitored head.

    The inherited monitor owns mutable audit state and still requires serialized
    use or separate instances. Sharing a local view adds no cross-call cache and
    does not make that monitor thread-safe.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Retain the exact existing children and names, without new RNG draws.
        # A monitor-first constructor registers its hook after this returns.
        self.feature = _SharedColumnsFeature(*self.feature.children())

    def configuration(self):
        result = super().configuration()
        result.update({
            "implementation_version": VERSION,
            "required_fill_parent_version": PARENT_VERSION,
            "required_fill_parent_source": {"path": REQUIRED_FILL_SOURCE, "sha256": REQUIRED_FILL_SHA256},
            "state_column_sharing": "One fresh differentiable weight[:,-2:] view per forward, explicit local typed terms, T unchanged state linear calls; no persistent tensor cache",
            "feature_monitor": "Inherited live belief/entropy argument0 and ordered checks; only private argument1 terms are unwrapped; ordinary feature/step calls delegate",
            "packing_reference_only_fields": result["packing_reference_only_fields"]+["state_column_reference_expressions"],
            "packing_view_only_fields": ["state_column_view_scalars"],
            "state_column_work_scope": "Source-level view expression and linear-use counts; view scalars share existing parameter storage, not allocated bytes or measured backward nodes. Full74 counters describe successful complete forwards",
        })
        return result

    @staticmethod
    def packing_work(valid, candidate_mask, token_mask, token_width=384, projection_width=64, hidden_width=64):
        work = DialogueTokenRequiredFillMemory.packing_work(
            valid, candidate_mask, token_mask, token_width, projection_width, hidden_width)
        return _shared_column_work(work, valid.shape[1], hidden_width)

    def forward(self, tokens, valid, query, candidates, candidate_mask, lexical, token_mask, token_prior, none_index=None):
        """Original validation and head order; one current-grad-context view."""
        self.last_packing_work = None
        b, q, c = self._schema(query, candidates, candidate_mask)
        self._check_tokens(tokens, valid, token_mask, token_prior, b, sequence=True)
        t = tokens.shape[1]
        self._float(lexical, (b, t, q, c, 10), "lexical sequence")
        state = self.initial(candidate_mask, none_index)
        qp, cp = self._project_schema(query, candidates, candidate_mask)
        schema_query = self._attention_query(query, candidates, candidate_mask)
        exogenous = self._pool_factored(tokens, valid, candidate_mask, token_mask, token_prior, schema_query, qp, cp, lexical)
        state_columns = self.feature[0].weight[:, -2:]
        rows = []
        for i in range(t):
            terms = _SharedHeadTerms(exogenous[:, i], state_columns)
            state, _ = self._advance(state, _SharedStateTurn(terms), valid[:, i], qp, cp,
                                    candidate_mask, lexical[:, i])
            rows.append(state["log_b"])
        self.last_packing_work = _shared_column_work(self.last_packing_work, t, self.hidden_dim)
        return torch.stack(rows, 1)
