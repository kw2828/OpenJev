"""Explicitly normalized numerical revision of candidate copy memories.

The old scalar transition assumes sum(exp(log_b)) == 1. Without projection its
computed mass follows S' = S*(S-m)+m, amplifying rounding error near S=1.
This additive version normalizes before belief features/transition and after
each real update. Padding retains the caller's original state exactly.

Same parameter schema does not make old fits corrected scientific evidence:
normalization changes the recurrent forward path and its training gradients.
"""
from __future__ import annotations

import torch

from openjev.research.dialogue_copy_memory import DialogueCopyMemory

VERSION = "dialogue-copy-memory-v2-normalized"


def normalize_log_b(log_b):
    """Normalize valid log support; -inf masked/zero entries remain -inf.

    Outputs are normalized to the working floating-point precision, not an
    assertion that exp(output).sum() is exactly one in real arithmetic.
    """
    total = torch.logsumexp(log_b, -1, keepdim=True)
    if not torch.isfinite(total).all():
        raise ValueError("Belief normalization requires finite nonempty support")
    return log_b - total


class DialogueCopyMemoryV2(DialogueCopyMemory):
    """Same learned heads and public API; different, explicit state arithmetic."""

    def configuration(self):
        configuration = super().configuration()
        configuration.update({
            "implementation_version": VERSION,
            "normalization": "log_b -= logsumexp(log_b) before features/transition and after each real update",
            "padding": "exact original state values; no normalization committed on padded turns",
            "parameter_schema": "same as DialogueCopyMemory for the same method and dimensions",
            "old_fit_scope": "parameter loading possible, but not corrected training or empirical efficacy evidence",
        })
        return configuration

    def _advance(self, state, turn, valid, query, candidates, mask, lexical):
        # This dictionary is private; neither the input state nor its tensors
        # are mutated. The inherited parent sees a normalized feature prior.
        normalized = {**state, "log_b": normalize_log_b(state["log_b"])}
        proposed, diagnostics = super()._advance(normalized, turn, valid, query, candidates, mask, lexical)
        log_b = normalize_log_b(proposed["log_b"])
        proposed["log_b"] = torch.where(valid[:, None, None], log_b, state["log_b"])
        return proposed, diagnostics
