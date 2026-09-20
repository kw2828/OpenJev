"""Shared geometry and actor assembly for the bounded token-alignment study.

No Torch, encoder, fitted output, or float cache is loaded on import. Actor
inputs consume only addresses, public schema support, and privileged prior state.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import study_dialogue_typed as typed

base = typed.base
ROOT = typed.ROOT
METHODS = ("flat_stratum", "token_mean", "token_aligned")
SEEDS = (6201, 6202, 6203)
ARM_ORDERS = (METHODS, METHODS[1:] + METHODS[:1], METHODS[2:] + METHODS[:2])
FIT_ORDER = [f"{m}-{s}" for s, methods in zip(SEEDS, ARM_ORDERS, strict=True) for m in methods]
CONFIG = {**base.CONFIG, "methods": list(METHODS), "seeds": list(SEEDS), "microbatch_size": 32}
LIMITS = dict(base.LIMITS)
require = base.require
SOURCES = ("scripts/dialogue_alignment_common.py", "tests/test_dialogue_alignment_common.py",
           "src/openjev/research/dialogue_token_alignment.py", "tests/test_dialogue_token_alignment.py",
           "scripts/prepare_dialogue_schema_tokens.py", "tests/test_prepare_dialogue_schema_tokens.py")


def source_map(check=lambda: None):
    return {**typed.source_map(check), **{n: base.sha(ROOT/n, check) for n in SOURCES}}


def load_metadata(check=lambda: None):
    return typed.load_metadata(SimpleNamespace(check=check), payloads=True)


def schema_metadata(index, offsets):
    """Adapt authenticated schema index and integer offsets to a common map."""
    require(offsets.dtype == np.int64 and offsets.ndim == 1 and offsets[0] == 0
            and np.all(np.diff(offsets) > 0) and len(offsets) == index["unique_texts"]+1,
            "Schema token offsets")
    queries = {q["query_index"]: q for q in index["queries"]}
    require(len(queries) == len(index["queries"]), "Unique schema query indices")
    return {"index": index, "queries": queries, "offsets": offsets}


def candidate_spans(row, schema):
    q = schema["queries"][row["query_index"]]
    require(q["candidate_feature_indices"] == row["cache"]["candidate_feature_indices"]
            and len(q["candidate_token_ids"]) == row["candidate_count"], "Actor schema address binding")
    offsets = schema["offsets"]
    return [(int(offsets[i]), int(offsets[i+1])) for i in q["candidate_token_ids"]]


def geometry(rows, schema):
    require(bool(rows), "Nonempty batch")
    b = len(rows)
    c = max(r["candidate_count"] for r in rows)
    context_lengths = [r["cache"]["token_stop"]-r["cache"]["token_start"] for r in rows]
    schema_lengths = [[stop-start for start, stop in candidate_spans(r, schema)] for r in rows]
    length, slength = max(context_lengths), max(max(v) for v in schema_lengths)
    return {"rows": b, "max_candidate_count": c, "max_token_length": length,
            "max_schema_token_length": slength,
            "supported_candidate_positions": sum(len(v) for v in schema_lengths),
            "padded_candidate_positions": b*c,
            "supported_context_token_positions": sum(context_lengths),
            "padded_context_token_positions": b*length,
            "supported_schema_token_positions": sum(map(sum, schema_lengths)),
            "padded_schema_token_positions": b*c*slength,
            "supported_pairwise_positions": sum(n*sum(v) for n, v in zip(context_lengths, schema_lengths, strict=True)),
            "padded_pairwise_positions": b*c*length*slength,
            "schema_float_input_bytes": 4*b*c*slength*(384+1),
            "schema_boolean_input_bytes": b*c*slength}


def effective_geometry(rows, schema):
    """Work is summed over actual row microbatches, with their local padding."""
    pieces = [geometry(rows[i:i+CONFIG["microbatch_size"]], schema)
              for i in range(0, len(rows), CONFIG["microbatch_size"])]
    total = Counter()
    for part in pieces:
        total.update({k: v for k, v in part.items() if not k.startswith("max_")})
    return {**dict(total), **{k: max(p[k] for p in pieces) for k in pieces[0] if k.startswith("max_")},
            "microbatches": len(pieces)}


def actor(rows, arrays, candidate_types, schema, method):
    require(method in METHODS, "Known arm")
    result, work = typed.actor(rows, arrays, candidate_types)
    if method == "flat_stratum":
        return result, work
    g = geometry(rows, schema)
    b, c, s = len(rows), g["max_candidate_count"], g["max_schema_token_length"]
    result.update(schema_tokens=np.zeros((b, c, s, 384), np.float32),
                  schema_mask=np.zeros((b, c, s), bool), schema_prior=np.zeros((b, c, s), np.float32))
    for i, row in enumerate(rows):
        for j, (start, stop) in enumerate(candidate_spans(row, schema)):
            result["schema_tokens"][i, j, :stop-start] = schema["tokens"][start:stop]
            result["schema_mask"][i, j, :stop-start] = True
            result["schema_prior"][i, j, :stop-start] = schema["priors"][start:stop]
    require(np.isfinite(result["schema_tokens"]).all() and np.isfinite(result["schema_prior"]).all(),
            "Finite schema actor")
    return result, {**work, **{"schema_"+k: v for k, v in g.items() if k not in ("rows",)}}


def init_models(torch, seed):
    from openjev.research.dialogue_token_alignment import (
        COMMON_TENSORS,
        DialogueTokenAlignment,
        copy_common_initialization,
        copy_initialization,
    )
    from openjev.research.dialogue_typed_observation import DialogueTypedObservation

    torch.manual_seed(seed)
    models = {"flat_stratum": DialogueTypedObservation("flat"),
              "token_mean": DialogueTokenAlignment("mean"),
              "token_aligned": DialogueTokenAlignment("aligned")}
    copy_common_initialization(models["flat_stratum"], models["token_mean"])
    copy_initialization(models["token_mean"], models["token_aligned"])
    initial = {m: base.tensor_digest(v, tuple(dict(v.named_parameters()))) for m, v in models.items()}
    common = {m: base.tensor_digest(v, COMMON_TENSORS) for m, v in models.items()}
    require(initial["token_mean"] == initial["token_aligned"] and len(set(common.values())) == 1,
            "Paired initializers")
    return models, {"full_state_sha256": initial, "common_state_sha256": common,
                    "common_tensors": list(COMMON_TENSORS)}
