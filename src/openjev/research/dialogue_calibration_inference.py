"""Restore final V2 weights and replay unchanged public dialogue trajectories.

The caller authenticates the campaign/cohort and owns execution limits. This
module additionally checks checkpoint/local-asset bytes before loading. It never
creates an optimizer, accepts a temperature, or passes evaluator fields into the
actor. Calibration fitting is deliberately outside this module.

Production callers use the default factories/device. Tiny synthetic tests may
inject both factories and use CPU; that distinction is explicit in witnesses.
The caller must put the repository scripts directory on sys.path before restore
so the unchanged study_dialogue_copy_v2 monitor can be imported normally.
"""
from __future__ import annotations

import copy
import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from openjev.research.dialogue_trainable_encoder import build_actor, encode_token_lists

VERSION = "dialogue-calibration-inference-v1"
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
ACTOR_FIELDS = ("tokens", "turn_text_ids", "query_text_ids", "candidate_text_ids", "candidate_ids", "lexical")
PUBLIC_FIELDS = frozenset((*ACTOR_FIELDS, "split", "source_split", "analysis_role", "dialogue_id", "query_ids",
                           "user_turn_indices", "original_feature_ids", "lexical_shape", "lexical_offset"))
ENCODING = {"chunk_tokens": 254, "chunk_batch_size": 32}
COUNT_FIELDS = ("checkpoint_load_attempts", "checkpoint_load_returns", "encoder_load_attempts", "encoder_load_returns",
                "public_forward_attempts", "public_forward_returns", "encoding_attempts", "encoding_returns",
                "encoder_forward_attempts", "encoder_forward_returns", "memory_forward_attempts", "memory_forward_returns")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_digest(module):
    """Exact named-state digest convention of the frozen V2 qualification."""
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        item = tensor.detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str((str(item.dtype), tuple(item.shape))).encode())
        value.update(item.view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def _monitor_api():
    from study_dialogue_copy_v2 import (
        MonitoredCopyMemoryV2,
        empty_invariants,
        merge_invariants,
    )

    return MonitoredCopyMemoryV2, empty_invariants, merge_invariants


def _strict_state(module, state, name):
    expected = module.state_dict()
    require(isinstance(state, Mapping) and set(state) == set(expected), "Exact " + name + " state keys")
    for key, target in expected.items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.layout == torch.strided and value.shape == target.shape and value.dtype == target.dtype,
                "Exact " + name + " tensor geometry/dtype: " + key)
        require(not (value.is_floating_point() or value.is_complex()) or torch.isfinite(value).all().item(),
                "Finite " + name + " tensor: " + key)
    module.load_state_dict(state, strict=True)


def _local_assets(snapshot, model_files):
    snapshot = Path(snapshot)
    require(snapshot.is_dir() and isinstance(model_files, Mapping) and model_files,
            "Nonempty pinned local model snapshot")
    require({"config.json", "model.safetensors"} <= set(model_files), "Pinned local config and safetensors")
    files = {}
    for name, entry in model_files.items():
        require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Safe local model member")
        require(isinstance(entry, Mapping) and {"path", "sha256"} <= set(entry) and _pin(entry["sha256"]),
                "Pinned local model descriptor")
        path = Path(entry["path"])
        require(path.resolve() == (snapshot/name).resolve() and path.is_file(), "Snapshot asset identity")
        require(file_digest(path) == entry["sha256"], "Local model asset hash: " + name)
        files[name] = {"path": str(path), "sha256": entry["sha256"]}
    return files


@dataclass(frozen=True)
class PublicForward:
    log_probs: torch.Tensor
    split: str
    analysis_role: str
    dialogue_id: str
    query_ids: tuple[int, ...]
    candidate_ids: tuple[tuple[str, ...], ...]
    user_turn_indices: tuple[int, ...] | None
    encoder_work: dict
    invariants: dict


class CalibrationInference:
    """Inference-only local modules; each call starts the original NONE state."""

    def __init__(self, encoder, memory, tokenizer_ids, progress, *, check, synthetic_injection):
        _, self.empty, self.merge = _monitor_api()
        self.encoder, self.memory = encoder, memory
        self.tokenizer_ids = dict(tokenizer_ids)
        self.progress, self.check = progress, check
        self.synthetic_injection = synthetic_injection
        self.counts = progress["counts"]
        self.work = Counter()
        self.audit = self.empty()
        self.attempted_encoder_work, self.returned_encoder_work = Counter(), Counter()
        self._active_encoder_work = None
        self.encoder.register_forward_pre_hook(self._encoder_before, with_kwargs=True)
        self.encoder.register_forward_hook(self._encoder_after, with_kwargs=True)

    def _encoder_before(self, _module, args, kwargs):
        require(not args and set(kwargs) == {"input_ids", "attention_mask"}, "Qualified encoder call signature")
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        require(ids.ndim == 2 and mask.shape == ids.shape and ids.shape[0] > 0, "Actual encoder geometry")
        n, width = ids.shape
        work = {"encoder_calls": 1, "encoder_sequences": n, "valid_token_positions": int(mask.sum().item()),
                "padded_token_positions": ids.numel(), "padded_attention_positions": n * width * width}
        self.counts["encoder_forward_attempts"] += 1
        self.attempted_encoder_work.update(work)
        self._active_encoder_work = work
        require(not torch.is_grad_enabled() and not self.encoder.training, "Encoder eval/no-grad path")
        self.check()

    def _encoder_after(self, _module, _args, _kwargs, _output):
        self.counts["encoder_forward_returns"] += 1
        require(self._active_encoder_work is not None, "Matching encoder attempt")
        self.returned_encoder_work.update(self._active_encoder_work)
        self._active_encoder_work = None
        self.check()

    def forward_public(self, payload):
        """Return raw full trajectories. Targets and temperature fields are rejected."""
        require(type(payload) is dict and set(payload) <= PUBLIC_FIELDS
                and {*ACTOR_FIELDS, "split", "analysis_role", "dialogue_id", "query_ids"} <= set(payload),
                "Public actor whitelist; no labels, targets, strata, or policy/oracle fields")
        require((payload["split"], payload["analysis_role"]) in (("train", "calibration"), ("dev", "qualification")),
                "Separate TRAIN calibration and DEV replay roles")
        require("source_split" not in payload or payload["source_split"] == payload["split"], "Original source split preserved")
        require(type(payload["dialogue_id"]) is str and payload["dialogue_id"], "Public dialogue identity")
        query_ids = payload["query_ids"]
        require(type(query_ids) is list and query_ids and all(_integer(q) for q in query_ids)
                and query_ids == sorted(set(query_ids)) and len(query_ids) == len(payload["query_text_ids"]),
                "Sorted supplied public query identities")
        shape = [len(payload["turn_text_ids"]), len(query_ids), max(map(len, payload["candidate_ids"])), 10]
        require("lexical_shape" not in payload or payload["lexical_shape"] == shape, "Public lexical geometry")
        user_indices = payload.get("user_turn_indices")
        require(user_indices is None or (type(user_indices) is list and len(user_indices) == shape[0]
                and all(_integer(t) for t in user_indices) and user_indices == sorted(set(user_indices))),
                "Public source turn indices")
        self.check()
        self.counts["public_forward_attempts"] += 1
        begun = False
        self.encoder.eval()
        self.memory.eval()
        require(all(not p.requires_grad and p.grad is None for m in (self.encoder, self.memory) for p in m.parameters()),
                "All restored parameters remain gradient-free")
        try:
            with torch.no_grad():
                self.counts["encoding_attempts"] += 1
                vectors, work = encode_token_lists(self.encoder, payload["tokens"], **self.tokenizer_ids,
                                                   trainable=False, **ENCODING)
                self.counts["encoding_returns"] += 1
                self.work.update(work)
                actor, none = build_actor(vectors, **{key: payload[key] for key in ACTOR_FIELDS if key != "tokens"},
                                         memory_device="cpu")
                self.memory.begin_batch(actor, [{"layout": {"shape": shape}}])
                begun = True
                self.counts["memory_forward_attempts"] += 1
                self.check()
                logs = self.memory(*actor, none_index=none)
                self.counts["memory_forward_returns"] += 1
                require(logs.dtype == torch.float32 and logs.device.type == "cpu"
                        and tuple(logs.shape) == (1, *shape[:3]) and not logs.requires_grad, "Raw float32 CPU trajectory")
                supported = actor[4][:, None].expand_as(logs)
                require(torch.isfinite(logs[supported]).all().item() and torch.isneginf(logs[~supported]).all().item(),
                        "Finite supported trajectory and exact padding")
                require((logs.double().exp().sum(-1)-1).abs().max().item() <= 2e-6, "Raw trajectory mass tolerance")
                self.check()
                self.counts["public_forward_returns"] += 1
                return PublicForward(logs, payload["split"], payload["analysis_role"], payload["dialogue_id"],
                                     tuple(query_ids), tuple(tuple(ids) for ids in payload["candidate_ids"]),
                                     tuple(user_indices) if user_indices is not None else None,
                                     dict(work), copy.deepcopy(self.memory.audit))
        finally:
            if begun:
                self.merge(self.audit, self.memory.audit)

    def snapshot(self):
        """Metadata only; available after a failed forward, without another clock read."""
        return copy.deepcopy({"version": VERSION, "arm": self.progress["arm"], "counts": dict(self.counts),
                              "encoder_work": dict(self.work), "invariants": self.audit,
                              "encoder_attempted_work": dict(self.attempted_encoder_work),
                              "encoder_returned_work": dict(self.returned_encoder_work),
                              "restored_sha256": self.progress["restored_sha256"],
                              "checkpoint_sha256": self.progress["checkpoint_sha256"],
                              "synthetic_injection": self.synthetic_injection,
                              "encoder_device": str(next(self.encoder.parameters()).device), "memory_device": "cpu",
                              "dtype": "float32", "chunk_tokens": 254, "encoder_batch": 32,
                              "optimizer_created": False, "temperature_applied": False,
                              "scope": "Work from public full-dialogue forwards only; evaluator collection is separate."})

    def verify_parameters_unchanged(self):
        current = {"memory": tensor_digest(self.memory), "encoder": tensor_digest(self.encoder)}
        require(current == self.progress["restored_sha256"], "Inference changed restored state")
        return current


def load_final_checkpoint(checkpoint_path, *, arm, checkpoint_sha256, snapshot, model_files, tokenizer_ids,
                          expected_encoder_sha256, encoder_factory=None, memory_factory=None,
                          encoder_device="mps", check=lambda: None, progress=None):
    """Strictly restore pinned final weights; injected CPU modules are tests only.

    ``progress`` is a caller-owned metadata dict retained even if restoration
    fails. No optimizer state is accepted and no new model choice is made.
    """
    require(arm in ARMS and _pin(checkpoint_sha256) and _pin(expected_encoder_sha256), "Fixed arm and external state pins")
    require(type(tokenizer_ids) is dict and set(tokenizer_ids) == {"cls_id", "sep_id", "pad_id"}
            and all(_integer(v) for v in tokenizer_ids.values()), "Exact pinned tokenizer special IDs")
    injected = encoder_factory is not None or memory_factory is not None
    require(not injected or (callable(encoder_factory) and callable(memory_factory)), "Both synthetic factories must be explicit")
    require(encoder_device == "mps" or (injected and encoder_device == "cpu"), "Production MPS without fallback")
    require(callable(check), "Caller budget check")
    if progress is None:
        progress = {}
    require(type(progress) is dict and not progress, "Fresh restoration progress mapping")
    progress.update(arm=arm, checkpoint_sha256=checkpoint_sha256, counts=Counter(dict.fromkeys(COUNT_FIELDS, 0)))
    check()
    checkpoint_path = Path(checkpoint_path)
    require(file_digest(checkpoint_path) == checkpoint_sha256, "Checkpoint bytes before deserialization")
    assets = _local_assets(snapshot, model_files)
    check()
    if not injected:
        require(torch.backends.mps.is_available(), "Qualified MPS backend required")
    progress["counts"]["checkpoint_load_attempts"] += 1
    state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    progress["counts"]["checkpoint_load_returns"] += 1
    require(type(state) is dict and set(state) == ({"memory", "encoder"} if arm.startswith("trainable_") else {"memory"}),
            "Exact final checkpoint fields; no optimizer or extra state")
    memory_class, _, _ = _monitor_api()
    if encoder_factory is None:
        from transformers import AutoModel

        encoder_factory = AutoModel.from_pretrained
    progress["counts"]["encoder_load_attempts"] += 1
    encoder = encoder_factory(str(snapshot), local_files_only=True, trust_remote_code=False,
                              use_safetensors=True, attn_implementation="eager", dtype=torch.float32)
    progress["counts"]["encoder_load_returns"] += 1
    memory = memory_factory() if memory_factory is not None else memory_class("scalar")
    require(isinstance(encoder, nn.Module) and isinstance(memory, memory_class)
            and memory.method == "scalar", "Qualified encoder and monitored normalized scalar memory")
    encoder = encoder.cpu()
    memory = memory.cpu()
    require(all(p.dtype == torch.float32 for m in (encoder, memory) for p in m.parameters()), "Exact float32 restored modules")
    _strict_state(memory, state["memory"], "memory")
    if arm.startswith("trainable_"):
        _strict_state(encoder, state["encoder"], "encoder")
    encoder.to(encoder_device).eval().requires_grad_(False)
    memory.eval().requires_grad_(False)
    for module in (encoder, memory):
        for p in module.parameters():
            p.grad = None
    restored = {"encoder": tensor_digest(encoder), "memory": tensor_digest(memory)}
    require(restored["encoder"] == expected_encoder_sha256, "Published final encoder digest")
    require(file_digest(checkpoint_path) == checkpoint_sha256, "Checkpoint end identity")
    for entry in assets.values():
        require(file_digest(entry["path"]) == entry["sha256"], "Local asset end identity")
    progress["restored_sha256"] = restored
    check()
    return CalibrationInference(encoder, memory, tokenizer_ids, progress, check=check, synthetic_injection=injected)


def collect_endpoints(result, endpoint_rows):
    """Evaluator-only gather: returns raw [N,12] logs, IDs and target indices.

    Rows must already have authoritative global ``row_index`` values. Their
    order is preserved; there is no implicit sorting, filtering or probability
    transformation. Targets do not enter or modify ``forward_public``.
    """
    require(isinstance(result, PublicForward) and type(endpoint_rows) is list and endpoint_rows, "Public result and explicit evaluator rows")
    logs = result.log_probs
    require(logs.dtype == torch.float32 and logs.device.type == "cpu" and not logs.requires_grad, "Detached raw trajectory")
    n, seen, row_ids, source_ids = len(endpoint_rows), set(), [], set()
    output = torch.full((n, 12), -torch.inf, dtype=torch.float32)
    labels = torch.empty(n, dtype=torch.int64)
    for position, row in enumerate(endpoint_rows):
        require(type(row) is dict and row["split"] == result.split and row["dialogue_id"] == result.dialogue_id,
                "Evaluator split-qualified dialogue identity")
        require("analysis_role" not in row or row["analysis_role"] == result.analysis_role, "Evaluator role identity")
        require("source_split" not in row or row["source_split"] == result.split, "Evaluator original source split")
        require(all(_integer(row[key]) for key in ("row_index", "source_row_index", "time", "query_position", "query_index", "label_index")),
                "Integral evaluator addresses")
        t, q = row["time"], row["query_position"]
        require(t < logs.shape[1] and q < len(result.query_ids) and result.query_ids[q] == row["query_index"], "Exact evaluator query/time address")
        require((t, q) not in seen and row["row_index"] not in row_ids and row["source_row_index"] not in source_ids, "Unique endpoint/global/source row identity")
        seen.add((t, q))
        row_ids.append(row["row_index"])
        source_ids.add(row["source_row_index"])
        ids = result.candidate_ids[q]
        require(3 <= len(ids) <= 12 and row["label_index"] < len(ids)
                and ids[row["label_index"]] == row["label_id"], "Exact evaluator label identity/support")
        require("candidate_ids" not in row or tuple(row["candidate_ids"]) == ids, "Canonical candidate order")
        if result.user_turn_indices is not None and "turn_index" in row:
            require(row["turn_index"] == result.user_turn_indices[t], "Original public turn identity")
        vector = logs[0, t, q, :len(ids)]
        require(torch.isfinite(vector).all().item() and abs(vector.double().exp().sum().item()-1) <= 2e-6,
                "Unrepaired raw endpoint probabilities")
        output[position, :len(ids)] = vector
        labels[position] = row["label_index"]
    return {"log_probs": output, "row_indices": torch.tensor(row_ids, dtype=torch.int64), "label_indices": labels}
