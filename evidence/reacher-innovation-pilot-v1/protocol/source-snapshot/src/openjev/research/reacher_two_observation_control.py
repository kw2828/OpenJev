"""Public-only geometry/CEM adapter for the exact two-observation history GRU.

No environment, random generator, checkpoint loader or class registry is owned
here. The caller supplies real packets, the previous ACTUALLY issued command,
and immutable search innovations. Native stepping and artifact serialization
belong to a future runner and are not included in this controller's wall time.

Unchanged scoring kernels retain the original learned heads/action-cost skip,
then use geometry with sequential float32 clipping for CEM. Every real boundary
pays all12 observation updates/all11 reconstruction transitions, even padding.
Hooks measure completed neural kernels, including failed prefixes; they do not
claim a failed model method completed or measure total FLOPs. The analytic skip
has no module hook: failures report conservative completed-call bounds.
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_adaptive_search import SearchInputs, SearchResult, search
from openjev.research.reacher_cache_control import (
    _plan,
    _state_arrays,
    _state_bytes,
    check_cap,
    require,
)
from openjev.research.reacher_geometry_control import (
    VERSION as KERNEL_VERSION,
)
from openjev.research.reacher_geometry_control import (
    _diagnostics,
    _journal,
    _preserve_failure,
    _score_bank,
    _selected_prediction,
)
from openjev.research.reacher_geometry_control import (
    scoring_configuration as _kernel_configuration,
)
from openjev.research.reacher_objective_training import canonical_tensor_hash
from openjev.research.reacher_two_observation_history import (
    REAL_KEYS,
    TwoObservationHistoryGRUWorldModel,
    parameter_and_operation_accounting,
)

VERSION = "reacher-two-observation-control-v1"
KIND = "two_observation_gru"


def scoring_configuration(noise_std):
    return {**_kernel_configuration("geometry", noise_std), "version": VERSION,
            "scoring_kernel_version": KERNEL_VERSION, "permitted_model_kinds": [KIND],
            "real_history": "last two actual observations and intervening issued commands",
            "timing_scope": "Controller validation, hooks, reconstruction, search, selected advance and snapshots; excludes caller native steps and artifact I/O.",
            "cost_limit": "Equal parameters/search proposals do not match total compute; buffer validation/copying and instrumentation remain charged."}


def model_identity(model, plan):
    _plan(plan)
    require(type(model) is TwoObservationHistoryGRUWorldModel, "Exact two-observation model class")
    require(not model.training and model.residual_reward is True
            and model.dt == plan["dt"] and model.noise_std == plan["noise_std"],
            "Eval-mode physical/residual configuration")
    require(not list(model.named_buffers()) and all(p.device.type == "cpu" and p.dtype == torch.float32
            and bool(torch.isfinite(p).all()) for p in model.parameters()), "Finite CPU float32 model without buffers")
    require("hidden_size" not in plan or plan["hidden_size"] == model.hidden_size, "Hidden size differs from plan")
    require(plan.get("score_modes", ["geometry"]) == ["geometry"]
            and plan.get("score_mode", "geometry") == "geometry", "Geometry scoring only")
    require(plan.get("engineering") is True or (plan["planning_horizon"] == 12 and plan["action_block"] == 3),
            "Production scorer uses horizon12/block3")
    return {"version": VERSION, "kind": KIND, "model_class": type(model).__name__,
            "configuration": model.configuration(), "width": model.hidden_size,
            "weight_tensor_sha256": canonical_tensor_hash(model.state_dict()),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters())}


def work_accounting(model, *, assimilate_samples, advance_samples):
    return {"assimilate_samples": assimilate_samples, "advance_samples": advance_samples,
            "operations": parameter_and_operation_accounting(model, assimilate_samples=assimilate_samples,
                                                              advance_samples=advance_samples),
            "compute_matched": False}


@contextmanager
def _neural_meter(model):
    """Temporary forward hooks, no method/registry/global replacement."""
    counts = {}
    handles = []
    try:
        for name, module in model.named_modules():
            if not isinstance(module, (torch.nn.GRUCell, torch.nn.Linear)):
                continue
            entry = counts[name] = {"attempted_sample_calls": 0, "completed_sample_calls": 0}

            def before(_module, inputs, entry=entry):
                entry["attempted_sample_calls"] += len(inputs[0])

            def after(_module, inputs, _output, entry=entry):
                entry["completed_sample_calls"] += len(inputs[0])

            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def _analytic_bounds(counts, completed_samples):
    # A returned parent advance has paid its analytic skip. A last completed
    # reward-head kernel alone cannot prove its subsequent non-module work ran.
    upper = counts.get("reward_head.2", {}).get("completed_sample_calls", 0)
    return {"completed_samples_lower_bound": completed_samples,
            "completed_samples_upper_bound": max(completed_samples, upper),
            "exact": completed_samples == upper}


def _validate_root(model, root, step=None):
    n = model._state_schema(root, real_boundary=False)
    require(bool((root["imagined_depth"] == 0).all()) and bool((root["real_index"] >= 0).all()),
            "Search requires an assimilated real root")
    index = int(root["real_index"][0, 0])
    require(bool((root["real_index"] == index).all()), "Search cases share a real clock")
    require(step is None or (type(step) is int and step == index), "Search step differs from real root")
    require(index < 50, "No decision at or beyond terminal50")
    return n, index


def _copy_work(root, records):
    n = len(root["packet"])
    completed = sum(r["model_advance_samples"] for r in records)
    attempted = sum(r["model_advance_attempted_samples"] for r in records)
    real_bytes = sum(root[key].numel() * root[key].element_size() for key in REAL_KEYS) // n
    return {"advance_state_schema_attempted_samples": attempted,
            "completed_advance_history_clone_tensor_bytes": completed * real_bytes,
            "completed_pending_action_clone_tensor_bytes": completed * 8,
            "candidate_root_repeat_scheduled_tensor_bytes": sum(_state_bytes(root) * r["bank"].shape[1] for r in records),
            "limits": "Payload accounting only. Failed-tail copies, temporaries, tensor objects and validators are not fully enumerated; all remain in measured wall time."}


def _failure(failure_out, records, error, metadata):
    metadata["banks"] = [_diagnostics(record)["metadata"] for record in records]
    error.two_observation_failure = copy.deepcopy(metadata)
    _preserve_failure(failure_out, records, error)
    if failure_out is not None:
        try:
            artifacts.write(Path(failure_out) / "adapter-work.json", metadata)
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Adapter work preservation failed: {preservation_error}")


@torch.no_grad()
def score_bank(plan, model, root, bank, deadline=float("inf"), *, failure_out=None):
    start, records, counts = time.perf_counter(), [], {}
    preserve = None
    try:
        if failure_out is not None:
            require(not Path(failure_out).exists(), "Failure artifact path must be exclusive")
            preserve = failure_out
        identity = model_identity(model, plan)
        _, index = _validate_root(model, root)
        require(isinstance(bank, np.ndarray) and bank.ndim == 4 and bank.shape[2] <= 50 - index,
                "Candidate horizon exceeds terminal boundary")
        check_cap(deadline)
        record = _journal(root, bank, "geometry")
        records.append(record)
        with _neural_meter(model) as counts:
            scores = _score_bank(plan, model, root, bank, "geometry", deadline, record)
        diagnostic = _diagnostics(record)
        diagnostic["metadata"].update({"model": identity, "configuration": scoring_configuration(plan["noise_std"]),
            "neural_kernels": counts, "history_work": _copy_work(root, records),
            "model_work": work_accounting(model, assimilate_samples=0, advance_samples=record["model_advance_samples"]),
            "wall_seconds": time.perf_counter() - start})
        return scores, diagnostic
    except BaseException as error:
        _failure(preserve, records, error, {"neural_kernels": counts, "wall_seconds": time.perf_counter() - start,
            "analytic_reward": _analytic_bounds(counts, sum(r["model_advance_samples"] for r in records))})
        raise


@torch.no_grad()
def score_search(plan, model, root, inputs, step, deadline=float("inf"), *, failure_out=None):
    """The frozen six-field geometry-search contract, with new class/work scope."""
    start, records, counts = time.perf_counter(), [], {}
    preserve = None
    try:
        if failure_out is not None:
            require(not Path(failure_out).exists(), "Failure artifact path must be exclusive")
            preserve = failure_out
        identity = model_identity(model, plan)
        _validate_root(model, root, step)
        require(type(inputs) is SearchInputs, "Explicit immutable SearchInputs required")
        check_cap(deadline)

        def score(bank):
            record = _journal(root, bank, "geometry")
            records.append(record)
            return _score_bank(plan, model, root, bank, "geometry", deadline, record)

        with _neural_meter(model) as counts:
            result = search("cem256", inputs, score, step=step, steps=plan["steps"],
                            planning_horizon=plan["planning_horizon"], action_block=plan["action_block"])
        sizes = [record["bank"].shape[1] for record in records]
        require(sizes == [64] * 4, "Exact four paid CEM256 batches")
        parts = [_diagnostics(record) for record in records]
        arrays = {key: np.concatenate([part["arrays"][key] for part in parts], axis=1)
                  for key in parts[0]["arrays"] if key != "root_target"}
        arrays.update(root_target=parts[0]["arrays"]["root_target"], selected_ids=result.selected_ids.copy(),
                      selected_actions=result.selected_actions.copy())
        work = {"candidate_evaluations": result.candidate_evaluations, "imagined_transitions": result.imagined_transitions,
            "root_tensor_bytes": _state_bytes(root),
            "max_single_candidate_state_tensor_bytes": max(record["max_state_bytes"] for record in records),
            "model_work": work_accounting(model, assimilate_samples=0, advance_samples=result.imagined_transitions),
            "neural_kernels": counts, "history_work": _copy_work(root, records),
            "geometry_samples": sum(record["geometry_samples"] for record in records),
            "geometry_seconds": sum(record["geometry_seconds"] for record in records),
            "model_advance_seconds": sum(record["model_advance_seconds"] for record in records),
            "diagnostic_array_bytes": sum(value.nbytes for value in arrays.values()),
            "tensor_bytes_are_not_peak_process_memory": True}
        metadata = {"version": VERSION, "model": identity, "step": step, "score_mode": "geometry",
            "configuration": scoring_configuration(plan["noise_std"]), "root_sha256": records[0]["root_sha256"],
            "callbacks": [{"candidate_start": i * 64, "candidate_stop": (i + 1) * 64, **part["metadata"]}
                          for i, part in enumerate(parts)], "work": work}
        check_cap(deadline)
        seconds = time.perf_counter() - start
        metadata["search_seconds"] = seconds
        return result, arrays["selected_rewards"], sizes, seconds, work, {"arrays": arrays, "metadata": metadata}
    except BaseException as error:
        _failure(preserve, records, error, {"neural_kernels": counts, "wall_seconds": time.perf_counter() - start,
            "analytic_reward": _analytic_bounds(counts, sum(r["model_advance_samples"] for r in records))})
        raise


@dataclass(frozen=True)
class Decision:
    action: np.ndarray
    search_result: SearchResult
    raw_rewards: np.ndarray
    callback_sizes: list[int]
    root_state: dict[str, np.ndarray]
    carried_state: dict[str, np.ndarray]
    diagnostics: dict
    metadata: dict


class TwoObservationController:
    """One episode of public decisions. Any failed decision is terminal.

    ``issued_action`` acknowledges what the caller actually sent after the last
    decision. An altered command requires a separately specified controller;
    silently retaining the planner's different command would corrupt history.
    Snapshots are independent copies, never writable handles into live state.
    """

    def __init__(self, plan, model, batch):
        start = time.perf_counter()
        self.plan = copy.deepcopy(plan)
        self.model = model
        self.identity = model_identity(model, self.plan)
        require(type(batch) is int and batch > 0, "Positive integer case count")
        self._state = model.initial(batch)
        self._issued = None
        self._step = 0
        self._failed = None
        self.setup_seconds = time.perf_counter() - start

    def snapshot(self):
        return {"next_step": self._step, "state": _state_arrays(self._state),
                "previous_selected_action": None if self._issued is None else self._issued.copy(),
                "failure": copy.deepcopy(self._failed), "setup_seconds": self.setup_seconds}

    @torch.no_grad()
    def decide(self, packet, inputs, *, issued_action=None, deadline=float("inf"), failure_out=None):
        require(self._failed is None, "Controller failed; no automatic retry")
        start = time.perf_counter()
        phase, root, diagnostic, selected = "validation", None, None, {}
        reconstruction_counts, selected_counts = {}, {}
        reconstruction_completed = False
        times = {"reconstruction_seconds": 0.0, "search_seconds": 0.0, "selected_advance_seconds": 0.0}
        n = len(self._state["packet"])
        can_preserve = False
        try:
            if failure_out is not None:
                require(not Path(failure_out).exists(), "Failure artifact path must be exclusive")
                can_preserve = True
            check_cap(deadline)
            require(self._step < 50, "No decision at or beyond terminal50")
            require(model_identity(self.model, self.plan) == self.identity, "Live model/configuration changed")
            if self._step == 0:
                require(issued_action is None, "No issued command before initial packet")
            else:
                require(isinstance(issued_action, np.ndarray) and issued_action.dtype == np.float32
                        and issued_action.shape == (n, 2) and np.isfinite(issued_action).all()
                        and np.array_equal(issued_action, self._issued), "Actual issued command differs from previous selection")
                require(np.array_equal(self._state["pending_action"].numpy(), issued_action), "Pending command differs from actual issued command")
            require(isinstance(packet, np.ndarray) and packet.dtype == np.float32 and packet.shape == (n, 8),
                    "Actual float32 public packet [case,8]")
            require(type(inputs) is SearchInputs and inputs.initial.shape[0] == n, "Explicit case-aligned innovations")
            phase, tick = "reconstruction", time.perf_counter()
            try:
                with _neural_meter(self.model) as reconstruction_counts:
                    root = self.model.assimilate(self._state, torch.from_numpy(packet.copy()))
                reconstruction_completed = True
            finally:
                times["reconstruction_seconds"] = time.perf_counter() - tick
            check_cap(deadline)
            phase, tick = "search", time.perf_counter()
            try:
                result, raw, sizes, _, work, diagnostic = score_search(
                    self.plan, self.model, root, inputs, self._step, deadline,
                    failure_out=None if failure_out is None else Path(failure_out) / "search")
            finally:
                times["search_seconds"] = time.perf_counter() - tick
            phase, tick = "selected_advance", time.perf_counter()
            try:
                with _neural_meter(self.model) as selected_counts:
                    carried, _, _ = _selected_prediction(self.plan, self.model, root,
                                                        result.selected_actions, "geometry", selected)
            finally:
                times["selected_advance_seconds"] = time.perf_counter() - tick
            phase = "snapshots_and_validation"
            diagnostic["arrays"].update(selected["arrays"])
            diagnostic["metadata"]["selected_advance"] = selected["metadata"]
            root_arrays, carried_arrays = _state_arrays(root), _state_arrays(carried)
            internal = {name: value.clone() for name, value in carried.items()}
            metadata = {"version": VERSION, "step": self._step, "model": copy.deepcopy(self.identity),
                "root_sha256": canonical_tensor_hash(root), "carried_sha256": canonical_tensor_hash(carried),
                "reconstruction_neural_kernels": reconstruction_counts, "selected_neural_kernels": selected_counts,
                "reconstruction_attempted_samples": n, "reconstruction_completed_samples": n,
                "model_work": work_accounting(self.model, assimilate_samples=n,
                                               advance_samples=n + result.imagined_transitions),
                "search": work, "selected_advance": copy.deepcopy(selected["metadata"]),
                "reconstruction_analytic_reward": _analytic_bounds(reconstruction_counts, 11 * n),
                "root_snapshot_tensor_bytes": _state_bytes(root), "carried_snapshot_tensor_bytes": _state_bytes(carried),
                "committed_state_clone_tensor_bytes": _state_bytes(carried),
                "real_boundary_schema_attempted_samples": n,
                "selected_history_work": {"advance_state_schema_attempted_samples": n,
                    "completed_history_clone_tensor_bytes": sum(root[key].numel() * root[key].element_size() for key in REAL_KEYS),
                    "completed_pending_action_clone_tensor_bytes": n * 8},
                "limits": scoring_configuration(self.plan["noise_std"])["timing_scope"], **times}
            decision = Decision(result.selected_actions.copy(), result, raw, sizes,
                                root_arrays, carried_arrays, diagnostic, metadata)
            require(model_identity(self.model, self.plan) == self.identity, "Inference changed model/configuration")
            check_cap(deadline)
            self._state, self._issued = internal, result.selected_actions.copy()
            self._step += 1
            metadata["controller_seconds"] = time.perf_counter() - start
            return decision
        except BaseException as error:
            # A next observation update proves the preceding reconstruction
            # transition (including analytic skip) returned successfully.
            guaranteed = (11 * n if reconstruction_completed else max(0,
                reconstruction_counts.get("observation_update", {}).get("attempted_sample_calls", 0) - n))
            self._failed = {"status": "failed", "step": self._step, "phase": phase,
                "exception_type": type(error).__name__, "message": str(error),
                "reconstruction_attempted_samples": n if reconstruction_counts else 0,
                "reconstruction_completed_samples": n if reconstruction_completed else 0,
                "reconstruction_neural_kernels": reconstruction_counts,
                "reconstruction_analytic_reward": _analytic_bounds(reconstruction_counts, guaranteed),
                "selected_neural_kernels": selected_counts, "selected_advance": selected.get("metadata"),
                "selected_analytic_reward": _analytic_bounds(selected_counts,
                    selected.get("metadata", {}).get("model_advance_samples", 0)),
                "search_failure": getattr(error, "two_observation_failure", None),
                "controller_seconds": time.perf_counter() - start, **times}
            if can_preserve:
                try:
                    out = Path(failure_out)
                    out.mkdir(parents=True, exist_ok=True)
                    arrays = {f"prior__{k}": v for k, v in _state_arrays(self._state).items()}
                    if root is not None:
                        arrays.update({f"root__{k}": v for k, v in _state_arrays(root).items()})
                    if diagnostic is not None:
                        arrays.update({f"scoring__{k}": v for k, v in diagnostic["arrays"].items()})
                    arrays.update({f"selected__{k}": v for k, v in selected.get("arrays", {}).items()})
                    artifacts.save_npz(out / "partial.npz", **arrays)
                    artifacts.write(out / "failed.json", self._failed)
                except BaseException as preservation_error:  # noqa: BLE001
                    error.add_note(f"Controller failure preservation failed: {preservation_error}")
            raise
