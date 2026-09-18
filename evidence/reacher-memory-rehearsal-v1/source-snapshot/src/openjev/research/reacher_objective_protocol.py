"""Engineering protocol support for the fixed-planner objective comparison.

No source authentication, freeze, training or evaluation is performed here.
The runner owns those boundaries. Manifests allocate isolated RNGs without
drawing samples; synthetic tests use explicit engineering namespaces only.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

import numpy as np
import torch

from openjev.research import reacher_search_protocol as search_protocol
from openjev.research.reacher_adaptive_search import ANCHORS
from openjev.research.reacher_random_streams import generator_manifest
from openjev.research.reacher_world_models import GRUWorldModel

STUDY = "reacher-objective-ablation-v1"
SCORED_NAMESPACE = "reacher-objective-ablation-v1-scored"
ENGINEERING_NAMESPACES = (
    "reacher-objective-engineering-unit-v1",
    "reacher-objective-engineering-runner-v1",
    "reacher-objective-engineering-capacity-v1",
    "reacher-objective-engineering-whole-tree-v1",
)
ARMS = ("anchor", "raw", "latent")
PAIRS = ("pair0", "pair1", "pair2")
PANELS = search_protocol.PANELS
REFERENCES = search_protocol.REFERENCES
INPUT_NAMES = search_protocol.INPUT_NAMES
INPUT_COUNTS = search_protocol.INPUT_COUNTS
require = search_protocol.require


def settings(*, engineering=False):
    """Fresh settings; no seed derivation or RNG allocation occurs here."""
    namespace = ENGINEERING_NAMESPACES[0] if engineering else SCORED_NAMESPACE
    plan = {
        "study": STUDY, "version": 1, "rng_namespace": namespace,
        "engineering_rng_namespaces": [name for name in ENGINEERING_NAMESPACES if name != namespace],
        "arms": list(ARMS), "pairs": list(PAIRS), "panels": list(PANELS), "references": list(REFERENCES),
        "train_episodes": 768, "prediction_episodes": 96, "control_episodes": 64,
        "steps": 50, "epochs": 48, "batch_size": 32, "learning_rate": .001,
        "hidden_size": 64, "dt": .02, "residual_reward": True, "noise_std": .05,
        "ordinary_gap": 6, "shift_gap": 10, "threads": 2,
        "rollout_horizon": 5, "rollout_weight": .5, "reward_scale": 4.,
        "kl_weight": .01, "kl_balance": .8, "free_nats": 1., "gradient_clip": 10.,
        "auxiliary_horizons": [1, 3, 7], "ema_momentum": .99,
        "variance_weight": 0., "covariance_weight": 0.,
        "optimizer": {"name": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
                      "amsgrad": False, "foreach": False, "fused": False, "maximize": False,
                      "capturable": False, "differentiable": False, "decoupled_weight_decay": False},
        "calibration": {"batches": 4, "target_fraction": .5, "clip": [.01, 100.],
                        "parameters": ["observation_update", "transition"]},
        "planner": "cem256", "candidates": 64, "planning_horizon": 12, "action_block": 3,
        "particles": 32, "filter_bandwidth": .02, "bootstrap_samples": 4096,
        "search_parameters": {
            "elites": 8, "minimum_std": .001, "reward_clip": [-2.5, 0.],
            "callback_sizes": {"cem256": [64, 64, 64, 64]}, "warm_start": False,
            "elite_carryover": False, "proposal_momentum": 0.,
        },
        "reset_policy": "before_last_visible_packet",
    }
    plan["fit_order"] = [row["name"] for row in fit_manifest(plan)]
    plan["execution_order"] = execution_order(plan)
    return plan


def fit_manifest(plan):
    require(plan["arms"] == list(ARMS) and plan["pairs"] == list(PAIRS), "All nine paired fits required")
    rows = []
    for index, pair in enumerate(PAIRS):
        for arm in (*ARMS[index:], *ARMS[:index]):
            rows.append({"name": f"{arm}-{pair}", "arm": arm, "pair": pair,
                         "initialization_role": f"fit/student/{pair}",
                         "predictor_role": None if arm == "anchor" else f"fit/predictor/{pair}",
                         "minibatch_role": f"fit/minibatch/{pair}"})
    return rows


def execution_order(plan):
    """Exact 45 learned rows plus 12 intact reference rows, in panel order."""
    rows = []
    for panel in PANELS:
        for fit in fit_manifest(plan):
            for reset in ((False,) if panel == "full" else (False, True)):
                label = fit["name"] + ("-reset" if reset else "")
                rows.append({"panel": panel, "fit": fit["name"], "arm": fit["arm"],
                             "pair": fit["pair"], "planner": "cem256", "reset": reset, "label": label,
                             "path": f"control/{panel}/{label}"})
        rows.extend({"panel": panel, "reference": name, "reset": False, "label": name,
                     "path": f"control/{panel}/{name}"} for name in REFERENCES)
    return rows


def validate_settings(plan, *, engineering=False):
    """Validate study-owned fields; the runner separately binds provenance/caps."""
    expected = settings(engineering=engineering)
    require(set(expected) <= set(plan), "Missing objective settings")
    flexible = {"rng_namespace", "engineering_rng_namespaces"}
    if engineering:
        flexible |= {"train_episodes", "prediction_episodes", "control_episodes", "epochs",
                     "batch_size", "hidden_size", "threads", "bootstrap_samples"}
    for key, value in expected.items():
        if key not in flexible:
            require(_identity(plan[key]) == _identity(value), f"Objective setting changed: {key}")
    for key in ("train_episodes", "prediction_episodes", "control_episodes", "epochs", "batch_size",
                "hidden_size", "threads", "bootstrap_samples"):
        require(type(plan[key]) is int and plan[key] > 0, f"Invalid positive integer: {key}")
    require(plan["train_episodes"] % plan["batch_size"] == 0, "Complete equal-size training batches required")
    require(isinstance(plan["rng_namespace"], str) and bool(plan["rng_namespace"].strip()), "RNG namespace")
    if engineering:
        require(plan["rng_namespace"] in ENGINEERING_NAMESPACES, "Declared engineering namespace required")
    else:
        require(plan["rng_namespace"] == SCORED_NAMESPACE, "Scored namespace changed")
        require(plan["engineering_rng_namespaces"] == list(ENGINEERING_NAMESPACES), "Engineering exclusions changed")
    namespaces = plan["engineering_rng_namespaces"]
    require(isinstance(namespaces, list) and all(isinstance(name, str) and name for name in namespaces)
            and len(set(namespaces)) == len(namespaces) and plan["rng_namespace"] not in namespaces,
            "Engineering exclusion membership")
    require(plan["fit_order"] == [row["name"] for row in fit_manifest(plan)], "Fit ordering changed")
    require(plan["execution_order"] == execution_order(plan), "Control ordering changed")
    return plan


def registry(plan):
    """NumPy seeds, preserving existing search-helper role names."""
    namespace = plan["rng_namespace"]
    require(isinstance(namespace, str) and bool(namespace.strip()), "Explicit RNG namespace required")
    roles = []
    for split in ("control", "prediction"):
        count = plan[f"{split}_episodes"]
        require(type(count) is int and count > 0, "Positive cohort count required")
        for purpose in ("reset", "actuator_noise", "sensor_schedule"):
            roles.extend(f"{split}/{purpose}/{i}" for i in range(count))
        if split == "prediction":
            roles.extend(f"prediction/exploration/{i}" for i in range(count))
    roles.extend(f"planner/particle_filter/{i}" for i in range(plan["control_episodes"]))
    roles.extend(("floor/uniform/0", "analysis/bootstrap/0"))
    for step in range(plan["steps"]):
        roles.extend(f"planner/control/{step}/{name}" for name in INPUT_NAMES)
    require(len(roles) == len(set(roles)), "Duplicate NumPy role")
    return {role: search_protocol.named_seed(namespace, role) for role in roles}


def seed32(namespace, role):
    """Restrict CPU Torch seeds to 32 bits, avoiding high-bit stream aliases."""
    return search_protocol.named_seed(namespace, role) & ((1 << 32) - 1)


def torch_registry(plan):
    require(isinstance(plan["rng_namespace"], str) and bool(plan["rng_namespace"].strip()), "Explicit RNG namespace required")
    require(plan["pairs"] == list(PAIRS), "All three initialization identities required")
    return {f"fit/{purpose}/{pair}": seed32(plan["rng_namespace"], f"fit/{purpose}/{pair}")
            for pair in PAIRS for purpose in ("student", "predictor", "minibatch")}


def torch_generator_manifest(values):
    """CPU initial-state hashes without sampling; normalize prior seed aliases."""
    result = {}
    for role, value in values.items():
        require(type(value) is int and 0 <= value < 1 << 64, "Invalid Torch seed")
        normalized = value & ((1 << 32) - 1)
        generator = torch.Generator(device="cpu").manual_seed(normalized)
        result[role] = {"seed": value, "effective_seed32": normalized, "device": "cpu",
                        "initial_state_sha256": hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                        "draws_for_manifest": 0}
    return result


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _states(manifest):
    return {row["initial_state_sha256"] for row in manifest.values()}


def stream_contract(plan, prior_registries=(), prior_torch_registries=(), prior_plans=()):
    """Enumerate current, prior and full-coverage engineering streams with no draws.

    The runner supplies all authenticated prior registries, including earlier
    engineering exclusions and original collection/shuffle/initialization
    seeds. This module does not read historical artifacts or infer authority.
    """
    current, torch_current = registry(plan), torch_registry(plan)
    require(len(set(current.values())) == len(current), "NumPy root seed collision")
    require(len(set(torch_current.values())) == len(torch_current), "Torch root seed collision")
    require(not (set(current.values()) & set(torch_current.values())), "Cross-runtime root seed collision")
    numpy_states, torch_states = generator_manifest(current), torch_generator_manifest(torch_current)
    require(len(_states(numpy_states)) == len(numpy_states), "NumPy generator-state collision")
    require(len(_states(torch_states)) == len(torch_states), "Torch generator-state collision")
    namespaces = plan["engineering_rng_namespaces"]
    require(isinstance(namespaces, list) and all(isinstance(name, str) and name for name in namespaces)
            and len(namespaces) == len(set(namespaces)), "Engineering namespace exclusion schema")
    exclusions = []
    for namespace in namespaces:
        full = settings()
        full["rng_namespace"] = namespace
        previous, torch_previous = registry(full), torch_registry(full)
        previous_states, torch_previous_states = generator_manifest(previous), torch_generator_manifest(torch_previous)
        require(not (set(current.values()) & set(previous.values())), "Engineering NumPy root collision")
        require(not (_states(numpy_states) & _states(previous_states)), "Engineering NumPy state collision")
        require(not (_states(torch_states) & _states(torch_previous_states)), "Engineering Torch state collision")
        exclusions.append({"namespace": namespace,
                           "coverage": {"control_episodes": 64, "prediction_episodes": 96, "steps": 50,
                                        "pairs": list(PAIRS)},
                           "registry": previous, "generators": previous_states,
                           "torch_registry": torch_previous, "torch_generators": torch_previous_states})
    for previous in prior_registries:
        require(not (set(current.values()) & set(previous.values())), "Prior NumPy root collision")
        require(not (_states(numpy_states) & _states(generator_manifest(previous))), "Prior NumPy state collision")
    for previous in prior_torch_registries:
        require(not (_states(torch_states) & _states(torch_generator_manifest(previous))), "Prior Torch state collision")
    return {
        "namespace": plan["rng_namespace"], "registry": current, "generators": numpy_states,
        "torch_registry": torch_current, "torch_generators": torch_states,
        "engineering_exclusions": exclusions, "priors": list(prior_plans),
        "prior_numpy_registry_sha256": [_identity(value) for value in prior_registries],
        "prior_torch_registry_sha256": [_identity(value) for value in prior_torch_registries],
        "draws_for_manifest": 0, "full_horizon_innovations": True,
        "unused_random_extra": "Generated and saved for SearchInputs compatibility; never scored by CEM256.",
        "intentional_reuse": [
            "Each paired triple shares student initialization and minibatch order; Raw/Latent share predictor initialization.",
            "All objective/fit/panel/reset rows share corresponding control resets, actuator noise and schedule phase.",
            "All rows share full-horizon innovations at each decision; adapted CEM sequences can differ.",
            "Reference particle-filter children retain separate initialization/process/resampling purposes.",
            "Uniform commands and bootstrap draws are paired; no prediction stream shares a control stream.",
        ],
    }


def seed(plan, role):
    key = "torch_registry" if role.startswith("fit/") else "registry"
    return plan["random_stream_contract"][key][role]


def torch_generator(plan, role):
    require(role in torch_registry(plan), "Unknown Torch RNG role")
    value = seed(plan, role)
    require(type(value) is int and 0 <= value < 1 << 32 and value == torch_registry(plan)[role],
            "Bound Torch seed differs from role-derived seed")
    return torch.Generator(device="cpu").manual_seed(value)


@contextmanager
def initialization_rng(plan, pair, component="student"):
    """Temporarily install an isolated CPU generator, restoring global RNG even on error."""
    require(pair in PAIRS and component in ("student", "predictor"), "Unknown initialization role")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch_generator(plan, f"fit/{component}/{pair}").get_state())
        yield


def minibatch_orders(plan, pair):
    require(pair in PAIRS, "Unknown minibatch pair")
    generator = torch_generator(plan, f"fit/minibatch/{pair}")
    return torch.stack([torch.randperm(plan["train_episodes"], generator=generator)
                        for _ in range(plan["epochs"])])


def phase(plan, index, split="control"):
    require(split in ("control", "prediction") and type(index) is int
            and 0 <= index < plan[f"{split}_episodes"], "Invalid cohort index")
    return search_protocol.phase(plan, index, split)


def schedule(plan, index, panel, split="control"):
    require(split in ("control", "prediction") and type(index) is int
            and 0 <= index < plan[f"{split}_episodes"], "Invalid cohort index")
    return search_protocol.schedule(plan, index, panel, split)


def draw_control_inputs(plan, step):
    require(type(step) is int and 0 <= step < plan["steps"], "No innovation draw beyond terminal boundary")
    return search_protocol.draw_inputs(plan, f"planner/control/{step}", plan["control_episodes"])


def common_bank(inputs, step, plan):
    """Same anchored distribution/arithmetic as the frozen search reference helper."""
    require(type(step) is int and 0 <= step < plan["steps"], "No candidate bank beyond terminal boundary")
    horizon = min(plan["planning_horizon"], plan["steps"] - step)
    chunks = (horizon + plan["action_block"] - 1) // plan["action_block"]
    value = inputs.initial[:, :, :chunks] * np.r_[np.full(32, .25), np.full(32, .75)][None, :, None, None]
    for index, anchor in enumerate(ANCHORS):
        value[:, index] = anchor
    return np.repeat(np.clip(value, -1, 1).astype(np.float32), plan["action_block"], axis=2)[:, :, :horizon]


def reset_steps(valid):
    """Precursor indices from the fixed mask only, never from future measurements."""
    valid = np.asarray(valid)
    require(valid.dtype == np.bool_ and valid.ndim == 1 and len(valid) >= 2 and bool(valid[0]),
            "Sensing schedule must be boolean, one-dimensional and initially observed")
    return tuple(np.flatnonzero(valid[:-1] & ~valid[1:]).tolist())


def assimilate_control(model, state, packets, step, schedules, *, reset=False):
    """Reset hidden before current assimilation; return fresh state and reset mask.

    Only public tensors and a fixed sensing mask are accepted. No physical
    state, native clock, noise generator, actions or future packet is supplied.
    Closed-loop paired runs may diverge after this intervention.
    """
    require(isinstance(model, GRUWorldModel), "Deterministic GRU interface required")
    require(type(reset) is bool and type(step) is int, "Reset flag/step types")
    parameter = next(model.parameters())
    require(isinstance(packets, torch.Tensor) and packets.ndim == 2 and packets.shape[1] == 8
            and packets.shape[0] > 0 and packets.dtype == parameter.dtype and packets.device == parameter.device,
            "Public packet shape/dtype/device")
    valid = np.asarray(schedules)
    require(valid.dtype == np.bool_ and valid.ndim == 2 and valid.shape[0] == len(packets)
            and valid.shape[1] >= 2 and 0 <= step < valid.shape[1] - 1, "Batch sensing schedule/step")
    precursors = [reset_steps(row) for row in valid]
    expected = torch.tensor(valid[:, step], device=packets.device)
    require(torch.all((packets[:, 6] == 0) | (packets[:, 6] == 1)).item()
            and torch.equal(packets[:, 6] > .5, expected), "Current packet disagrees with sensing schedule")
    public = torch.cat((torch.where(expected[:, None], packets[:, :4], 0), packets[:, 4:]), dim=-1)
    require(torch.isfinite(public).all().item() and torch.all(public[:, 7] >= 0).item(), "Invalid public packet")
    require(set(state) == {"hidden", "packet"} and state["hidden"].shape == (len(packets), model.hidden_size)
            and state["packet"].shape == packets.shape, "GRU state schema")
    require(all(value.dtype == parameter.dtype and value.device == parameter.device for value in state.values()),
            "GRU state dtype/device")
    mask = torch.tensor([reset and step in times for times in precursors], dtype=torch.bool, device=packets.device)
    require(not torch.any(mask & ~expected).item(), "Reset precursor must be observed")
    working = {name: value.clone() for name, value in state.items()}
    if mask.any():
        initial = model.initial(len(packets), packets.device)
        working["hidden"] = torch.where(mask[:, None], initial["hidden"], working["hidden"])
    return model.assimilate(working, public), mask


def reset_events(step, mask, packets):
    """Small saved-event schema: step, case identity and current public packet."""
    require(type(step) is int and step >= 0 and mask.dtype == torch.bool
            and mask.shape == (len(packets),), "Reset event shape/types")
    selected = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    require(all(packets[index, 6].item() == 1 and torch.isfinite(packets[index]).all().item()
                for index in selected), "Reset event requires an observed finite packet")
    return [{"step": step, "case": index, "packet": packets[index].detach().cpu().tolist()} for index in selected]
