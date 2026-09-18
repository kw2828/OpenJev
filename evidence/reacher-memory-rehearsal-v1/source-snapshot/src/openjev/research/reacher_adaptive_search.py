"""Fixed-budget action search over explicit, immutable innovation arrays.

This component has no RNG, model, simulator, or optimizer dependencies. A scorer
receives float32 commands shaped [case, candidate, native_step, action] and must
return finite summed rewards [case, candidate], with larger values preferred.
Scoring must use the same root belief for every call; the callback must not
advance the real belief or inspect native state. Reward clipping belongs to that
callback, not to this generic search component.

Inputs retain the full planning horizon, even near the episode boundary. Search
uses their leading action blocks, repeats each block, and truncates at the
remaining native steps. Reported transition counts describe these scheduled
rollouts; they cannot verify operations performed inside an arbitrary callback.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

METHODS = ("rs64", "rs256", "cem256")
ANCHORS = ((0., 0.), (.1, 0.), (-.1, 0.), (0., .1), (0., -.1), (.2, .2), (-.2, -.2))
ELITES = 8
MIN_STD = 1e-3


def _immutable(value, dtype=None):
    """A bytes-backed copy cannot be made writable by toggling NumPy flags."""
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = _immutable(array, np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be representable in float64")
    return result


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identity(value):
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True)
    return hashlib.sha256(header.encode() + b"\n" + value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, eq=False)
class SearchInputs:
    """All prospective standard-normal draws, copied and frozen on creation.

    Shapes are [N, 64, C, 2], [N, 192, C, 2], and a tuple containing
    [N, 64, C, 2], [N, 64, C, 2], [N, 63, C, 2]. Here C is
    ceil(planning_horizon / action_block), including unused trailing blocks.
    The first seven initial draws are deliberately overwritten by fixed anchors,
    matching the original 64-candidate bank's allocation. No unused draw is
    converted into an additional evaluated candidate.
    """

    initial: np.ndarray
    random_extra: np.ndarray
    cem: tuple[np.ndarray, np.ndarray, np.ndarray]

    def __post_init__(self):
        initial = _real(self.initial, "initial innovations")
        if initial.ndim != 4 or initial.shape[1] != 64 or initial.shape[-1] != 2:
            raise ValueError("initial innovations must have shape [N,64,C,2]")
        n, _, chunks, _ = initial.shape
        if n < 1 or chunks < 1:
            raise ValueError("initial innovations need at least one case and block")
        extra = _real(self.random_extra, "random_extra innovations")
        if extra.shape != (n, 192, chunks, 2):
            raise ValueError("random_extra innovations must have shape [N,192,C,2]")
        if not isinstance(self.cem, tuple) or len(self.cem) != 3:
            raise ValueError("cem innovations must be a tuple of three arrays")
        cem = tuple(_real(value, f"cem generation {i + 1} innovations")
                    for i, value in enumerate(self.cem))
        for value, count in zip(cem, (64, 64, 63), strict=True):
            if value.shape != (n, count, chunks, 2):
                raise ValueError("cem innovation shapes must be [N,64,C,2], [N,64,C,2], [N,63,C,2]")
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "random_extra", extra)
        object.__setattr__(self, "cem", cem)

    def identities(self):
        """Identities include every supplied draw, even draws unused by an arm."""
        return tuple((name, _identity(value)) for name, value in (
            ("initial", self.initial), ("random_extra", self.random_extra),
            *((f"cem/{i + 1}", value) for i, value in enumerate(self.cem)),
        ))


@dataclass(frozen=True, eq=False)
class SearchStage:
    """Global candidate interval and parameters used before its one score call."""

    name: str
    start: int
    stop: int
    innovation_source: str
    innovation_count: int
    proposal_mean: np.ndarray | None = None
    proposal_std: np.ndarray | None = None
    fixed_scales: np.ndarray | None = None
    source_elite_ids: np.ndarray | None = None
    mean_candidate_id: int | None = None


@dataclass(frozen=True, eq=False)
class SearchResult:
    method: str
    horizon: int
    action_block: int
    selected_ids: np.ndarray
    selected_actions: np.ndarray
    selected_sequences: np.ndarray
    sequences: np.ndarray
    scores: np.ndarray
    candidate_ids: tuple[str, ...]
    stages: tuple[SearchStage, ...]
    input_identities: tuple[tuple[str, str], ...]
    candidate_evaluations_per_case: int
    candidate_evaluations: int
    imagined_transitions_per_case: int
    imagined_transitions: int


def search(
    method: str,
    inputs: SearchInputs,
    score: Callable[[np.ndarray], np.ndarray],
    *,
    step: int,
    steps: int = 50,
    planning_horizon: int = 12,
    action_block: int = 3,
) -> SearchResult:
    """Run RS64, RS256, or four-generation CEM256 without hidden evaluations.

    Initial slot 0..6 contains the fixed anchors. Slots 7..31 use std .25,
    slots 32..63 use std .75. RS256 adds 96 proposals at each scale. CEM
    refits from eight current-generation elites using population standard
    deviation (ddof=0), floor 1e-3, and no smoothing or retained elites. Its
    final generation has 63 random proposals and one evaluated proposal mean.
    All methods execute their best globally evaluated sequence, with ties
    resolved by earliest global candidate ID. The callback receives an
    immutable copy; outputs are validated and copied before the next call.
    Callback schedules are exactly [64], [256], and [64,64,64,64] for RS64,
    RS256, and CEM256 respectively. Full-horizon inputs preserve the initial
    bank's marginal distribution, not legacy shortened-array RNG assignment.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not isinstance(inputs, SearchInputs):
        raise TypeError("inputs must be SearchInputs")
    if not callable(score):
        raise TypeError("score must be callable")
    step = _integer(step, "step", 0)
    steps = _integer(steps, "steps", 1)
    planning_horizon = _integer(planning_horizon, "planning_horizon", 1)
    action_block = _integer(action_block, "action_block", 1)
    if step >= steps:
        raise ValueError("no planning is allowed at or beyond the terminal boundary")
    if inputs.initial.shape[2] != (planning_horizon + action_block - 1) // action_block:
        raise ValueError("innovation blocks must cover exactly the full planning horizon")
    horizon = min(planning_horizon, steps - step)
    chunks = (horizon + action_block - 1) // action_block
    n = inputs.initial.shape[0]
    all_chunks, all_sequences, all_scores, stages, identities = [], [], [], [], []

    def evaluate(raw, stage, candidate_ids):
        # Quantize before computing CEM moments so fitting sees scored commands.
        commands = np.clip(raw, -1., 1.).astype(np.float32)
        if not np.isfinite(commands).all():
            raise ValueError("nonfinite candidate commands")
        bank = _immutable(np.repeat(commands, action_block, axis=2)[:, :, :horizon])
        values = _real(score(bank), "scorer output")
        if values.shape != (n, stage.stop - stage.start):
            raise ValueError("scorer output must have exact shape [case,candidate]")
        all_chunks.append(commands)
        all_sequences.append(bank)
        all_scores.append(values)
        stages.append(stage)
        identities.extend(candidate_ids)

    scales = np.r_[np.full(32, .25), np.full(32, .75)]
    initial = inputs.initial[:, :, :chunks] * scales[None, :, None, None]
    for i, anchor in enumerate(ANCHORS):
        initial[:, i] = anchor
    initial_ids = [f"initial/anchor/{i}" if i < 7 else f"initial/random/{i}" for i in range(64)]

    if method == "rs256":
        extra_scales = np.r_[np.full(96, .25), np.full(96, .75)]
        extra = inputs.random_extra[:, :, :chunks] * extra_scales[None, :, None, None]
        evaluate(np.concatenate((initial, extra), axis=1), SearchStage(
            "single_bank", 0, 256, "initial+random_extra", 256,
            fixed_scales=_immutable(np.concatenate((scales, extra_scales))),
        ), initial_ids + [f"random_extra/{i}" for i in range(192)])
    else:
        evaluate(initial, SearchStage("initial", 0, 64, "initial", 64,
                                     fixed_scales=_immutable(scales)), initial_ids)

    if method == "cem256":
        for generation, innovations in enumerate(inputs.cem, 1):
            previous = all_chunks[-1]
            elite_local = np.argsort(-all_scores[-1], axis=1, kind="stable")[:, :ELITES]
            elite = previous[np.arange(n)[:, None], elite_local].astype(np.float64)
            mean = elite.mean(axis=1)
            std = np.maximum(elite.std(axis=1, ddof=0), MIN_STD)
            # Extreme finite innovations may overflow intermediate arithmetic;
            # clipping still defines finite saturated commands. NaNs fail below.
            with np.errstate(over="ignore", invalid="ignore"):
                proposed = mean[:, None] + std[:, None] * innovations[:, :, :chunks]
            start = generation * 64
            mean_id = start + 63 if generation == 3 else None
            if mean_id is not None:
                proposed = np.concatenate((proposed, mean[:, None]), axis=1)
            evaluate(proposed, SearchStage(
                f"cem/{generation}", start, start + 64, f"cem/{generation}", innovations.shape[1],
                proposal_mean=_immutable(mean), proposal_std=_immutable(std),
                source_elite_ids=_immutable(elite_local + stages[-1].start),
                mean_candidate_id=mean_id,
            ), [f"cem/{generation}/random/{i}" for i in range(innovations.shape[1])]
               + ([f"cem/{generation}/mean"] if mean_id is not None else []))

    sequences = _immutable(np.concatenate(all_sequences, axis=1))
    scores = _immutable(np.concatenate(all_scores, axis=1))
    selected = scores.argmax(axis=1)
    chosen = sequences[np.arange(n), selected]
    count = sequences.shape[1]
    return SearchResult(
        method=method, horizon=horizon, action_block=action_block,
        selected_ids=_immutable(selected), selected_actions=_immutable(chosen[:, 0]),
        selected_sequences=_immutable(chosen), sequences=sequences, scores=scores,
        candidate_ids=tuple(identities), stages=tuple(stages), input_identities=inputs.identities(),
        candidate_evaluations_per_case=count, candidate_evaluations=n * count,
        imagined_transitions_per_case=count * horizon, imagined_transitions=n * count * horizon,
    )
