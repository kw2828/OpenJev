"""Four candidate-scoring controls using exact native chess successors.

This is an untrained architecture prototype, not an efficacy result. All arms
share the frozen residual root initializer. The three refinement arms reuse its
core and candidate head after an action-conditioned initialization. Delta means
a shared linear projection of the complete nineteen-channel input difference,
not an exact nonlinear latent transition or a learned world model. Every call
starts afresh; the neural encoding contains no cross-move memory.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.chess_anchor import AnchorChess
from openjev.research.chess_spatial import (
    DEPTH,
    ENCODING_VERSION,
    INPUT_CHANNELS,
    WIDTH,
    _check_candidates,
    _check_plan_hash,
    action_planes,
    encode_board,
    encode_candidates,
)

ARMS = ("direct", "action_only", "delta", "full_afterstate")
CHECKPOINT_VERSION = "openjev-chess-candidate-v1"
ARCHITECTURE = "residual-root-shared-core-linear-input-delta-root-perspective-v1"
BRANCH_DEPTH = 2
CANDIDATE_CHUNK_SIZE = 128


def _arm(arm):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("Unknown candidate-scoring arm")


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def encode_batch(boards, arm, *, device="cpu"):
    """Return ``(tensor_dict, menus)`` with no targets or privileged fields.

    The tensor dictionary feeds ``CandidateChess.forward`` directly. Successors
    exist only for delta/full-afterstate and are flattened in row-major order of
    ``legal_mask.nonzero()``. If a caller permutes candidates, the corresponding
    successor rows must be permuted too. Padding has no successor rows.

    Successors come from isolated full-history native board copies. Their nineteen
    channels retain the PRE-MOVE root player's perspective and literal EP state.
    All input boards, move stacks and attached attributes remain unchanged.
    """
    _arm(arm)
    boards = list(boards)
    if not boards:
        raise ValueError("At least one nonterminal board is required")
    menus, features, observations, successors = [], [], [], []
    for board in boards:
        ids, candidates = encode_candidates(board)
        if not ids or board.is_game_over(claim_draw=False):
            raise ValueError("Candidate batches require nonterminal positions")
        perspective = board.turn
        menus.append(ids)
        features.append(candidates)
        observations.append(encode_board(board, perspective=perspective))
        if arm in ("delta", "full_afterstate"):
            for uci in ids:
                child = board.copy(stack=True)
                child.push_uci(uci)
                successors.append(encode_board(child, perspective=perspective))
    size = max(map(len, menus))
    candidates = torch.zeros((len(boards), size, 5), dtype=torch.long)
    mask = torch.zeros((len(boards), size), dtype=torch.bool)
    for index, value in enumerate(features):
        candidates[index, : len(value)] = torch.from_numpy(value)
        mask[index, : len(value)] = True
    batch = {
        "observations": torch.from_numpy(np.stack(observations)).to(device),
        "candidates": candidates.to(device),
        "legal_mask": mask.to(device),
    }
    if arm in ("delta", "full_afterstate"):
        batch["successors"] = torch.from_numpy(np.stack(successors)).to(device)
    return batch, tuple(menus)


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class CandidateChess(AnchorChess):
    def __init__(self, arm="direct", seed=17, width=WIDTH, root_depth=DEPTH, branch_depth=BRANCH_DEPTH):
        _arm(arm)
        _positive_int(branch_depth, "branch_depth")
        super().__init__("residual", seed=seed, width=width, depth=root_depth)
        self.arm, self.branch_depth = arm, branch_depth
        # Added only after the identical parent initializer, without changing the
        # caller's RNG or any common tensor. The projection exists in every arm.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.action_projection = nn.Conv2d(3, width, 1)

    @property
    def root_depth(self):
        return self.depth

    def parameter_counts(self):
        """Count parameters active in CE + root-value MSE, excluding unused heads."""
        auxiliary = sum(p.numel() for p in self.aux_head.parameters())
        action = sum(p.numel() for p in self.action_projection.parameters())
        stored = self.parameter_count()
        inactive_action = action if self.arm == "direct" else 0
        return {
            "stored": stored,
            "active": stored - auxiliary - inactive_action,
            "inactive_auxiliary": auxiliary,
            "inactive_action_projection": inactive_action,
        }

    def recurrence_cost(self, legal_count, *, depth=None):
        """Exact core iteration counts for one position, not a FLOP estimate."""
        _positive_int(legal_count, "legal_count")
        root_depth = self._depth(depth)
        refinement = 0 if self.arm == "direct" else legal_count * self.branch_depth
        child_root = legal_count * root_depth if self.arm == "full_afterstate" else 0
        return {
            "root_core_iterations": root_depth,
            "candidate_refinement_iterations": refinement,
            "successor_root_iterations": child_root,
            "total_core_iterations": root_depth + refinement + child_root,
        }

    def _root(self, observations, depth):
        hidden = self.encoder(observations)
        for _ in range(depth):
            hidden = self.core(hidden)
        return hidden

    def _candidate_scores(self, hidden, candidates):
        squares = hidden.flatten(2).transpose(1, 2)
        index = torch.arange(len(hidden), device=hidden.device)
        features = torch.cat(
            (
                squares[index, candidates[:, 0]],
                squares[index, candidates[:, 1]],
                hidden.mean(dim=(2, 3)),
                self.promotion_embedding(candidates[:, 2]),
                self.dx_embedding(candidates[:, 3]),
                self.dy_embedding(candidates[:, 4]),
            ),
            dim=-1,
        )
        return self.policy_head(features).squeeze(-1)

    def forward(
        self,
        observations,
        candidates,
        legal_mask,
        *,
        successors=None,
        depth=None,
        candidate_chunk_size=CANDIDATE_CHUNK_SIZE,
    ):
        """Return masked logits, one root value per position, and root hidden.

        Only valid candidates enter a refinement branch. Candidate chunks bound forward
        workspace, but training still retains their autograd graphs until backward.
        Memory-bounded training should accumulate gradients over complete positions in
        the same logical minibatch, not split a single sibling softmax into local losses.
        The unused frozen reconstruction head remains stored and receives no loss here.
        """
        root_depth = self._depth(depth)
        _positive_int(candidate_chunk_size, "candidate_chunk_size")
        if observations.ndim != 4 or observations.shape[1:] != (INPUT_CHANNELS, 8, 8):
            raise ValueError("Observations must have shape [batch,19,8,8]")
        if candidates.ndim != 3 or candidates.shape[0] != observations.shape[0]:
            raise ValueError("Candidates must have shape [batch,moves,5]")
        _check_candidates(candidates)
        if (
            legal_mask.dtype != torch.bool
            or legal_mask.shape != candidates.shape[:2]
            or len(observations) == 0
            or not legal_mask.any(dim=-1).all()
        ):
            raise ValueError("Every position requires a nonempty boolean legal-move mask")
        parameter = next(self.parameters())
        if (
            observations.device != candidates.device
            or legal_mask.device != candidates.device
            or parameter.device != observations.device
            or parameter.dtype != observations.dtype
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("Inputs must be finite and match the model device/dtype")
        valid = legal_mask.nonzero(as_tuple=False)
        if self.arm in ("delta", "full_afterstate"):
            if (
                successors is None
                or successors.shape != (len(valid), INPUT_CHANNELS, 8, 8)
                or successors.device != observations.device
                or successors.dtype != observations.dtype
                or not torch.isfinite(successors).all()
            ):
                raise ValueError("Expected one finite root-perspective successor per valid candidate")
        elif successors is not None:
            raise ValueError("Direct and action-only controls must not receive native successors")
        if self.arm == "direct":
            # Retain the exact frozen head path, including its padded layout.
            return super().forward(observations, candidates, legal_mask, depth=root_depth)

        root = self._root(observations, root_depth)
        value = torch.tanh(self.value_head(root.mean(dim=(2, 3)))).squeeze(-1)
        actions = candidates[legal_mask]
        scores = []
        for start in range(0, len(valid), candidate_chunk_size):
            end = min(start + candidate_chunk_size, len(valid))
            root_index = valid[start:end, 0]
            moves = actions[start:end]
            action = self.action_projection(action_planes(moves, dtype=observations.dtype))
            if self.arm == "full_afterstate":
                hidden = self._root(successors[start:end], root_depth) + action
            else:
                hidden = root[root_index] + action
                if self.arm == "delta":
                    delta = successors[start:end] - observations[root_index]
                    hidden = hidden + F.conv2d(delta, self.encoder[0].weight, bias=None, padding=1)
            for _ in range(self.branch_depth):
                hidden = self.core(hidden)
            scores.append(self._candidate_scores(hidden, moves))
        logits = torch.full(legal_mask.shape, -torch.inf, device=root.device, dtype=root.dtype)
        logits = logits.masked_scatter(legal_mask, torch.cat(scores))
        return logits, value, root

    @torch.no_grad()
    def choose(self, board, depth=None, *, candidate_chunk_size=CANDIDATE_CHUNK_SIZE):
        """Score the entire native legal menu with unchanged argmax selection."""
        self.eval()
        device = next(self.parameters()).device
        _sync(device)
        started = time.perf_counter()
        batch, menus = encode_batch([board], self.arm, device=device)
        logits, value, _ = self(**batch, depth=depth, candidate_chunk_size=candidate_chunk_size)
        ids = menus[0]
        probabilities = logits.softmax(-1).cpu()[0]
        if not torch.isfinite(probabilities).all() or not torch.isfinite(value).all():
            raise ValueError("Nonfinite model decision")
        result = {
            "choice": ids[int(probabilities.argmax())],
            "probabilities": {uci: float(probabilities[i]) for i, uci in enumerate(ids)},
            "value": float(value.cpu()[0]),
            "arm": self.arm,
            "seed": self.seed,
            "width": self.width,
            "depth": self._depth(depth),
            "branch_depth": self.branch_depth,
            "recurrence": "residual",
            "mode": "recurrent",
            "candidate_evaluations": len(ids),
            "candidate_branch_evaluations": 0 if self.arm == "direct" else len(ids),
            "native_successors": len(ids) if self.arm in ("delta", "full_afterstate") else 0,
            **self.recurrence_cost(len(ids), depth=depth),
            "probability_semantics": "uncalibrated legal-move softmax",
            "value_semantics": "one root-side value; no negated successor value scoring",
            "model_kind": "candidate-conditioned policy controls; no multi-ply search, learned dynamics or cross-move memory",
            "transition_semantics": {
                "direct": "root policy; no native successor inputs",
                "action_only": "root and action refinement; no native successor inputs",
                "delta": "native one-ply lookahead in root perspective; shared linear nineteen-channel input correction, not an exact latent transition",
                "full_afterstate": "native one-ply lookahead; re-encode every successor in fixed root perspective",
            }[self.arm],
            "timing_scope": "full CPU/native preparation, device transfer, inference, synchronization and response construction; model load excluded",
        }
        _sync(device)
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        return result

    def save(self, path, *, plan_sha256):
        """Exclusively save a tensor-only, CPU-float32 architecture checkpoint."""
        _check_plan_hash(plan_sha256)
        state = self.state_dict()
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Candidate checkpoints require finite float32 tensors")
        payload = {
            "format_version": CHECKPOINT_VERSION,
            "architecture": ARCHITECTURE,
            "encoding": ENCODING_VERSION,
            "arm": self.arm,
            "recurrence": "residual",
            "seed": self.seed,
            "width": self.width,
            "root_depth": self.depth,
            "branch_depth": self.branch_depth,
            "plan_sha256": plan_sha256,
            "state_dict": {name: value.detach().cpu() for name, value in state.items()},
        }
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(
        cls,
        path,
        *,
        expected_plan_sha256=None,
        expected_arm=None,
        expected_seed=None,
        expected_width=None,
        expected_root_depth=None,
        expected_branch_depth=None,
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "format_version",
            "architecture",
            "encoding",
            "arm",
            "recurrence",
            "seed",
            "width",
            "root_depth",
            "branch_depth",
            "plan_sha256",
            "state_dict",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != required
            or payload["format_version"] != CHECKPOINT_VERSION
            or payload["architecture"] != ARCHITECTURE
            or payload["encoding"] != ENCODING_VERSION
            or payload["recurrence"] != "residual"
        ):
            raise ValueError("Candidate checkpoint format or architecture mismatch")
        _check_plan_hash(payload["plan_sha256"])
        if expected_plan_sha256 is not None:
            _check_plan_hash(expected_plan_sha256)
            if payload["plan_sha256"] != expected_plan_sha256:
                raise ValueError("Candidate checkpoint plan mismatch")
        if expected_arm is not None:
            _arm(expected_arm)
            if payload["arm"] != expected_arm:
                raise ValueError("Candidate checkpoint arm mismatch")
        for name, expected in (
            ("seed", expected_seed),
            ("width", expected_width),
            ("root_depth", expected_root_depth),
            ("branch_depth", expected_branch_depth),
        ):
            if expected is not None and (type(expected) is not int or payload[name] != expected):
                raise ValueError(f"Candidate checkpoint {name} mismatch")
        model = cls(
            payload["arm"], payload["seed"], payload["width"], payload["root_depth"], payload["branch_depth"]
        )
        state = payload["state_dict"]
        reference = model.state_dict()
        if (
            not isinstance(state, dict)
            or set(state) != set(reference)
            or any(
                not isinstance(value, torch.Tensor)
                or value.shape != reference[name].shape
                or value.dtype != reference[name].dtype
                or not torch.isfinite(value).all()
                for name, value in state.items()
            )
        ):
            raise ValueError("Candidate checkpoint state tensors do not match the architecture")
        model.load_state_dict(state, strict=True)
        return model.eval()
