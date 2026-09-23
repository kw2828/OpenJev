"""Small causal predictors between fixed, externally scheduled planner queries.

The caller owns authenticated data, episode/query boundaries, fitting and costs.
Each forward starts a fresh query window: there is no persistent module state,
teacher forcing at skipped ages, planner import, filesystem access or simulator.
Only the supplied query scores are visible. Four raw costs divided by64 are the
prediction coordinates; only their centered values enter a network.
"""
from __future__ import annotations

import math

VERSION = "otto-recurrent-scores-v1"
KINDS = ("residual_gru", "direct_gru", "history_mlp", "current_mlp")
FEATURE_DIM, WINDOW, SCORE_DIM, SCALE = 31, 4, 4, 64.0
SHAPES = {
    "residual_gru": {"recurrent.weight_ih": (87, 35), "recurrent.weight_hh": (87, 29),
                     "recurrent.bias_ih": (87,), "recurrent.bias_hh": (87,),
                     "output.weight": (4, 29), "output.bias": (4,)},
    "history_mlp": {"hidden.weight": (43, 132), "hidden.bias": (43,),
                    "output.weight": (4, 43), "output.bias": (4,)},
    "current_mlp": {"hidden.weight": (82, 66), "hidden.bias": (82,),
                    "output.weight": (4, 82), "output.bias": (4,)},
}
SHAPES["direct_gru"] = dict(SHAPES["residual_gru"])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameter_count(kind):
    require(kind in KINDS, "declared score predictor")
    return sum(math.prod(shape) for shape in SHAPES[kind].values())


def valid_mask(lengths):
    """Boolean [B,4] mask; query age0 is valid but is not a forecast target."""
    import torch

    require(isinstance(lengths, torch.Tensor) and lengths.dtype == torch.int64
            and lengths.device.type == "cpu" and lengths.ndim == 1 and lengths.numel() > 0
            and bool(((lengths >= 1) & (lengths <= WINDOW)).all()), "CPU int64 lengths in1..4")
    return torch.arange(WINDOW, device="cpu")[None, :] < lengths[:, None]


def make_head(kind, seed):
    """Construct a locally seeded CPU float32 head without changing caller RNG.

    Default pinned-Torch hidden initialization is shared exactly by both GRUs.
    All output tensors begin at zero. Every valid output consequently holds the
    query scores exactly for the admitted power-of-two-roundtrip score domain.
    The protocol, rather than this pure factory, fixes the permitted fit seeds.
    """
    require(kind in KINDS and type(seed) is int and 0 <= seed < 2**32, "declared kind and uint32 seed")
    import torch

    class ScoreHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kind, self.seed = kind, seed
            if kind in ("residual_gru", "direct_gru"):
                self.recurrent = torch.nn.GRUCell(35, 29, device="cpu", dtype=torch.float32)
                width = 29
            else:
                inputs, width = (132, 43) if kind == "history_mlp" else (66, 82)
                self.hidden = torch.nn.Linear(inputs, width, device="cpu", dtype=torch.float32)
            self.output = torch.nn.Linear(width, SCORE_DIM, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                self.output.weight.zero_()
                self.output.bias.zero_()

        @staticmethod
        def centered(value):
            return value - value.mean(dim=-1, keepdim=True)

        def forward(self, features, query_scores, lengths):
            """Return raw float32 costs [B,4,4], with zeros at padded ages.

            Features are float32[B,4,31], query_scores float32[B,4], and lengths
            CPU int64[B] in1..4. Age0 equals query_scores and is not predicted.
            Active features are validated and read chronologically. Padding is
            never consumed and may contain arbitrary/nonfinite sentinels. Use
            valid_mask(lengths), excluding age0, for forecast loss/metrics.
            Neither features, scores nor lengths is mutated; windows in a batch
            and successive calls share weights only, never recurrent state.

            Extremely subnormal scores that lose bits on divide/multiply64 are
            rejected explicitly. This avoids a false exact-hold claim outside
            the ordinary finite teacher-cost domain. No scores are repaired.
            """
            mask = valid_mask(lengths)
            batch = lengths.numel()
            require(isinstance(features, torch.Tensor) and features.dtype == torch.float32
                    and features.device.type == "cpu" and features.shape == (batch, WINDOW, FEATURE_DIM),
                    "CPU float32[B,4,31] features")
            require(isinstance(query_scores, torch.Tensor) and query_scores.dtype == torch.float32
                    and query_scores.device.type == "cpu" and query_scores.shape == (batch, SCORE_DIM)
                    and bool(torch.isfinite(query_scores).all()), "finite CPU float32[B,4] query scores")
            require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in self.parameters()),
                    "CPU float32 predictor parameters")
            anchor = query_scores / SCALE
            require(torch.equal(anchor * SCALE, query_scores), "query scores must roundtrip through divide/multiply64")
            require(bool(torch.isfinite(features[:, 0]).all()), "finite active age0 features")
            anchor_center = self.centered(anchor)
            previous = anchor
            hidden = None
            if self.kind in ("residual_gru", "direct_gru"):
                hidden = self.recurrent(torch.cat((features[:, 0], anchor_center), dim=-1),
                                        torch.zeros(batch, 29, dtype=torch.float32, device="cpu"))
                require(bool(torch.isfinite(hidden).all()), "finite initialized recurrent state")
            outputs = [query_scores.clone()]
            for age in range(1, WINDOW):
                index = torch.nonzero(mask[:, age], as_tuple=False).flatten()
                full = torch.zeros_like(query_scores)
                if index.numel():
                    current = features[index, age]
                    require(bool(torch.isfinite(current).all()), "finite active forecast features")
                    if self.kind in ("residual_gru", "direct_gru"):
                        incoming = previous[index] if self.kind == "residual_gru" else anchor[index]
                        step_hidden = self.recurrent(torch.cat((current, self.centered(incoming)), dim=-1), hidden[index])
                        prediction = incoming + self.centered(self.output(step_hidden))
                        hidden = hidden.index_copy(0, index, step_hidden)
                    else:
                        if self.kind == "history_mlp":
                            prefix = torch.zeros(index.numel(), WINDOW, FEATURE_DIM, dtype=torch.float32, device="cpu")
                            prefix[:, :age + 1] = features[index, :age + 1]
                            observed = (torch.arange(WINDOW)[None, :] <= age).expand(index.numel(), -1).to(torch.float32)
                            inputs = torch.cat((prefix.flatten(1), observed, anchor_center[index]), dim=-1)
                        else:
                            inputs = torch.cat((features[index, 0], current, anchor_center[index]), dim=-1)
                        step_hidden = torch.tanh(self.hidden(inputs))
                        prediction = anchor[index] + self.centered(self.output(step_hidden))
                    raw = prediction * SCALE
                    require(bool(torch.isfinite(step_hidden).all()) and bool(torch.isfinite(raw).all()),
                            "finite forecast state and scores")
                    previous = previous.index_copy(0, index, prediction)
                    full = full.index_copy(0, index, raw)
                outputs.append(full)
            return torch.stack(outputs, dim=1)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ScoreHead()
    require({name: tuple(value.shape) for name, value in model.named_parameters()} == SHAPES[kind],
            "exact declared tensor geometry")
    require(sum(p.numel() for p in model.parameters()) == parameter_count(kind), "exact parameter budget")
    return model
