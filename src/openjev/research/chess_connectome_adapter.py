"""Untrained topology adapters on a frozen four-step direct chess backbone.

Only the interface, edge magnitudes and node biases are trainable. The complete
root encoder/recurrence and readouts are copied and frozen. Every call starts a
new graph state; there is no successor input, search, learned world model or
cross-move memory. A zero output projection reproduces the unchanged direct
model at initialization while allowing that projection to learn immediately.

All three controls deliberately execute dense matrix multiplications. Sparse
parameterization is not a sparse-kernel speed claim. Anatomical synapse counts
never initialize or scale weights. This module neither downloads nor saves graph
assets or weights; external graph-derived license terms remain applicable.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.chess_candidate import CandidateChess, _sync, encode_batch
from openjev.research.chess_spatial import INPUT_CHANNELS, _check_candidates
from openjev.research.connectome_graph import SignedGraph, graph_sha256

MODES = ('sparse', 'dense', 'node_local')
MAX_NODES = 4096
ROOT_DEPTH = 4
# Degree normalization starts each nonempty row at absolute edge mass one
# before any signed parallel-edge contributions are combined.
# The tanh/leaky updates bound node states without suppressing messages by 10x.
INITIAL_MAGNITUDE = 1.0


def _integer(value, name: str, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer at least {minimum}')
    return value


def _seed(value, name: str):
    value = _integer(value, name)
    if value > 2**63-1:
        raise ValueError(f'{name} must fit in a nonnegative int64')
    return value


def _rng(seed: int, purpose: str) -> np.random.Generator:
    derived = int.from_bytes(hashlib.sha256(f'{seed}:{purpose}'.encode()).digest()[:8], 'little')
    return np.random.Generator(np.random.PCG64(derived))


def _balanced_slots(node_ids: np.ndarray, slots: int, seed: int) -> np.ndarray:
    """Assign packed identities to balanced channels-first spatial slots."""
    rng = _rng(seed, 'connectome-slot-mapping-v1')
    values = np.resize(rng.permutation(slots), len(node_ids))
    values = rng.permutation(values)
    result = np.empty(len(node_ids), dtype=np.int64)
    result[np.argsort(node_ids)] = values
    return result


class ConnectomeChessAdapter(nn.Module):
    """Residual graph adapter with explicit synthetic or anatomical topology.

    ``backbone`` must be a direct, residual CandidateChess with root depth four.
    Width 32 is the research proposal; smaller widths and graphs support unit
    fixtures. Slot ordering is channel * 64 + rank * 8 + file. Node assignment
    depends only on original packed IDs, count, slot count and ``mapping_seed``.
    It is an artificial board interface, not a biological sensory mapping.

    Sparse mode retains all supplied signed edges, including parallel/self edges.
    Dense mode uses every directed pair including self, with fixed seeded random
    signs. Node-local mode uses a positive self edge per node. Only a separately
    matched rewired sparse graph isolates higher-order biological topology.
    """

    def __init__(self, backbone: CandidateChess, graph: SignedGraph, *, mode='sparse',
                 seed=17, mapping_seed=0, slot_channels=16, steps=4, alpha=0.5):
        super().__init__()
        if not isinstance(backbone, CandidateChess):
            raise TypeError('backbone must be a CandidateChess')
        if (backbone.arm != 'direct' or backbone.root_depth != ROOT_DEPTH
                or backbone.recurrence != 'residual'):
            raise ValueError('Expected a direct residual backbone with root depth four')
        if not isinstance(graph, SignedGraph):
            raise TypeError('graph must be a SignedGraph')
        if type(mode) is not str or mode not in MODES:
            raise ValueError('Unknown connectome adapter mode')
        seed, mapping_seed = _seed(seed, 'seed'), _seed(mapping_seed, 'mapping_seed')
        slot_channels = _integer(slot_channels, 'slot_channels', minimum=1)
        steps = _integer(steps, 'steps', minimum=1)
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('alpha must be finite and in (0,1]')
        if len(graph.node_ids) > MAX_NODES:
            raise ValueError(f'Dense-execution prototype supports at most {MAX_NODES} nodes')
        self.mode, self.seed, self.mapping_seed = mode, seed, mapping_seed
        self.slot_channels, self.steps, self.alpha = slot_channels, steps, float(alpha)
        self.nodes, self.width = len(graph.node_ids), backbone.width
        self.slot_count = 64*slot_channels
        self.backbone = copy.deepcopy(backbone)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.grad = None
        reference = next(self.backbone.parameters())
        if any(parameter.device != reference.device or parameter.dtype != reference.dtype
               or not torch.isfinite(parameter).all() for parameter in self.backbone.parameters()):
            raise ValueError('Backbone parameters must be finite with a common device and dtype')

        # Module constructors use CPU RNG internally; restore it and then use a
        # private CPU generator. No manual_seed call touches global/device RNGs.
        with torch.random.fork_rng(devices=[]):
            self.input_projection = nn.Conv2d(self.width, slot_channels, 1, device='cpu')
            self.output_projection = nn.Conv2d(slot_channels, self.width, 1, device='cpu')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        nn.init.kaiming_uniform_(self.input_projection.weight, a=math.sqrt(5), generator=generator)
        bound = 1/math.sqrt(self.width)
        nn.init.uniform_(self.input_projection.bias, -bound, bound, generator=generator)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        if mode == 'sparse':
            sources = graph.sources.copy()
            destinations = graph.destinations.copy()
            signs = graph.signs.copy()
        elif mode == 'dense':
            sources = np.tile(np.arange(self.nodes), self.nodes)
            destinations = np.repeat(np.arange(self.nodes), self.nodes)
            signs = _rng(seed, 'connectome-dense-signs-v1').choice([-1, 1], self.nodes**2)
        else:
            sources = np.arange(self.nodes)
            destinations = np.arange(self.nodes)
            signs = np.ones(self.nodes, dtype=np.int8)
        self.edges = len(sources)
        mapping = _balanced_slots(graph.node_ids, self.slot_count, mapping_seed)
        self.register_buffer('node_ids', torch.tensor(graph.node_ids.copy(), dtype=torch.long))
        self.register_buffer('groups', torch.tensor(graph.groups.copy(), dtype=torch.long))
        self.register_buffer('sources', torch.tensor(sources, dtype=torch.long))
        self.register_buffer('destinations', torch.tensor(destinations, dtype=torch.long))
        self.register_buffer('signs', torch.tensor(signs, dtype=torch.float32))
        self.register_buffer('indegree', torch.tensor(
            np.bincount(destinations, minlength=self.nodes), dtype=torch.float32))
        self.register_buffer('node_slots', torch.tensor(mapping, dtype=torch.long))
        self.register_buffer('slot_occupancy', torch.tensor(
            np.bincount(mapping, minlength=self.slot_count), dtype=torch.float32))
        self.edge_log_gain = nn.Parameter(torch.full((self.edges,), math.log(math.expm1(INITIAL_MAGNITUDE))))
        self.node_bias = nn.Parameter(torch.zeros(self.nodes))
        self._source_graph_sha256 = graph_sha256(graph)
        self._graph_provenance = dict(graph.provenance)
        self._source_anatomical_counts = graph.anatomical_counts is not None
        self._mapping_sha256 = hashlib.sha256(mapping.astype('<i8').tobytes()).hexdigest()
        self.to(device=reference.device, dtype=reference.dtype)
        self.train(True)

    def train(self, mode=True):
        if type(mode) is not bool:
            raise ValueError('training mode must be boolean')
        super().train(mode)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        return self

    @property
    def config(self):
        return {
            'architecture': 'frozen-direct-root4-signed-graph-residual-adapter-v1',
            'mode': self.mode, 'seed': self.seed, 'mapping_seed': self.mapping_seed,
            'slot_channels': self.slot_channels, 'slot_count': self.slot_count,
            'graph_steps': self.steps, 'alpha': self.alpha, 'nodes': self.nodes,
            'edges': self.edges, 'backbone_width': self.width,
            'backbone_seed': self.backbone.seed, 'backbone_root_depth': ROOT_DEPTH,
            'mapping_sha256': self._mapping_sha256,
            'source_graph_sha256': self._source_graph_sha256,
            'execution': 'Dense matrix assembly and dense matmul in every mode',
            'normalization': 'Each signed edge divided by destination stored-edge indegree, clamped at one',
            'node_interface': 'Artificial balanced channels-first spatial slot mapping, all nodes driven/read',
            'state': 'Fresh h0=tanh(projected board drive) per call; no cross-move state',
            'residual': 'Occupancy-mean of final-minus-initial node state; zero-initialized output projection',
            'anatomical_counts_used': False,
        }

    @property
    def provenance(self):
        return {
            **self._graph_provenance, 'source_graph_sha256': self._source_graph_sha256,
            'source_anatomical_counts_present': self._source_anatomical_counts,
            'anatomical_counts_used': False, 'pretrained_graph_weights_used': False,
            'graph_control': self.mode, 'source_packed_node_identity_preserved': True,
            'derivative_asset_policy': 'No graph or weight persistence; external graph terms remain applicable',
        }

    def parameter_counts(self):
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        backbone = self.backbone.parameter_counts()
        return {
            'stored': count(self),
            'trainable': sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            'frozen_backbone': count(self.backbone),
            'active_frozen_backbone': backbone['active'],
            'inactive_frozen_backbone': backbone['stored']-backbone['active'],
            'input_projection': count(self.input_projection),
            'output_projection': count(self.output_projection),
            'edge_magnitudes': self.edge_log_gain.numel(), 'node_bias': self.node_bias.numel(),
        }

    def computation_counts(self, batch_size=1, legal_candidates=1):
        """Exact selected MAC/work counts, not FLOPs or a latency estimate.

        ``legal_candidates`` is the padded candidate-slot count PER position;
        the frozen head also processes masked padding. Graph matrix assembly is
        performed once per forward, then reused for all graph steps. Recorded
        MACs omit biases, activations, softplus, normalization, gathers, pooling,
        tensor validation and native encoding; full choose latency includes them.
        """
        batch = _integer(batch_size, 'batch_size', minimum=1)
        candidates = _integer(legal_candidates, 'legal_candidates', minimum=1)
        width, channels = self.width, self.slot_channels
        root_macs = batch*64*(INPUT_CHANNELS*width*9+ROOT_DEPTH*2*width*width*9)
        projection_macs = batch*64*width*channels
        graph_macs = batch*self.steps*self.nodes*self.nodes
        policy_macs = batch*candidates*((3*width+24)*2*width+2*width)
        value_macs = batch*(width*width+width)
        return {
            'batch_size': batch, 'candidate_slots_per_position': candidates,
            'candidate_score_slots': batch*candidates,
            'frozen_root_iterations': batch*ROOT_DEPTH,
            'frozen_root_conv_macs': root_macs,
            'input_projection_macs': projection_macs,
            'output_projection_macs': projection_macs,
            'graph_dense_matmul_macs': graph_macs,
            'graph_structural_message_products': batch*self.steps*self.edges,
            'graph_matrix_entries_materialized': self.nodes*self.nodes,
            'graph_edge_gain_evaluations': self.edges,
            'node_state_updates': batch*self.steps*self.nodes,
            'slot_gather_values': batch*self.nodes,
            'slot_pool_accumulations': batch*self.nodes,
            'slot_occupancy_divisions': batch*self.slot_count,
            'policy_head_macs': policy_macs, 'value_head_macs': value_macs,
            'accounted_dense_mac_total': root_macs+2*projection_macs+graph_macs+policy_macs+value_macs,
            'native_successors': 0, 'candidate_branch_evaluations': 0,
            'scope': 'Selected actual dense MACs plus separate structural counts; not total FLOPs or measured latency',
        }

    def _message_matrix(self):
        """Assemble W[destination,source], retaining parallel-edge contributions."""
        values = self.signs*F.softplus(self.edge_log_gain)
        values = values/self.indegree.clamp_min(1)[self.destinations]
        matrix = values.new_zeros((self.nodes, self.nodes))
        return matrix.index_put((self.destinations, self.sources), values, accumulate=True)

    def _messages(self, hidden, matrix=None):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
                or hidden.shape[1] != self.nodes or len(hidden) == 0
                or hidden.device != self.edge_log_gain.device
                or hidden.dtype != self.edge_log_gain.dtype or not torch.isfinite(hidden).all()):
            raise ValueError('Expected finite graph state [batch,nodes] on the model device and dtype')
        if matrix is None:
            matrix = self._message_matrix()
        return F.linear(hidden, matrix)

    def _check_inputs(self, observations, candidates, legal_mask):
        if (not isinstance(observations, torch.Tensor) or observations.ndim != 4
                or observations.shape[1:] != (INPUT_CHANNELS, 8, 8) or len(observations) == 0):
            raise ValueError('Observations must be nonempty [batch,19,8,8]')
        if (not isinstance(candidates, torch.Tensor) or candidates.ndim != 3
                or candidates.shape[0] != len(observations) or candidates.shape[-1] != 5):
            raise ValueError('Candidates must have shape [batch,moves,5]')
        _check_candidates(candidates)
        if (not isinstance(legal_mask, torch.Tensor) or legal_mask.dtype != torch.bool
                or legal_mask.shape != candidates.shape[:2] or not legal_mask.any(dim=-1).all()):
            raise ValueError('Every observation requires a nonempty boolean legal-move mask')
        if (observations.device != self.edge_log_gain.device
                or observations.dtype != self.edge_log_gain.dtype
                or candidates.device != observations.device or legal_mask.device != observations.device
                or not torch.isfinite(observations).all()):
            raise ValueError('Inputs must be finite and match the model device and dtype')

    def forward(self, observations, candidates, legal_mask):
        """Return masked candidate logits, root-side value and adapted spatial state."""
        self._check_inputs(observations, candidates, legal_mask)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        # Freeze the complete four-step root path, not just its first convolution.
        with torch.no_grad():
            root = self.backbone._root(observations, ROOT_DEPTH)
        drive_slots = self.input_projection(root).flatten(1)
        drive = drive_slots[:, self.node_slots]
        initial = torch.tanh(drive)
        hidden = initial
        matrix = self._message_matrix()
        for _ in range(self.steps):
            messages = self._messages(hidden, matrix)
            proposal = torch.tanh(drive+messages+self.node_bias)
            hidden = (1-self.alpha)*hidden+self.alpha*proposal
        pooled = drive_slots.new_zeros(drive_slots.shape)
        pooled.scatter_add_(1, self.node_slots.expand(len(root), -1), hidden-initial)
        pooled = pooled/self.slot_occupancy.clamp_min(1)
        correction = self.output_projection(pooled.reshape(len(root), self.slot_channels, 8, 8))
        hidden = root+correction

        # Match AnchorChess.forward's candidate layout exactly, including padding.
        squares = hidden.flatten(2).transpose(1, 2)
        batch = torch.arange(len(hidden), device=hidden.device).unsqueeze(1)
        pooled = hidden.mean(dim=(2, 3))
        features = torch.cat((
            squares[batch, candidates[..., 0]], squares[batch, candidates[..., 1]],
            pooled.unsqueeze(1).expand(-1, candidates.shape[1], -1),
            self.backbone.promotion_embedding(candidates[..., 2]),
            self.backbone.dx_embedding(candidates[..., 3]),
            self.backbone.dy_embedding(candidates[..., 4]),
        ), dim=-1)
        logits = self.backbone.policy_head(features).squeeze(-1).masked_fill(~legal_mask, -torch.inf)
        value = torch.tanh(self.backbone.value_head(pooled)).squeeze(-1)
        if (not torch.isfinite(hidden).all() or not torch.isfinite(logits[legal_mask]).all()
                or not torch.isfinite(value).all()):
            raise ValueError('Nonfinite connectome adapter output')
        return logits, value, hidden

    @torch.no_grad()
    def choose(self, board):
        """Time full native preparation, transfers, dense inference and response construction."""
        self.eval()
        device = self.edge_log_gain.device
        _sync(device)
        started = time.perf_counter()
        batch, menus = encode_batch([board], 'direct', device=device)
        batch['observations'] = batch['observations'].to(dtype=self.edge_log_gain.dtype)
        logits, value, _ = self(**batch)
        ids = menus[0]
        probabilities = logits.softmax(-1).cpu()[0]
        result = {
            'choice': ids[int(probabilities.argmax())],
            'probabilities': {uci: float(probabilities[index]) for index, uci in enumerate(ids)},
            'value': float(value.cpu()[0]), 'mode': self.mode, 'seed': self.seed,
            'mapping_seed': self.mapping_seed, 'graph_steps': self.steps,
            'backbone_root_depth': ROOT_DEPTH, 'candidate_evaluations': len(ids),
            'probability_semantics': 'uncalibrated legal-move softmax',
            'value_semantics': 'one root-side value; no successor value scoring',
            'model_kind': 'frozen direct chess backbone with topology residual adapter; no search or cross-move memory',
            'computation': self.computation_counts(legal_candidates=len(ids)),
            'timing_scope': 'Native encoding, legal menu, transfer, matrix assembly, projections, inference, synchronization and response construction; model/graph loading excluded',
        }
        _sync(device)
        result['latency_ms'] = (time.perf_counter()-started)*1000
        return result
