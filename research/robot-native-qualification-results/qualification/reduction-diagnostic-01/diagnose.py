"""One source-derived float32 reduction witness, fabricated vectors only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from openjev.research.structured_robot_transition import StructuredRobotTransition

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def scalar(values):
    result = np.float32(0)
    for value in values:
        result = np.float32(result + value)
    return result


def lanes4(values):
    lanes = np.zeros(4, dtype=np.float32)
    for start in (0, 4, 8):
        for lane in range(4):
            lanes[lane] = np.float32(lanes[lane] + values[start + lane])
    return scalar(lanes)


def pairwise(values):
    values = list(values)
    while len(values) > 1:
        values = [np.float32(values[i] + values[i + 1])
                  if i + 1 < len(values) else values[i]
                  for i in range(0, len(values), 2)]
    return values[0]


def bits(value):
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def compare(values):
    expected = torch.from_numpy(values.copy()).sum(-1).numpy()
    results = {name: np.asarray([fn(row) for row in values], dtype=np.float32)
               for name, fn in [('scalar', scalar), ('lanes4', lanes4), ('pairwise', pairwise)]}
    return {
        'rows': len(values),
        'exact_matches': {name: int(np.sum(result.view(np.uint32) == expected.view(np.uint32)))
                          for name, result in results.items()},
        'first_scalar_mismatch': next(({'products': values[i].tolist(),
            'torch_bits': bits(expected[i]),
            **{name + '_bits': bits(result[i]) for name, result in results.items()}}
            for i in range(len(values)) if bits(expected[i]) != bits(results['scalar'][i])), None),
    }


torch.set_num_threads(1)
sources = [Path(__file__), HERE / 'SumKernel.cpp',
    ROOT / 'src/openjev/research/structured_robot_transition.py',
    ROOT / 'rust/robot_transition/src/lib.rs',
    ROOT / '.venv/lib/python3.12/site-packages/torch/include/ATen/cpu/vec/vec128/vec128_float_neon.h']
pins = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
(HERE / 'definition.json').write_text(json.dumps({
    'scope': 'Fabricated 12D products only; no model checkpoints or measured inputs.',
    'candidate_order': 'Torch SumKernel contiguous inner float32 sum, four NEON lanes.',
    'source_url': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/aten/src/ATen/native/cpu/SumKernel.cpp',
    'pins': pins,
}, indent=2) + '\n')
wave = np.arange(1024 * 12, dtype=np.float64).reshape(1024, 12)
products = (np.sin(wave * .37) * np.cos(wave * .071) * 2 ** (wave % 9 - 4)).astype(np.float32)
cell = StructuredRobotTransition('householder', 97241)
q = (np.sin(np.arange(2 * 32 * 6, dtype=np.float64).reshape(2, 32, 6) / 19) * .3).astype(np.float32)
u = (np.cos(np.arange(2 * 32 * 6, dtype=np.float64).reshape(2, 32, 6) / 23) * .4).astype(np.float32)
future = (np.sin(np.arange(2 * 128 * 6, dtype=np.float64).reshape(2, 128, 6) / 17) * .5).astype(np.float32)
with torch.no_grad():
    state = cell.condition(torch.from_numpy(q), torch.from_numpy(u))
    prepared = cell._prepare()
    mixing = cell._mix(state, torch.from_numpy(future[:, 0]))
    vectors = cell._vectors(mixing, prepared)
    value = state * prepared.scale
    reflection_records = []
    for index in range(4):
        vector = vectors[:, index]
        reflection_records.append({
            'reflection': index,
            'numerator': compare((vector * value).numpy()),
            'denominator': compare(vector.square().numpy()),
        })
        value = cell._reflect(value, vector)
result = {'torch': torch.__version__, 'fabricated_products': compare(products),
          'initial_reflections': reflection_records, 'pins': pins}
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha for name, sha in pins.items())
(HERE / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2, allow_nan=False))
