"""Fabricated storage-order regression; no measured data or model inference."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[2]
audit_source = root / 'scripts/audit_robot_coupling.py'
source = audit_source.read_text()
assert "value = {key: data[key].copy(order='K') for key in data.files}" in source
# A transposed solve-shaped coefficient has the producer's F-contiguous layout.
original = (np.arange(150, dtype=np.float64).reshape(25, 6) / 17).T
assert original.flags.f_contiguous and not original.flags.c_contiguous
archive = io.BytesIO()
np.savez_compressed(archive, coefficient=original)
archive.seek(0)
with np.load(archive, allow_pickle=False) as data:
    recorded = data['coefficient']
    old = recorded.copy()
    repaired = recorded.copy(order='K')
assert np.array_equal(original, old) and np.array_equal(original, repaired)
assert not np.shares_memory(recorded, repaired)
assert repaired.strides == original.strides and old.strides != original.strides
# Torch's parameter clone preserves the input layout. load_state_dict copies
# values into that existing layout, so numeric checkpoint values alone cannot
# repair the incorrect constructor stride.
def parameter(value):
    return torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32).clone())

source_parameter = parameter(original)
old_parameter = parameter(old)
repaired_parameter = parameter(repaired)
assert source_parameter.stride() == repaired_parameter.stride() == (1, 6)
assert old_parameter.stride() == (25, 1)
with torch.no_grad():
    old_parameter.copy_(source_parameter.contiguous())
    repaired_parameter.copy_(source_parameter.contiguous())
assert old_parameter.stride() == (25, 1)
assert repaired_parameter.stride() == source_parameter.stride()
x = torch.arange(125, dtype=torch.float32).reshape(5, 25) / 11
assert torch.equal(x @ source_parameter.T, x @ repaired_parameter.T)
print(json.dumps({'status': 'PASS', 'scope': 'fabricated coefficient storage and parameter stride only',
    'producer_stride': list(source_parameter.stride()), 'old_loader_stride': list(old_parameter.stride()),
    'repaired_loader_stride': list(repaired_parameter.stride()), 'model_inference_calls': 0,
    'measured_array_decodes': 0, 'auditor_sha256': hashlib.sha256(audit_source.read_bytes()).hexdigest()}, indent=2))
