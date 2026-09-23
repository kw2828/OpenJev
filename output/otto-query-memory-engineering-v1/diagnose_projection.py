"""Fabricated diagnostic for attempt-01; no training or empirical inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests/test_otto_query_memory_integration.py"
spec = importlib.util.spec_from_file_location("integration", TEST)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = []
for period in (4, 8):
    model = module.predictor.make_head("frozen", 73, period)
    features, scores, mask = module.packet(period)
    full = model(features, scores, torch.tensor([65]), mask, episode_ends=torch.tensor([True]))
    parts, carry = [], None
    for start, stop in ((0, 32), (32, 64), (64, 65)):
        part = model(features[:, start:stop], scores[:, start:stop], torch.tensor([stop - start]),
                     mask[:, start:stop], episode_ends=torch.tensor([stop == 65]), carry=carry)
        parts.append(part)
        carry = module.predictor.detach_carry(part.carry)
    key_parts = torch.cat([part.keys_hidden for part in parts], 1)
    projected_full = full.keys_hidden @ module.projection()
    projected_parts = torch.cat([part.keys_hidden @ module.projection() for part in parts], 1)
    rows.append({"period": period, "hidden_keys_exact": torch.equal(key_parts, full.keys_hidden),
                 "shadow_prior_exact": torch.equal(torch.cat([p.shadow_prior for p in parts], 1), full.shadow_prior),
                 "projection_max_absolute_difference": float((projected_full - projected_parts).abs().max()),
                 "projection_differing_coordinates": torch.nonzero(projected_full != projected_parts).tolist()})
result = {"scope": "fabricated engineering diagnostic only", "rows": rows,
          "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (Path(__file__), TEST, Path(module.predictor.__file__))}}
with (Path(__file__).parent / "projection-diagnostic-01.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(rows))
