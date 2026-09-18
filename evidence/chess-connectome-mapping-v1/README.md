# Connectome spatial-mapping comparison

Status: input validation and protocol preparation are running. Training has not
started. There is no new policy-quality, game-strength or connectome-advantage
result in this package.

The [completed original comparison](../../docs/chess-connectome.md) failed its
continuation rule, with 5 of 30 checks passing. This follow-up asks whether
learning how board squares connect to neurons benefits biological wiring more
than matched controls. It preserves the previous negative result.

## Comparison

- Five topologies: biological, three signed-degree-preserving rewires and
  node-local recurrence.
- Fixed and learned spatial mappings for every topology, each under the same
  three seeds: 30 fresh fits, plus three unchanged direct-model references.
- The original 32,768 training positions, frozen backbones, six epochs and
  1,536 optimizer updates per fit.
- Exactly 320 proposed square swaps per fit. Fixed controls pay for both
  proposal forwards but retain their initial mapping.
- All fits precede benchmark evaluation. Final checkpoints and all seeds are
  included. Forty predefined checks must all pass before a separate
  game-strength follow-up is justified.

See the [study description](../../research/chess-connectome-mapping-study.md)
for the mechanism, matching, objective, timing and continuation criteria.

## Engineering evidence

The [test receipt](tests.json) records 135 passing checks, including synthetic
MPS training, checkpoint round trips, proposal replay, evaluation identity and
deadline handling. These are implementation checks, not performance results.

The [synthetic profile](../chess-connectome-mapping-preflight-v1/README.md)
uses invented targets and an artificial graph. Its roughly 20-minute linear
update-loop projection excludes input preparation, checkpointing, journals and
evaluation; it is not the measured runtime of this study.

## Execution

On the original host, with the pinned original assets available:

```bash
.venv/bin/python scripts/chess_connectome_mapping_study.py prepare \
  --out evidence/chess-connectome-mapping-v1/protocol
.venv/bin/python scripts/chess_connectome_mapping_study.py launch \
  --plan evidence/chess-connectome-mapping-v1/protocol/plan.json \
  --execution runs/chess-connectome-mapping-v1/execution \
  --audit-out evidence/chess-connectome-mapping-v1/audit \
  --out runs/chess-connectome-mapping-v1/launcher
```

Output paths are exclusive. These commands describe the original launch, not
a command to rerun an existing study. The supervisor limits primary execution
to two hours and saved-output auditing to thirty minutes. Failed attempts retain
their records; no retry, replacement fit, resume or time extension is permitted
within this comparison.

The inherited development panels have already been exposed. They are not fresh
confirmation data. No gameplay or Elo evaluation belongs to this screen.
Derivative weights, biological arrays, original training data and raw execution
artifacts remain local under `runs/` with their original terms. This package
does not claim a self-contained execution archive. Chess dependencies are in
[research/requirements-chess.txt](../../research/requirements-chess.txt).
