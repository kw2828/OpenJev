# Mystery Path memory qualification

This study asks whether retaining public route discoveries helps beyond strong finite-history controls. It uses the official `MysteryPath-Grid-v0` at upstream commit `a94f2b60d1769ea44df3226561488768e1dff9f4`, unchanged defaults. It does not train a model or establish architectural novelty.

The protocol and 64 fixed layout seeds precede qualification. Five controllers use each layout under four compass-priority orders, totaling 1,280 episodes. An episode has at most 128 native actions. The four orders are correlated repeats, not independent layouts. All episodes remain in the denominator.

The public input is the actual RGB observation decoded into visible location, heading and failure cue. Public controllers receive no simulator state, path, goal, seed or evaluator information. The known-route reference is isolated. This is a structured-observation qualification, not a pixels-only neural benchmark.

## Clarifications made before outcomes

The earlier direction note called for the shortest safe route. Source inspection showed that native actions are wait, turn left, turn right and forward. We therefore minimize primitive-action distance over location and heading, charging every turn. Four priority orders break ties between compass frontier directions, not native action IDs. All public controllers share this planner.

A failure frame shows the agent at the unsafe destination. The next action is ignored, consumes one step and returns to the origin with heading preserved. Full memory survives this within-episode return. The erase-on-failure control discards old evidence before retaining the newly observed failure transition. History controls retain exactly their last 16 or 32 transitions, including both endpoint observations, and reconstruct the map each decision.

Public observations are read in upstream x/y/channel order. The parser was checked against all 392 position, heading and failure combinations rendered by the actual upstream sprite code. Four separately declared engineering seeds (0, 1, 2, 3) tested the parser, deliberate failure, ignored return action and privileged reference in 87 native steps. They are absent from qualification inputs. `preflight-01/started.json` calls the aggregate four-episode step cap `max_steps: 512`; each episode still has the default 128-step limit.

## Decision rule

All 17 fixed criteria must pass: the reference succeeds on at least 244/256 layout/order pairs; full memory on at least 205/256; full memory exceeds each of last-16, last-32 and erase-on-failure by at least 39 successes pooled and four successes in each order. Any failed criterion closes this recipe. No windows, layouts, action budgets, cues or thresholds will be changed after observing outcomes.

## Evidence and reproduction

`protocol.json` defines the comparison; `inputs.json` allocates every qualification seed. `environment.json` pins the runtime and verifies all 14 installed upstream Python files against the checkout. `bindings.json` binds the experiment source, tests and inputs before execution. The upstream source and license are retained in `upstream/`.

Install the pinned upstream checkout in a separate Python 3.10 environment with the versions in `environment.json`. Then, from the repository root:

```sh
PYTHONPATH=src:scripts .venv-memory/bin/python scripts/run_mystery_path_qualification.py run --out output/mystery-path-qualification-v1/attempt-01
PYTHONPATH=src:scripts .venv-memory/bin/python scripts/run_mystery_path_qualification.py audit --out output/mystery-path-qualification-v1/attempt-01
.venv-memory/bin/python scripts/report_mystery_path_qualification.py --attempt output/mystery-path-qualification-v1/attempt-01 --out output/mystery-path-qualification-v1/review-01
```

Each output is exclusive. Execution and audit have separate 900-second caps, no retries, and at most 163,840 environment steps each. Completed episodes preserve every actual frame, action, reward, terminal flag, public observation, timing and state-size measurement. Replay reconstructs controller actions and checks every native frame and result. It uses the same official simulator, not an independent physics implementation.

Serialized controller state and recursively measured Python object sizes exclude code and temporary search workspace; neither is process memory. The reference records its supplied safe cells and goal, while public controllers also retain their mode/priority configuration. Instrumented timings include parsing, planning, updates, storage and resets. These are shared-host research timings, not deployment latency claims.

Passing would justify a learned memory comparison. It would not establish connectome superiority, a recurrent world-model contribution, or a paper-ready result. Prior Reacher and Pendulum results remain unchanged.
