# Separate history encoding from transition learning

The [completed finite-world study](finite-observation-learning-results.md)
does not justify scaling this recipe or adding RL to it. Exact targets and a
representable world were insufficient under the registered budget. That does
not prove the architecture cannot learn: optimization, latent identification,
history encoding and objective balance remain possible explanations.

The next useful diagnostic is a separately registered two-by-two comparison at
the world's eight-state dimension:

| Prefix state | Transition operators | Question |
| --- | --- | --- |
| Exact posterior | Exact world | Does the full prediction path reproduce the oracle? |
| Learned from public history | Exact world | Can the encoder recover a useful state in a fixed basis? |
| Exact posterior | Learned | Can short forecasts identify useful transition/observation operators? |
| Learned from public history | Learned | Does their joint learning introduce an additional failure? |

Use fresh cases and fixed equal update budgets for the learned comparisons.
Retain the ordinary GRU and the uniform-state reference. Freeze the decoder in
the known state basis in all four cells, and report the resulting
privileged information explicitly. An exact posterior is an oracle input, not
a deployable recurrent policy. This comparison would diagnose a training
bottleneck; it would not establish a competitive architecture.

Use the base sensing law as the primary diagnostic. Supplying exact shifted
operators would give the model privileged knowledge of the new sensor, so it
would not measure unannounced transfer.

First require the exact/exact arm to reconstruct costs, surviving mass and
observation probabilities within numerical tolerance. Before seeing new data,
register separate learning and long-gap criteria for the other arms, along with
parameter counts, updates and actual time. A successful privileged arm alone
cannot admit a robotics, Doom, chess or connectome claim.

If each separately learned component passes but joint learning fails, that
would support a coupled-training explanation. If either isolated component
fails, investigate that training path first. The current result does not tell
us which explanation is correct.

This proposal has not been executed. The prior failed criteria remain closed.
Calibration and conformal prediction can be investigated after the underlying
forecasts and decisions show useful held-out performance.
