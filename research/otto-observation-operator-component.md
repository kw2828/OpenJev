# Observation-operator recurrence: component specification

**Prospective implementation and fabricated qualification only.** This does not
train on OTTO, read the running conditional-label experiment, admit a new
scientific trial, or establish a novel architecture. The
[mechanism proposal](otto-predictive-moment-mechanism-draft.md) describes the
related work and the evidence a later comparison would need.

The component separates a learned update after seeing an odor from the update
used while observations are unavailable. For action `a` and nonterminal odor
`o`, `B[a,o]` maps nonnegative latent mass into nonnegative next-state mass.
The four observation branches and a found row have unit outgoing column mass.
The tied model's blind matrix is their sum. A control uses independently learned
blind matrices, initialized to the same marginal transition.

Observed nonterminal updates normalize by the evidence of the observed branch.
Blind updates retain unnormalized surviving mass. Observed found creates an
absorbing zero state. A centered, bias-free linear cost readout commutes with
branch summation and returns zero on a zero state. Forecasts precede observation
assimilation, so a future odor cannot affect its own prior forecast. Impossible
observed evidence fails explicitly; it is not repaired with an arbitrary floor.

The initial state comes from the same public prefix interface: nine 31-feature
rows and their lengths. A width 28 observation GRU feeds a projection into 14
latent coordinates. This is a chosen prototype size, not a selected winning
width. Actions and subsequent observed odor/found indicators are the only
continuation inputs. There is no simulator, known sensor law, teacher call,
hidden source or checkpoint loading inside the component.
The current GRU's normal-observation API accepts 31 continuation features,
including belief-derived summaries. A future observed-update comparison must
match this input interface or explicitly account for that extra external
computation; sharing a public-history origin does not make those interfaces
equivalent at inference cost.

Use float32 prefix encoding and float64 operator/state/readout arithmetic for
the structural prototype. Fourteen float64 latent coordinates and 28 float32
coordinates both occupy 112 bytes per carried state. This does not make total
memory, parameters or runtime equal: the encoder, transition parameters,
temporary branches, training activations and readout also count. Report exact
parameter dtypes and byte counts separately. A future deployment comparison
must charge actual work, including normalization and operator construction.
The normalized 14-coordinate state has only 13 independent coordinates; blind
survival mass adds one degree of freedom. Its linear future readouts also impose
a finite predictive basis. Equal byte counts therefore do not mean equal
representational capacity. The tied-versus-untied comparison isolates the
proposed relationship more directly than the GRU comparison does.

Fabricated qualification must establish:

1. Nonnegative branch probabilities and conservation of outgoing mass, including
   found, against an independently specified small finite-state example.
2. Equality between blind propagation and explicit observation-tree aggregation
   for the linear readout, within a declared floating-point tolerance.
3. The distinction between normalized observed states and unnormalized blind
   states, including absorbing found and impossible-evidence failure.
4. No future-observation or padding leakage; deterministic local initialization
   that preserves the caller's RNG and matches tied/untied initial behavior.
5. Gradients reach the intended operators, while the untied blind transition can
   change independently of its observed transition. A nonlinear-head example
   demonstrates why readout linearity matters.

These are internal structural properties. They do not show that a 14-coordinate
state can represent the history-dependent teacher, identify observation
operators from endpoint-only labels, or improve decisions. A later trial needs
matched branch/multihorizon supervision for all controls, a common information
budget, a GRU control, the untied operator control, fresh longer gaps and a
sensing shift. The current conditional-label trial keeps its original protocol
and outcome regardless of this component's qualification.
