# Start with a competent planner, then learn when to use it

**Recommendation: retain the released OTTO network as the working policy and test recurrent allocation of its computation.** Do not make another randomly initialized small head responsible for learning search from the same 558 labels. This is a prospective experiment, not an admission or an architecture result. No neural calls were made for this review.

## The starting policy already exists

Use the original 13,390,849-parameter TensorFlow value network, original sixteen-branch/eight-symmetry `RLPolicy`, and [RestrictedPolicyActor](../src/openjev/research/otto_restricted_policy.py). The [boundary comparison](otto-boundary-control-results.md) passed all six competence conditions: 192/192 neural searches succeeded, with 9.67%/13.24% fewer weighted moves than analytic control at lengths three/four. Complete controller cost was approximately 1.08/1.38 seconds per search, versus 0.012/0.016 seconds analytically.

That is a useful starting reference, not a universally superior teacher. The same study failed its teacher-consistency and computation rules; restriction changed no action. The [earlier shifted failure](otto-released-reference-results.md) remains evidence against robustness. Length-five competence is not established for this checkpoint.

Practical assets are already local: `tmp/otto-pretrained-reference-01/zoo_model_2_3_2`, weight SHA `1efb73aa38e0fd8b08d6d03059c0db4664da3afb8eaff1d7ab9b363f8e7ad37d`; original classes under `tmp/otto-source-review-01/isotropic/classes`; and `.venv-otto-released-native/bin/python`. Reuse the qualified construction/loading sequence in [study_otto_released_reference.py](../scripts/study_otto_released_reference.py), including legacy Keras, CPU1 and exact tensor verification. Do not substitute the NumPy port: it failed exact-action parity. These paths/runtime need current authentication before execution, not reinstallation or new large-model training.

## One mechanism: recurrent query allocation

At each decision, compute the existing in-bounds analytic action. A small gate decides whether to use it or pay for the complete frozen neural planner. The gate never observes the current neural score before deciding to query. Initialize it to always query, reproducing the qualified reference before fitting. Once trained, switching can change trajectories and must earn competence afresh.

Use one GRU32 gate with a scalar sigmoid output. Its inexpensive public inputs are position, legal mask, known sensing length, hit, normalized entropy, expected source distance, the four analytic costs centered over eligible actions, and their changes since the last step. Fix scaling prospectively. State resets each episode; both policies assimilate every actual public observation. Compute scores through public views without committing two pending actions; commit only the selected action. Preserve the exact frozen neural arithmetic and tie rule.

This recurrence predicts when expensive planning matters. It does not reconstruct missing source information: the actor already maintains the exact sufficient Bayesian belief. No biological or connectome interpretation is warranted.

## Supervision and matched experiment

Collect one fresh TRAIN cohort of 96 complete trajectories, balanced across lengths three/four and initial hits, half controlled by analytic policy and half by the frozen neural policy. Query the frozen neural planner once at every visited state and retain its four costs, analytic choice, complete public sequence and all censored tails. Both gate families receive this same cohort. Do not train on old EVAL paths or enforce a selected-state quota that can underfill.

The target is whether the analytic action lies outside the neural policy's eligible near-minimum set. Retain the continuous neural cost gap descriptively; it is not a measured continuation regret. This supervision has no Monte Carlo continuation-label noise, though the planner can still be wrong. Train GRU32 and a parameter-matched stateless MLP with identical current inputs, episode weights and three paired seeds: 80 epochs, Adam 0.0003, clipping five, chronological 32-step windows with detached carried state and identical row exposure. Retain partial windows with a loss mask. Use a fixed 0.05 predicted-disagreement query threshold. No autonomous tuning.

Evaluate both three-seed gate families, always-neural and always-analytic on 24 fresh paired cases per length: 384 episodes for lengths three/four. A separately reported length-five extension adds 192 episodes and tests transfer. Keep the 2,188-move horizon and failures. The strongest architecture control is the stateless learned gate; always-analytic prevents a cheap switcher from receiving credit for merely recovering existing behavior.

The falsifiable claim is **preserved competence with fewer paid neural queries**, with recurrent allocation outperforming stateless allocation. Prospectively require at least 95% success and moves within 105% of both references for each primary seed/setting, plus at least 50% lower complete controller time than always-neural. A meaningful search-quality advantage over always-analytic must also be shown before calling this a useful new controller. Query savings alone are insufficient.

## Bounded cost and prior art

Historical neural choices cost roughly 0.034 seconds. About 4,000 TRAIN decisions would therefore cost roughly 136 seconds of neural computation; 10,000 cost roughly 340 seconds, before filtering, collection, fitting and evidence. These are projections, not guarantees. Propose 900 seconds for collection/fitting and 3,600 seconds for the complete autonomous comparison, CPU1/4 GiB, with no automatic extension. Count gate, analytic, neural, filtering and cold-load costs separately. A failed cap is retained, not repaired through subsampling.

Exact-belief neural planning is established [OTTO prior art](https://arxiv.org/html/2302.00706v2); visited-policy supervision is established [DAgger](https://proceedings.mlr.press/v15/ross11a.html); learned expert-query selection is established [SafeDAgger](https://ojs.aaai.org/index.php/AAAI/article/view/10857). The proposed contribution would have to be demonstrated computation allocation and transfer, not renaming these components. Start with this executable strong-policy comparison before attempting a recurrent world model or biological wiring.
