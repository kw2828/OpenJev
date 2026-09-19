# Exact candidate memory with learned correction

Prospective source review, 2026-09-19. No training, dataset inspection, or model calls. The completed [dialogue-memory result](dialogue-memory-results.md) remains closed: 8/11 checks passed, and literal carry's unseen macro accuracy was 65.76%, versus 53.98% for innovation Kalman. This motivates testing learned interpretation around an explicit candidate ledger; it does not establish a new memory principle.

## Four closest sources

| Primary source and inspected version | Established mechanism | Consequence for OpenJev |
|---|---|---|
| [TRADE, ACL 2019, pp. 808-819](https://aclanthology.org/P19-1078.pdf), Section 2, equations 1-7; [author implementation](https://github.com/jasonwu0731/trade-dst) | A shared domain/slot-conditioned decoder mixes vocabulary generation with attention-based copying from dialogue history. A separate gate handles NONE, DONTCARE, and generated values. | Exact-value copying and schema-conditioned transfer are established. Our supplied finite candidate set is an easier, different task than open-vocabulary generation; copying candidate IDs is not a novel pointer generator. |
| [SOM-DST, ACL 2020, pp. 567-582](https://aclanthology.org/2020.acl-main.53.pdf), Sections 3-4; [author implementation](https://github.com/clovaai/som-dst) | Explicit slot memory with CARRYOVER, DELETE, DONTCARE, and UPDATE; generation occurs only for updated slots. Training uses ground-truth operations and previous states. | A learned overwrite gate around exact memory is already standard. A fair local comparator should use the same public inputs and our own predicted prior state, not paper-style gold previous states or additional operation supervision. |
| [Thomson and Young, Bayesian update of dialogue state](https://farm2.user.srcf.net/research/papers/csl-buds.pdf), inspected author preprint dated 17 October 2008, Sections 2.1-2.3; [2010 journal identity](https://doi.org/10.1016/j.csl.2009.07.003) | Recurrent categorical beliefs combine state-transition and observation factors; a factored dialogue model uses approximate inference to handle the joint state space. | Predict-then-correct categorical memory is old. Our small single-slot candidate distribution needs no loopy inference. A neural scoring factor trained only through state cross-entropy is not automatically a calibrated observation likelihood. |
| [Gated Delta Networks, arXiv:2412.06464v1](https://arxiv.org/html/2412.06464v1), Section 3.1, equation 8; [author implementation](https://github.com/NVlabs/GatedDeltaNet) | Matrix memory combines decay and a residual delta write: `S_t = alpha S_(t-1)(I-beta kk^T) + beta vk^T`. The paper evaluates language modeling and retrieval, not this supplied-candidate DST task. | Selective fast-weight replacement is established. It does not guarantee exact preservation of other identities: a write affects a queried key in proportion to its overlap with the written key. Exact candidate indexing removes that compression/interference tradeoff by design. |

The linked repositories establish implementation provenance; this review did not execute them or pin their moving branches. The paper versions above are the methodological sources. No claim of exhaustive novelty clearance follows from four papers.

## One implementation worth testing

Use a **candidate-indexed categorical correction model**. For each supplied service/slot, retain `b[c]` over the valid candidate IDs, including separate NOT_MENTIONED and DONTCARE states. Initialize `b_0` to NOT_MENTIONED. IDs index storage only; shared scorers consume candidate descriptions, never learned index-specific embeddings.

Let `u[c]=1/C` over valid candidates and compute a scalar change hazard `h_t` from the previous belief, query description, and preceding SYSTEM text. Keep the current USER text out of this prior. Then:

```text
bminus[c] = (1-h_t) * bprev[c] + h_t * u[c]
e[c] = shared_score(current USER + preceding SYSTEM, query, candidate[c], mention[c])
bnext = softmax(log(bminus) + e)
```

`mention[c]` is a public, deterministic exact-string feature with the existing Unicode-boundary convention; it is not a gold span or operation label. Give every learned comparator the same features. Keep ambiguous multiple matches rather than silently selecting a gold value. Reserved states need explicit schema features, not literal matching of their internal sentinel names.

This is differentiable through the entire predicted belief sequence. A positive hazard lets previously zero-probability candidates recover; implement the mixture in log space rather than adding an undocumented probability floor. Padding never updates state. A missing supervision record never resets memory. Candidate reordering must permute the result exactly, including the reserved-state indices.

The distinction from scalar carry is the update to relative odds:

```text
log(bnext[i]/bnext[j]) = log(bminus[i]/bminus[j]) + e[i] - e[j]
```

Thus the score can weaken a previously plausible identity without erasing its stored identity or requiring a correct replacement first. This is an inductive-bias hypothesis, not an expressivity theorem: a sufficiently flexible carry/write model can reproduce the same final distributions. Separate unrestricted "support" and "reject" heads would also be non-identifiable; their difference is just another score.

The current USER enters the evidence factor once. If it instead controls both the prior and the evidence score, call the result a discriminative recurrent factorization, not two independent Bayesian observations. Even the separated version needs calibration evidence before any Bayesian-confidence claim. The hazard can itself cause unwanted forgetting on irrelevant turns, which is a concrete risk to measure.

## Small comparison and falsification

Use three paired initializations for four trained arms: the proposed filter; a strong same-input operation mixture `bnext=(1-g)*bprev+g*softmax(e)`; the proposed filter without literal features; and its nonrecurrent `h=1` ablation. Match shared scorer shape, candidate features, data/order, epochs, and losses; disclose any gate-parameter difference. Retain literal carry and always-NOT_MENTIONED as deterministic references. Existing neural results supply historical context, not a matched new control.

Train only on state labels, through predicted prior beliefs. The operation mixture must also see its own previous belief and all public features, so its weakness cannot be manufactured by withholding inputs. Use fresh confirmation evidence after the already inspected development results. Report seen/unseen three-stratum macro accuracy, micro NLL/Brier, revision-only accuracy with counts, and retention versus corrections where the literal rule is right or wrong. Keep any annotation-defined breakdown evaluator-only. The old development set has no clears, so it cannot establish deletion ability.

Disconfirm the proposed mechanism if it fails to beat the same-input mixture on held-out corrections, if removing literal features explains essentially the whole gain, or if revision improvement comes from damaging retained states or probability quality. Failure to beat literal carry on unseen macro accuracy blocks a practical superiority claim. An improved aggregate alone does not establish learned negation handling; that needs separately labeled, prospectively specified evidence.

State costs scale as `O(number of supplied slots * candidates)` and each update scores those candidates. Charge cached schema/encoder features, all slot updates, initialization, and full inference time. This is deliberately more explicit memory than a shared compressed matrix. Any useful contribution would be a demonstrated correction/generalization tradeoff under these controls, not the invention of copying, carry, Bayes filtering, or test-time weight updates.
