# Does the existing dialogue cohort contain delayed literal evidence?

Prospective diagnostic, 20 September 2026. The completed lexical-input ablation
failed its continuation rule. This diagnostic determines whether the existing
autonomous training streams contain cases worth examining for delayed proposal
memory. It does not fit, rank or select models.

## Fixed population and inputs

Use every endpoint in the original encoder packet's **TRAIN cohort: 2,017
dialogues and 51,741 labeled questions**. Preserve complete public dialogue
streams, including unscored turns, and every supplied query. Do not reuse the
later adjacent-endpoint admission filter. The supplied query inventory defines
a conditional task; it does not demonstrate service routing.

Authenticate `runs/sgd-state-v1/features-02/packet.json` against its original
completion receipt and `runs/sgd-state-v1/data/train-dialogues.jsonl` against
its data completion receipt. The packet also contains previously exposed DEV
records; those records are not audited or scored here. No DEV dialogue file,
official TEST content, feature array, checkpoint or model output is opened.
Record exact source, selected input and protocol hashes before execution.

## Definitions fixed before the pass

At each labeled USER endpoint, search ordinary candidate literals using the
existing casefolded, stripped-value, whole-word regular-expression semantics.
Count age in USER exchanges: the current USER and immediately preceding SYSTEM
are age zero. The previous three exchanges are recent; age four or greater is
distant. Process public utterances in order, without future access. Distinguish
consecutive SYSTEM turns if present: each belongs to its following USER
exchange, so the audit does not silently discard earlier SYSTEM text. Report
consecutive same-role pairs as a structural check. Distinguish
no literal match from a distant match. NOT_MENTIONED and DONTCARE have no
literal value and receive a separate nonliteral category. TRUE/FALSE literal
matches are reported separately from other values and do not interpret yes/no.

For each target, report separate USER and SYSTEM ages, their joint table,
transition bin, service, and whether the previous annotated endpoint was the
immediately preceding USER turn. Verify transition bins against previous
annotated targets, starting at NOT_MENTIONED. Gaps are retained and identified;
they do not assert what the gold state was during unannotated turns.

The primary **delayed SYSTEM-only proxy** is an OTHER-valued first assignment
or revision whose target has no USER literal match anywhere in its prefix and
whose latest SYSTEM literal match is distant. Also report all OTHER-valued
targets whose latest literal evidence across either speaker is distant, split
by changed and retained state. Retained remote evidence mainly concerns carry,
which the existing scalar model already provides. Include row, distinct
dialogue and service support. No proxy is described as an accepted proposal,
semantic absence, causal memory dependence or annotation correctness.

## Execution and interpretation

One model-free pass, one CPU thread, **180 seconds**, **2 GiB process RSS** and
**128 MiB output** caps. Time starts before input authentication. Use exclusive
output paths, retain failures and partial artifacts, and do not retry a failed
frozen attempt. Synthetic tests and source review precede the pass. Freeze
the helper, runner, both test files and this protocol; commit the plan before
execution. There is no model/plant allocation or external model call.

Save aggregate counts and losslessly compressed endpoint metadata without
utterance text. Preserve every cohort row. The metadata are evaluator-side
diagnostics and must never become model inputs. Independently check aggregate
coverage and inspect any primary-proxy support before proposing a training
study. Publish the complete counts, even if the motivating proxy is absent.

There is no numerical pass threshold or automatic training admission. Counts
can locate review candidates, but cannot establish a learned-memory advantage.
If this cohort supplies no convincing delayed-proposal cases, stop that branch
on this cohort and prioritize observation learning with autonomous scalar
tracking. Any later architecture study needs its own matched controls and
untouched confirmation; this diagnostic changes no earlier failed result.
