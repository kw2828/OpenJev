# Review of all 38 delayed SYSTEM-only literal proxies

Independent AI-assisted, label-aware source inspection, 20 September 2026. **This set does not justify a source-separated proposal-memory fit.** Most cases are digit/number-word mismatches with incidental older numerical matches. Four cases plausibly reuse a previously specified party size across services, but none supplies a clear unresolved SYSTEM proposal that is accepted only after a long delay.

I read every complete public prefix in the fixed local review packet ([packet receipt](../output/dialogue-history-support-v1/review-preparation-01/receipt.json)), following the [all-case review plan](dialogue-history-support-review-plan.md). The packet contains 38 endpoints from 37 dialogues and 12 services. All targets are counts from one through four. No future turns, other source data or model predictions were consulted. No experimental model inference, model API calls, checkpoint loading or encoder execution occurred. Official labels are unchanged.

## Complete findings

| Assessment | Cases | Meaning |
|---|---:|---|
| Recent explicit value for the requested slot | 32 | The current USER states the target count in words. Older context is unnecessary for that count. |
| Recent number with service or transfer ambiguity | 2 | P09 and P10 contain the target number inside history4, but assigning it to the requested service involves ambiguous wording or an implicit group transfer. |
| Plausible transfer from an older confirmed USER request | 4 | P05-P08 have an older event-ticket quantity but no party size in the recent restaurant exchanges. |

Thus 34/38 have the target number explicitly expressed inside history4. This is semantic availability to this reviewer, **not 34 measured correct model answers**. The 32 direct cases plus two ambiguous cases should remain separate. The complete [case judgments](../output/dialogue-history-support-v1/case-review-01.json) retain every case ID, supplied target, recent boundary, decisive public-turn indices and a concise paraphrase.

The older digit usually belongs to a different fact: search-result counts, bedrooms or bathrooms, hotel duration, a phone country code, an address, a departure hour, a rating or a hotel name. These hits are faithful to the frozen literal matcher. They do not show that the corresponding state value was proposed earlier. For example, P16's older match is a phone component at turn 9, while turn 20 explicitly requests one flight ticket. P20's older match is a departure hour at turn 7, while turn 18 requests four bus tickets. P31 has a hotel-annotation gap, yet turn 20 directly gives the room count.

## The six cases needing qualification

| Case | Decisive public turns | Assessment |
|---|---|---|
| P05 | 2, 3, 4; 12-14 | An older three-ticket USER request is confirmed. The later restaurant acceptance gives no party size. Reusing the same party is plausible but unstated. |
| P06 | 0, 3, 4; 12-14 | An older one-ticket USER request is confirmed. A later restaurant booking does not explicitly specify a solo party. |
| P07 | 0, 2, 3, 4; 15-16 | The USER previously corrects two event tickets to one. The later restaurant reservation supplies a time, not its party size. |
| P08 | 0, 3, 4; 8-12 | An older three-ticket USER request is confirmed. The restaurant request supplies a time; transferring the group remains implicit. |
| P09 | 12, 17, 18 | Four people is explicit at turn 18, but the USER says car while the active dialogue concerns buses. The older rental-car result count is incidental. |
| P10 | 18-20 | Four people is explicit within the recent window. The new flight request does not restate passenger count, so retaining the group is plausible but not explicit. |

P05-P08 may require older context to recover a plausible party size, conditional on a cross-service continuity assumption. Their older evidence originates in an explicit USER quantity and a subsequent confirmation. It is already committed information in the event discussion, not a pending SYSTEM-only proposal. P09 and P10 share a dialogue and must not be treated as independent demonstrations.

## Decision and limits

Do not admit a proposal/commitment accumulator from this proxy set. Its main apparent support disappears once number words are interpreted. At most, P05-P08 motivate a small, separately specified cross-service party-transfer diagnostic against an explicit ledger, after deciding whether such implicit transfer is part of the intended target contract. They do not establish enough clear examples for training or a distinct learned recurrence.

For the broader development task, these observations favor resolving observation interpretation and evaluating a normalized scalar with its own state before adding source-separated memory. They do not establish that a particular encoder, fine-tuning recipe or number-normalization feature will improve task accuracy. The original literal audit and all earlier failed experiment rules remain unchanged.

This is one review with targets visible, on an error-motivated proxy set. The descriptive categories were formed during inspection, not frozen as a new success rule. No prevalence estimate, inter-rater agreement, annotation correction, causal memory effect or general absence of delayed references is inferred. Parent judgments were not consulted for these classifications.

Packet SHA-256: `39961ad5959acece2efe226f88425ac937a80430afa9efb1f8e9010287e2bb52`.

Case-review JSON SHA-256: `84c58a9ec8834d4a1d240c20576d92db8499fe6bbd2258b1613b3ef68732e7ae`.
