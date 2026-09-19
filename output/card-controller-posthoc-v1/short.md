# Remaining C-policy errors: exploratory saved-output review

All 192 completed C-policy games for each of Kalman, local innovation and gated delta are included, covering all three fitted seeds and the same 64 deck draws. The existing gates are unchanged.

| Model | Successes | Failed games with all 52 positions discovered | Failed games with repeated mismatches | Failed-game selected hidden recall | Failed-game all hidden recall | Failed-game hidden recall, age >32 |
|---|---:|---:|---:|---|---|---|
| kalman | 83/192 | 0/109 | 89/109 | 98.8% (6,090/6,162) | 99.5% (66,532/66,870) | 96.5% (1,777/1,841) |
| innovation_local | 80/192 | 0/112 | 90/112 | 98.6% (6,258/6,345) | 99.4% (68,667/69,105) | 94.0% (1,770/1,883) |
| gated_delta | 68/192 | 0/124 | 105/124 | 99.0% (6,930/7,001) | 99.5% (75,322/75,708) | 96.7% (2,090/2,162) |

Accuracy uses saved raw pre-action probabilities and only ranks revealed by earlier public frames. The age threshold is strictly greater than 32 actions since last public visibility. Each denominator is eligible position-time queries, including repeated queries; per-game macro accuracies and all success/failure subset totals are in the JSON.

| Model | Successful-game all hidden recall | Successful-game hidden recall, age >32 | Failed-game mean visited positions | Failed-game remaining publicly known pairs, mean |
|---|---|---|---:|---:|
| kalman | 99.4% (39,410/39,646) | 93.8% (807/860) | 46.70 | 0.19 |
| innovation_local | 99.5% (37,807/37,992) | 92.2% (744/807) | 46.59 | 0.19 |
| gated_delta | 99.6% (31,556/31,698) | 93.6% (684/731) | 46.90 | 0.17 |

Every failed game left at least one position undiscovered. Failed-game raw hidden-card recall nevertheless reaches 99.37% to 99.49%, while repeated mismatches occur in 80.36% to 84.68% of failures. Final public histories still contain known matching pairs in 21 failed games per family. These observations favor investigating exploration and repeated decision waste with unchanged weights before attributing the remaining failures to memory architecture. A small number of recall errors could itself cause repeated choices and incomplete discovery. Confidence on repeated mismatching pairs was not analyzed.

A future comparison needs a separately fixed controller rule and fresh cases. This analysis cannot establish how correcting recall or changing decisions would affect success. Successful and failed games are different selected populations; their accuracy contrast is not a causal effect. Full discovery can also occur too late to finish within 104 actions.

Execution completion SHA-256: `cc0d22c85fe8c9dadb36ce4b6c4fb047ad920c05a929752b1d0df2be94df22ba`.
Diagnostic JSON SHA-256: `ddc706b6944baef4802e8fd441aeb8fd06161c56b2585a3e68f271a149fba316`.

No model, simulator or RNG calls were made. No main-run, release, source or gate files were changed.
