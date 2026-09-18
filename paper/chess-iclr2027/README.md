# Chess architecture manuscript

This is a development draft targeting ICLR 2027, not a submitted or accepted
paper. The ongoing goal remains incomplete until the scientific claim and its
supporting evidence, as well as the manuscript, withstand review.

The updated [nineteen-page PDF](../../output/pdf/openjev-chess-pin-quality-v3-development.pdf)
includes the completed 24-fit joint-pin comparison and full-decision timing.
Only 2/16 quality checks passed; Joint was 13.1% slower than WLDN by the median
paired timing ratio. All pages were rendered and changed pages visually reviewed.
The [new build receipt](build-pin-quality-v3-receipt.json) binds the sources,
both completed audits and rendered pages. It records artifact checks, not
scientific acceptance. The [earlier PDF](../../output/pdf/openjev-chess-iclr2027-development.pdf)
and [receipt](build-receipt.json) remain preserved.

The current draft combines the candidate-consequence comparison, the biological
topology comparison, a new factorized locality architecture, and the two-endpoint relation transport
comparison with a separately frozen post-hoc engine assessment and a nine-fit
simpler-head comparison plus an eighteen-fit joint-training follow-up. Appendices retain the separate stronger-engine baseline
assessment, the failed untrained child-operator screen, and the input-overlap
audit. The candidate-operator appendix also records a dense shared-weight graph
contrast prototype and its shared-root implementation with numerical and
artificial-update replay, plus a fully reconstructed lossless graph cache.
Its twelve-fit quality study now completes with four of eight original checks
passed and all 73,728 predictions replayed. The related-work discussion
identifies WLDN candidate ranking and molecular difference encoders as closer
precedents; the completed ingredient comparison does not establish novelty against
those methods. The WLDN baseline has now completed its three fits and a full
24,576-prediction replay audit: agreement is 37.01% ordinary and 31.43% shifted,
against base 31.75% and 26.81%. The condensed-graph baseline also completed, with
agreement 34.08% / 28.74% and all 24,576 predictions replayed. The combined
comparison fails with four of fourteen checks passed; WLDN exceeds scalar
contrast by 3.60 / 3.48 points. Two new tables retain all scalar-contrast controls
and the same-process six-method quality/cost comparison. The stronger
NBFNet baseline and native attack-edit audit are also retained in the appendices.
A fixed-weight WLDN pathway appendix now retains five interventions and the native model, with all 86,016 fresh/copied predictions replayed. Zeroing node differences costs 5.45 / 5.18 agreement points; candidate shuffling costs 10.63 / 9.15. These are diagnostic effects, not retrained architecture results.
The appendix now also records the union/edit head engineering screen and the verified linear equivalence of edit-class versus union/root/child encodings. Its twelve-fit quality study now completes with 2 of 10 original checks passed. All 86,016 predictions replay with zero discrepancy. Edits reaches 37.22% / 31.92% agreement, versus WLDN 37.01% / 31.43%; both source-game intervals for that difference include zero. A new table retains all arms, NLL and the 1,728 complete-native timings. The edits/WLDN median paired time ratio is 1.510, with all 288 distinct native choices replayed. No quality or speed advantage meeting the declared criteria is established. The full input/control audit independently replays all 1,087,523 cached candidates and confirms that the corruption changes every candidate while breaking reverse-direction consistency in 99.98%, limiting the causal interpretation of that control.
Main text ends on page9; total page count includes references/appendices. Earlier spatial,
recurrence and capacity studies provide development context. Each study retains
its own panel and fixed gate. Cross-study numbers must not be pooled into a
leaderboard.

A separate reverse-consistent control now has a bounded engineering result in
the appendix: 3,920 reconstructed native candidates, 627 score checks and three
exactly replayed artificial updates. It preserves reverse consistency by
construction. It has no trained quality result and does not replace the
original frozen corruption control.

## Build

The original biological report and the separately frozen locality engineering
screen are complete; both failed their continuation gates. A separately frozen
batched full-recomputation follow-up also rejects the current locality speed
claim. To regenerate the
source-bound tables, run from the OpenJev directory:

```sh
.venv/bin/python scripts/write_chess_program_paper.py
```

That historical generator does not include the later pin study. Its separate
saved-artifact generator is `paper/chess-iclr2027/write_pin_quality_v3.py`, run
with `.venv-robotics/bin/python`. It authenticates the completed quality and
cost receipts before generating the new tables and paired-seed figure.

This generator authenticates the saved summaries against completion receipts
and writes tables and a source-hash receipt. It does not independently repeat
every original experiment audit or run new inference. The original biological
report performs the raw prediction/training audit. The locality screen retains
per-candidate equality and node-count records plus full-loop timings. The transport
audit replays all 86,016 saved neural predictions. The separate engine audit checks
817 calls and 4,608 policy-position records; its post-hoc status does not alter the
failed primary criterion. The simpler-head audit replays all61,440 baseline and
reference predictions. Transport has a positive margin in every paired seed,
but fails all six fixed1-point requirements. The separate eighteen-fit joint
representation study also fails its criterion (4/12 comparisons pass). Its
transport margin over equally fine-tuned direct falls from1.51 points ordinary
to0.02 points shifted; all98,304 predictions replay exactly.

Compile `main.tex` with `TEXINPUTS=./style:` so the pinned official ICLR 2027
style is used. `style/source.json` records the public download URL and hash.
The draft changes only the title's status text through a local macro patch;
it must not falsely say it is under review. The template source itself stays
unchanged.

## Completion requirements still open

- A defensible novel mechanism beyond chess GNNs, recurrence, biological wiring,
  NNUE and established exact incremental GNN inference.
- A demonstrated routing or locality advantage: the trained transport head failed
  its ingredient/topology criterion (3/10 comparisons passed). The separate engine
  assessment improves bounded loss over the backbone but worsens shifted raw
  centipawn loss; rewired transport has lower shifted bounded loss. No uniform
  quality advantage is established.
- Strong, appropriately matched baselines, including batched full recomputation
  for speed and task-specific baselines for quality.
- Fresh confirmation separated from the sequential development program;
  gameplay and any cross-domain claims require their own evidence.
- Mechanism interventions and uncertainty appropriate to paired seeds and
  source-game dependence.
- Complete full-text/implementation review of the nearest prior work.
- Human scientific review, verified author list and approved AI-use disclosure.
- An anonymous, reproducible supplementary package with external data terms
  resolved. Local biological graph derivatives are not an MIT release.

The official call checked September 18, 2026 gives September 18 at 23:59 AoE for
abstract registration and September 25 at 23:59 AoE for the full paper. Building
or reviewing this draft does not register an abstract or submit a paper.
