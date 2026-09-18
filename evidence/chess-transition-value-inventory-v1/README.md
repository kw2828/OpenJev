# Saved training continuation inventory

96,758 usable behavior continuations from 98,304 accepted training roots; 1,546 explicitly skipped.

Every records.jsonl line represents one accepted historical TRAIN root. Skipped rows have a null continuation value, never an invented zero. Nonterminal values negate the accepted next TRAIN root teacher value in the same source and game; terminal values use native chess outcomes after replaying the complete recorded move history with claim_draw=False.

Teacher values are tanh(centipawns / 600), not calibrated win probabilities. Random-sourced moves can coincide with teacher choices; both fields are retained. All source-file hashes and inherited call costs are in summary.json. Role costs overlap. No dev/shift label is used as a target.

Exposed historical development data only. One recorded behavior action per root; no alternative-action values, calibrated win probabilities, new teacher calls, training, evaluation or performance claim. Source teacher costs are inherited, not newly incurred; root/child role costs overlap and must not be added.
