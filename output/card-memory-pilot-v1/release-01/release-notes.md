Completed native ConcentrationHard experiment: 18 learned fits, 2,304 optimizer updates, 1,280 evaluation games and 130,730 evaluation actions.

The proposed local innovation memory fails the fixed continuation rule (1/6 checks). All learned families finish 0/192 games; both public symbolic references finish 64/64. Local mean return is -0.078125 versus -0.080228 for gated delta, below the required improvement.

Saved-prefix analysis also exposes numerical sensitivity: normalizing the local model's probabilities changes 40.33% of first-card choices within tiny score ties. No alternate trajectories were run; improved returns are not established. This limits architectural interpretation.

The archive contains all raw public trajectories, 18 initial and final checkpoints, optimizer states, training logs, source snapshots, protocols, audits, figures and the first scheduled game GIF. Native/model calls during reporting, visualization and packaging: zero.

Archive SHA256: dbdc50851322e2dc02b744e43eac83b315f83bcfa607b9931ef932e2f69b980f
Archive bytes: 430005762
Members: 3457

Prospective execution commit: 3b9c5e689daa223c27b2aa89c1734a5dc377fd94
