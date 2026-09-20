# Observation learning qualification artifacts

Code and synthetic checks are MIT. The source observations come from the
[Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78),
licensed CC BY-SA 4.0. This dataset consists of paraphrased simulated dialogues.
The experiment uses original TRAIN only and synthetic optimization targets.

The prepared plan and workload metadata are published. The reversible token
payload `preparation-01/representatives.json` remains local, following the
original dataset protocol's source-text publication boundary. Its 514,537 bytes
are authenticated by SHA256
`f9bc4147965972285883fa6ae0cfc7d391feb8b1c0592006b634bde9106705ea`.
The published completion manifest describes the full local preparation;
it does not mean every manifested file is in Git. The preparation script can
reconstruct it from the authenticated public dataset and local model assets.

`preflight-01` preserves 53 passing synthetic tests and a failed lint check.
`preflight-02` records the eight affected tests and lint passing after test
formatting corrections. Neither preflight used a pretrained model or corpus.
Source review was independent and read-only.

`watchdog-01.py` is the stdlib process supervisor. Its launch records bind the
exact command, process group and supervisor hash; terminal records contain
the actual return code and process-group cleanup observation.
