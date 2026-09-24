"""Default collection policy for explicitly superseded frozen test files.

The v1 audit test preserves a known selector defect and its original failure:
output/finite-joint-reuse-throughput-v1/engineering-01.receipt.json
The published archive retains that receipt and the complete failed attempt:
research/finite-joint-reuse-throughput-results.md

test_finite_joint_reuse_throughput_audit_v2.py replaces its coverage and is
collected normally. To reproduce the archived assertion failure explicitly,
select the v1 file with --noconftest, as its original registration did. This
file cannot affect either frozen qualification: both used --noconftest.

The original head-initialization test likewise preserves its wrong exception
assertion and three recorded failures. Its corrected v2 file replaces that
coverage. Both attempts are retained in research/finite-head-learning-results.md;
their frozen qualifications also use --noconftest.
"""

collect_ignore = [
    'test_finite_joint_reuse_throughput_audit.py',
    'test_finite_head_initialization.py',
]
