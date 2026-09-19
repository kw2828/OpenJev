"""Plot only the independently audited failure aggregates, with source binding."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "failure-audit/summary.json"
DIGEST = "f4c65283a6497a70721d8d4372c7a59c524c0edc44a966176cba5ffa43e0ce00"


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == DIGEST
    summary = json.loads(SOURCE.read_text())
    rows = summary["normalization"]["by_public_time"]
    t = np.array([r["public_user_time_index"] for r in rows])
    n = np.array([r["rows"] for r in rows])
    bad = np.array([r["old_b_normalization_violations"] for r in rows])
    assert int(n.sum()) == 62329 and int(bad.sum()) == 7989
    low = [r["old_b_sum_min"] for r in rows]
    high = [r["old_b_sum_max"] for r in rows]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Reproduced predictions, invalid internal memory", fontsize=16, x=.065, ha="left")
    axes[0].fill_between(t, low, high, color="#e59b91", alpha=.55, label="Observed min-to-max range")
    axes[0].plot(t, high, color="#ac3029", linewidth=1.5)
    axes[0].plot(t, low, color="#ac3029", linewidth=1.5)
    axes[0].axhline(1, color="#263b4e", linestyle="--", label="Required probability mass: 1")
    axes[0].set(ylabel="Sum of internal candidate values", ylim=(.97, 1.21), title="Scalar-4101 only; no new training")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].bar(t, n-bad, color="#c5d0d8", label="Within the fixed tolerance")
    axes[1].bar(t, bad, bottom=n-bad, color="#ac3029", label="Outside tolerance (7,989 rows)")
    axes[1].set(ylabel="Scored query rows", title="Later turns have much smaller support")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.set(xlabel="Public USER turn index (zero-based)", xlim=(-.7, 28.7), xticks=range(0, 29, 4))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.065, .025, "All 62,329 original outputs match bitwise. One fit inspected; no gate-effectiveness conclusion.\n"
             "Different turns contain different query rows; the range is not a confidence interval or a paired trajectory.", fontsize=8)
    fig.subplots_adjust(left=.065, right=.985, bottom=.24, top=.80, wspace=.28)
    target = HERE / "normalization-failure.png"
    if target.exists():
        raise FileExistsError(target)
    fig.savefig(target, dpi=160, facecolor="white")
    plt.close(fig)
    receipt = {"source_sha256": DIGEST, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "image_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "new_model_calls": 0}
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
