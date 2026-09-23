"""Plot complete saved replay counts and historical time, not a speedup."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/otto-exact-cache-v1"


def main():
    names = ["proxy-01/summary.json", "proxy-01/receipt.json", "proxy-supervisor-01.terminal.json",
             "replay-01/summary.json", "replay-01/receipt.json", "replay-supervisor-01.terminal.json"]
    raw = {name: (BASE / name).read_bytes() for name in names}
    records = {name: json.loads(value) for name, value in raw.items()}
    for name in names:
        if name.endswith(("receipt.json", "terminal.json")):
            assert records[name]["status"] == "completed"
        if name.endswith("terminal.json"):
            assert records[name]["returncode"] == 0 and records[name]["group_absent"]
    proxy, replay = records[names[0]], records[names[3]]
    assert replay["agreement"] and replay["all_gzip_eof_verified"]
    assert proxy["totals"]["queries"] == replay["totals"]["requests"] == 22296
    out = BASE / "figure-01"
    out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("Can exact caching remove the teacher bottleneck?", x=.025, ha="left", weight="bold", fontsize=17)
    for y, (label, hits) in enumerate([("Full-precision state proxy", proxy["totals"]["hits"]),
                                      ("Rebuilt model input", replay["totals"]["hits"])]):
        axes[0].barh(y, 22296, color="#e2e8f0", height=.5)
        axes[0].barh(y, hits, color="#167d8d", height=.5)
        axes[0].text(22296 * .99, y, f"{hits:,} / 22,296 hits", ha="right", va="center", weight="bold")
    axes[0].set_yticks([0, 1], ["Full-precision state proxy", "Rebuilt model input"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Returned requests, fixed 64-entry episode cache")
    axes[0].set_xlim(0, 22296)
    total = replay["totals"]["original_forward_seconds"]
    eligible = replay["totals"]["hit_forward_seconds"]
    axes[1].barh(0, total, color="#e2e8f0", height=.45)
    axes[1].barh(0, eligible, color="#167d8d", height=.45)
    axes[1].axvline(.4 * total, color="#b45309", linestyle="--", label="40% continuation threshold")
    axes[1].text(total * .98, 0, f"{eligible:.2f} / {total:.2f} s\n{100 * eligible / total:.2f}% eligible",
                 ha="right", va="center", weight="bold")
    axes[1].set_xlim(0, total)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Historical TensorFlow time attached to hits")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    decision = "PASS" if replay["continuation"] else "FAIL"
    fig.text(.025, .04, f"Engineering screen: {decision}. All returned requests, including one partial episode.\n"
             "Historical eligible time excludes future cache overhead and is not a measured speedup.", fontsize=10)
    fig.tight_layout(rect=(0, .16, 1, .9))
    fig.savefig(out / "exact-cache.png", dpi=160, facecolor="white")
    plt.close(fig)
    for name, data in raw.items():
        assert (BASE / name).read_bytes() == data
    png = (out / "exact-cache.png").read_bytes()
    receipt = {"status": "completed", "scope": "Plotting complete frozen replay outputs only.",
               "inputs": {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "figure": {"sha256": hashlib.sha256(png).hexdigest(), "bytes": len(png)}}
    with (out / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
