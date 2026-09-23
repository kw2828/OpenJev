"""Reuse the qualified opaque-byte packager for sampled forecast evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts/package_otto_score_forecasts.py"
PIN = "c5a7099bf42c4ddb11c7d25b555a8c4ff237ff60721a028f7bc0758449916429"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(PACKAGER.read_bytes()).hexdigest() != PIN:
        raise ValueError("unchanged original opaque-byte packager required")
    spec = importlib.util.spec_from_file_location("_sampled_forecast_packager", PACKAGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = "otto-sampled-forecast-package-v1"
    module.NAME = "openjev-otto-sampled-forecast-v1.tar.gz"
    module.execute(args)


if __name__ == "__main__":
    main()
