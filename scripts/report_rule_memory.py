"""Serialize NumPy scalar counters in the unchanged frozen memory reporter.

The original report computed/replayed all metrics, then failed to serialize its
NumPy seed-win counter. This adapter changes JSON types only, not selection.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from openjev.research.text_distillation import write_new


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def main(args):
    source = Path(__file__).with_name('rule_memory_study.py')
    spec = importlib.util.spec_from_file_location('frozen_rule_memory', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_new = lambda path, value: write_new(path, plain(value))
    module.report(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'plan', 'features', 'runs', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
