# SPDX-License-Identifier: GPL-3.0-only
"""Rebind the unchanged pre-outcome native-cost comparison to fresh quality v3."""
import argparse
import copy
import importlib.util
import math
import time
from pathlib import Path

import chess_pin_quality_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]
OLD_PLAN = 'evidence/chess-pin-trained-cost-v1/protocol/plan.json'
OLD_PLAN_SHA = '4e8e50b2c414673471bae87bfce89c0fdb8bdfd62d6f870cf6b78695ee7d3bab'
QUALITY_PLAN = 'evidence/chess-pin-quality-v3/protocol/plan.json'
QUALITY_PLAN_SHA = 'b0b3da433d1f507d012208624c4231ee777febe7d6843685f9dacd169c691582'
EXECUTION = 'runs/chess-pin-quality-v3/execution'
AUDIT = 'evidence/chess-pin-quality-v3/audit/receipt.json'
VERSION = 'trained-pin-complete-native-cost-v2'
CODE = ['scripts/chess_pin_trained_cost.py', 'tests/test_chess_pin_trained_cost.py',
        'scripts/chess_pin_trained_cost_recovery.py', 'tests/test_chess_pin_trained_cost_recovery.py']


def load_engine():
    spec = importlib.util.spec_from_file_location('_isolated_pin_trained_cost', ROOT/'scripts/chess_pin_trained_cost.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_engine()
cost = load_engine()
cost.study = recovery.execution_engine()
cost.PLAN, cost.EXECUTION, cost.AUDIT = QUALITY_PLAN, EXECUTION, AUDIT
cost.PROTOCOL = copy.deepcopy(legacy.PROTOCOL)
cost.PROTOCOL['version'] = VERSION
cost.CODE = list(CODE)
native_signature = cost.signature


def prior_plan():
    path = ROOT/OLD_PLAN
    if recovery.sha(path) != OLD_PLAN_SHA:
        raise ValueError('Frozen predecessor cost plan changed')
    old, digest = legacy.validate_plan(path)
    if digest != OLD_PLAN_SHA:
        raise ValueError('Predecessor cost validation identity differs')
    return old


def signature():
    old = prior_plan()
    value = native_signature()
    quality = value['quality_signature']
    if (value['quality_plan_sha256'] != QUALITY_PLAN_SHA
            or quality['prior_signature'] != old['quality_signature']
            or quality['prior_plan_sha256'] != old['quality_plan_sha256']):
        raise ValueError('Recovery quality lineage differs from predecessor cost study')
    expected_protocol = copy.deepcopy(old['protocol'])
    expected_protocol['version'] = VERSION
    if value['protocol'] != expected_protocol:
        raise ValueError('Scientific cost protocol changed')
    if any(value[key] != old[key] for key in ('panel', 'pin_input_coverage', 'input_sha256')):
        raise ValueError('Original timing panel or pin strata changed')
    return {**value, 'prior_cost_plan_sha256': OLD_PLAN_SHA,
            'prior_cost_signature': {k: v for k, v in old.items() if k != 'prepared_unix'},
            'quality_bindings': {'plan': QUALITY_PLAN, 'execution': EXECUTION, 'audit': AUDIT},
            'recovery_scope': 'Same methods, roots, order, native paths, aggregates and budgets. Fresh freeze before quality-v3 evaluation; only fresh v3 final checkpoints after its complete replay audit. No v2 partial checkpoint reuse. Prior costs and failed gates remain retained.'}


def valid_time(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def validate_plan(path):
    plan = cost.study.read(path)
    expected = signature()
    if (set(plan) != set(expected) | {'freeze_state', 'prepared_unix'}
            or any(plan[k] != v for k, v in expected.items())):
        raise ValueError('Frozen recovery cost protocol or sources changed')
    state = plan['freeze_state']
    if (not isinstance(state, dict)
            or set(state) != {'completed_fits', 'quality_started_unix', 'evaluation_outputs_absent'}
            or state['evaluation_outputs_absent'] is not True
            or type(state['completed_fits']) is not int
            or not 0 <= state['completed_fits'] < len(cost.SEEDS)*(len(cost.METHODS)-1)
            or not valid_time(state['quality_started_unix']) or not valid_time(plan['prepared_unix'])
            or not state['quality_started_unix'] <= plan['prepared_unix'] <= time.time()):
        raise ValueError('Invalid fresh pre-outcome freeze evidence')
    return plan, recovery.sha(path)


cost.signature = signature
cost.validate_plan = validate_plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'run', 'audit'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command != 'prepare' and args.plan is None:
        parser.error('--plan required')
    if args.command != 'verify' and args.out is None:
        parser.error('--out required')
    if args.command == 'audit' and args.execution is None:
        parser.error('--execution required for audit')
    if args.command == 'prepare':
        cost.prepare(args.out)
    elif args.command == 'verify':
        _, digest = validate_plan(args.plan)
        print(digest, flush=True)
    else:
        cost.execute(args.plan, args.out, args.execution if args.command == 'audit' else None)
