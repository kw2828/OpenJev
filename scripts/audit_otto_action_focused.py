"""Independent saved-record action-focused audit, without model or simulator calls.

The hash-pinned old auditor supplies only process/file lifecycle and complete
VALID-window reconstruction. This version independently checks chronological
TRAIN query/selected-label bytes, saved fit accounting and all 41 condition records and three named gates.
Teacher/filter/inference/optimizer truth is inherited, never re-executed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = "scripts/audit_otto_score_forecasts.py"
BASE_PIN = "c944b2edf7b053c65cb43b3e24ee81a549a1584808cb3fbf0553c28593e9d2ce"
SELF = "scripts/audit_otto_action_focused.py"
TEST = "tests/test_audit_otto_action_focused.py"
VERSION = "otto-action-focused-saved-audit-v1"
PRODUCER = "scripts/train_otto_action_focused.py"
PRODUCER_VERSION = "otto-action-focused-training-v1"
COLLECTOR = "scripts/collect_otto_action_focused.py"
COLLECTOR_VERSION = "otto-action-focused-collection-v1"
SEEDS = (295000001, 295000002, 295000003)
SELECTION_START = 296000001
COLLECTION_SECONDS = 7200
CAPACITY_SCOPE = "Inherited shared-architecture evidence only; new SPO+ arithmetic and current total runtime were not measured"
CAPACITY_FAMILIES = ("innovation_aux", "innovation_mse", "innovation_gru_mse", "innovation_gru_aux")
CAPACITY_CELLS = {k: ("innovation_gru" if k.startswith("innovation_gru") else "innovation",
                      "query_aux" if k.endswith("aux") else "mse") for k in CAPACITY_FAMILIES}
CAPACITY_PARAMETERS = {k: 5996 if a == "innovation_gru" else 5978 for k,(a,_) in CAPACITY_CELLS.items()}


def _base():
    path = ROOT / BASE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_PIN:
        raise ValueError("unchanged qualified forecast audit base")
    spec = importlib.util.spec_from_file_location("_prequery_calibration_audit_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _base()
FAMILIES = ("innovation_aux", "innovation_spo", "gru_aux", "gru_spo")
CELLS = {kind: ("innovation" if kind.startswith("innovation_") else "innovation_gru",
                "shared", "spo" if kind.endswith("_spo") else "aux") for kind in FAMILIES}
REGIMES, ARMS = ("lambda3", "lambda4"), ("analytic", "neural", "period4_hold")
PARAMETERS = {k: 5978 if a == "innovation" else 5996 for k, (a, _, _) in CELLS.items()}
LIMITATIONS = [
    "Saved neural forecasts, optimizer/gradient truth and physical timings are authenticated but not rerun.",
    "Teacher values, original native trajectories and public-filter numerical truth remain inherited evidence.",
    "Independent checks reconstruct full query inputs, selected targets/IPW, saved metrics, 41 condition records and three gates and process/file joins.",
    "Forced-path teacher imitation does not establish autonomous utility, biological learning or architectural novelty.",
    "Initial recurrent equality uses saved hashes, without regenerating initial model tensors.",
    "Inherited capacity evidence covers shared architectures and the old MSE recipe, not SPO+ cost or current total runtime.",
]
# Explicit prospective configuration; no producer imports.
CONFIG = {"families": list(FAMILIES), "fit_seeds": list(SEEDS), "epochs": 80,
          "batch_episodes": 6, "chunk": 32, "learning_rate": .003, "gradient_clip": 5.,
          "weight_decay": 0., "scale": 64., "training_episodes": 54,
          "validation_episodes": 36, "required_conditions": 41,
          "checkpoint": "last", "device": "cpu", "dtype": "float32",
          "selection_seed_start": SELECTION_START,
          "training_weight": "(W/k)/(54*full_episode_nonquery_rows)",
          "batch_scale": "54/actual_batch_episodes", "tbptt": "detach every32; accumulate then one Adam step",
          "loss": "eligible-centered nonquery MSE plus all-four prequery MSE; spo adds nonquery SPO+; scores /64",
          "cells": {k: {"architecture": v[0], "readout": v[1], "objective": v[2]} for k, v in CELLS.items()},
          "prior_coefficient": 1., "spo_coefficient": 1., "prior_weight": "1/(54*full_episode_later_query_rows)",
          "compute_matching": "same episode exposure, updates and backward chunks; actual objective costs counted",
          "capacity_evidence_scope": CAPACITY_SCOPE}
for _name in ("SELF", "TEST", "VERSION", "PRODUCER", "PRODUCER_VERSION", "COLLECTOR", "COLLECTOR_VERSION",
              "SEEDS", "FAMILIES", "REGIMES", "ARMS", "PARAMETERS", "LIMITATIONS", "CONFIG"):
    setattr(base, _name, globals()[_name])
require, f32, near_set = base.require, base.f32, base.near_set


def cohort():
    result, global_case = [], 0
    for stage, size, starts in (("train", 9, (291000001, 292000001)), ("valid", 6, (293000001, 294000001))):
        for regime, start in zip(REGIMES, starts, strict=True):
            for case in range(size):
                shift = global_case % 3
                for arm in ARMS[shift:] + ARMS[:shift]:
                    result.append({"stage": stage, "episode_index": len(result), "regime": regime,
                        "seed": start + case, "case": case, "initial_hit": 1 + case % 3,
                        "arm": arm, "episode_id": f"{stage}:{regime}:{start + case}:{arm}"})
                global_case += 1
    return result


base.cohort = cohort


def journal_exhausted(rows):
    """A saved JSON null is an extra record, never the end-of-stream marker."""
    marker = object()
    return next(rows, marker) is marker


def prior_arrays(np, query_scores, offsets, *, training):
    """Independent exact target/mask/weight construction, preserving all episodes."""
    total, episodes = len(query_scores), len(offsets) - 1
    require(episodes > 0 and int(offsets[0]) == 0 and int(offsets[-1]) == total,
            'complete prior offsets')
    targets = np.zeros((total, 4), np.float32)
    weights, mask = np.zeros(total, np.float64), np.zeros(total, np.bool_)
    supported = 0
    for low, high in pairwise(offsets):
        low, high = int(low), int(high)
        require(1 <= high - low <= 2188, 'prior episode length')
        rows = list(range(low + 4, high, 4))
        if rows:
            require(bool(np.isfinite(query_scores[rows]).all()), 'finite observed prior targets')
            targets[rows] = query_scores[rows]
            mask[rows] = True
            if training:
                weights[rows] = 1 / (episodes * len(rows))
            supported += 1
    return {'prior_targets': targets, 'prior_weights': weights, 'prior_mask': mask}, supported


def scalar_prior_metrics(history, prior, identities, *, check=None):
    """Independent scalar all-four raw squared-error metric; inactive rows ignored."""
    ids, regimes, offsets = history['episode_ids'], history['episode_regimes'], history['episode_offsets']
    require(len(ids) == len(regimes) == len(identities) > 0 and len(set(ids)) == len(ids)
            and len(offsets) == len(ids) + 1 and int(offsets[0]) == 0
            and int(offsets[-1]) == len(prior), 'complete prior metric geometry')
    losses, cases, seen = [], [], set()
    for index, (low, high) in enumerate(pairwise(offsets)):
        low, high = int(low), int(high)
        identity = identities[index]
        require(identity['episode_id'] == ids[index] and identity['regime'] == regimes[index]
                and type(identity['case']) is int and identity['case'] >= 0 and identity['arm'] in ARMS,
                'aligned prior case/collector')
        key = (regimes[index], identity['case'], identity['arm'])
        require(key not in seen and 1 <= high-low <= 2188, 'unique complete prior episode')
        seen.add(key); cases.append(key[:2])
        rows = []
        for step, row in enumerate(range(low, high)):
            require(bool(history['query_mask'][row]) == (step % 4 == 0), 'exact true-query chronology')
            if step < 4 or step % 4:
                continue
            p, q = tuple(map(float, prior[row])), tuple(map(float, history['query_scores'][row]))
            require(len(p) == len(q) == 4 and all(math.isfinite(x) for x in (*p, *q)),
                    'finite active all-four prior values')
            pm, qm = math.fsum(p) / 4, math.fsum(q) / 4
            rows.append(math.fsum(((a-pm)-(b-qm))**2 for a, b in zip(p, q, strict=True)) / 4)
        losses.append(rows)
        if check is not None:
            check()

    def group(indices):
        active = [i for i in indices if losses[i]]
        numerator = math.fsum(math.fsum(losses[i]) / len(losses[i]) for i in active)
        declared, supported = {cases[i] for i in indices}, {cases[i] for i in active}
        def records(items):
            return [{'regime': regime, 'case': case} for regime, case in sorted(items)]
        return {'episodes': len(indices), 'supported_episodes': len(active),
            'zero_support_episode_ids': [ids[i] for i in indices if not losses[i]],
            'prior_rows': sum(len(losses[i]) for i in indices), 'weight_mass': len(active) / len(indices),
            'episode_weighted_centered_mse': numerator / len(indices),
            'supported_episode_centered_mse': numerator / len(active) if active else None,
            'declared_case_count': len(declared), 'supported_case_count': len(supported),
            'supported_cases': records(supported), 'zero_support_cases': records(declared-supported)}
    return {'version': 'otto-prequery-metrics-v1',
        'scope': 'raw-score calibration on fixed paths; no autonomous efficacy',
        'prior_mask': 'query and absolute_step >= 4',
        'coordinates': 'all four actions; no legal mask; separately centered',
        'overall': group(list(range(len(ids)))),
        'by_regime': {r: group([i for i, v in enumerate(regimes) if v == r]) for r in dict.fromkeys(regimes)},
        'by_case': [{'regime': r, 'case': c, **group([i for i, v in enumerate(cases) if v == (r, c)])}
                    for r, c in sorted(set(cases))],
        'by_episode': [{'episode_id': ids[i], 'regime': regimes[i], 'case': cases[i][1],
                        'arm': identities[i]['arm'], **group([i])} for i in range(len(ids))]}


def scalar_training_loss(predictions, targets, weights, legal=None, *, check=None):
    """Real arithmetic reference plus an IEEE-f32 operation-error envelope.

    Saved f32 weights are the actual loss inputs. Each four-coordinate center,
    difference, square and mean has at most 24 rounded operations. A gamma24
    bound on sum magnitudes covers cancellation; direct squared deviations then
    propagate this error. Gamma(N) covers any nonnegative reduction order. The
    minimum-subnormal allowance covers underflow. This is not Torch replay.
    """
    u, eta = 2.**-24, 2.**-149
    def gamma(n):
        return n*u / (1-n*u)
    values, errors = [], []
    require(len(predictions) == len(targets) == len(weights), 'saved loss row geometry')
    for row, weight in enumerate(weights):
        weight = float(f32(float(weight)))
        require(math.isfinite(weight) and weight >= 0, 'finite saved loss weight')
        if not weight:
            continue
        allowed = range(4) if legal is None else [i for i in range(4) if legal[row][i]]
        require(bool(allowed), 'weighted row has action support')
        p = [float(predictions[row][i]) / 64 for i in allowed]
        q = [float(targets[row][i]) / 64 for i in allowed]
        require(all(math.isfinite(v) for v in (*p, *q)), 'finite active saved loss scores')
        k = len(p); pm, qm = math.fsum(p)/k, math.fsum(q)/k
        d = [(a-pm)-(b-qm) for a,b in zip(p,q,strict=True)]
        magnitude = math.fsum(abs(v) for v in (*p,*q))
        delta = gamma(24)*magnitude + 24*eta
        value = math.fsum(v*v for v in d)/k
        error = math.fsum(2*abs(v)*delta + delta*delta + u*(abs(v)+delta)**2 + eta for v in d)/k
        error += gamma(4)*(value+error) + 4*eta
        values.append(value*weight)
        errors.append(error*weight + u*(value+error)*weight + eta)
        if check is not None and row % 512 == 0:
            check()
    n = len(weights)
    require(n*u < 1, 'bounded saved loss reduction')
    total, error = math.fsum(values), math.fsum(errors)
    return total, error + gamma(n)*(total+error) + n*eta


def scalar_spo_loss(predictions, targets, weights, legal, *, check=None):
    """Independent real SPO+ reference with a prospective float32 bound.

    Exact raw teacher minima define a uniform reference before division by64.
    The max is 1-Lipschitz: a bound for every affine margin also bounds its max.
    At most32 rounded operations with total coefficient magnitude bounded by8
    cover scale/subtract/reference-mean/margin construction. gamma32*8*L1 plus
    64 minimum subnormals conservatively covers this route, including ties and
    cancellation. Weighted multiplication and any nonnegative N-term reduction
    add their standard gamma bounds. This does not replay Torch or gradients.
    No minibatch54/B factor belongs in the final whole-cohort rescore.
    """
    u, eta = 2.**-24, 2.**-149

    def gamma(n):
        require(n*u < 1, 'bounded SPO reduction')
        return n*u/(1-n*u)

    require(len(predictions) == len(targets) == len(weights) == len(legal), 'saved SPO geometry')
    values, errors = [], []
    for index, original_weight in enumerate(weights):
        if check is not None and index % 512 == 0:
            check()
        original_weight = float(original_weight)
        weight = float(f32(original_weight))
        require(math.isfinite(original_weight) and original_weight >= 0 and math.isfinite(weight)
                and (weight > 0 if original_weight > 0 else weight == 0), 'finite positive f32 importance weight')
        if not weight:
            continue
        actions = [a for a in range(4) if legal[index][a]]
        require(bool(actions), 'weighted SPO row has legal actions')
        q = [float(targets[index][a]) for a in actions]
        p = [float(predictions[index][a]) for a in actions]
        require(all(math.isfinite(v) for v in (*p,*q)), 'finite active SPO scores')
        minimum = min(q)
        best = [i for i, value in enumerate(q) if value == minimum]
        z, c = [v/64 for v in p], [v/64 for v in q]
        reference = math.fsum(z[i] for i in best)/len(best)
        row = max((teacher - minimum/64) - 2*(prediction-reference)
                  for teacher,prediction in zip(c,z,strict=True))
        magnitude = math.fsum(abs(v) for v in (*z,*c))
        bound = gamma(32)*8*magnitude + 64*eta
        require(row >= -bound and math.isfinite(row), 'finite SPO reference and rounding enclosure')
        values.append(row*weight)
        errors.append(bound*weight + u*(abs(row)+bound)*weight + eta)
    total, error = math.fsum(values), math.fsum(errors)
    return total, error + gamma(len(weights))*(math.fsum(abs(v) for v in values)+error) + len(weights)*eta


def scalar_metrics(predictions, windows, metadata, identities, *, check=None):
    """Scalar per-episode reduction, including unsupported episodes and cases."""
    ids, regimes = metadata['episode_ids'], metadata['episode_regimes']
    require(len(ids) == len(identities) == len(regimes) and len(set(ids)) == len(ids), 'unique aligned identities')
    cases = []
    for index, identity in enumerate(identities):
        require(identity['episode_id'] == ids[index] and identity['regime'] == regimes[index]
                and type(identity['case']) is int and identity['arm'] in ARMS, 'original case identity')
        cases.append((identity['regime'], identity['case']))
    samples = [[] for _ in ids]
    for window in range(len(windows['lengths'])):
        index = int(windows['episode_index'][window])
        for age in range(1, int(windows['lengths'][window])):
            scores, teacher, legal = predictions[window][age], windows['targets'][window][age], windows['legal'][window][age]
            proposed, _ = near_set(scores, legal)
            optimal, best = near_set(teacher, legal)
            action = proposed[0]
            allowed = [i for i in range(4) if legal[i]]
            ps, ts = [float(scores[i]) for i in allowed], [float(teacher[i]) for i in allowed]
            pm, tm = math.fsum(ps) / len(ps), math.fsum(ts) / len(ts)
            error = math.fsum(((p - pm) - (t - tm))**2 for p, t in zip(ps, ts, strict=True)) / len(ps)
            samples[index].append((int(windows['step_offsets'][window]) + age, age,
                float(action in optimal), float(teacher[action]) - best, float(action == optimal[0]), error))
        if check is not None and window % 256 == 0:
            check()

    def reduce(indices, minimum_step, age=None, maximum_step=None):
        active, empty, row_count = [], [], 0
        totals = [[], [], [], []]
        for index in indices:
            rows = [row for row in samples[index] if row[0] >= minimum_step and (maximum_step is None or row[0] <= maximum_step) and (age is None or row[1] == age)]
            row_count += len(rows)
            if not rows:
                empty.append(ids[index])
            else:
                active.append(index)
                for column in range(4):
                    totals[column].append(math.fsum(row[column + 2] for row in rows) / len(rows))
        sums = [math.fsum(values) for values in totals]
        denominator, supported = len(indices), len(active)
        declared_cases, active_cases = {cases[i] for i in indices}, {cases[i] for i in active}
        def records(values):
            return [{'regime': regime, 'case': case} for regime, case in sorted(values)]
        return {'episodes': denominator, 'supported_episodes': supported, 'zero_support_episode_ids': empty,
            'nonquery_rows': row_count, 'weight_mass': supported / denominator,
            'episode_weighted_agreement': sums[0] / denominator, 'episode_weighted_raw_gap': sums[1] / denominator,
            'episode_weighted_first_argmin_match': sums[2] / denominator,
            'episode_weighted_centered_mse': sums[3] / denominator,
            'supported_episode_agreement': sums[0] / supported if supported else None,
            'supported_episode_raw_gap': sums[1] / supported if supported else None,
            'supported_episode_centered_mse': sums[3] / supported if supported else None,
            'declared_case_count': len(declared_cases), 'supported_case_count': len(active_cases),
            'supported_cases': records(active_cases), 'zero_support_cases': records(declared_cases - active_cases)}

    def scope(first, last=None):
        def group(indices):
            return {**reduce(indices, first, maximum_step=last), 'by_age': {str(age): reduce(indices, first, age, last) for age in (1, 2, 3)}}
        return {'overall': group(list(range(len(ids)))),
            'by_regime': {r: group([i for i, v in enumerate(regimes) if v == r]) for r in dict.fromkeys(regimes)},
            'by_case': [{'regime': r, 'case': c, **group([i for i, v in enumerate(cases) if v == (r, c)])}
                        for r, c in sorted(set(cases))]}
    return {'version': 'otto-separate-prior-metrics-v1',
            'scope': 'saved teacher imitation on fixed paths; no autonomous efficacy',
            'primary_mask': 'nonquery and absolute_step >= 5', 'full': scope(0), 'postcorrection': scope(5), 'initial': scope(1, 3)}


def continuation_rules(models, hold, *, technical_complete):
    """Independent fixed-seed objective and architecture decisions."""
    require(type(technical_complete) is bool, 'strict technical state')
    by = {(row['family'], row['seed']): row for row in models}
    require(len(models) == len(by) == 12 and set(by) == {(f,s) for f in FAMILIES for s in SEEDS},
            'all final models without selection')
    agreement, gap = 'episode_weighted_agreement', 'episode_weighted_raw_gap'
    means = {(f,r,scope,key): math.fsum(by[f,seed]['metrics'][scope]['by_regime'][r][key]
                 for seed in SEEDS)/3 for f in FAMILIES for r in REGIMES
                 for scope in ('full','initial','postcorrection') for key in (agreement,gap)}
    rows = [{'name':'common.technical_complete','value':technical_complete,'relation':'==',
             'threshold':True,'passes':technical_complete}]

    def add(name, value, relation, threshold, strict=None):
        require(math.isfinite(value) and math.isfinite(threshold), 'finite decision operands')
        passed = value >= threshold if relation == '>=' else value > threshold if relation == '>' else value <= threshold
        row = {'name':name,'value':value,'relation':relation,'threshold':threshold,'passes':bool(passed)}
        if strict is not None:
            require(math.isfinite(strict), 'finite strict comparison')
            row.update(relation='<= and <',strict_upper_bound=strict,passes=bool(passed and value < strict))
        rows.append(row)

    for regime in REGIMES:
        add(f'common.{regime}.initial.case_support',hold['initial']['by_regime'][regime]['supported_case_count'],'>=',4)
        for age in ('1','2','3'):
            add(f'common.{regime}.age{age}.case_support',
                hold['postcorrection']['by_regime'][regime]['by_age'][age]['supported_case_count'],'>=',4)
        add(f'common.{regime}.hold.postcorrection.positive_gap',
            hold['postcorrection']['by_regime'][regime][gap],'>',0.)
    for architecture in ('innovation','gru'):
        candidate, control = architecture+'_spo', architecture+'_aux'
        for regime in REGIMES:
            prefix = f'objective.{architecture}.{regime}'
            own, other = means[candidate,regime,'postcorrection',gap],means[control,regime,'postcorrection',gap]
            add(prefix+'.postcorrection.gap',own,'<=',.9*other,strict=other)
            for scope in ('full','initial'):
                add(prefix+'.'+scope+'.agreement',means[candidate,regime,scope,agreement],'>=',means[control,regime,scope,agreement])
            for seed in SEEDS:
                add(prefix+f'.{seed}.postcorrection.gap_nonregression',
                    by[candidate,seed]['metrics']['postcorrection']['by_regime'][regime][gap],'<=',
                    by[control,seed]['metrics']['postcorrection']['by_regime'][regime][gap])
    for regime in REGIMES:
        other = min(means[f,regime,'postcorrection',gap] for f in ('gru_aux','gru_spo'))
        add(f'architecture.{regime}.postcorrection.gap',means['innovation_spo',regime,'postcorrection',gap],
            '<=',.9*other,strict=other)
        for scope in ('full','initial'):
            add(f'architecture.{regime}.{scope}.agreement',means['innovation_spo',regime,scope,agreement],'>=',
                max(means[f,regime,scope,agreement] for f in ('gru_aux','gru_spo')))
    require(len(rows) == len({r['name'] for r in rows}) == 41, 'all 41 unique decision records')
    return rows


def gate_decisions(rows):
    require(len(rows) == len({r['name'] for r in rows}) == 41
            and all(type(r['passes']) is bool for r in rows), 'complete Boolean named-gate records')
    result = {}
    for gate, prefixes, size in (
            ('innovation_objective',('common.','objective.innovation.'),23),
            ('gru_objective',('common.','objective.gru.'),23),
            ('architecture',('common.','objective.innovation.','architecture.'),29)):
        selected = [r for prefix in prefixes for r in rows if r['name'].startswith(prefix)]
        require(len(selected) == size, 'complete named gate '+gate)
        result[gate] = {'conditions':[r['name'] for r in selected],'passed':sum(r['passes'] for r in selected),
                        'total':size,'passes':all(r['passes'] for r in selected)}
    return result

CLOCK = base.CLOCK

CLOCK_PIN = base.CLOCK_PIN

SUPERVISOR = base.SUPERVISOR

SUPERVISOR_PIN = base.SUPERVISOR_PIN

LIMITS = {"seconds": 240, "rss_bytes": 2 * 1024**3, "output_bytes": 256 * 1024**2}
base.LIMITS = LIMITS

write = base.write


class Audit(base.Audit):
    def admit(self):
        self.require(self.sha(self.path(CLOCK)) == CLOCK_PIN, "qualified clock before import")
        spec = importlib.util.spec_from_file_location("_forecast_audit_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            self.require(self.clock.now_ns() - self.start < 5 * 10**9, "original audit launch available")
            time.sleep(.01)
        self.launch = self.read(self.args.supervision)
        cmd = list(self.launch["command"])
        if cmd[1:2] == ["-u"]:
            cmd.pop(1)
        self.require(cmd == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                     and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                     and self.launch["cwd"] == str(ROOT) == str(Path.cwd())
                     and self.launch["cap_seconds"] == LIMITS["seconds"]
                     and self.launch["clock_backend"] == self.clock.backend
                     and self.launch["clock_source_sha256"] == CLOCK_PIN
                     and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
                     and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
                     and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["seconds"] * 10**9,
                     "original bounded saved-audit process")
        self.inputs = {}
        for role in ("plan", "worker", "terminal"):
            path = self.path(getattr(self.args, role))
            descriptor = self.descriptor(path)
            self.require(descriptor["sha256"] == getattr(self.args, role + "_sha256"), "external " + role)
            self.inputs[role] = descriptor
        self.plan = self.read(self.args.plan)
        self.require(self.plan["version"] == PRODUCER_VERSION and self.plan["status"] == "frozen_before_fitting"
                     and self.plan["config"] == CONFIG
                     and self.plan["limits"] == {"seconds": 14400, "rss_bytes": 4 * 1024**3,
                                                 "output_bytes": 2 * 1024**3}, "fixed training allocation")
        self.require({SELF, TEST, CLOCK, SUPERVISOR, BASE_PATH} <= self.plan["sources"].keys(),
                     "audit source frozen before fitting")
        for name, pin in self.plan["sources"].items():
            self.require(self.sha(self.path(name)) == pin, "frozen source " + name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.require(os.environ.get(name) == "1", "single numerical thread")
        self.receipt.update(plan_sha256=self.args.plan_sha256, producer_inputs=self.inputs,
                            sources={name: self.plan["sources"][name] for name in (SELF, TEST, CLOCK, SUPERVISOR)},
                            supervision_sha256=self.sha(self.args.supervision))
        write(self.out / "started.json", {"started_ns": self.start, "launch": self.launch,
                                           "producer_inputs": self.inputs})

    def authenticate(self):
        _, self.worker, self.parent, self.run = self.phase(
            self.args.plan, self.args.worker, self.args.terminal, PRODUCER, ".venv/bin/python", 14400)
        self.require(self.worker["version"] == PRODUCER_VERSION and self.worker["fits_completed"] == 12
                     and self.worker["teacher_calls"] == self.worker["native_calls"] == 0,
                     "complete sampled forecast fitting without scientific collection")
        roles = self.plan["inputs"]
        self.require(set(roles) == {"collection_plan", "collection_receipt", "collection_terminal", "engineering", "capacity_plan", "capacity_receipt", "capacity_terminal"},
                     "exact training source roles")
        self.collection_plan, self.collection_worker, self.collection_parent, self.collection = self.phase(
            self.path(roles["collection_plan"]["path"]), self.path(roles["collection_receipt"]["path"]),
            self.path(roles["collection_terminal"]["path"]), COLLECTOR,
            ".venv-otto-released-native/bin/python", COLLECTION_SECONDS)
        c = self.collection_worker
        payloads = {name + ".jsonl.gz" for name in ("work", "weights", "forwards", "transitions", "samples", "annotations")}
        payloads.update({"started.json", "runtime.json", "setup.json", "deployment.json", "cohort.json",
            "episode-boundaries.jsonl", "episodes.jsonl", "train.npz", "valid.npz", "train-selection.json", "costs.json", "summary.json"})
        self.require(self.collection_plan["version"] == COLLECTOR_VERSION
                     and self.collection_plan["status"] == "frozen_before_collection"
                     and self.collection_plan["limits"] == {"native_seconds": COLLECTION_SECONDS,
                         "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
                     and set(c["files"]) == set(self.collection_plan["payloads"]) == payloads,
                     "new complete collection allocation and eighteen payloads")
        self.require(c["version"] == COLLECTOR_VERSION and c["completed_episodes"] == 90
                     and c["train_episodes"] == 54 and c["valid_episodes"] == 36
                     and c["training_updates"] == 0, "complete new ninety-path collection")
        self.equal(self.collection_plan["cohort"], cohort(), "exact prospective collection cohort")
        self.episodes = list(self.rows(self.collection / "episodes.jsonl"))
        self.require(len(self.episodes) == 90, "all complete episode records")
        for row, identity in zip(self.episodes, cohort(), strict=True):
            self.equal({key: row[key] for key in identity}, identity, "ordered episode identity")
            steps = row["steps"]
            self.require(type(steps) is int and 1 <= steps <= 2188 and row["rows"] == row["updates"] == steps
                         and row["final_update_assimilated"] is True and type(row["found"]) is bool
                         and row["censored"] is (not row["found"]) and (row["found"] or steps == 2188)
                         and row["corrections"] == (steps + 3) // 4, "all complete path tails and final updates")
            self.require(all(type(row[key]) is int and row[key] >= 0 for key in
                             ("teacher_calls", "deployed_queries", "annotation_only", "deferred_annotations", "unlabeled_rows"))
                         and row["teacher_calls"] == row["deployed_queries"] + row["annotation_only"]
                         and row["teacher_calls"] + row["unlabeled_rows"] == steps
                         and row["deferred_annotations"] <= row["annotation_only"], "one returned score per labeled state")
            if row["stage"] == "valid":
                self.require(row["teacher_calls"] == steps and row["unlabeled_rows"] == row["deferred_annotations"] == 0,
                             "full VALID teacher census unchanged")
        self.capacity()
        engineering = self.read(roles["engineering"]["path"])
        self.require(engineering["status"] == "passed"
                     and engineering["sources_before"] == engineering["sources_after"]
                     and all(row["exit_code"] == 0 for row in engineering["results"]), "qualified engineering scope")

    def sampled_training(self):
        np = self.np
        flat = self.arrays(self.collection / "train.npz")
        selections = self.read(self.collection / "train-selection.json")
        identities = [row for row in self.episodes if row["stage"] == "train"]
        self.require(len(identities) == len(selections) == 54, "all TRAIN episodes and selections")
        self.require(set(flat) == {"features", "raw_q", "legal", "actions", "correction", "episode_offsets", "label_mask"},
                     "exact sampled flat array keys")
        offsets = flat["episode_offsets"]
        self.require(offsets.dtype == np.int64 and offsets.shape == (55,) and offsets[0] == 0
                     and bool((np.diff(offsets) >= 1).all()) and bool((np.diff(offsets) <= 2188).all()), "full TRAIN offsets")
        total = int(offsets[-1])
        for name, dtype, shape in (("features", np.float32, (total, 31)), ("raw_q", np.float32, (total, 4)),
                                   ("legal", np.bool_, (total, 4)), ("actions", np.int64, (total,)),
                                   ("correction", np.bool_, (total,)), ("label_mask", np.bool_, (total,))):
            self.require(flat[name].dtype == dtype and flat[name].shape == shape, "exact TRAIN " + name)
        self.require(bool(np.isfinite(flat["features"]).all()) and bool(flat["legal"].any(axis=1).all()), "finite public features and legal support")
        missing = flat["raw_q"][~flat["label_mask"]]
        self.require(missing.tobytes() == np.zeros(missing.shape, np.float32).tobytes(), "unscored positive-zero placeholders")
        lengths = [int(hi - lo) for lo, hi in pairwise(offsets)]
        records = []
        for index, (identity, length) in enumerate(zip(identities, lengths, strict=True)):
            population = (length + 3) // 4
            count = min(8, population)
            seed = SELECTION_START + index
            rng = np.random.Generator(np.random.PCG64(seed))
            starts = sorted(int(value) * 4 for value in rng.choice(population, size=count, replace=False))
            expected = {"episode_id": identity["episode_id"], "version": "otto-sampled-forecast-data-v1",
                "length": length, "seed": seed, "period": 4, "population_windows": population,
                "selected_windows": count, "start_offsets": starts, "inclusion_numerator": count,
                "inclusion_denominator": population, "inclusion_probability": count / population,
                "algorithm": "numpy.Generator(PCG64(seed)).choice(W,size=k,replace=False); sorted"}
            self.equal(selections[index], expected, "independent fixed uniform TRAIN selection")
            self.require(all(type(selections[index][key]) is type(value) for key, value in expected.items())
                         and all(type(value) is int for value in selections[index]["start_offsets"]), "exact selection scalar types")
            records.append(expected)
        n = sum(record["selected_windows"] for record in records)
        full_counts = [length - (length + 3) // 4 for length in lengths]
        output = {"features": np.zeros((n, 4, 31), np.float32), "query_scores": np.zeros((n, 4), np.float32),
            "targets": np.zeros((n, 4, 4), np.float32), "legal": np.zeros((n, 4, 4), np.bool_),
            "valid_mask": np.zeros((n, 4), np.bool_), "nonquery_mask": np.zeros((n, 4), np.bool_),
            "lengths": np.zeros(n, np.int64), "episode_index": np.zeros(n, np.int64),
            "step_offsets": np.zeros(n, np.int64), "nonquery_weights": np.zeros((n, 4), np.float64),
            "episode_lengths": np.asarray(lengths, np.int64), "episode_nonquery_counts": np.asarray(full_counts, np.int64)}
        window = 0
        for index, (identity, record, length) in enumerate(zip(identities, records, lengths, strict=True)):
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            self.require(identity["start_row"] == lo and identity["end_row"] == hi and identity["steps"] == length,
                         "complete full-history row ranges")
            x, actions = flat["features"][lo:hi], flat["actions"][lo:hi]
            steps = np.arange(length)
            self.require(np.array_equal(x[:, 15], (steps / 2188).astype(np.float32))
                         and np.array_equal(x[:, 16], ((steps % 4) / 2188).astype(np.float32))
                         and bool((x[:, 17] == 1).all())
                         and np.array_equal(flat["correction"][lo:hi], steps % 4 == 0)
                         and bool(((actions >= 0) & (actions < 4)).all())
                         and bool(flat["legal"][lo:hi][steps, actions].all()), "full public chronology and legal actions")
            self.require(int(flat["label_mask"][lo:hi].sum()) == identity["teacher_calls"], "all physically labeled rows accounted")
            row_weight = (record["population_windows"] / record["selected_windows"]) / (54 * full_counts[index]) if full_counts[index] else 0.
            for start in record["start_offsets"]:
                size = min(4, length - start)
                sl = slice(lo + start, lo + start + size)
                self.require(bool(flat["label_mask"][sl].all()) and bool(np.isfinite(flat["raw_q"][sl]).all()), "selected labels complete and finite")
                for key, source in (("features", "features"), ("targets", "raw_q"), ("legal", "legal")):
                    output[key][window, :size] = flat[source][sl]
                output["query_scores"][window] = flat["raw_q"][lo + start]
                output["valid_mask"][window, :size] = True
                output["nonquery_mask"][window, 1:size] = True
                output["lengths"][window] = size
                output["episode_index"][window] = index
                output["step_offsets"][window] = start
                output["nonquery_weights"][window, 1:size] = row_weight
                window += 1
            self.check()
        selected_rows = int(output["valid_mask"].sum())
        counts = {"episodes": 54, "windows": n, "rows": selected_rows, "query_rows": n,
            "nonquery_rows": selected_rows - n, "zero_support_episodes": sum(value == 0 for value in full_counts),
            "query_only_windows": int((output["lengths"] == 1).sum()), "full_rows": total,
            "population_windows": sum(record["population_windows"] for record in records), "full_nonquery_rows": sum(full_counts)}
        meta = {"version": "otto-score-forecast-data-v1", "episode_ids": [row["episode_id"] for row in identities],
            "episode_regimes": [row["regime"] for row in identities], "episode_splits": ["train"] * 54,
            "counts": counts, "sampling": {"version": "otto-sampled-forecast-data-v1",
                "unit": "disjoint period-four TRAIN windows", "episode_denominator": 54, "window_cap": 8,
                "selections": records, "weight_formula": "(population_windows/selected_windows)/(54*full_episode_nonquery_rows)",
                "realized_weight_mass": float(output["nonquery_weights"].sum()), "renormalized": False,
                "shared_across_fits": True, "validation": "unchanged full census"}}
        self.counts.update(train_rows=total, train_windows=n, selected_train_rows=selected_rows,
                           selected_train_nonquery_rows=selected_rows - n)
        return output, meta, identities

    def saved_windows(self, stage, windows, meta):
        saved = self.arrays(self.run / f"{stage}.npz")
        self.require(set(saved) == set(windows), "exact persisted window keys")
        for key, expected in windows.items():
            value = saved[key]
            self.require(value.dtype == expected.dtype and value.shape == expected.shape
                         and value.tobytes() == expected.tobytes(), "independent persisted window bytes " + key)
        self.equal(self.read(self.run / f"{stage}.json"), meta, "persisted identities, selection and weights")

    def reconstruct(self, stage):
        if stage != 'train':
            return super().reconstruct(stage)
        np = self.np
        selected, selected_meta, identities = self.sampled_training()
        flat = self.arrays(self.collection / 'train.npz')
        total = len(flat['features'])
        result = {k: flat[k].copy() for k in ('features', 'legal', 'actions', 'episode_offsets')}
        result.update(query_scores=np.zeros((total, 4), np.float32), targets=np.zeros((total, 4), np.float32),
                      query_mask=flat['correction'].copy(), weights=np.zeros(total, np.float64))
        query_rows = result['query_mask']
        self.require(bool(flat['label_mask'][query_rows].all())
                     and bool(np.isfinite(flat['raw_q'][query_rows]).all()), 'all chronological queries have finite returned labels')
        result['query_scores'][query_rows] = flat['raw_q'][query_rows]
        self.require(np.array_equal(result['query_scores'] / np.float32(64) * np.float32(64), result['query_scores']),
                     'qualified exact query-score scale domain')
        for window, index in enumerate(selected['episode_index']):
            low = int(flat['episode_offsets'][index]) + int(selected['step_offsets'][window])
            length = int(selected['lengths'][window])
            result['targets'][low+1:low+length] = selected['targets'][window, 1:length]
            result['weights'][low+1:low+length] = selected['nonquery_weights'][window, 1:length]
        prior, supported = prior_arrays(np, result['query_scores'], result['episode_offsets'], training=True)
        result.update(prior)
        meta = {'version': 'otto-prequery-data-v1',
            **{k: selected_meta[k] for k in ('episode_ids', 'episode_regimes', 'episode_splits', 'sampling')},
            'counts': {'episodes': 54, 'rows': total, 'query_rows': int(query_rows.sum()),
                'selected_nonquery_rows': int((result['weights'] > 0).sum()),
                'selected_windows': selected_meta['counts']['windows'],
                'zero_support_episodes': selected_meta['counts']['zero_support_episodes'],
                'prior_rows': int(prior['prior_mask'].sum()), 'prior_supported_episodes': supported,
                'prior_zero_support_episodes': 54-supported,
                'prior_weight_mass': float(prior['prior_weights'].sum())}}
        return result, meta, identities

    def capacity(self):
        roles = self.plan['inputs']
        paths = {r: self.path(roles['capacity_' + r]['path']) for r in ('plan', 'receipt', 'terminal')}
        for role, path in paths.items():
            self.pinned(path, roles['capacity_' + role])
        plan, worker, terminal = (self.read(paths[r]) for r in ('plan', 'receipt', 'terminal'))
        command = list(terminal['command'])
        if command[1:2] == ['-u']:
            command.pop(1)
        self.require(command[:3] == [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/qualify_otto_prequery_capacity.py'), 'run']
                     and len(command) == 11 and len(set(command[3::2])) == 4, 'original capacity process command')
        opts = dict(zip(command[3::2], command[4::2], strict=True))
        self.require(set(opts) == {'--plan', '--plan-sha256', '--supervision', '--output'}
                     and opts['--plan'] == str(paths['plan']) and opts['--plan-sha256'] == self.sha(paths['plan'])
                     == worker['plan_sha256'] and opts['--output'] == str(paths['receipt'].parent), 'capacity input/output joins')
        launch = self.read(opts['--supervision'])
        self.require(self.sha(self.path(opts['--supervision'])) == worker['supervision_sha256'], 'original capacity launch pin')
        base.closed_parent(worker, terminal, launch, 120)
        directory = paths['receipt'].parent
        self.equal(self.read(directory / 'started.json')['launch'], launch, 'capacity saved launch')
        self.require(plan['version'] == worker['version'] == 'otto-prequery-capacity-v1'
                     and plan['status'] == 'frozen_before_synthetic_work'
                     and plan['limits'] == {'seconds': 120, 'rss_bytes': 4*1024**3, 'output_bytes': 128*1024**2}
                     and worker['admitted'] is True and worker['optimizer_updates'] == 4
                     and worker['completed_families'] == list(CAPACITY_FAMILIES)
                     and worker['pending'] is worker['pending_emission'] is None
                     and worker['new_teacher_calls'] == worker['new_native_calls'] == worker['empirical_payloads_read'] == 0,
                     'complete admitted synthetic scope')
        self.equal(worker['sources'], plan['sources'], 'capacity source map')
        self.require(set(worker['files']) == {'started.json','runtime.json','synthetic.json','work.jsonl','summary.json'}
                     and {p.name for p in directory.iterdir()} == set(worker['files']) | {'receipt.json'},
                     'complete five-payload capacity closure')
        for name, pin in worker['files'].items():
            self.require(Path(name).name == name, 'flat capacity payload')
            self.pinned(directory / name, pin)
        for name, pin in plan['sources'].items():
            self.require(self.sha(self.path(name)) == pin == self.plan['sources'].get(name),
                         'same qualified capacity source in training closure')
        self.require(type(worker['peak_rss_bytes']) is int and 0 < worker['peak_rss_bytes'] <= 4*1024**3
                     and sum(row['bytes'] for row in worker['files'].values()) <= 128*1024**2,
                     'bounded original synthetic resources')
        self.pinned(self.path(plan['engineering']['path']), plan['engineering'])
        summary = self.read(directory / 'summary.json')
        rows = summary['families']
        self.require([r['family'] for r in rows] == list(CAPACITY_FAMILIES) and all(r['chunks'] == 69 and r['forward_rows'] == 6*2188
                     and r['optimizer_updates'] == 1 and math.isfinite(r['batch_seconds']) and r['batch_seconds'] > 0 for r in rows),
                     'complete maximum-length capacity batches')
        for row in rows:
            architecture, objective = CAPACITY_CELLS[row['family']]
            self.require(row['architecture'] == architecture and row['objective'] == objective
                         and row['parameter_count'] == CAPACITY_PARAMETERS[row['family']] and row['prior_rows'] == 6*546
                         and row['optimizer_step'] == 1 and row['forward_chunks'] == 69
                         and row['no_grad_chunks'] + row['backward_chunks'] == 69
                         and (row['backward_chunks'] == 69 if objective == 'query_aux' else row['prior_loss'] == 0),
                         'capacity objective and complete later-query work')
        seconds = math.fsum(r['batch_seconds'] for r in rows)
        self.equal(summary['sum_batch_seconds'], seconds, 'independent capacity total')
        self.equal(summary['projected_seconds'], 1.5*3*720*seconds+120, 'prospective capacity projection')
        self.require(plan['configuration']['training_cap_seconds'] == 10800
                     and plan['configuration']['admission_seconds'] == 8100
                     and summary['threshold_seconds'] == 8100 and summary['admitted'] is True
                     and 1.5*3*720*seconds+120 <= 8100, 'capacity admission unchanged')
        self.counts['closed_capacity_phases'] = 1

    def metrics(self, predictions, windows, meta, identities):
        value = scalar_metrics(predictions, windows, meta, identities, check=self.check)
        value["by_collector"] = {}
        for arm in ARMS:
            episodes = [i for i, row in enumerate(identities) if row["arm"] == arm]
            remap = {old: new for new, old in enumerate(episodes)}
            chosen = [i for i, index in enumerate(windows["episode_index"]) if int(index) in remap]
            sub = {key: windows[key][chosen] for key in ("targets", "legal", "lengths", "step_offsets")}
            sub["episode_index"] = [remap[int(windows["episode_index"][i])] for i in chosen]
            metadata = {"episode_ids": [meta["episode_ids"][i] for i in episodes],
                        "episode_regimes": [meta["episode_regimes"][i] for i in episodes]}
            value["by_collector"][arm] = scalar_metrics(predictions[chosen], sub, metadata,
                [identities[i] for i in episodes], check=self.check)
        return value

    def validation_history(self, windows, meta):
        np = self.np
        flat = self.arrays(self.collection / 'valid.npz')
        query = np.zeros_like(flat['raw_q'])
        query[flat['correction']] = flat['raw_q'][flat['correction']]
        values = {key: flat[key].copy() for key in ('features', 'legal', 'actions', 'episode_offsets')}
        values.update(query_scores=query, targets=flat['raw_q'].copy(), query_mask=flat['correction'].copy(),
                      weights=np.zeros(len(query), np.float64))
        prior, _ = prior_arrays(np, query, values['episode_offsets'], training=False)
        values.update(prior)
        metadata = {'version': 'otto-prequery-data-v1',
            **{key: meta[key] for key in ('episode_ids', 'episode_regimes', 'episode_splits')},
            'counts': {'episodes': 36, 'rows': len(query), 'query_rows': int(flat['correction'].sum()),
                       'prior_rows': int(prior['prior_mask'].sum())}}
        self.saved_windows('validation-history', values, metadata)
        return values, metadata

    def expected_batches(self, train, orders, objective):
        np = self.np
        self.require(objective in ('aux','spo'), 'fixed objective')
        offsets = train['episode_offsets']
        lengths = [int(offsets[i+1] - offsets[i]) for i in range(54)]
        supported = np.zeros((54, 69), np.bool_)
        for i, length in enumerate(lengths):
            low = int(offsets[i])
            for start in range(0, length, 32):
                selected = slice(low+start, low+min(length,start+32))
                supported[i, start//32] = bool((train['weights'][selected] > 0).any()) or (
                    objective in ('aux', 'spo') and bool((train['prior_weights'][selected] > 0).any()))
        result = []
        for epoch, order in enumerate(orders, start=1):
            for batch in range(9):
                indices = order[batch*6:(batch+1)*6]
                chunks = (max(lengths[i] for i in indices)+31)//32
                backwards = int(supported[indices,:chunks].any(axis=0).sum())
                result.append({'epoch': epoch, 'batch': batch, 'episode_indices': indices,
                    'forward_chunks': chunks, 'backward_chunks': backwards, 'no_grad_chunks': chunks-backwards,
                    'forward_rows': sum(lengths[i] for i in indices), 'optimizer_step': len(result)+1,
                    'objective': objective, 'prior_rows': sum((lengths[i]+3)//4-1 for i in indices),
                    'spo_loss_calls': backwards if objective == 'spo' else 0,
                    'spo_weighted_rows': sum(int((train['weights'][int(offsets[i]):int(offsets[i+1])] > 0).sum())
                                             for i in indices) if objective == 'spo' else 0})
        return result

    def checkpoint(self, family, filename):
        np = self.np
        architecture, readout, _ = CELLS[family]
        width, inputs = (28,40) if architecture == 'innovation_gru' else (29,35)
        shapes = {'recurrent.weight_ih_l0': (3*width,inputs), 'recurrent.weight_hh_l0': (3*width,width),
            'recurrent.bias_ih_l0': (3*width,), 'recurrent.bias_hh_l0': (3*width,),
            'output.weight': (4,width), 'output.bias': (4,)}
        if architecture == 'innovation':
            shapes['correction.weight'] = (29,4)
        if readout == 'separate':
            shapes.update({'prior_output.weight': (4,width), 'prior_output.bias': (4,)})
        values = self.arrays(self.run / filename)
        self.require(set(values) == set(shapes), 'exact final tensor names')
        for key, shape in shapes.items():
            self.require(values[key].dtype == np.float32 and values[key].shape == shape
                         and bool(np.isfinite(values[key]).all()), 'finite correctly shaped final tensor')
        self.require(sum(v.size for v in values.values()) == PARAMETERS[family], 'exact final parameter count')

    def training(self, train, meta):
        np = self.np
        saved = self.read(self.run / 'fits.json')
        self.equal(saved['train_counts'], meta['counts'], 'all chronological training exposure')
        fits = saved['fits']
        self.require([(r['family'],r['seed']) for r in fits] == [(f,s) for s in SEEDS for f in FAMILIES],
                     'all twelve fixed fits in order')
        progress, work = iter(self.rows(self.run / 'progress.jsonl')), iter(self.rows(self.run / 'work.jsonl'))
        serial, totals = 0, {'forward_chunks':0,'backward_chunks':0,'forward_rows':0,
                            'spo_loss_calls':0,'spo_weighted_rows':0}
        for fit in fits:
            kind, seed = fit['family'], fit['seed']
            self.equal(next(progress), {'event':'fit_start','family':kind,'seed':seed}, 'fit entry')
            rng, digest = np.random.default_rng(seed), hashlib.sha256()
            orders = []
            for _ in range(80):
                order = rng.permutation(54).astype(np.int64)
                digest.update(order.tobytes()); orders.append(order.tolist())
            self.equal(fit['episode_orders'], orders, 'all 80 independently reconstructed episode orders')
            self.require(fit['permutation_sha256'] == digest.hexdigest(), 'exact episode order digest')
            architecture, readout, objective = CELLS[kind]
            batches = self.expected_batches(train, orders, objective)
            times, sums = [], {'forward_chunks':0,'backward_chunks':0,'no_grad_chunks':0,'forward_rows':0,
                              'spo_loss_calls':0,'spo_weighted_rows':0}
            for expected in batches:
                serial += 1
                identity = {'call_id':serial,'family':kind,'seed':seed,'epoch':expected['epoch']-1,
                    'batch':expected['batch'],'episode_indices':expected['episode_indices']}
                self.equal(next(work), {'event':'attempt',**identity}, 'original batch attempt')
                returned = next(work)
                self.require(set(returned) == {'event',*identity,'seconds','result'}, 'exact batch return fields')
                self.equal({k:returned[k] for k in identity}, identity, 'matching original batch return')
                self.require(returned['event'] == 'return' and type(returned['seconds']) in (int,float)
                             and math.isfinite(returned['seconds']) and returned['seconds'] >= 0, 'finite returned batch interval')
                result = returned['result']
                expected_result = {k:expected[k] for k in ('episode_indices','optimizer_step','objective','prior_rows',*sums)}
                for key in ('loss','nonquery_loss','prior_loss','spo_loss','gradient_norm_before_clip'):
                    self.require(type(result[key]) in (int,float) and math.isfinite(result[key]) and result[key] >= 0,
                                 'finite inherited optimization scalar')
                    expected_result[key] = result[key]
                expected_result['all_parameter_gradients_finite'] = True
                self.equal(result, expected_result, 'complete batch geometry and acknowledgment')
                component_sum = result['nonquery_loss'] + result['prior_loss'] + result['spo_loss']
                self.require(abs(result['loss'] - component_sum) <= 2**-22 * component_sum
                             + 2 * expected['backward_chunks'] * 2**-149,
                             'batch component sum within f32-addition roundoff')
                if objective == 'aux':
                    self.require(result['spo_loss'] == 0, 'AUX objective has no trained SPO term')
                for key in sums:
                    sums[key] += result[key]
                times.append(returned['seconds'])
            for epoch in (20,40,60,80):
                self.equal(next(progress), {'event':'epoch','family':kind,'seed':seed,'epoch':epoch,
                    'optimizer_steps':9*epoch,'episode_exposures':54*epoch}, 'fixed epoch coverage')
            self.equal(next(progress), {'event':'fit_complete',**fit}, 'durable final fit')
            name = f'{kind}-{seed}.npz'
            self.require(fit['checkpoint_path'] == name and fit['epochs'] == 80 and fit['steps'] == 720
                         and fit['episode_exposures'] == 4320 and fit['episodes_per_epoch'] == 54
                         and fit['parameter_count'] == PARAMETERS[kind]
                         and fit['architecture'] == architecture and fit['readout'] == readout and fit['objective'] == objective,
                         'complete matched training recipe')
            for key, value in sums.items():
                self.require(fit[key] == value, 'complete per-fit work ' + key)
                if key in totals:
                    totals[key] += value
            for key in ('initial_sha256','initial_core_sha256'):
                self.require(type(fit[key]) is str and len(fit[key]) == 64
                             and all(c in '0123456789abcdef' for c in fit[key]), 'saved initial hash witness')
            self.equal(fit['checkpoint'], self.worker['files'][name], 'published final checkpoint pin')
            self.checkpoint(kind,name)
            initial = fit['initial_tensors']
            keys = {'recurrent.weight_ih_l0','recurrent.weight_hh_l0','recurrent.bias_ih_l0',
                    'recurrent.bias_hh_l0','output.weight','output.bias'}
            if architecture == 'innovation':
                keys.add('correction.weight')
            if readout == 'separate':
                keys.update(('prior_output.weight','prior_output.bias'))
            self.require(set(initial) == keys and all(type(v) is str and len(v) == 64
                         and all(c in '0123456789abcdef' for c in v) for v in initial.values()), 'complete initial tensor hash witnesses')
            for key in ('final_train_loss','final_nonquery_loss','final_prior_loss','final_spo_loss','fit_seconds',
                        'train_rescore_seconds','checkpoint_seconds','wall_seconds'):
                self.require(type(fit[key]) in (int,float) and math.isfinite(fit[key]) and fit[key] >= 0, 'finite fit value ' + key)
            self.require(math.fsum(times) <= fit['fit_seconds'] + 1e-9
                         and math.fsum(fit[k] for k in ('fit_seconds','train_rescore_seconds','checkpoint_seconds'))
                         <= fit['wall_seconds'] + 1e-9, 'physical work intervals within full fit')
            offsets = train['episode_offsets']
            chunks = sum((max(int(offsets[i+1]-offsets[i]) for i in range(first,first+6))+31)//32
                         for first in range(0,54,6))
            self.equal(fit['train_rescore'], {'forward_chunks':chunks,'forward_rows':meta['counts']['rows'],
                'prior_rows':meta['counts']['prior_rows']},
                       'all final TRAIN rows rescored')
            self.training_predictions(fit, train, meta)
        for seed in SEEDS:
            for architecture in ('innovation','innovation_gru'):
                pair = [fit for fit in fits if fit['seed'] == seed and fit['architecture'] == architecture]
                self.require(len(pair) == 2 and {r['objective'] for r in pair} == {'aux','spo'}
                             and all(r['readout'] == 'shared' for r in pair), 'matched shared-head objective pair')
                for key in ('initial_core_sha256','initial_sha256','permutation_sha256'):
                    self.require(pair[0][key] == pair[1][key], 'same paired initialization and order')
                self.equal(pair[0]['initial_tensors'],pair[1]['initial_tensors'],'same initial tensors')
                self.require(all(not k.startswith('prior_output.') for k in pair[0]['initial_tensors']), 'no separate readout')
        self.equal(next(progress), {'event':'all_checkpoints_closed_before_VALID','fits_completed':12,
            'checkpoints':{f['checkpoint_path']:f['checkpoint'] for f in fits}}, 'complete final barrier before VALID')
        self.require(journal_exhausted(progress) and journal_exhausted(work), 'no omitted or extra training events')
        self.require(serial == self.worker['optimizer_steps'] == 8640, 'all planned optimizer updates')
        for key, value in totals.items():
            self.require(self.worker['training_'+key] == value, 'complete worker batch accounting')
        self.require(self.worker['training_rescore_chunks'] == sum(f['train_rescore']['forward_chunks'] for f in fits)
                     and self.worker['training_rescore_rows'] == 12*meta['counts']['rows'], 'complete final TRAIN rescoring')
        self.counts.update(fits=12,optimizer_steps=8640,training_events=73,work_events=17280)
        return fits

    def checked_prior(self, saved, history):
        np = self.np
        prior, mask = saved['prior'], saved['prior_mask']
        self.require(prior.dtype == np.float32 and prior.shape == history['query_scores'].shape
                     and bool(np.isfinite(prior).all()) and mask.dtype == np.bool_
                     and mask.shape == history['prior_mask'].shape
                     and mask.tobytes() == history['prior_mask'].tobytes(),
                     'exact complete finite later-query saved forecasts')
        inactive = prior[~mask]
        self.require(inactive.tobytes() == np.zeros(inactive.shape, np.float32).tobytes(),
                     'unexposed prior rows are exact positive zero')
        return prior

    def training_predictions(self, fit, train, meta):
        np = self.np
        saved = self.arrays(self.run / ('training-prediction-' + fit['checkpoint_path']))
        self.require(set(saved) == {'predictions','prior','prior_mask'}, 'exact final TRAIN forecast fields')
        predictions = saved['predictions']
        self.require(predictions.dtype == np.float32 and predictions.shape == train['targets'].shape
                     and bool(np.isfinite(predictions).all()), 'complete finite final TRAIN forecasts')
        mask = train['query_mask']
        self.require(predictions[mask].tobytes() == train['query_scores'][mask].tobytes(),
                     'TRAIN query outputs remain observed scores')
        prior = self.checked_prior(saved, train)
        nonquery, nq_error = scalar_training_loss(predictions, train['targets'], train['weights'],
                                                  train['legal'], check=self.check)
        auxiliary, prior_error = scalar_training_loss(prior, train['prior_targets'], train['prior_weights'],
                                                       check=self.check)
        spo, spo_error = scalar_spo_loss(predictions,train['targets'],train['weights'],train['legal'],check=self.check)
        for key, expected, error in (('final_nonquery_loss',nonquery,nq_error),
                                     ('final_prior_loss',auxiliary,prior_error),('final_spo_loss',spo,spo_error)):
            self.require(abs(fit[key]-expected) <= error,
                         'independent saved TRAIN loss within derived f32 bound: ' + key)
        objective = fit['objective']
        expected = fit['final_nonquery_loss'] + fit['final_prior_loss'] + (fit['final_spo_loss'] if objective == 'spo' else 0.)
        self.require(fit['final_prior_loss_scope'] == 'trained objective component in every cell'
                     and fit['final_objective_prior_loss'] == fit['final_prior_loss']
                     and fit['final_spo_loss_scope'] == 'post-fit diagnostic for every cell; trained only by spo cells'
                     and fit['final_objective_spo_loss'] == (fit['final_spo_loss'] if objective == 'spo' else 0.)
                     and fit['final_rescore_spo_loss_calls'] == 1,
                     'paid descriptive SPO rescore separated from trained objective')
        self.equal(fit['final_train_loss'], expected, 'fixed final objective component arithmetic')
        identities = [row for row in self.episodes if row['stage'] == 'train']
        history = {**train, **{key: meta[key] for key in ('episode_ids','episode_regimes')}}
        self.train_reports.append({'family': fit['family'], 'seed': fit['seed'],
            'prior_metrics': scalar_prior_metrics(history, prior, identities, check=self.check),
            'loss_check': {'nonquery_float64': nonquery, 'nonquery_roundoff_bound': nq_error,
                          'prior_float64': auxiliary, 'prior_roundoff_bound': prior_error,
                          'spo_float64': spo, 'spo_roundoff_bound': spo_error,
                          'scope': 'saved f32 predictions and f32 loss weights; no Torch or model replay'}})
        self.counts['training_prediction_files'] = self.counts.get('training_prediction_files',0)+1

    def predictions(self, windows, meta, identities, fits, history):
        np = self.np
        models, hold = [], None
        for kind, seed in [('hold',None), *[(row['family'],row['seed']) for row in fits]]:
            filename = 'prediction-hold.npz' if kind == 'hold' else f'prediction-{kind}-{seed}.npz'
            saved = self.arrays(self.run / filename)
            self.require(set(saved) == ({'predictions'} if kind == 'hold' else {'predictions','prior','prior_mask'}),
                         'complete saved final prediction fields')
            predictions = saved['predictions']
            self.require(predictions.dtype == np.float32 and predictions.shape == windows['targets'].shape
                         and bool(np.isfinite(predictions).all()), 'complete finite final VALID forecasts')
            self.require(predictions[:,0].tobytes() == windows['query_scores'].tobytes(),
                         'exact observed query outputs are not prequery forecasts')
            padding = predictions[~windows['valid_mask']]
            self.require(padding.tobytes() == np.zeros(padding.shape,np.float32).tobytes(), 'exact zero final padding')
            metrics = self.metrics(predictions,windows,meta,identities)
            if kind == 'hold':
                expected = np.repeat(windows['query_scores'][:,None,:],4,axis=1)
                expected[~windows['valid_mask']] = 0
                self.require(predictions.tobytes() == expected.tobytes(), 'exact whole hold-score baseline')
                hold = metrics
            else:
                prior = self.checked_prior(saved,history)
                models.append({'family':kind,'architecture':CELLS[kind][0],'readout':CELLS[kind][1],'objective':CELLS[kind][2],
                    'seed':seed,'metrics':metrics,
                    'prior_metrics':scalar_prior_metrics(history,prior,identities,check=self.check)})
            self.counts['predictions'] += 1
            self.counts['forecast_metric_rows'] += meta['counts']['nonquery_rows']
        return models,hold

    def body(self):
        self.authenticate()
        import numpy as np
        self.np = np
        runtime = self.read(self.run / 'runtime.json')
        self.require(runtime['threads'] == runtime['interop_threads'] == 1 and runtime['deterministic'] is True
                     and runtime['cuda_used'] is runtime['mps_used'] is False, 'qualified deterministic CPU runtime')
        train, train_meta, _ = self.reconstruct('train')
        self.saved_windows('training-history',train,train_meta)
        self.train_reports = []
        fits = self.training(train,train_meta)
        del train
        windows, meta, identities = self.reconstruct('valid')
        self.saved_windows('validation-windows',windows,meta)
        history, history_meta = self.validation_history(windows,meta)
        history.update({key: meta[key] for key in ('episode_ids','episode_regimes')})
        models, hold = self.predictions(windows,meta,identities,fits,history)
        lengths = windows['episode_lengths']
        chunks = sum((max(int(v) for v in lengths[first:first+6])+31)//32 for first in range(0,36,6))
        prediction_work = {'forward_chunks':chunks,'forward_rows':meta['counts']['rows'],
                           'prior_rows':history_meta['counts']['prior_rows']}
        for row in models:
            row['prediction_work'] = dict(prediction_work)
        self.require(self.worker['validation_forward_chunks'] == 12*chunks
                     and self.worker['validation_forward_rows'] == 12*meta['counts']['rows'], 'complete census inference accounting')
        rules = continuation_rules(models,hold,technical_complete=False)
        expected = {'version':PRODUCER_VERSION,'models':models,'hold':hold,'required':rules,
            'required_passed':sum(r['passes'] for r in rules),'required_conditions':41,
            'gates':gate_decisions(rules),
            'technical_complete_pending_saved_audit':True,'scientific_conditions':40,
            'scientific_passed':sum(r['passes'] for r in rules[1:]),
            'requires_successful_original_supervisor_and_saved_audit':True,
            'train_counts':train_meta['counts'],'validation_counts':meta['counts'],
            'validation_history_counts':history_meta['counts'],
            'capacity_evidence_scope':CAPACITY_SCOPE,
            'scope':'Forced-path forecasts only, with separate objective and architecture gates. No omnibus all-41 gate, autonomous performance, new-shift evidence or true action regret.'}
        summary = self.read(self.run / 'summary.json')
        for key in ('setup_seconds','fitting_seconds','validation_seconds'):
            self.require(type(summary[key]) in (int,float) and math.isfinite(summary[key]) and summary[key] >= 0,
                         'finite original physical stage')
            expected[key] = summary[key]
        self.require(math.fsum(f['wall_seconds'] for f in fits) <= summary['fitting_seconds']+1e-9
                     and math.fsum(summary[k] for k in ('setup_seconds','fitting_seconds','validation_seconds'))
                     <= self.worker['wall_seconds']+1e-9, 'whole-phase physical cost containment')
        self.equal(summary,expected,'independent provisional producer summary and all 40 scientific records')
        payloads = {'started.json','runtime.json','progress.jsonl','work.jsonl','fits.json','summary.json',
            'training-history.npz','training-history.json','validation-history.npz','validation-history.json',
            'validation-windows.npz','validation-windows.json','prediction-hold.npz'}
        for fit in fits:
            payloads.update((fit['checkpoint_path'],'prediction-'+fit['checkpoint_path'],
                             'training-prediction-'+fit['checkpoint_path']))
        self.require(len(payloads) == 49 and set(self.worker['files']) == payloads, 'exact final 49-payload closure')
        for directory, worker in ((self.run,self.worker),(self.collection,self.collection_worker)):
            for name,pin in worker['files'].items():
                self.pinned(directory/name,pin)
        for descriptor in self.plan['inputs'].values():
            self.pinned(self.path(descriptor['path']),descriptor)
        final_rules = continuation_rules(models,hold,technical_complete=True)
        final = {**expected,'required':final_rules,'required_passed':sum(r['passes'] for r in final_rules),
            'gates':gate_decisions(final_rules),'technical_complete_pending_saved_audit':False}
        self.counts.update(required_conditions=41,required_passed=final['required_passed'],collection_episodes=90,
            prediction_files=25,source_files=len(self.plan['sources']),training_payloads=49,collection_payloads=18)
        self.result = {'version':VERSION,'agreement':True,'producer_summary':expected,'summary':final,
            'fits':fits,'training_forecast_checks':self.train_reports,'counts':dict(self.counts),'limitations':LIMITATIONS,
            'technical_condition_requires_successful_original_audit_supervisor':True}


base.Audit = Audit

if __name__ == '__main__':
    base.main()
