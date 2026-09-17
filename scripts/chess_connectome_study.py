"""Frozen biological-wiring comparison on a shared pretrained chess backbone.

All final adapters are fitted before neural evaluation. Graph assets and model
derivatives stay in local runs/. Reports audit saved outputs without inference.
"""
import argparse
import dataclasses
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import torch

from openjev.research import chess_connectome_data as data
from openjev.research import chess_connectome_study as study
from openjev.research.chess_candidate import CandidateChess
from openjev.research.chess_candidate_eval import CachedPositions, audit_cache_metadata
from openjev.research.connectome_graph import (
    ASSET_SHA256,
    META_SHA256,
    graph_sha256,
    induced_subgraph,
    load_pinned_graph,
    rewire_signed_degrees,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'runs/chess-inputs/stockfish/stockfish-macos-universal'
ASSETS = 'runs/chessfly-source-audit/assets'
GRAPH_AUDIT = 'evidence/connectome-graph-v1/descending-controls.json'
PREFLIGHT = 'evidence/connectome-adapter-preflight-v1'
PREFLIGHT_SHA = 'dca5986c6e5661884acbec92f6dc35b6224454ba8ae2c42c066221544e98d5b0'
CANDIDATE_PLAN = 'evidence/chess-candidate-v2/protocol/plan.json'
CANDIDATE_PLAN_SHA = '0cdcf5b9f9e64b5dde6f0a951fa4249475d050470e9036ce5c8ab76cf181c106'
NEW_SOURCES = [
    'scripts/chess_connectome_study.py', 'tests/test_chess_connectome_runner.py',
    'src/openjev/research/chess_connectome_study.py', 'tests/test_chess_connectome_study.py',
    'src/openjev/research/chess_connectome_data.py', 'tests/test_chess_connectome_data.py',
    'src/openjev/research/chess_connectome_adapter.py', 'tests/test_chess_connectome_adapter.py',
    'src/openjev/research/connectome_graph.py', 'tests/test_connectome_graph.py',
    'src/openjev/research/chessbench_data.py', 'tests/test_chessbench_data.py',
]
PROTOCOL = {
    'version': 'chess-connectome-v1', 'paired_seeds': [97, 109, 127],
    'variants': ['biological', 'rewire151', 'rewire163', 'rewire179', 'dense', 'node_local'],
    'fit_order_seed': 11500017, 'batch_order_seed_base': 11500029,
    'torch_threads': 2, 'deterministic_algorithms': False,
    'dtype': 'float32', 'all_final_fits_before_neural_evaluation': True,
    'regret_positions_per_split': 128, 'regret_selection_seed': 11500043,
    'regret_nodes': 20000, 'regret_call_ceiling': 5632, 'regret_node_ceiling': 112640000,
    'latency_positions_per_split': 64, 'latency_selection_seed': 11500059, 'latency_warmups': 3,
    'relative_loss_reduction': .10, 'gate_absolute_tolerance': 1e-12,
    'gate': 'Biological mean signed bounded engine-score loss at least10% lower than EACH of three rewires on BOTH panels, all reference means positive. Biological strictly lower in EVERY paired seed than each rewire, and no worse than its own unchanged direct baseline in EVERY paired seed on both panels. Absolute arithmetic tolerance1e-12, incomplete/nonfinite evidence cannot pass.',
    'graph': 'All1409 descending nodes and44090 signed edges; fixed original identities, degree normalization, no anatomical-count initialization. Three audited degree-preserving rewires151/163/179.',
    'interface': 'Frozen complete direct backbone width32/rootdepth4. Paired seeded balanced1409node-to1024slot mapping;16channels per square; four scalar leaky graph steps alpha0.5; zero-initialized residual output projection; fresh states per decision.',
    'matching': 'All eighteen fits use original32768 training positions, labels and paired minibatch order, sixepochs,1536updates. Primary biological-vs-rewire topology comparison matches parameters and operations. Dense/local parameter counts differ. Every mode uses dense matmuls.',
    'pretraining': 'All three direct candidate-v2 final backbones are reused. Original backbone pretraining and earlier failed attempts remain part of the cost/provenance; no low-data sample-efficiency claim.',
    'arena': 'Stage1 has no arena or Elo estimate. Only a passed topology screen can justify a separately frozen arena replication; no game-strength claim from this stage.',
    'selection': 'All fixed seeds and final fits, no best epoch, retries, replacements, resume or budget extension. A failed attempt is retained.',
    'uncertainty': 'Descriptive equal-seed means and paired seed differences on two development panels; no independent-position or seed-population confidence interval.',
    'scope': 'Wiring-prior mechanism screen, not an intact fly circuit, learned world model, search policy, calibrated confidence or established novelty. Historical generators have been studied.',
    'asset_policy': 'Graph arrays, rewires and derived checkpoints remain local in runs/; no MIT redistribution of external biological assets or derivatives.',
    'timing': 'Full CPU choose call including native board/menu construction. Three starting-board warmups separately charged; fixed64positions per split. No concurrent planned heavy study during timing. Shared interactive host, not isolated latency; record load averages before and after each measurement.',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tree(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Nonregular evidence entry')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'machine': platform.machine(), 'mps_available': torch.backends.mps.is_available(),
            'dependencies': {n: importlib.metadata.version(n) for n in ('torch', 'chess', 'numpy')},
            'lock_sha256': sha(ROOT/'uv.lock'), 'pyproject_sha256': sha(ROOT/'pyproject.toml')}


def configurations():
    result = study.configurations()
    random.Random(PROTOCOL['fit_order_seed']).shuffle(result)
    return result


def baselines():
    return study.baseline_configurations()


def name(config):
    return f"{config['variant']}-{config['seed']}"


def binding():
    if sha(ROOT/CANDIDATE_PLAN) != CANDIDATE_PLAN_SHA:
        raise ValueError('Original backbone plan changed')
    candidate = read(ROOT/CANDIDATE_PLAN)
    sources = dict(candidate['sources'])
    for path, digest in sources.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Frozen prerequisite source changed: {path}')
    sources.update({p: sha(ROOT/p) for p in NEW_SOURCES})
    if sha(ROOT/PREFLIGHT/'summary.json') != PREFLIGHT_SHA:
        raise ValueError('Preflight summary changed')
    preflight = read(ROOT/PREFLIGHT/'protocol.json')
    for p, digest in preflight['source_sha256'].items():
        if sha(ROOT/p) != digest:
            raise ValueError('Preflight source changed')
    summary = read(ROOT/PREFLIGHT/'summary.json')
    if summary['status'] != 'completed' or summary['protocol_sha256'] != sha(ROOT/PREFLIGHT/'protocol.json'):
        raise ValueError('Preflight is not complete')
    for case in summary['cases']:
        if case['status'] != 'completed' or sha(ROOT/PREFLIGHT/case['receipt']) != case['receipt_sha256']:
            raise ValueError('Preflight case changed')
    graph_audit = read(ROOT/GRAPH_AUDIT)
    if graph_audit['status'] != 'completed':
        raise ValueError('Graph controls incomplete')
    for p, digest in graph_audit['sources'].items():
        if sha(ROOT/p) != digest:
            raise ValueError('Graph control source changed')
    assets = {**ASSET_SHA256, 'meta.json': META_SHA256}
    if any(sha(ROOT/ASSETS/p) != h for p, h in assets.items()):
        raise ValueError('Biological source bytes changed')
    transfer = read(ROOT/data.TRANSFER_PLAN)
    if sha(ROOT/data.TRANSFER_PLAN) != data.TRANSFER_PLAN_SHA:
        raise ValueError('Transfer plan changed')
    manifest_path = ROOT/'evidence/chess-candidate-v2/results/manifest.json'
    if sha(manifest_path) != transfer['publication_manifest_sha256']:
        raise ValueError('Published pretraining manifest changed')
    manifest = read(manifest_path)
    backbones = {str(c['seed']): c for c in transfer['configurations'] if c['arm'] == 'direct'}
    if set(backbones) != {str(s) for s in PROTOCOL['paired_seeds']}:
        raise ValueError('Missing backbone seed')
    for config in backbones.values():
        if sha(ROOT/config['path']) != config['sha256']:
            raise ValueError('Published backbone changed')
    pretraining_costs = {}
    for seed in PROTOCOL['paired_seeds']:
        member = f'execution/direct-{seed}/training.json'
        receipt_path = ROOT/'runs/chess-candidate-v2'/member
        digest = sha(receipt_path)
        if digest != manifest['members'][member]['sha256']:
            raise ValueError('Backbone training differs from published evidence')
        receipt = read(receipt_path)
        pretraining_costs[str(seed)] = {'receipt': receipt_path.relative_to(ROOT).as_posix(),
            'sha256': digest, **{k: receipt[k] for k in ('updates', 'examples_seen', 'training_seconds', 'computation')}}
    return {'sources': dict(sorted(sources.items())), 'environment': environment(),
            'backbones': backbones, 'backbone_plan_sha256': CANDIDATE_PLAN_SHA,
            'backbone_pretraining_protocol': candidate['protocol'],
            'backbone_pretraining_costs': pretraining_costs,
            'backbone_execution_receipt_sha256': sha(ROOT/'runs/chess-candidate-v2/execution/completed.json'),
            'preflight_files': tree(ROOT/PREFLIGHT), 'graph_audit_sha256': sha(ROOT/GRAPH_AUDIT),
            'graph_hashes': {'biological': graph_audit['source_graph_sha256'],
                **{f"rewire{c['seed']}": c['graph_sha256'] for c in graph_audit['controls']}},
            'asset_files': {f'{ASSETS}/{p}': h for p, h in assets.items()},
            'engine': ENGINE, 'engine_sha256': sha(ROOT/ENGINE),
            'training_source': data.TRAIN, 'training_source_sha256': sha(ROOT/data.TRAIN)}


def graphs(expected):
    full = load_pinned_graph(ROOT/ASSETS)
    graph = induced_subgraph(full, full.node_ids[full.groups == 3])
    del full
    result = {'biological': graph}
    for seed in (151, 163, 179):
        result[f'rewire{seed}'] = rewire_signed_degrees(
            graph, seed=seed, accepted_swaps=440900, max_attempts=44090000).graph
    if {key: graph_sha256(value) for key, value in result.items()} != expected:
        raise ValueError('Reconstructed graph differs from independently audited graph')
    result['dense'] = result['node_local'] = graph
    return result


def load_backbone(plan, seed):
    item = plan['binding']['backbones'][str(seed)]
    if sha(ROOT/item['path']) != item['sha256']:
        raise ValueError('Backbone file changed')
    return CandidateChess.load(ROOT/item['path'], expected_plan_sha256=CANDIDATE_PLAN_SHA,
        expected_arm='direct', expected_seed=seed, expected_width=32,
        expected_root_depth=4, expected_branch_depth=2)


def prepare(out):
    if Path(out).exists():
        raise FileExistsError(out)
    frozen = binding()
    # Only graph construction and old-board exclusion work precedes this freeze.
    graphs(frozen['graph_hashes'])
    excluded = data.collect_exclusions(ROOT)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(excluded['states'], separators=(',', ':'))+'\n').encode()
    with (out/'excluded-states.json.gz').open('xb') as stream:
        stream.write(gzip.compress(raw, mtime=0))
    plan = {'protocol': PROTOCOL, 'training_settings': dataclasses.asdict(study.TrainingSettings()),
            'data_config': data.CONFIG, 'configurations': configurations(), 'baselines': baselines(),
            'binding': frozen, 'exclusion_files': excluded['files'], 'exposure': excluded['counts'],
            'exclusion_scope': excluded['scope'], 'exclusion_file': 'excluded-states.json.gz',
            'exclusion_sha256': sha(out/'excluded-states.json.gz'),
            'exclusion_uncompressed_sha256': hashlib.sha256(raw).hexdigest(),
            'new_panel_generation_or_training_started': False}
    write(out/'plan.json', plan)
    write(out/'prepared.json', {'status': 'prepared', 'plan_sha256': sha(out/'plan.json'),
                              'created_unix': time.time(), 'neural_evaluation_started': False})
    return out/'plan.json'


def verify(plan_path):
    plan = read(plan_path)
    prepared = read(Path(plan_path).parent/'prepared.json')
    if (prepared['status'] != 'prepared' or prepared['plan_sha256'] != sha(plan_path)
            or prepared['neural_evaluation_started'] is not False):
        raise ValueError('Prepared plan identity changed')
    if (plan['protocol'] != PROTOCOL or plan['data_config'] != data.CONFIG
            or plan['training_settings'] != dataclasses.asdict(study.TrainingSettings())
            or plan['configurations'] != configurations() or plan['baselines'] != baselines()
            or plan['binding'] != binding() or plan['new_panel_generation_or_training_started'] is not False):
        raise ValueError('Frozen protocol, code, environment or prerequisites changed')
    for path, digest in plan['exclusion_files'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Exposure source changed')
    snapshot = Path(plan_path).parent/'excluded-states.json.gz'
    if plan['exclusion_file'] != snapshot.name or sha(snapshot) != plan['exclusion_sha256']:
        raise ValueError('Exclusion snapshot changed')
    raw = gzip.decompress(snapshot.read_bytes())
    if hashlib.sha256(raw).hexdigest() != plan['exclusion_uncompressed_sha256']:
        raise ValueError('Exclusion contents changed')
    excluded = json.loads(raw)
    if excluded != sorted(set(excluded)) or len(excluded) != plan['exposure']['total_natural_states']:
        raise ValueError('Exclusion membership inconsistent')
    return plan, excluded


def panels(eval_rows):
    secondary, latency, combined = {}, [], []
    for offset, split in enumerate(('dev', 'shift')):
        n = len(eval_rows[split])
        secondary[split] = sorted(random.Random(PROTOCOL['regret_selection_seed']+offset).sample(
            range(n), PROTOCOL['regret_positions_per_split']))
        selected = sorted(random.Random(PROTOCOL['latency_selection_seed']+offset).sample(
            range(n), PROTOCOL['latency_positions_per_split']))
        latency.extend(len(combined)+i for i in selected)
        combined.extend(eval_rows[split])
    return {'secondary_indices': secondary, 'latency_indices': latency}, combined


def completed(directory, plan_hash=None):
    actual = tree(directory)
    receipt = read(Path(directory)/'completed.json')
    actual.pop('completed.json')
    if (receipt['status'] != 'completed' or receipt['files'] != actual
            or 'failed.json' in actual or plan_hash is not None and receipt['plan_sha256'] != plan_hash):
        raise ValueError('Incomplete or changed output directory')
    return receipt


def artifact_membership(execution, plan):
    """Reject even correctly rehashed extra fits, checkpoints or predictions."""
    execution = Path(execution)
    required = {'data', 'fits', 'evaluation', 'regret', 'started.json', 'training-cache.json',
                'panels.json', 'all-fits-completed.json', 'evaluation-cache.json', 'completed.json'}
    if {p.name for p in execution.iterdir()} != required:
        raise ValueError('Execution artifact membership differs')
    expected_fits = {name(c) for c in plan['configurations']}
    expected_evaluation = {name(c) for c in plan['baselines']+plan['configurations']}
    for directory, names in ((execution/'fits', expected_fits), (execution/'evaluation', expected_evaluation)):
        if {p.name for p in directory.iterdir()} != names or any(not p.is_dir() for p in directory.iterdir()):
            raise ValueError('Fit/evaluation model membership differs')
    for identity in expected_evaluation:
        if {p.name for p in (execution/'evaluation'/identity).iterdir()} != {
                'dev.jsonl', 'shift.jsonl', 'latency.json', 'completed.json'}:
            raise ValueError('Evaluation artifact membership differs')


def private_output(out):
    out = Path(out).resolve()
    if not out.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Graph-derived artifacts must remain in local runs/')
    return out


def run(plan_path, out):
    out = private_output(out)
    if out.exists():
        raise FileExistsError(out)
    plan, excluded = verify(plan_path)
    plan_hash = sha(plan_path)
    out.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    write(out/'started.json', {'status': 'started', 'plan_sha256': plan_hash, 'started_unix': time.time()})
    try:
        torch.set_num_threads(PROTOCOL['torch_threads'])
        torch.use_deterministic_algorithms(PROTOCOL['deterministic_algorithms'])
        topology = graphs(plan['binding']['graph_hashes'])
        data.generate(out/'data', ROOT/ENGINE, excluded)
        eval_rows = data.validate(out/'data', excluded)
        del excluded
        training = data.training_rows(ROOT)
        cache = CachedPositions(training, include_successors=False)
        write(out/'training-cache.json', cache.metadata)
        selection, combined = panels(eval_rows)
        write(out/'panels.json', selection)
        (out/'fits').mkdir()
        fit_records = []
        for config in plan['configurations']:
            identity = name(config)
            backbone = load_backbone(plan, config['seed'])
            print(json.dumps({'fit_started': identity, 'unix': time.time()}), flush=True)
            study.fit(config, backbone, topology[config['variant']], cache, out/'fits'/identity,
                plan_sha256=plan_hash, data_sha256=plan['binding']['training_source_sha256'])
            fit_records.append({'configuration': config,
                'training_sha256': sha(out/'fits'/identity/'training.json'),
                'weights_sha256': sha(out/'fits'/identity/'weights.pt')})
            print(json.dumps({'fit_completed': identity, 'completed': len(fit_records)}), flush=True)
            del backbone
            gc.collect()
            torch.mps.empty_cache()
        write(out/'all-fits-completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'fits': fit_records, 'completed_unix': time.time(), 'evaluation_started': False})
        del cache, training
        gc.collect()
        prepared = {split: CachedPositions(records, include_successors=False) for split, records in eval_rows.items()}
        write(out/'evaluation-cache.json', {key: cache.metadata for key, cache in prepared.items()})
        (out/'evaluation').mkdir()
        decisions = []
        for config in plan['baselines']+plan['configurations']:
            identity = name(config)
            backbone = load_backbone(plan, config['seed'])
            if config['variant'] == 'direct':
                model = backbone
            else:
                checkpoint = out/'fits'/identity/'weights.pt'
                record = next(r for r in fit_records if r['configuration'] == config)
                model = study.load_checkpoint(checkpoint, backbone, topology[config['variant']],
                    expected_plan_sha256=plan_hash, expected_config=config,
                    expected_backbone_sha256=study.state_sha256(backbone),
                    expected_checkpoint_sha256=record['weights_sha256'])
            directory = out/'evaluation'/identity
            directory.mkdir()
            measured = {}
            for split, cache in prepared.items():
                measured[split] = study.evaluate(model, cache, directory/f'{split}.jsonl',
                                                configuration=config, plan_sha256=plan_hash)
                decisions.extend({'configuration': identity, 'split': split, 'panel_index': i,
                    'id': p['id'], 'choice': p['choice']} for i, p in enumerate(rows(directory/f'{split}.jsonl')))
            host_before = list(os.getloadavg())
            write(directory/'latency.json', study.measure_latency(model, combined, selection['latency_indices'],
                                                                 configuration=config, plan_sha256=plan_hash,
                                                                 warmups=PROTOCOL['latency_warmups']))
            write(directory/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
                'configuration': config, 'evaluation': measured,
                'host_load_average_before_latency': host_before,
                'host_load_average_after_latency': list(os.getloadavg()), 'files': tree(directory)})
            print(json.dumps({'evaluated': identity, 'metrics': {s: r['metrics'] for s, r in measured.items()}}), flush=True)
            del model, backbone
        regret = out/'regret'
        grading_plan = {'engine_path': str((ROOT/ENGINE).resolve()), 'panels': eval_rows,
            'secondary_indices': selection['secondary_indices'],
            'configurations': [{'id': name(c)} for c in plan['baselines']+plan['configurations']]}
        graded = study.score_stronger(grading_plan, regret, decisions, plan_sha256=plan_hash,
            expected_engine_sha256=plan['binding']['engine_sha256'],
            call_ceiling=PROTOCOL['regret_call_ceiling'], requested_node_ceiling=PROTOCOL['regret_node_ceiling'])
        cost = graded['cost']
        if cost['calls'] > PROTOCOL['regret_call_ceiling'] or cost['requested_nodes'] > PROTOCOL['regret_node_ceiling']:
            raise ValueError('Stronger-grading budget exceeded')
        write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash,
            'wall_seconds': time.perf_counter()-begun, 'files': tree(out)})
    except Exception as error:
        write(out/'failed.json', {'status': 'failed', 'plan_sha256': plan_hash, 'error': str(error),
                                 'wall_seconds': time.perf_counter()-begun})
        raise
    return out/'completed.json'


def continuation(losses):
    """Exact preregistered all-control/all-panel/all-seed rule, no selection."""
    identities = {name(c) for c in baselines()+configurations()}
    if set(losses) != identities or any(set(splits) != {'dev', 'shift'} for splits in losses.values()):
        raise ValueError('Incomplete topology comparison')
    if any(type(v) not in (int, float) or not math.isfinite(v)
           for splits in losses.values() for v in splits.values()):
        raise ValueError('Nonfinite or invalid losses')
    seeds, tol = PROTOCOL['paired_seeds'], PROTOCOL['gate_absolute_tolerance']
    means = {v: {s: math.fsum(losses[f'{v}-{seed}'][s] for seed in seeds)/len(seeds)
                 for s in ('dev', 'shift')} for v in ['direct']+PROTOCOL['variants']}
    checks = []
    for split in ('dev', 'shift'):
        biological = means['biological'][split]
        for comparator in ('rewire151', 'rewire163', 'rewire179'):
            reference = means[comparator][split]
            reduction = (reference-biological)/reference if reference > 0 else None
            checks.append({'split': split, 'comparator': comparator, 'criterion': 'mean_relative_reduction',
                'reduction': reduction, 'passed': reduction is not None and reduction+tol >= PROTOCOL['relative_loss_reduction']})
            for seed in seeds:
                difference = losses[f'biological-{seed}'][split]-losses[f'{comparator}-{seed}'][split]
                checks.append({'split': split, 'comparator': comparator, 'seed': seed,
                    'criterion': 'strict_paired_improvement', 'difference': difference, 'passed': difference < -tol})
        for seed in seeds:
            difference = losses[f'biological-{seed}'][split]-losses[f'direct-{seed}'][split]
            checks.append({'split': split, 'comparator': 'direct', 'seed': seed,
                'criterion': 'no_paired_degradation', 'difference': difference, 'passed': difference <= tol})
    return {'means': means, 'checks': checks, 'continuation_passed': all(c['passed'] for c in checks)}


def report(plan_path, execution, out):
    plan, excluded = verify(plan_path)
    plan_hash, execution, out = sha(plan_path), Path(execution), Path(out)
    finished = completed(execution, plan_hash)
    artifact_membership(execution, plan)
    started = read(execution/'started.json')
    if (started['status'] != 'started' or started['plan_sha256'] != plan_hash
            or type(started['started_unix']) not in (int, float) or not math.isfinite(started['started_unix'])
            or type(finished['wall_seconds']) not in (int, float)
            or not math.isfinite(finished['wall_seconds']) or finished['wall_seconds'] < 0):
        raise ValueError('Execution start or timing receipt differs')
    eval_rows = data.validate(execution/'data', excluded)
    del excluded
    selection, combined = panels(eval_rows)
    if read(execution/'panels.json') != selection:
        raise ValueError('Fixed evaluation selection changed')
    training = data.training_rows(ROOT)
    train_cache = read(execution/'training-cache.json')
    audit_cache_metadata(train_cache, training, include_successors=False)
    evaluation_cache = read(execution/'evaluation-cache.json')
    for split in ('dev', 'shift'):
        audit_cache_metadata(evaluation_cache[split], eval_rows[split], include_successors=False)
    topology = graphs(plan['binding']['graph_hashes'])
    fit_boundary = read(execution/'all-fits-completed.json')
    if (fit_boundary['status'] != 'completed' or fit_boundary['plan_sha256'] != plan_hash
            or fit_boundary['evaluation_started'] is not False
            or type(fit_boundary['completed_unix']) not in (int, float)
            or not math.isfinite(fit_boundary['completed_unix'])
            or fit_boundary['completed_unix'] < started['started_unix']
            or [v['configuration'] for v in fit_boundary['fits']] != plan['configurations']):
        raise ValueError('All-final-fits boundary differs')
    fit_costs, metrics, latency, decisions = {}, {}, {}, []
    for config in plan['configurations']:
        identity = name(config)
        directory = execution/'fits'/identity
        record = next(r for r in fit_boundary['fits'] if r['configuration'] == config)
        if record['training_sha256'] != sha(directory/'training.json') or record['weights_sha256'] != sha(directory/'weights.pt'):
            raise ValueError('Fit changed after final-fit boundary')
        backbone = load_backbone(plan, config['seed'])
        fit_costs[identity] = study.audit_training(directory, config, plan_hash,
            plan['binding']['training_source_sha256'], train_cache, backbone, topology[config['variant']],
            training_rows=training)
    for config in plan['baselines']+plan['configurations']:
        identity = name(config)
        directory = execution/'evaluation'/identity
        receipt = completed(directory, plan_hash)
        if receipt['configuration'] != config:
            raise ValueError('Evaluation arm identity changed')
        backbone = load_backbone(plan, config['seed'])
        if config['variant'] == 'direct':
            model = backbone
        else:
            checkpoint = execution/'fits'/identity/'weights.pt'
            record = next(r for r in fit_boundary['fits'] if r['configuration'] == config)
            model = study.load_checkpoint(checkpoint, backbone, topology[config['variant']],
                expected_plan_sha256=plan_hash, expected_config=config,
                expected_backbone_sha256=study.state_sha256(backbone),
                expected_checkpoint_sha256=record['weights_sha256'])
        metrics[identity] = {}
        for split in ('dev', 'shift'):
            result, predictions = study.audit_evaluation(receipt['evaluation'][split],
                directory/f'{split}.jsonl', eval_rows[split], configuration=config,
                plan_sha256=plan_hash, model=model, cache_metadata=evaluation_cache[split])
            metrics[identity][split] = result
            decisions.extend({'configuration': identity, 'split': split, 'panel_index': i,
                'id': p['id'], 'choice': p['choice']} for i, p in enumerate(predictions))
        measurement = read(directory/'latency.json')
        study.audit_latency(measurement, combined, selection['latency_indices'],
            configuration=config, plan_sha256=plan_hash, expected_state_sha256=study.state_sha256(model),
            warmups=PROTOCOL['latency_warmups'], model=model)
        for key in ('host_load_average_before_latency', 'host_load_average_after_latency'):
            if (type(receipt[key]) is not list or len(receipt[key]) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in receipt[key])):
                raise ValueError('Invalid host-load observation')
        latency[identity] = {'mean_ms': measurement['total_wall_ms']/len(measurement['records']),
            'total_wall_ms': measurement['total_wall_ms'], 'warmup_wall_ms': measurement['warmup_wall_ms'],
            **{k: receipt[k] for k in ('host_load_average_before_latency', 'host_load_average_after_latency')}}
    grading_plan = {'engine_path': str((ROOT/ENGINE).resolve()), 'panels': eval_rows,
        'secondary_indices': selection['secondary_indices'],
        'configurations': [{'id': name(c)} for c in plan['baselines']+plan['configurations']]}
    graded = study.audit_stronger(grading_plan, execution/'regret', decisions,
        expected_plan_sha256=plan_hash, expected_engine_sha256=plan['binding']['engine_sha256'],
        call_ceiling=PROTOCOL['regret_call_ceiling'], requested_node_ceiling=PROTOCOL['regret_node_ceiling'])
    cost = graded['cost']
    values = rows(execution/'regret/regret.jsonl')
    losses = {name(c): {split: math.fsum(r['bounded_regret'] for r in values
        if r['configuration'] == name(c) and r['split'] == split)/PROTOCOL['regret_positions_per_split']
        for split in ('dev', 'shift')} for c in plan['baselines']+plan['configurations']}
    summary = {'status': 'completed', 'plan_sha256': plan_hash,
        'execution_receipt_sha256': sha(execution/'completed.json'),
        'metrics': metrics, 'engine_loss': losses, 'topology_comparison': continuation(losses),
        'latency': latency, 'training_costs': fit_costs, 'grading_costs': cost,
        'backbone_pretraining_costs': plan['binding']['backbone_pretraining_costs'],
        'data_costs': read(execution/'data/completed.json'), 'wall_seconds': finished['wall_seconds'],
        'limits': [PROTOCOL[k] for k in ('matching', 'pretraining', 'arena', 'uncertainty', 'scope', 'asset_policy', 'timing')],
        'report_new_model_or_engine_calls': 0}
    out.mkdir(parents=True, exist_ok=False)
    write(out/'summary.json', summary)
    write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash, 'files': tree(out)})
    return out/'summary.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'run', 'report'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.out)
    elif args.command == 'verify':
        plan, exclusions = verify(args.plan)
        result = {'verified': True, 'excluded_states': len(exclusions), 'fits': len(plan['configurations'])}
    elif args.command == 'run':
        result = run(args.plan, args.out)
    else:
        result = report(args.plan, args.execution, args.out)
    print(result)
