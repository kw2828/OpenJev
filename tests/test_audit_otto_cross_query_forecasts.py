"""Independent fabricated sampled-window reconstruction, no scientific calls."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_cross_query_audit", ROOT / "scripts/audit_otto_cross_query_forecasts.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(tmp_path):
    obj = audit.Audit(SimpleNamespace(output=tmp_path))
    obj.np = np
    obj.collection = tmp_path
    obj.run = tmp_path
    obj.check = lambda: None
    lengths = [37] + [1] * 53
    offsets = np.asarray([0, *np.cumsum(lengths)], np.int64)
    n = int(offsets[-1])
    x = np.zeros((n, 31), np.float32)
    correction = np.zeros(n, np.bool_)
    records, episodes = [], []
    for index, (identity, length) in enumerate(zip(audit.cohort()[:54], lengths, strict=True)):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        steps = np.arange(length)
        x[lo:hi, 15] = steps / 2188
        x[lo:hi, 16] = steps % 4 / 2188
        x[lo:hi, 17] = 1
        correction[lo:hi] = steps % 4 == 0
        w, seed = (length + 3) // 4, audit.SELECTION_START + index
        k = min(w, 8)
        starts = sorted(int(v) * 4 for v in np.random.Generator(np.random.PCG64(seed)).choice(w, k, replace=False))
        records.append({"episode_id": identity["episode_id"], "version": "otto-sampled-forecast-data-v1",
            "length": length, "seed": seed, "period": 4, "population_windows": w, "selected_windows": k,
            "start_offsets": starts, "inclusion_numerator": k, "inclusion_denominator": w,
            "inclusion_probability": k / w, "algorithm": "numpy.Generator(PCG64(seed)).choice(W,size=k,replace=False); sorted"})
        episodes.append({**identity, "start_row": lo, "end_row": hi, "steps": length, "teacher_calls": length})
    flat = {"features": x, "raw_q": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
            "legal": np.ones((n, 4), np.bool_), "actions": np.zeros(n, np.int64),
            "correction": correction, "episode_offsets": offsets, "label_mask": np.ones(n, np.bool_)}
    obj.episodes = episodes
    obj.arrays = lambda path: flat
    obj.read = lambda path: records
    return obj, flat, records


def test_all_query_inputs_but_only_selected_nonquery_targets(tmp_path):
    obj, flat, records = fixture(tmp_path)
    history, meta, _ = obj.reconstruct('train')
    assert meta['counts'] == {'episodes': 54, 'rows': 90, 'query_rows': 63,
        'selected_nonquery_rows': sum(min(3, 36-s) for s in records[0]['start_offsets']),
        'selected_windows': 61, 'zero_support_episodes': 53}
    assert history['query_scores'][::1][history['query_mask']].tobytes() == flat['raw_q'][flat['correction']].tobytes()
    selected = {step for start in records[0]['start_offsets'] for step in range(start+1, min(start+4,37))}
    for step in range(37):
        assert (history['weights'][step] > 0) == (step in selected)
        if step in selected:
            assert history['weights'][step] == (10/8)/(54*27)
            assert history['targets'][step].tobytes() == flat['raw_q'][step].tobytes()
        else:
            assert history['targets'][step].tobytes() == np.zeros(4,np.float32).tobytes()


def test_query_outside_sample_requires_positive_provenance(tmp_path):
    obj, flat, records = fixture(tmp_path)
    step = next(i for i in range(0,37,4) if i not in records[0]['start_offsets'])
    flat['label_mask'][step] = False
    flat['raw_q'][step] = 0
    obj.episodes[0]['teacher_calls'] -= 1
    with pytest.raises(ValueError, match='chronological queries'):
        obj.reconstruct('train')


def test_incidental_nonquery_labels_do_not_enter_history(tmp_path):
    obj, flat, records = fixture(tmp_path)
    before, _, _ = obj.reconstruct('train')
    selected = {i for s in records[0]['start_offsets'] for i in range(s,min(s+4,37))}
    step = next(i for i in range(37) if i%4 and i not in selected)
    flat['raw_q'][step] = np.nan  # authenticated producer would reject; audit must not use incidental scores.
    after, _, _ = obj.reconstruct('train')
    for key in before:
        assert before[key].tobytes() == after[key].tobytes()


def metric_fixture():
    # One two-window path and a query-only sibling case, with no postcorrection support.
    windows = {'lengths': np.array([4,4,1]), 'episode_index': np.array([0,0,1]),
        'step_offsets': np.array([0,4,0]), 'targets': np.tile(np.array([0,2,8,9],np.float32),(3,4,1)),
        'legal': np.ones((3,4,4),np.bool_)}
    predictions = windows['targets'].copy()
    predictions[1,1:,0] = 3 # choose action1, teacher gap2 at absolute steps5..7
    ids = [{'episode_id':'a','regime':'lambda3','case':0,'arm':'analytic'},
           {'episode_id':'b','regime':'lambda3','case':1,'arm':'analytic'}]
    meta = {'episode_ids':['a','b'],'episode_regimes':['lambda3','lambda3']}
    return predictions, windows, meta, ids


def test_scalar_postcorrection_domain_and_zero_support_denominator():
    result = audit.scalar_metrics(*metric_fixture())
    full, primary = result['full']['overall'], result['postcorrection']['overall']
    assert full['episode_weighted_agreement'] == .25
    assert primary['episode_weighted_agreement'] == 0
    assert primary['episode_weighted_raw_gap'] == 1
    assert primary['supported_episode_raw_gap'] == 2
    assert primary['weight_mass'] == .5 and primary['zero_support_episode_ids'] == ['b']
    assert primary['declared_case_count'] == 2 and primary['supported_case_count'] == 1
    assert primary['by_age']['3']['episode_weighted_raw_gap'] == 1
    assert result['postcorrection']['by_case'][1]['supported_episode_agreement'] is None


def test_float32_ties_and_legal_centered_common_offset():
    predictions, windows, meta, ids = metric_fixture()
    windows['targets'][:] = np.array([0,5e-11,999,999],np.float32)
    windows['legal'][:,:,2:] = False
    predictions[:] = np.array([2e-10,0,-999,-999],np.float32)
    result = audit.scalar_metrics(predictions,windows,meta,ids)
    assert result['postcorrection']['overall']['episode_weighted_agreement'] == .5
    assert result['postcorrection']['overall']['episode_weighted_first_argmin_match'] == 0
    assert result['postcorrection']['overall']['episode_weighted_raw_gap'] == float(np.float32(5e-11))/2
    windows['targets'][:] = np.array([0,2,999,999],np.float32)
    predictions[:] = np.array([100,102,-999,-999],np.float32)
    assert audit.scalar_metrics(predictions,windows,meta,ids)['full']['overall']['episode_weighted_centered_mse'] == 0


def reports():
    p,w,m,ids=metric_fixture()
    one = audit.scalar_metrics(p,w,m,ids)
    for domain in ('full','postcorrection'):
        group=one[domain]['by_regime']['lambda3']
        group['episode_weighted_agreement']=1.
        group['episode_weighted_raw_gap']=0.
        for age in group['by_age'].values():
            age['episode_weighted_raw_gap']=0.
            age['supported_case_count']=4
        one[domain]['by_regime']['lambda4']=copy.deepcopy(group)
    models=[{'family':f,'seed':s,'metrics':copy.deepcopy(one)} for s in audit.SEEDS for f in audit.FAMILIES]
    return models,one


def test_all_53_rules_include_third_control_full_path_and_technical_gate():
    models,hold=reports()
    rules=audit.continuation_rules(models,hold,technical_complete=True)
    assert len(rules)==53 and all(r['passes'] for r in rules)
    models[0]['metrics']['postcorrection']['by_regime']['lambda3']['episode_weighted_raw_gap']=1e-12
    rules=audit.continuation_rules(models,hold,technical_complete=False)
    assert not rules[0]['passes']
    by={r['name']:r for r in rules}
    assert not by['lambda3.mean_gap_vs_innovation_gru']['passes']
    assert not by['lambda3.mean_gap_vs_reset_direct']['passes']
    assert by['lambda3.full.mean_gap_vs_reset_direct']['passes']
    with pytest.raises(ValueError,match='all final'):
        audit.continuation_rules(models[:-1],hold,technical_complete=True)


def test_renormalized_persisted_history_weights_rejected(tmp_path):
    obj,_,_=fixture(tmp_path)
    values,meta,_=obj.reconstruct('train')
    saved={k:v.copy() for k,v in values.items()}
    saved['weights'] /= saved['weights'].sum()
    obj.arrays=lambda path:saved
    obj.read=lambda path:meta
    with pytest.raises(ValueError,match='window bytes weights'):
        obj.saved_windows('training-history',values,meta)


def test_complete_batch_work_includes_zero_target_lanes_and_full_tails(tmp_path):
    obj = audit.Audit(SimpleNamespace(output=tmp_path)); obj.np = np
    lengths = [1,32,33,64,65,2]+[1]*48
    offsets = np.asarray([0,*np.cumsum(lengths)],np.int64)
    weights = np.zeros(int(offsets[-1]),np.float64)
    weights[int(offsets[4])+33] = 1
    rows = obj.expected_batches({'episode_offsets':offsets,'weights':weights},[list(range(54))]*80)
    assert len(rows) == 720 and rows[-1]['optimizer_step'] == 720
    assert rows[0] == {'epoch':1,'batch':0,'episode_indices':list(range(6)),
        'forward_chunks':3,'backward_chunks':1,'no_grad_chunks':2,'forward_rows':197,'optimizer_step':1}
    assert rows[1]['forward_chunks'] == 1 and rows[1]['backward_chunks'] == 0
    assert sum(r['forward_rows'] for r in rows) == 80*sum(lengths)


def test_successful_worker_cannot_replace_failed_original_parent():
    root=str(audit.ROOT)
    launch={'cwd':root,'cap_seconds':120,'clock_source_sha256':audit.CLOCK_PIN,
        'watchdog_sha256':audit.SUPERVISOR_PIN,'started_ns':0,'deadline_ns':120_000_000_000}
    worker={'status':'completed','complete':True,'started_ns':1_000_000_000,
        'finished_ns':2_000_000_000,'wall_seconds':1.}
    terminal={**launch,'status':'completed','returncode':0,'timed_out':False,'error':None,
        'clock_error':None,'group_absent':True,'cleanup':{'reaped':True,'errors':[]},
        'finished_ns':3_000_000_000,'elapsed_ns':3_000_000_000,'wall_seconds':3.}
    audit.base.closed_parent(worker,terminal,launch,120)
    terminal['returncode']=1
    with pytest.raises(ValueError,match='successful fully closed'):
        audit.base.closed_parent(worker,terminal,launch,120)


def test_extra_json_null_is_not_treated_as_journal_eof():
    assert audit.journal_exhausted(iter([]))
    assert not audit.journal_exhausted(iter([None]))
    assert not audit.journal_exhausted(iter([{}]))
