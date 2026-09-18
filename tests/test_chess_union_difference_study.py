# SPDX-License-Identifier: GPL-3.0-only
import copy
import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import chess_union_difference_study as study


def passing():
    return {f'{arm}-{seed}':{split:{'agreement':.35 if arm=='edits' else .33} for split in study.SPLITS}
            for arm in ('edits','child','union','rotated','wldn','base') for seed in study.SEEDS}


def test_gate_requires_every_comparator_panel_and_seed_floor():
    metrics=passing();assert len(study.gate(metrics))==10 and all(r['passed'] for r in study.gate(metrics))
    metrics['wldn-97']['shift']['agreement']=.356
    failed=[r for r in study.gate(metrics) if not r['passed']]
    assert len(failed)==1 and failed[0]['split']=='shift' and failed[0]['comparator']=='wldn'
    assert failed[0]['mean_gain']>.01 # Seed floor still rejects the comparison.
    metrics=passing();metrics['union-97']['dev']['agreement']=.349;metrics['union-109']['dev']['agreement']=.349
    assert not next(r for r in study.gate(metrics) if r['split']=='dev' and r['comparator']=='union')['passed']


def test_learning_audit_rejects_reordered_batches_and_missing_updates():
    records=[]
    for epoch in range(6):
        order=np.random.default_rng(700000+100*97+epoch).permutation(32768)
        for j in range(256):
            records.append({'update':epoch*256+j+1,'epoch':epoch,'examples':128,'loss':1.,'gradient_norm':.5,
                'indices_sha256':hashlib.sha256(order[j*128:(j+1)*128].tobytes()).hexdigest()})
    study.check_learning(records,97)
    bad=copy.deepcopy(records);bad[0]['indices_sha256']=bad[1]['indices_sha256']
    with pytest.raises(AssertionError):study.check_learning(bad,97)
    with pytest.raises(AssertionError):study.check_learning(records[:-1],97)
    with pytest.raises(AssertionError):study.check_learning(records,109)


def test_corrupted_evaluation_loads_edits_weights_and_checks_identity(tmp_path):
    path=tmp_path/'edits-97';path.mkdir();head=study.ChessUnionDifferenceHead('edits',seed=1197)
    payload={'version':study.VERSION,'seed':97,'arm':'edits','plan_sha256':'p','state_dict':head.state_dict()}
    torch.save(payload,path/'weights.pt')
    loaded=study.load(tmp_path,97,'edits_corrupted','p')
    assert loaded.arm=='rotated' and all(torch.equal(v,loaded.state_dict()[k]) for k,v in head.state_dict().items())
    with pytest.raises(AssertionError):study.load(tmp_path,97,'edits_corrupted','wrong')


def test_protocol_retains_full_training_and_prediction_budget():
    p=study.PROTOCOL
    assert len(p['arms'])*len(p['seeds'])*p['epochs']*(p['training_examples']//p['batch_size'])==p['total_updates']==18432
    assert len(study.MODES)*3*2*2048==p['new_predictions']==61440
    assert p['copied_predictions']==2*3*2*2048
