"""Small synthetic data-flow check; no experiment panel or efficacy result."""
import importlib.util
from pathlib import Path
import time

import chess
import torch

from openjev.research.chess_candidate import CandidateChess

path=Path(__file__).resolve().parents[1]/'scripts/chess_transport_study.py'
spec=importlib.util.spec_from_file_location('transport_study_fixture',path)
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_pack_train_reload_evaluate_matches_complete_decision(tmp_path):
    torch.set_num_threads(2)
    board=chess.Board(); second=board.copy();second.push_uci('e2e4')
    rows=[{'id':f'fixture-{i}','fen':b.fen(en_passant='fen'),'game_id':i,
           'target_uci':sorted(m.uci() for m in b.legal_moves)[0]} for i,b in enumerate([board,second])]
    data,menus=study.pack(rows,'fixture',tmp_path)
    model=CandidateChess('direct',seed=97).eval().requires_grad_(False)
    original={k:v.clone() for k,v in model.state_dict().items()}
    cache=study.root_cache(model,data)
    study.train_head(97,'transport',model,data,cache,tmp_path/'transport-97','a'*64,time.time())
    head=study.load_head(tmp_path,97,'transport','a'*64)
    result=study.evaluate(head,model,data,cache,rows,menus,'transport',tmp_path/'predictions.jsonl')
    records=study.rows(tmp_path/'predictions.jsonl')
    assert result['examples']==2
    for row,record in zip(rows,records):
        assert record['choice']==study.decision(model,head,row['fen'],'transport')
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_fit_order_contains_each_arm_once_per_paired_backbone():
    first=study.fit_order()
    assert first==study.fit_order()
    assert set(first)=={'97','109','127'}
    for arms in first.values():
        assert len(arms)==5 and set(arms)==set(study.ARMS)
