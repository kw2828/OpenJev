import importlib.util
import json
import threading
import time
from pathlib import Path

import chess
import pytest

from openjev.research.chess_external import CodexChessPolicy


def test_bridge_request_has_no_gold_and_response_matches(tmp_path):
    policy = CodexChessPolicy(tmp_path, timeout_seconds=2)
    def worker():
        request = tmp_path/'request-00000.json'
        while not request.exists():
            time.sleep(.005)
        row=json.loads(request.read_text())
        assert 'gold' not in row
        assert len(row['candidates']) == 20
        temp=tmp_path/'answer.tmp'
        temp.write_text(json.dumps({'request_id':row['request_id'], 'choice':'e2e4'}))
        temp.rename(row['response_file'])
    thread=threading.Thread(target=worker)
    thread.start()
    assert policy(chess.Board())['choice'] == 'e2e4'
    thread.join()
    policy.close()
    assert json.loads((tmp_path/'closed.json').read_text())['requests'] == 1
    with pytest.raises(FileExistsError):
        CodexChessPolicy(tmp_path)


def test_timeout_is_preserved_without_fallback(tmp_path):
    policy=CodexChessPolicy(tmp_path, timeout_seconds=.01)
    with pytest.raises(TimeoutError):
        policy(chess.Board())
    assert (tmp_path/'request-00000.json').exists()
    assert not (tmp_path/'response-00000.json').exists()
    assert (tmp_path/'expired-00000.json').exists()
    spec=importlib.util.spec_from_file_location('bridge', Path(__file__).parents[1]/'scripts/chess_agent_bridge.py')
    bridge=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    assert bridge.next_packet(tmp_path, seconds=0) == {'status':'waiting'}
    policy.close()
    assert bridge.next_packet(tmp_path, seconds=0) == {'status':'closed'}


@pytest.mark.parametrize('filename',['closed.json','response-00000.json','expired-00000.json'])
def test_stale_bridge_state_rejected(tmp_path, filename):
    (tmp_path/filename).write_text('{}')
    with pytest.raises(FileExistsError):
        CodexChessPolicy(tmp_path)
