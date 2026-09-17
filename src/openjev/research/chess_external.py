"""Auditable filesystem bridge to an explicitly selected Codex chess player.

The parent dispatches the model; this module does not impersonate an API or
verify provider identity. Workers receive observable positions, never targets.
Elapsed time includes Codex orchestration and tool use, not just inference.
"""

import json
import time
from pathlib import Path

from openjev.research.chess_arena import chess_record


class CodexChessPolicy:
    name = "GPT-6 Astra / Codex"

    def __init__(self, bridge_dir, timeout_seconds=120):
        self.directory = Path(bridge_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.glob('request-*')) or any(self.directory.glob('response-*')) or (
                self.directory/'closed.json').exists() or any(self.directory.glob('expired-*')):
            raise FileExistsError('Bridge already has execution state; create a fresh execution')
        self.index = 0
        self.timeout = float(timeout_seconds)
        if not 0 < self.timeout <= 300:
            raise ValueError('Bridge timeout must be in (0, 300] seconds')
        self.metadata = {'model': 'gpt-6-astra', 'route': 'Codex agent filesystem bridge',
                         'provider_attested_model': False, 'calibration': 'not returned',
                         'timing': 'wall time including agent orchestration and tool calls',
                         'probabilities': 'not requested or fabricated',
                         'timeout_seconds': self.timeout}

    def __call__(self, board):
        index = self.index
        self.index += 1
        request_id = f'{index:05d}'
        start = time.perf_counter()
        packet = {'request_id': request_id, 'fen': board.fen(),
                  **chess_record(board),
                  'response_file': str(self.directory/f'response-{request_id}.json')}
        temporary = self.directory/f'request-{request_id}.tmp'
        temporary.write_text(json.dumps(packet, indent=2))
        temporary.rename(self.directory/f'request-{request_id}.json')
        response_path = self.directory/f'response-{request_id}.json'
        while not response_path.exists():
            if time.perf_counter()-start >= self.timeout:
                (self.directory/f'expired-{request_id}.json').write_text(json.dumps(
                    {'request_id': request_id, 'reason': 'transport_timeout'}))
                raise TimeoutError(f'No Codex decision for {request_id}; no retry')
            time.sleep(.025)
        answer = json.loads(response_path.read_text())
        if set(answer) != {'request_id', 'choice'} or answer['request_id'] != request_id:
            raise ValueError('Invalid response identity or fields')
        return {'choice': answer['choice'], 'request_id': request_id,
                'latency_ms': (time.perf_counter()-start)*1000, 'policy': self.name,
                'timing_route': self.metadata['timing']}

    def close(self):
        (self.directory/'closed.json').write_text(json.dumps({'requests': self.index}))
