"""Archive a closed Codex chess bridge; never dispatch or retry a decision."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path


def archive(directory, out):
    directory, out = Path(directory), Path(out)
    receipt_path = out.with_name(out.name.removesuffix('.jsonl.gz')+'.json')
    if out.exists() or receipt_path.exists():
        raise FileExistsError('Choose a fresh archive destination')
    closed = json.loads((directory/'closed.json').read_text())
    requests = sorted(directory.glob('request-*.json'))
    if len(requests) != closed['requests']:
        raise ValueError('Closed request count differs from files')
    expected_ids = {f'{i:05d}' for i in range(len(requests))}
    for prefix in ('response', 'expired'):
        if any(p.stem.split('-')[1] not in expected_ids for p in directory.glob(prefix+'-*.json')):
            raise ValueError('Orphan response or expiration record')
    rows = []
    for index, path in enumerate(requests):
        packet = json.loads(path.read_text())
        request_id = f'{index:05d}'
        if packet['request_id'] != request_id or path.name != f'request-{request_id}.json':
            raise ValueError('Request identity or order differs')
        response = directory/f'response-{request_id}.json'
        expired = directory/f'expired-{request_id}.json'
        if not response.exists() and not expired.exists():
            raise ValueError('Unresolved request in closed bridge')
        # A deadline/submission race can leave both records. Preserve both.
        rows.append({'request': packet, 'response': json.loads(response.read_text()) if response.exists() else None,
                     'expired': json.loads(expired.read_text()) if expired.exists() else None})
    raw = ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as stream:
        stream.write(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buffer.getvalue())
    receipt = {'requested_model': 'gpt-6-astra', 'worker': '/root/astra_chess_player',
               'dispatch_source': 'Explicit model selection in parent Codex collaboration tool call',
               'provider_api_attestation': False, 'scope': 'Persistent Codex agent with instruction-limited isolation',
               'requests': len(rows), 'responses': sum(r['response'] is not None for r in rows),
               'expired': sum(r['expired'] is not None for r in rows),
               'archive_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
               'uncompressed_sha256': hashlib.sha256(raw).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'closed': closed}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.dir, args.out)))
