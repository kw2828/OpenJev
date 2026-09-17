"""Expose only unanswered board packets to a separately dispatched chess player."""

import argparse
import json
import time
from pathlib import Path


def next_packet(directory, seconds=20):
    deadline=time.monotonic()+seconds
    while True:
        if (directory/'closed.json').exists():
            return {'status':'closed'}
        for path in sorted(directory.glob('request-*.json')):
            number=path.stem.split('-')[1]
            if not (directory/f'response-{number}.json').exists() and not (
                    directory/f'expired-{number}.json').exists():
                return json.loads(path.read_text())
        if time.monotonic() >= deadline:
            return {'status':'waiting'}
        time.sleep(.05)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['next','submit','ready'])
    parser.add_argument('--dir',type=Path,required=True)
    parser.add_argument('--id')
    parser.add_argument('--move')
    args=parser.parse_args()
    args.dir.mkdir(parents=True,exist_ok=True)
    if args.command == 'ready':
        with (args.dir/'worker-ready.json').open('x') as f:
            json.dump({'ready':True},f)
        result={'status':'ready'}
    elif args.command == 'next':
        result=next_packet(args.dir)
    else:
        if not args.id or not args.id.isdigit() or len(args.id)!=5:
            parser.error('Expected the five-digit request ID')
        request=json.loads((args.dir/f'request-{args.id}.json').read_text())
        if (args.dir/'closed.json').exists() or (args.dir/f'expired-{args.id}.json').exists():
            print(json.dumps({'status':'discarded_late', 'next':next_packet(args.dir)}),flush=True)
            return
        if args.move not in {c['id'] for c in request['candidates']}:
            parser.error('Choose a supplied legal move')
        dest=args.dir/f'response-{args.id}.json'
        if dest.exists():
            raise FileExistsError(dest)
        temp=args.dir/f'response-{args.id}.tmp'
        with temp.open('x') as f:
            json.dump({'request_id':args.id,'choice':args.move},f)
        temp.rename(dest)
        result=next_packet(args.dir)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
