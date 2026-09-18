"""Lossless disk cache of native root/child attack bits, without teacher labels.

This is ordinary bit packing, not an algorithmic or inference-speed claim.
Decoded child graphs are compact and follow legal-mask row order. Operators
are normalized in floating point only after decoding, so no approximate graph
delta is introduced. A completed cache is immutable and content authenticated.
"""
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import torch

from openjev.research.chess_graph_contrast import candidate_graphs

VERSION = 'native-child-attack-bits-v1'
BYTES_PER_GRAPH = 1024
FILES = ('roots.bin', 'children.bin', 'offsets.npy', 'roots.jsonl')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_cache(rows, out, *, provenance):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    offsets = [0]; roots = 0
    try:
        with (out/'roots.bin').open('xb') as root_stream, (out/'children.bin').open('xb') as child_stream, (out/'roots.jsonl').open('x') as metadata:
            for index, row in enumerate(rows):
                graph = candidate_graphs([chess.Board(row['fen'])])
                names = graph['menus'][0]; count = len(names)
                root_stream.write(np.packbits(graph['root'].numpy().reshape(1, -1), axis=-1, bitorder='little').tobytes())
                child_stream.write(np.packbits(graph['children'][0, :count].numpy().reshape(count, -1), axis=-1, bitorder='little').tobytes())
                offsets.append(offsets[-1]+count); roots += 1
                record = {'index': index, 'fen': row['fen'], 'moves': names,
                          'permutation': graph['permutation'][0, :count].tolist()}
                metadata.write(json.dumps(record, separators=(',', ':'))+'\n')
        if not roots:
            raise ValueError('Cannot write an empty graph cache')
        with (out/'offsets.npy').open('xb') as stream:
            np.save(stream, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
        manifest = {'status': 'completed', 'version': VERSION, 'roots': roots,
                    'children': offsets[-1], 'bits_per_graph': 8192, 'bitorder': 'little',
                    'provenance': provenance, 'files': {name: file_hash(out/name) for name in FILES},
                    'stored_graph_bytes': BYTES_PER_GRAPH*(roots+offsets[-1]),
                    'unpacked_uint8_graph_bytes': 8192*(roots+offsets[-1]),
                    'scope': 'Native attacks only; root-player frame; no labels or learned features stored. Disk bit packing is not an inference-speed or peak-memory result.'}
        with (out/'manifest.json').open('x') as stream:
            json.dump(manifest, stream, indent=2); stream.write('\n')
        return manifest
    except BaseException as error:
        with (out/'failed.json').open('x') as stream:
            json.dump({'status': 'failed', 'error': repr(error), 'roots_written': roots}, stream, indent=2)
        raise


class ChildGraphCache:
    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory/'manifest.json').read_text())
        if (manifest['status'] != 'completed' or manifest['version'] != VERSION
                or manifest['bitorder'] != 'little' or manifest['bits_per_graph'] != 8192
                or set(manifest['files']) != set(FILES)):
            raise ValueError('Unsupported or incomplete child-graph cache')
        if any(file_hash(directory/name) != digest for name, digest in manifest['files'].items()):
            raise ValueError('Child-graph cache hash mismatch')
        count, children = manifest['roots'], manifest['children']
        if (type(count) is not int or type(children) is not int or count < 1 or children < count
                or (directory/'roots.bin').stat().st_size != count*BYTES_PER_GRAPH
                or (directory/'children.bin').stat().st_size != children*BYTES_PER_GRAPH):
            raise ValueError('Invalid cache sizes')
        offsets = np.load(directory/'offsets.npy', allow_pickle=False)
        records = [json.loads(line) for line in (directory/'roots.jsonl').open()]
        if (offsets.dtype != np.int64 or offsets.shape != (count+1,) or offsets[0] != 0
                or offsets[-1] != children or not (np.diff(offsets) > 0).all() or len(records) != count):
            raise ValueError('Invalid cache offsets')
        for i, row in enumerate(records):
            length = int(offsets[i+1]-offsets[i])
            if (row['index'] != i or len(row['moves']) != length
                    or row['moves'] != sorted(set(row['moves']))
                    or sorted(row['permutation']) != list(range(length))):
                raise ValueError('Invalid cached legal menu')
        self.manifest, self.offsets, self.records = manifest, offsets, records
        self.roots = np.memmap(directory/'roots.bin', mode='r', dtype=np.uint8,
                               shape=(count, BYTES_PER_GRAPH))
        self.children = np.memmap(directory/'children.bin', mode='r', dtype=np.uint8,
                                  shape=(children, BYTES_PER_GRAPH))

    def batch(self, indices):
        indices = indices.tolist() if hasattr(indices, 'tolist') else list(indices)
        if (not indices or any(type(i) is not int or i < 0 or i >= len(self.records) for i in indices)):
            raise ValueError('Invalid root indices')
        records = [self.records[i] for i in indices]
        roots = np.unpackbits(self.roots[indices], axis=-1, bitorder='little').reshape(-1, 2, 64, 64)
        packed = np.concatenate([self.children[self.offsets[i]:self.offsets[i+1]] for i in indices], axis=0)
        children = np.unpackbits(packed, axis=-1, bitorder='little').reshape(-1, 2, 64, 64)
        lengths = torch.tensor([len(row['moves']) for row in records])
        width = int(lengths.max()); mask = torch.arange(width)[None] < lengths[:, None]
        permutation = torch.arange(width).expand(len(indices), -1).clone()
        for b, row in enumerate(records):
            permutation[b, :len(row['moves'])] = torch.tensor(row['permutation'])
        return {'root': torch.from_numpy(roots), 'children': torch.from_numpy(children),
                'mask': mask, 'permutation': permutation, 'menus': [tuple(r['moves']) for r in records],
                'fens': [r['fen'] for r in records]}
