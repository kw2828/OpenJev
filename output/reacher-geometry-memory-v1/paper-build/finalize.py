from pathlib import Path
from hashlib import sha256
import json
from datetime import datetime, timezone
import shutil
from pypdf import PdfReader

root = Path('/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev')
build = root / 'output/reacher-geometry-memory-v1/paper-build'
auth = json.loads((build / 'input-authentication.json').read_text())
def digest(path):
    return sha256(path.read_bytes()).hexdigest()
def identity(path):
    return {'sha256': digest(path), 'bytes': path.stat().st_size}
snapshots = build / 'input-snapshots'
snapshots.mkdir(exist_ok=False)
input_members = {}
for relative, expected in auth['inputs_sha256'].items():
    source = root / relative
    if digest(source) != expected:
        raise ValueError(f'Input changed before snapshot: {relative}')
    target = snapshots / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if digest(target) != expected or digest(source) != expected:
        raise ValueError(f'Input changed during snapshot: {relative}')
    input_members[relative] = identity(target)
source = root / 'paper/reacher-geometry-memory-study.tex'
shutil.copyfile(source, build / 'final-source.tex')
pdf = build / 'reacher-geometry-memory-study.pdf'
reader = PdfReader(pdf)
assert len(reader.pages) == 4
texts = [page.extract_text() for page in reader.pages]
assert all(text.strip() for text in texts)
assert '25/25' in '\n'.join(texts)
assert '\u2014' not in source.read_text()
log = (build / 'reacher-geometry-memory-study.log').read_text()
assert 'Overfull' not in log and 'Underfull' not in log
assert 'undefined references' not in log
command = json.loads((build / 'build-command.json').read_text())
assert command['exit_code'] == 0
output = root / 'output/pdf/openjev-reacher-geometry-memory-study.pdf'
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('xb') as stream:
    stream.write(pdf.read_bytes())
assert digest(output) == digest(pdf)
qa = {
    'pages': 4,
    'rotations_degrees': [page.rotation for page in reader.pages],
    'text_words_per_page': [len(t.split()) for t in texts],
    'latex_overfull_boxes': 0, 'latex_underfull_boxes': 0,
    'undefined_references': 0,
    'benign_warnings': ['Two caption hypcap warnings: captionof outside a float; visible captions and hyperlinks inspected.'],
    'render': {'tool': 'pdftoppm', 'dpi': 120, 'exit_code': 0},
    'visual_review': {
        'method': 'All four final rendered PNGs inspected with view_image by the authoring agent.',
        'pages_inspected': [1, 2, 3, 4],
        'findings': 'No clipped or overlapping text, broken glyphs, illegible figure labels, or table overflow. Portrait pages 1/4 and landscape pages 2/3 retain page numbering.'
    },
    'table_validation': 'All nine controller means across three panels checked against authenticated audit summary; all fit-level figures reused unchanged.',
    'new_analysis_draws': 0, 'new_model_calls': 0, 'new_native_calls': 0,
}
(build / 'qa.json').write_text(json.dumps(qa, indent=2) + '\n')
files = {str(p.relative_to(build)): identity(p) for p in sorted(build.rglob('*')) if p.is_file() and p.name != 'receipt.json'}
receipt = {
    'status': 'completed',
    'scope': 'Four-page development report from completed authenticated saved artifacts; not an ICLR submission or new scientific run.',
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'latex_source': {str(source.relative_to(root)): identity(source)},
    'pdf': {str(output.relative_to(root)): identity(output)},
    'inputs': input_members,
    'input_snapshot_root': str(snapshots.relative_to(root)),
    'build': command,
    'retained_failed_compile': {
        'command': 'build-command-01.json', 'log': 'compile-01.log', 'source': 'source-at-first-compile.tex',
        'reason': 'Title used display-math delimiter before spacing instead of a LaTeX line break; corrected before successful compile. No input or scientific result changed.'
    },
    'qa': qa,
    'files': files,
    'limits': 'This receipt binds rendered report artifacts. It reuses the existing scientific audit; it does not rerun or independently certify neural predictions.'
}
(build / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'pdf': identity(output), 'source': identity(source), 'receipt': identity(build / 'receipt.json'), 'pages': 4, 'input_count': len(input_members), 'build_members': len(files)}, indent=2))
