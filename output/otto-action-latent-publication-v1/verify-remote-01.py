import base64
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from PIL import Image

ROOT=Path.cwd()
OUT=ROOT/'output/otto-action-latent-publication-v1'
COMMIT='a6310629ffa70f9ce50a792eb6cf8771b251648b'

def api(path):
    return json.loads(subprocess.check_output(['gh','api',path],text=True))

def sha(data):
    return hashlib.sha256(data).hexdigest()

release=api('repos/kw2828/OpenJev/releases/tags/otto-action-latent-v1')
assert release['target_commitish']==COMMIT and release['draft'] is False
assets={a['name']:a for a in release['assets']}
names={'otto-action-latent-v1.tar.gz','archive-manifest.json','archive-verification.json','SHA256SUMS'}
assert set(assets)==names
asset_checks=[]
for name in sorted(names):
    path=OUT/name
    expected=sha(path.read_bytes())
    remote=assets[name]
    assert remote['state']=='uploaded' and remote['size']==path.stat().st_size and remote['digest']=='sha256:'+expected
    asset_checks.append({'name':name,'bytes':remote['size'],'sha256':expected,'remote_digest':remote['digest']})
readme=(ROOT/'README.md').read_text()
images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',readme)
assert len(images)==4
verified=[]
for name in ['README.md',*images]:
    path=ROOT/name
    tree=subprocess.check_output(['git','ls-tree',COMMIT,'--',name],text=True).strip()
    assert tree
    blob=tree.split()[2]
    remote=api('repos/kw2828/OpenJev/git/blobs/'+blob)
    data=base64.b64decode(remote['content'])
    assert remote['encoding']=='base64' and data==path.read_bytes() and remote['size']==len(data)
    row={'path':name,'blob':blob,'bytes':len(data),'sha256':sha(data)}
    if name!='README.md':
        with Image.open(io.BytesIO(data)) as im:
            row.update(format=im.format,size=list(im.size),frames=getattr(im,'n_frames',1))
            if path.suffix=='.gif': assert row['frames']>1
    verified.append(row)
result={'status':'passed','release':release['html_url'],'commit':COMMIT,'assets':asset_checks,'readme_and_images':verified,
        'scope':'GitHub release server digests and exact Git blobs, local decoding of remote image bytes; not browser animation playback',
        'new_model_or_native_calls':0}
with (OUT/'release-verification-01.json').open('x') as stream:
    json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'status':'passed','release':release['html_url'],'assets':len(asset_checks),'readme_images':len(images)}))
