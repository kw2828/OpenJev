"""Verify the uploaded draft against local hashes before publishing it."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'output/reacher-memory-ablation-v1'
OUT = ROOT/'evidence/reacher-memory-ablation-v1/scored-publication'
REPO = 'kw2828/OpenJev'
TAG = 'research-reacher-memory-ablation-v1'
RELEASE_ID = 391858368
COMMIT = '607a7d29d9ad615df3973e7f82c30eb02e21e529'


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def api(*args):
    result = subprocess.run(['gh','api',*args],check=True,text=True,capture_output=True)
    return json.loads(result.stdout)


def verify(release, expected):
    assert release['tag_name'] == TAG and release['target_commitish'] == COMMIT
    rows = {row['name']:row for row in release['assets']}
    assert set(rows) <= set(expected), 'Unexpected release asset'
    for name, row in rows.items():
        want = expected[name]
        assert row['state'] == 'uploaded' and row['size'] == want['bytes'], name
        assert row['digest'] == 'sha256:'+want['sha256'], name
    return rows


def main():
    started = json.loads((BASE/'upload-started.json').read_text())
    expected = {row['name']:row for row in started['assets']}
    assert len(expected) == 10
    endpoint = f'repos/{REPO}/releases/{RELEASE_ID}'
    deadline = time.monotonic()+1200
    previous = -1
    try:
        while True:
            release = api(endpoint)
            assert release['draft'] is True, 'Expected unpublished draft'
            rows = verify(release, expected)
            if len(rows) != previous:
                print(json.dumps({'verified_uploaded_assets':len(rows),'expected':10}),flush=True)
                previous = len(rows)
            if set(rows) == set(expected):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Upload verification deadline; preserve draft')
            time.sleep(30)
        write(BASE/'draft-upload-verification.json',{'status':'all_assets_verified','release_id':RELEASE_ID,
            'expected_assets':expected,'remote':release,'verified_at_utc':datetime.now(UTC).isoformat()})
        request = BASE/'publish-request.json'
        write(request,{'draft':False,'make_latest':'false'})
        api('--method','PATCH',endpoint,'--input',str(request))
        final = api(endpoint)
        assert final['draft'] is False and final['published_at']
        rows = verify(final,expected)
        assert set(rows) == set(expected)
        downloads=[]
        for name in ('receipt.json','manifest.json','SHA256SUMS'):
            with urllib.request.urlopen(rows[name]['browser_download_url'],timeout=60) as response:
                data=response.read()
            assert len(data)==expected[name]['bytes']
            assert hashlib.sha256(data).hexdigest()==expected[name]['sha256']
            downloads.append({'name':name,'bytes':len(data),'sha256':expected[name]['sha256']})
        result={'status':'published_and_verified','repository':REPO,'release_id':RELEASE_ID,
            'release_tag':TAG,'release_url':final['html_url'],'target_commit':COMMIT,
            'published_at':final['published_at'],'verified_at_utc':datetime.now(UTC).isoformat(),
            'verification_scope':'All10 GitHub-reported upload states, byte sizes and SHA-256 digests match local assets;3 public sidecars separately downloaded and hashed. Large archive bytes were reopened and verified locally, not downloaded again.',
            'assets':[{**expected[name],'asset_id':rows[name]['id'],
                'browser_download_url':rows[name]['browser_download_url']} for name in expected],
            'public_sidecar_downloads':downloads,
            'package_receipt_sha256':'911d4397c5b5f43b2cbe9b03b8290b22462e02d5dccdcecf381d94f948b96b39',
            'audit_receipt_sha256':'2589383326a2312d7f738821da7fd11f82b9bee1b7168aba2e7f44aaa4946d5d',
            'scientific_gate':{'passed':True,'checks_passed':25,'checks_total':25},
            'new_model_calls':0,'new_native_calls':0,
            'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'earlier_rejected_creation':{'http_status':422,'short_target':'607a7d2',
                'reason':'Release.target_commitish is invalid','resolution':'Verified remote full commit, then created draft successfully.'}}
        write(OUT/'release-verification.json',result)
        print(json.dumps({'status':result['status'],'url':result['release_url'],'assets':len(rows)}),flush=True)
    except BaseException as error:
        write(BASE/'release-verification-failed.json',{'status':'failed','error':repr(error),
            'release_id':RELEASE_ID,'time_utc':datetime.now(UTC).isoformat(),
            'scope':'Publication only. Scientific execution and audit are unchanged.'})
        raise


if __name__=='__main__':
    main()
