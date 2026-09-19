"""One execution and saved-output audit of the already frozen memory study."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'evidence/reacher-memory-ablation-v1/protocol/plan.json'
DIGEST = '05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786'
OUT = ROOT / 'output/reacher-memory-ablation-v1/supervision'
EXECUTION = ROOT / 'runs/reacher-memory-ablation-v1/attempt'
AUDIT = ROOT / 'evidence/reacher-memory-ablation-v1/audit'


def write(name, data):
    with (OUT / name).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    stage = 'validate-launch'
    try:
        assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == DIGEST
        assert not EXECUTION.exists() and not AUDIT.exists()
        plan = json.loads(PLAN.read_text())
        write('started.json', {'plan_sha256': DIGEST,
            'started_at_utc': datetime.now(UTC).isoformat(),
            'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'execution': str(EXECUTION.relative_to(ROOT)), 'audit': str(AUDIT.relative_to(ROOT)),
            'frozen_execution_cap_seconds':plan['cap_seconds'],
            'frozen_audit_cap_seconds':plan['audit_cap_seconds'],
            'external_timeout_grace_seconds':30,
            'grace_scope':'Only kill a hung process; internal frozen caps still determine validity.',
            'retries':0})
        stage = 'execution'
        with (OUT/'execution.log').open('x') as log:
            subprocess.run([sys.executable, '-u', 'scripts/reacher_memory_study.py', 'run',
                '--plan', str(PLAN), '--expected-plan-sha256', DIGEST, '--out', str(EXECUTION)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=plan['cap_seconds']+30)
        print(json.dumps({'phase':'execution-complete'}), flush=True)
        stage = 'audit'
        with (OUT/'audit.log').open('x') as log:
            subprocess.run([sys.executable, '-u', 'scripts/audit_reacher_memory_study.py',
                '--plan', str(PLAN), '--expected-plan-sha256', DIGEST,
                '--execution', str(EXECUTION), '--out', str(AUDIT)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=plan['audit_cap_seconds']+30)
        summary=json.loads((AUDIT/'summary.json').read_text())
        result={'status':'completed','plan_sha256':DIGEST,
            'continuation_passed':summary['continuation_gate']['passed'],
            'passed_checks':sum(r['passed'] for r in summary['continuation_gate']['checks']),
            'native_transitions_checked':summary['native_transitions_checked'],
            'whole_supervision_seconds':time.monotonic()-begin,
            'audit_receipt_sha256':hashlib.sha256((AUDIT/'receipt.json').read_bytes()).hexdigest()}
        write('completed.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as error:
        write('failed.json',{'status':'failed','stage':stage,'plan_sha256':DIGEST,
            'error':repr(error),'whole_supervision_seconds':time.monotonic()-begin,
            'retry_or_resume_permitted':False})
        raise


if __name__=='__main__':
    main()
