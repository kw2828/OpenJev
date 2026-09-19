"""Capture this prospective study after tests/review, before any scientific run."""
from pathlib import Path
import datetime
import json
import shutil
import subprocess
from card_memory_common import runtime, sha, write_json
from card_calibration_common import authenticate

root=Path(__file__).resolve().parents[3]
evidence=root/'evidence/card-calibration-v1'
paths=[
 'evidence/card-calibration-v1/protocol.json','evidence/card-calibration-v1/inputs.json',
 'scripts/card_calibration_common.py','scripts/fit_card_calibration.py',
 'scripts/evaluate_card_calibration.py','scripts/report_card_calibration.py','scripts/visualize_card_calibration.py',
 'src/openjev/research/card_probability_calibration.py',
 'tests/test_card_probability_calibration.py','tests/test_fit_card_calibration.py',
 'tests/test_evaluate_card_calibration.py','tests/test_report_card_calibration.py',
 'tests/test_visualize_card_calibration.py',
 'research/card-calibration-study.md','output/card-calibration-v1/driver-review.md',
 'output/card-calibration-v1/preparation/freeze.py',
 'output/card-confidence-direction-v1/calibration-design.md','output/card-confidence-direction-v1/architecture-prior-art.md',
]
paths += [str(p.relative_to(root)) for p in sorted((root/'output/card-confidence-diagnostic-v1').rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
assert not any((root/'output/card-calibration-v1'/p).exists() for p in ('calibration-01','evaluation-01'))
for legacy in ('evidence/card-memory-pilot-v1/source-bindings.json','evidence/card-controllers-v1/source-bindings.json'):
    for name,digest in json.loads((root/legacy).read_text())['files'].items():
        assert sha(root/name)==digest, name
bindings={'version':'card-calibration-v1','runtime':runtime(),
 'files':{p:sha(root/p) for p in paths},
 'inherited_bindings_sha256':sha(root/'evidence/card-controllers-v1/source-bindings.json'),
 'inherited_source_count':88}
write_json(evidence/'source-bindings.json',bindings)
for relative in paths:
    dest=evidence/'source-snapshot'/relative; dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/relative,dest)
    assert sha(dest)==bindings['files'][relative]
protocol,inputs,binding=evidence/'protocol.json',evidence/'inputs.json',evidence/'source-bindings.json'
checkpoint=root/'output/card-memory-pilot-v1/training-01/checkpoint-map.json'
recipe,cases,weights,fits=authenticate(root,protocol,sha(protocol),inputs,sha(inputs),binding,sha(binding),checkpoint,sha(checkpoint))
write_json(evidence/'freeze.json',{'status':'prospective','version':'card-calibration-v1',
 'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'protocol_sha256':sha(protocol),'inputs_sha256':sha(inputs),'bindings_sha256':sha(binding),
 'checkpoint_map_sha256':sha(checkpoint),'new_bound_files':len(paths),'inherited_source_files':88,
 'calibration_fits_before_freeze':0,'fresh_native_calls_before_freeze':0,'new_weight_updates':0,
 'native_retries_authorized':0,'all_inherited_fits_authenticated':len(fits),
 'all_new_seeds_disjoint':len(cases['evaluation']),
 'tests':'See bound driver-review.md for final focused synthetic check counts.'})
print(json.dumps({'status':'prospective_frozen','files':len(paths),'protocol':sha(protocol),'bindings':sha(binding)}))
