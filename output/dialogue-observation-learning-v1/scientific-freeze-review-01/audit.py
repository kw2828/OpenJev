"""One saved-metadata freeze check. No project, model or numerical-library imports."""
import hashlib
import json
import math
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'output/dialogue-observation-learning-v1'
OUT = Path(__file__).resolve().parent
FREEZE = BASE / 'scientific-freeze-01'
PREP = BASE / 'preparation-02'
PINS = {
 'scientific-freeze-01/completed.json': 'c3a53a94dd4a36e677b36d4e358bb779abd6419711941a792e50ce48feab36c3',
 'scientific-freeze-01/plan.json': 'acb79b4600c66966762895d28eb2dc1d2be15c761d677c5e87c5750dde47f237',
 'scientific-freeze-process-01.launch.json': '53c141423d9edb17b2951d73f978014846ebada34b49a3f591558987780605c8',
 'scientific-freeze-process-01.terminal.json': '2c6a4661238d865fa79b3b6aba9e9da64db9f7ebba216558a8751e5ac169b15a',
 'scientific-allocation-01.json': 'f1348237d4955cc2f3bb8b6d7b24071c7ed13bea6a5e8b1ce9ace10460ca8b69',
 'scientific-source-review-01/receipt.json': '45921e885a9cd1cf67036e5d2dfd065e5a3e3e767777d4c62210e12a599fd815',
 'preparation-02/completed.json': 'd1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83',
 'preparation-02/plan.json': '4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e',
 'preparation-audit-02/receipt.json': '4fefa46325c8c8e4a74a6f14dcf8a7961f13c875faeb14b3f03ee35169c2c23c',
}


def need(ok, message):
 if not ok:
  raise ValueError(message)


def sha(path):
 need(path.is_file() and not path.is_symlink(), 'Regular file required')
 h = hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda: f.read(1024*1024), b''):
   h.update(block)
 return h.hexdigest()


def read(path):
 return json.loads(path.read_text())


def write(path, obj):
 with path.open('x') as f:
  json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
  f.write('\n')


def manifest(folder, files):
 all_paths = list(folder.rglob('*'))
 need(not any(p.is_symlink() for p in all_paths), 'No symlinks')
 actual = {p.relative_to(folder).as_posix() for p in all_paths if p.is_file()}
 need(actual == set(files) | {'completed.json'}, 'Exact closed payload set')
 for name, entry in files.items():
  need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Safe member path')
  p = folder/name
  need(p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], 'Payload identity: '+name)


def inspect():
 for name, pin in PINS.items():
  need(sha(BASE/name) == pin, 'External pin: '+name)
 done, plan = read(FREEZE/'completed.json'), read(FREEZE/'plan.json')
 parent, prepared = read(PREP/'plan.json'), read(PREP/'completed.json')
 manifest(FREEZE, done['files'])
 manifest(PREP, prepared['files'])
 prior = read(BASE/'preparation-audit-02/receipt.json')
 need(prior['agreement'] is True and prior['preparation_completed_sha256'] == PINS['preparation-02/completed.json'], 'Inherited prepared audit')
 review = read(BASE/'scientific-source-review-01/receipt.json')
 need(review['status'] == 'clear', 'Final source clearance')
 sources = plan['source_sha256']
 need(len(sources) == 53 and sources == done['source_sha256'] == review['scientific_source_sha256'], 'All53 reviewed sources')
 for name, pin in sources.items():
  need(sha(ROOT/name) == sha(FREEZE/'sources'/name) == pin, 'Source/current snapshot match: '+name)
 expected_files = {'started.json','allocation.json','evaluation-rows.jsonl','references.json','plan.json'} | {'sources/'+n for n in sources}
 need(set(done['files']) == expected_files and len(expected_files) == 58, 'Freeze58payload/59total closure')
 need(done['version'] == plan['version'] == 'dialogue-observation-scientific-v1'
      and done['status'] == 'completed' and done['phase'] == 'freeze'
      and done['model_calls'] == done['encoder_calls'] == 0
      and done['plan_sha256'] == PINS['scientific-freeze-01/plan.json'], 'Completed model-free freeze')
 spec = read(BASE/'scientific-allocation-01.json')
 need(plan['allocation'] == spec and sha(FREEZE/'allocation.json') == plan['allocation_sha256'] == PINS['scientific-allocation-01.json'], 'Exact allocation copy')
 need(spec['limits'] == {'wall_seconds':28800,'rss_bytes':8*1024**3,'mps_driver_bytes':8*1024**3,'output_bytes':2*1024**3}
      and spec['freeze_limits'] == {'wall_seconds':300,'rss_bytes':8*1024**3,'output_bytes':512*1024**2}, 'Prospective phase allocations')
 for key in ('cost_completed','cost_audit','protocol'):
  need(sha(Path(spec[key]['path'])) == spec[key]['sha256'], 'Allocation prerequisite pin')
 audit = read(Path(spec['cost_audit']['path']))
 need(audit['status'] == 'completed' and audit['agreement'] is True
      and audit['execution_completed_sha256'] == spec['cost_completed']['sha256'], 'Successful cost/audit binding')
 need(plan['prepared_completed_sha256'] == PINS['preparation-02/completed.json']
      and plan['prepared_plan_sha256'] == PINS['preparation-02/plan.json']
      and Path(plan['prepared']).resolve() == PREP.resolve(), 'Corrected full preparation binding')
 need(plan['config'] == parent['config'] and plan['loss_counts'] == parent['loss_counts']
      and plan['loss_weights'] == parent['loss_weights'] and plan['runtime'] == parent['runtime'], 'Prepared recipe/objective/runtime')
 need(plan['quality_scoring_in_runner'] is False and plan['no_retry'] is True, 'No selection/retry')
 arms = ['frozen_original','frozen_numbers','trainable_original','trainable_numbers']
 seeds = [6901,6902,6903]
 need(plan['fit_order'] == [f'{arm}-{seed}' for seed in seeds for arm in arms], 'All12 paired-fit order')
 need(plan['config']['epochs'] == 20 and plan['config']['effective_batch'] == 32 and plan['config']['microbatch'] == 1, 'Epoch/batch recipe')
 need(sha(PREP/'orders.json') == plan['orders_sha256'], 'Exact prepared orders')
 orders = read(PREP/'orders.json')
 need(orders['seeds'] == seeds and orders['epochs'] == 20 and len(set(orders['dialogue_ids'])) == 2017, 'Order inventory')
 for seed in seeds:
  sequence = orders['orders'][str(seed)]
  need(len(sequence) == 20 and all(sorted(o) == list(range(2017)) for o in sequence), 'Every epoch a complete permutation')
 need(sha(FREEZE/'evaluation-rows.jsonl') == plan['evaluation_rows_sha256']
      and sha(FREEZE/'references.json') == plan['references_sha256'], 'Opaque evaluator/reference hash joins')
 profiles = read(PREP/'workloads.json')['profiles']
 index = read(PREP/'index.json')
 bykey = {(p['split'],p['dialogue_id']):p['work'] for p in profiles}
 need(len(bykey) == len(profiles) == len(index) == 4380, 'Complete4380work/index records')
 need(set(bykey) == {(p['split'],p['dialogue_id']) for p in index}, 'Split-qualified layout/work join')
 cohorts = Counter(p['split'] for p in profiles)
 need(cohorts == {'train':2017,'dev':2363}, 'Complete fixed cohorts')
 endpoints = {s:sum(p['scored_rows'] for p in index if p['split'] == s) for s in cohorts}
 need(endpoints == {'train':51741,'dev':62329}, 'Original endpoint counts')
 need(all(endpoints[s] == sum(parent['loss_counts'][s].values()) for s in cohorts), 'Loss count endpoint join')
 keys = ('input_texts','content_tokens','encoder_sequences','encoder_calls','special_token_positions',
         'valid_token_positions','padded_token_positions','padded_attention_positions','padding_token_positions',
         'overlength_texts_chunked','truncated_tokens')
 def visits(key):
  return sum(p['work'][key] * (20 if p['split']=='train' else 1) for p in profiles)
 enc = {k:visits(k) for k in keys}
 per = {'optimizer_updates':20*math.ceil(2017/32),'training_dialogue_visits':20*2017,'backward_calls':20*2017,
        'evaluation_forwards':2363,'training_endpoints':20*51741,'evaluation_endpoints':62329,
        'memory_forwards':20*2017+2363,'encoding_passes':20*2017+2363,'encoder_calls':enc['encoder_calls']}
 steps = dict.fromkeys(('incoming_checks','feature_checks','result_checks','mass_checks',
                       'executed_valid_question_slots','real_question_updates'),visits('real_question_updates'))
 steps.update(dict.fromkeys(('advance_calls','advance_returned','valid_turns'),visits('public_user_turns')))
 steps.update(dict.fromkeys(('forward_calls','forward_returned'),per['memory_forwards']))
 expected = {'per_fit':per,'all_fits':{k:v*12 for k,v in per.items()},'fits':12,'encoder_work_per_fit':enc,'state_counts_per_fit':steps}
 need(expected == plan['expected'] == done['expected'], 'All operation/work/monitor totals independently reconstructed')
 started = read(FREEZE/'started.json')
 request = {'command':'freeze','allocation':str((BASE/'scientific-allocation-01.json').resolve()),
            'allocation_sha256':PINS['scientific-allocation-01.json'],'prepared':str(PREP.resolve()),'out':str(FREEZE.resolve())}
 need(started['version'] == plan['version'] and started['runtime'] == plan['runtime'] and started['request'] == request, 'Exact freeze request/runtime')
 launch=read(BASE/'scientific-freeze-process-01.launch.json'); terminal=read(BASE/'scientific-freeze-process-01.terminal.json')
 command=['.venv/bin/python','scripts/study_dialogue_observation.py','freeze','--allocation',request['allocation'],
          '--allocation-sha256',request['allocation_sha256'],'--prepared',request['prepared'],'--out',request['out']]
 need(launch['command'] == terminal['command'] == command and launch['cap_seconds'] == spec['freeze_limits']['wall_seconds'], 'Supervised freeze command/cap')
 need(launch['watchdog_sha256'] == sha(ROOT/'output/dialogue-finetune-qualification-v1/watchdog-01.py')
      == 'd3992f019cc5135d162b074327a73bdbd3b4c160ddc89f990165cd345c578242', 'Frozen300s supervisor')
 need(launch['pid'] > 0 and launch['pid'] == launch['pgid'] == terminal['pgid']
      and terminal['returncode'] == 0 and terminal['group_absent'] is True and terminal['error'] is None
      and terminal['timed_out'] is False, 'Successful recorded process cleanup')
 size=sum(p.stat().st_size for p in FREEZE.rglob('*') if p.is_file())
 need(0 <= done['wall_seconds'] <= terminal['wall_seconds'] <= 300
      and 0 < done['peak_rss_bytes'] <= 8*1024**3 and size <= 512*1024**2, 'Freeze resource limits')
 return {'status':'completed','agreement':True,'external_pins':PINS,'script_sha256':sha(Path(__file__)),
         'scientific_sources_verified':53,'freeze_payloads_verified':58,'freeze_total_files':59,
         'allocation':spec,'cohorts':dict(cohorts),'endpoints':endpoints,'expected_work':expected,
         'evaluator_rows_sha256':plan['evaluation_rows_sha256'],'references_sha256':plan['references_sha256'],
         'worker_wall_seconds':done['wall_seconds'],'supervisor_wall_seconds':terminal['wall_seconds'],
         'freeze_peak_rss_bytes':done['peak_rss_bytes'],'freeze_output_bytes':size,
         'recorded_process_group_absent':True,'recorded_returncode':0,'reported_model_calls':0,'reported_encoder_calls':0,
         'reviewer_experimental_model_calls':0,'task_metrics_computed':False,
         'limits':['Evaluator rows, references, token/label streams and float caches are hash-checked only, never decoded here.',
                   'Token-profile reconstruction, semantic joins and exact canonical content inherit the independently audited preparation and reviewed frozen authoring code. This review recomputes work totals and endpoint denominators from small metadata.',
                   'Resource readings and process absence are authenticated producer/supervisor witnesses, not fresh runtime observations.',
                   'This is metadata readiness only. No training, task scoring or empirical model claim is made.']}


start=time.perf_counter()
def timeout(*_):
 raise TimeoutError('60second saved-review cap')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,60)
try:
 write(OUT/'started.json',{'external_pins':PINS,'script_sha256':sha(Path(__file__)),'wall_cap_seconds':60})
 result=inspect()
 result['review_wall_seconds']=time.perf_counter()-start
 result['review_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
 need(result['review_wall_seconds']<=60 and result['review_peak_rss_bytes']<=2*1024**3,'Review resource cap')
 write(OUT/'receipt.json',result)
except BaseException as error:
 signal.setitimer(signal.ITIMER_REAL,0)
 try:
  write(OUT/'failed.json',{'status':'failed','external_pins':PINS,'error':str(error),'error_type':type(error).__name__})
 except BaseException as secondary:
  if hasattr(error,'add_note'):error.add_note('Failure preservation error: '+repr(secondary))
 raise
finally:
 signal.setitimer(signal.ITIMER_REAL,0)
print(json.dumps({'status':'completed','agreement':True,'receipt_sha256':sha(OUT/'receipt.json')}))
