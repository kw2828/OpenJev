"""Close the three original bounded processes without scientific execution."""
import json
import sys
sys.path.insert(0, 'scripts')
import otto_belief_distillation_common as c
plan_path=c.OUT/'registration-01.json'
plan=c.read(plan_path); c.check_plan(plan)
processes={}
previous=None
for name,directory,prefix in [('collection','collection-01','collection-native-01'),('fit','fit-01','fit-native-01'),('audit','audit-01','audit-native-01')]:
    receipt_path=c.OUT/directory/'receipt.json'; terminal_path=c.OUT/(prefix+'.terminal.json')
    receipt=c.closed(c.OUT/directory,terminal_path); terminal=c.read(terminal_path)
    c.require(receipt['plan_sha256']==c.desc(plan_path)['sha256'],'same registration')
    c.require(previous is None or previous<=terminal['started_ns'],'closed sequential phases')
    previous=terminal['finished_ns']
    processes[name]={'receipt':c.desc(receipt_path),'terminal':c.desc(terminal_path),'worker_seconds':receipt['wall_seconds']}
audit=c.read(c.OUT/'audit-01/audit.json')
c.require(audit['agreement'] and audit['requires_original_supervisor_closure'] and not audit['technical_complete'],'independent audit state')
status='DEV_PASS' if audit['gate']['passed'] else 'DEV_FAIL'
closure={'version':c.VERSION,'technical_complete':True,'independent_audit_passed':True,'status':status,
         'gate':audit['gate'],'audit_counts':audit['counts'],'processes':processes,
         'registration':c.desc(plan_path),'audit':c.desc(c.OUT/'audit-01/audit.json'),
         'new_execution_admitted':False,'old_test_admitted':False}
c.write(c.OUT/'closure-01.json',closure)
print(json.dumps({'status':status,'passed_cells':sum(x['passed'] for x in audit['gate']['cells']), 'cells':len(audit['gate']['cells']),'closure':c.desc(c.OUT/'closure-01.json')}))
