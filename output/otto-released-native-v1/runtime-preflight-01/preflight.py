import os,json,sys,platform,contextlib,io,warnings
from pathlib import Path
import numpy as np
out=Path(sys.argv[1]);rows=[]
for phase in ['before_tensorflow','after_tensorflow']:
 if phase=='after_tensorflow':
  import tensorflow as tf
  tf.config.set_visible_devices([], 'GPU');tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
  assert not tf.config.get_visible_devices('GPU') and tf.keras.Model.__module__.startswith('tf_keras.')
  assert tf.keras.backend.floatx()=='float32' and tf.keras.mixed_precision.global_policy().name=='float32' and tf.keras.backend.image_data_format()=='channels_last'
 for k in [11025,1024]:
  for left in [0.,1/16]:
   a=np.full((1,k),left,dtype=np.float32);b=np.full((k,1024),1/16,dtype=np.float32)
   with warnings.catch_warnings(record=True) as caught,np.errstate(all='warn'):
    warnings.simplefilter('always');c=a@b
   expected=left*k/16;record={'phase':phase,'k':k,'left':left,'expected':expected,'finite':bool(np.isfinite(c).all()),'exact':bool(np.all(c==expected)),'warnings':[str(w.message) for w in caught]}
   rows.append(record)
   with (out/'primitive-checks.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   assert record['finite'] and record['exact'] and not caught
buffer=io.StringIO()
with contextlib.redirect_stdout(buffer):np.show_config()
(out/'numerical-configuration.txt').write_text(buffer.getvalue())
(out/'runtime.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'tensorflow':tf.__version__,'keras_model_module':tf.keras.Model.__module__,'physical_devices':[str(d) for d in tf.config.list_physical_devices()],'visible_devices':[str(d) for d in tf.config.get_visible_devices()],'platform':platform.platform(),'model_calls':0,'checkpoint_reads':0,'primitive_calls':len(rows)},indent=2)+'\n')
