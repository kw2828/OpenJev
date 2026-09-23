import hashlib,json,platform,time
import numpy as np
from openjev.research.otto_sampled_forecast_data import select_windows
start=time.monotonic(); overall=hashlib.sha256(); seeds=[]
for seed in range(23600001,23600055):
 h=hashlib.sha256()
 for length in range(1,2189):
  data=(json.dumps(select_windows(length,seed),sort_keys=True,separators=(",",":"))+"\n").encode(); h.update(data); overall.update(data)
 seeds.append({"seed":seed,"sha256":h.hexdigest()})
print(json.dumps({"python":platform.python_version(),"numpy":np.__version__,"draws":54*2188,"sha256":overall.hexdigest(),"seeds":seeds,"elapsed_seconds":time.monotonic()-start},sort_keys=True))
