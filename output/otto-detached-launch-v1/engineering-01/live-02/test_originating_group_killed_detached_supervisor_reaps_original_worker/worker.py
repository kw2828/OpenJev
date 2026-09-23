import json, os, pathlib, sys, time
base = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 6
launch = base / 'phase.launch.json'
while not launch.exists():
    if time.monotonic() >= deadline: raise TimeoutError('no original launch')
    time.sleep(.01)
record = json.loads(launch.read_text())
assert record['pid'] == record['pgid'] == os.getpid() == os.getpgrp()
assert record['parent_pid'] == os.getppid()
ready = {'pid': os.getpid(), 'ppid': os.getppid(), 'pgid': os.getpgrp(),
         'sid': os.getsid(0), 'threads': os.environ['OMP_NUM_THREADS']}
def marker(name, value):
    temporary = base / (name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, base / name)
    temporary.unlink()
marker('worker-ready.json', ready)
while not (base / 'release').exists():
    if time.monotonic() >= deadline: raise TimeoutError('fixture release not received')
    time.sleep(.01)
marker('worker-finished.json',
       {'pid': os.getpid(), 'parent_still_original': os.getppid() == record['parent_pid']})
