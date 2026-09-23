import importlib.util, json, pathlib, sys, time
spec = importlib.util.spec_from_file_location('detached_fixture', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
base = pathlib.Path(sys.argv[2]); prefix = base / 'phase'
command = [sys.executable, str(base / 'worker.py'), str(base), '--supervision', f'{prefix}.launch.json']
module.launch(prefix, 8, command)
time.sleep(10)
