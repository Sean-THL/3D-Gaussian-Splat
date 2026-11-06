"""Run Kit in a child and require a passing report as well as process success.

Kit shutdown can terminate Python before code after SimulationApp.close runs.
The parent therefore owns the final operational exit status.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--output', type=Path)
options, _ = parser.parse_known_args()
root = Path(__file__).resolve().parents[3]
output = options.output or root/'data/scene_20261025_200034/isaac_sim'/('native_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
isaac_root = Path(os.environ.get('ISAAC_SIM_ROOT', str(Path.home()/'isaacsim')))
import sys
command = [str(isaac_root/'python.sh'), str(Path(__file__).with_name('native_rollout.py')), '--output', str(output), *sys.argv[1:]]
result = subprocess.run(command)
if '--help' in sys.argv or '-h' in sys.argv:
    raise SystemExit(result.returncode)
try:
    report = json.loads((output/'report.json').read_text())
except (OSError, ValueError):
    print('NATIVE_VALIDATION_FAILED: missing or invalid report', file=sys.stderr)
    raise SystemExit(1)
if result.returncode == 0 and report.get('status') == 'interactive_closed':
    print('NATIVE_VIEWER_CLOSED: '+str(output), flush=True)
    raise SystemExit(0)
passed = result.returncode == 0 and report.get('status') == 'passed'
print('NATIVE_VALIDATION_'+('PASSED' if passed else 'FAILED')+': '+str(output), flush=True)
raise SystemExit(0 if passed else 1)
