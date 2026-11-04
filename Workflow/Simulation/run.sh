#!/usr/bin/env bash
set -euo pipefail
simulation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$simulation_dir/../.."
export MAX_JOBS="${MAX_JOBS:-1}"
simulation_env="${SIMULATION_ENV:-$HOME/tools/3dgrut/.venv}"
if [[ -f "$simulation_env/bin/activate" ]]; then
  source "$simulation_env/bin/activate"
  python Workflow/Simulation/run.py "$@"
elif command -v conda >/dev/null 2>&1; then
  if [[ -n "${SIMULATION_ENV:-}" ]]; then
    conda run --no-capture-output --prefix "$simulation_env" python Workflow/Simulation/run.py "$@"
  else
    conda run --no-capture-output -n 3dgrut python Workflow/Simulation/run.py "$@"
  fi
else
  echo "Activate a 3DGUT environment or set SIMULATION_ENV to its path." >&2
  exit 1
fi
