#!/usr/bin/env bash
set -euo pipefail
native_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$native_dir/../../.." && pwd)"
isaac_root="${ISAAC_SIM_ROOT:-$HOME/isaacsim}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo 'Deactivate Conda: native physics uses Isaac bundled Python.' >&2
  exit 1
fi
export PYTHONPATH="$project_root/data/scene_20261025_200034/isaac_dependencies${PYTHONPATH:+:$PYTHONPATH}"
export ISAAC_SIM_ROOT="$isaac_root"
exec python3 "$native_dir/native_supervisor.py" "$@"
