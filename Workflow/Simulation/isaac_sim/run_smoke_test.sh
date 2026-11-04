#!/usr/bin/env bash
set -euo pipefail
isaac_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$isaac_script_dir/../../.."
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$HOME/isaacsim}"
if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
  echo "Missing standalone launcher: $ISAAC_SIM_ROOT/python.sh" >&2
  exit 1
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Deactivate Conda first (conda deactivate); this test uses Isaac Sim's bundled Python." >&2
  exit 1
fi
exec "$ISAAC_SIM_ROOT/python.sh" "$isaac_script_dir/smoke_test.py" "$@"
