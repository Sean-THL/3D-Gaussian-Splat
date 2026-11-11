#!/usr/bin/env bash
set -euo pipefail
bridge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_root="${ISAAC_SIM_ROOT:-$HOME/isaacsim}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo 'Deactivate Conda before launching Isaac; Gaussian rendering runs separately.' >&2
  exit 1
fi
if [[ ! -x "$isaac_root/isaac-sim.streaming.sh" ]]; then
  echo "Missing Isaac streaming launcher: $isaac_root/isaac-sim.streaming.sh" >&2
  exit 1
fi
exec "$isaac_root/isaac-sim.streaming.sh" --exec "$bridge_dir/start_live_bridge.py" "$@"
