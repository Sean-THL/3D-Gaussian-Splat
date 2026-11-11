# Collect thrust/body-rate demonstrations

The current collection choice is native Isaac rigid-body physics with NMPC;
follow [NATIVE_BACKEND.md](isaac_sim/NATIVE_BACKEND.md#run-native-isaac-collection).
The instructions below retain the custom-simulator alternative.

The default custom profile now uses a numerical force/torque rigid-body plant.
Use `--quadrotor-plant rate_response` for the earlier simplified response model.
NMPC outputs collective thrust in newtons and XYZ body rates in radians/second.
The 3DGUT renderer uses the achieved position and orientation, including tilt.
Randomization samples task offsets, heading, duration, initial errors, and
physical response parameters once per episode. Appearance randomization is
not implemented. The camera mount/alignment remains provisional.

The native Isaac backend is also available; see
[NATIVE_BACKEND.md](isaac_sim/NATIVE_BACKEND.md). The standalone A-to-B
linear-MPC/geometric-controller experiment is retained in `animate_ab_flight.py`;
that experimental combination is not wired into the RGB dataset collector.

## 1. Prepare the remote shell

Connect from your laptop:

```bash
ssh tk@141.212.76.61
```

In that remote terminal, with the latest project code already transferred:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
export PATH="$HOME/anaconda3/bin:$PATH"
export SIMULATION_ENV="$HOME/anaconda3/envs/3dgrut"
"$SIMULATION_ENV/bin/python" -c 'import torch, casadi; assert torch.cuda.is_available(), "CUDA unavailable"; print("CUDA and NMPC dependencies ready")'
```

If CasADi is missing in the selected project environment, install the pinned
solver dependency there:

```bash
"$SIMULATION_ENV/bin/python" -m pip install -r Workflow/Simulation/control/requirements-nmpc.txt
```

No Isaac process, WebRTC client, browser server, or separate pose-render service
is needed for custom headless collection. The collector loads its own CUDA
renderer. Keep the SSH connection open while it runs.

## 2. Collect a small pilot

```bash
collection_root="data/scene_20261025_200034/datasets/pilot_$(date -u +%Y%m%dT%H%M%SZ)"
bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/quadrotor_nmpc.json \
  --trajectory lissajous_figure8 --yaw-mode velocity_aligned \
  --randomize --seed 1000 --episodes 10 \
  --headless --dataset --record \
  --output "$collection_root"
```

This runs seeds 1000–1009 sequentially. `--dataset` saves synchronized RGB
and expert commands at the 20 Hz control rate. `--record` additionally saves
an inspection video; it is optional for later large batches. Do not use
`--no-render` for visual-policy collection. Omitting `--duration` runs the
complete randomized route and its settling interval.

Each output directory must be new; existing directories are refused. An error
stops the batch. Do not treat an interrupted batch as completed.

## 3. Audit and inspect the pilot

After successful collection, in the same shell:

```bash
for episode_dir in "$collection_root"/episode_*; do
  "$SIMULATION_ENV/bin/python" Workflow/Simulation/audit_dataset.py "$episode_dir" || break
done
```

The audit checks image/action timing, reference shape, solved expert labels,
finite states/actions, quaternion norms, action bounds, image checksums and
sample counts. Each successful audit writes `dataset_audit.json`.

Also inspect several `simulation.mp4` files and each episode's `summary.json`:
verify plausible camera views, adequate scene coverage and good tracking.
The collection gate currently stops if position error exceeds 0.2 m; this is
a development tracking threshold, not obstacle clearance or a proof of fidelity.
Dataset timing is pre-action; inspection video frames are rendered separately
after stepping and should not be used as synchronized training images.

## 4. Collect the first development dataset

After pilot review, use separate batches and non-overlapping seed ranges.
These counts are an initial experiment, not a claim of sufficient training data:

| Split | Start seed | Episodes | Seeds |
| --- | ---: | ---: | --- |
| Train | 2000 | 100 | 2000–2099 |
| Validation | 10000 | 20 | 10000–10019 |
| Test | 20000 | 20 | 20000–20019 |

Use the pilot command with a new `--output`, the corresponding `--seed` and
`--episodes`, and omit `--record` if inspection videos are unnecessary.
Keep whole episodes together when splitting. Never randomly split adjacent
frames from the same flight across training and evaluation sets. Preserve the
configuration/calibration for the collection campaign; changed tasks or models
should get a separate campaign directory. Future test variants can assess
conditions beyond the training distribution.

The default figure eight lasts 30 seconds, with a hard minimum of 30 seconds.
Duration randomization is currently 1.0–1.1 times the nominal duration (30–33 s).
Including two settling seconds gives about 640–700 samples per episode at 20 Hz. Estimate storage and
wall time from the pilot before starting larger batches.

## Output layout

```text
pilot_<timestamp>/
  config.json
  provenance.json
  episodes.json                 # Written after the batch completes
  episode_0001/
    config.json
    episode_parameters.json     # Seed, actual task, plant and initial offsets
    rgb/000000.png
    rgb/000001.png
    rgb_actions.jsonl           # Image paths + pre-action state/reference/labels
    dataset_metadata.json       # Frames, camera provenance, limits and timing
    dataset_count.json
    dataset_audit.json          # Added by successful audit
    demonstrations.jsonl
    telemetry.csv
    summary.json
    simulation.mp4              # Only when --record is supplied
```

The future student loader will consume `rgb_actions.jsonl` and its linked PNG
images. The target is `[thrust_n, body_rate_x, body_rate_y, body_rate_z]`.
Physical randomization parameters are retained for analysis, not student inputs.
The network and training loader have not yet been implemented.

When collection finishes, type `exit` to close this SSH session.

## Visualize completed data

On a machine with the complete downloaded run, NumPy, SciPy, Pillow, Matplotlib
and FFmpeg installed:

```bash
python3 Workflow/Simulation/visualize_dataset.py PATH_TO_COMPLETED_RUN
```

This audits each episode, then creates `visualization/rollout.mp4`, a PNG/PDF
overview, an RGB contact sheet and a summary with the source-label checksum.
All views use saved pre-action data; physics and 3DGS rendering are not rerun.
The video previews at 10 fps; all original 20 Hz training samples remain intact.
Use `--plots-only` to skip video or `--fps 20` to display every sample at 20 Hz.
Synchronized position errors in these figures use pre-action timestamps;
the native `report.json` tracking statistics use post-step telemetry.
