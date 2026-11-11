# Project map: recording → scene → drone demonstrations

This is the current index for the October 25 scene. Run paths and historical
provenance retain their original names. Descriptive shortcuts group the assets
by stage without breaking existing scripts or duplicating large files.

## Start here: named assets

Open [the organized scene folder](../data/scene_20261025_200034/organized/README.md).

```text
data/scene_20261025_200034/organized/
  01_recording/       original bag, decoded recording and preview MP4s
  02_colmap/          296 input images, sparse reconstruction, undistorted input
  03_gaussian_scene/ final PLY export, renderer checkpoint, training run
  04_alignment/      pinned candidate Vicon ↔ 3DGS calibration
  05_fidelity/       measured-depth evaluation and comparison images
  06_simulation/     custom/native runs and confirmed training-data pilot
```

Each item is a relative symlink. Original data are unchanged. These local assets
are ignored by Git; a clone alone will not contain them. Recreate shortcuts on
another machine after copying the data:

```bash
python3 Workflow/organize_scene.py --create-links --verify-hashes
```

Omit `--create-links` for a read-only presence check. Missing assets are reported
per machine. The tracked [scene asset registry](scene_oct25_assets.json) lists
exact original paths, names, purposes and known PLY hashes.

## 1. Which recording belongs to this scene?

| Item | Location relative to repository root | Role |
| --- | --- | --- |
| Original reconstruction recording | `Workflow/Data/rosbags/flight_rgbd_20261025_024629/` | ROS 2 metadata + DB3; RGB, measured depth, Vicon, PX4, calibration and TF |
| Retained scene-local copy | `data/scene_20261025_200034/raw/flight_rgbd_20261025_024629/` | Copy of the same recording, not another flight |
| Full decoded export | `Workflow/Data/exports/flight_rgbd_20261025_024629/native/` | 1,009 RGB frames, 1,010 depth frames, source timestamps and telemetry |
| Sparse inspection export | `data/scene_20261025_200034/reference_sample/` | 0.2 Hz inspection sample; not all reconstruction frames |
| RGB-only preview | `.../native/video/rgb.mp4` | Watch the recording; not COLMAP input |
| Original two-panel preview | `.../raw/flight_rgbd_20261025_024629/flight_rgbd_20261025_024629_two_panel.mp4` | Recording inspection; not COLMAP input |

The scene/model ID `scene_20261025_200034` and recording ID
`flight_rgbd_20261025_024629` are different historical identifiers. Preparation
provenance and the matched final PLY link them; do not infer separate flights
from the differing names. Keep the bag directory intact, including metadata.

Data code: [extract_rosbag.py](Data/extract_rosbag.py) decodes recordings;
[dataset_tools.py](Data/dataset_tools.py) selects frames and creates calibrated
crop/resize variants. See [Data/README.md](Data/README.md) and
[DATASET_WORKFLOW.md](DATASET_WORKFLOW.md).

## 2. Which MP4 was used for COLMAP?

**None.** For the confirmed scene, the preparation command extracted PNGs
directly from the RGB rosbag stream, using recording-relative times 12–130 s
and a maximum sampling rate of 3 Hz. It produced **296 images at 1280×720**:

```text
Workflow/Reconstruction/datasets/flight_20261025_native/
  images/                       the 296 COLMAP input PNGs
  calibration.json              recorded/fixed lens calibration
  preparation.json              original bag, extraction command and image count
  source_export/
    manifest.json
    rgb/timestamps.csv           original timestamps for the selected images
```

The full recording MP4, two-panel MP4, trimmed AnySplat videos and saved Gaussian
preview MP4 are different products. None is the source of these COLMAP images.

`oct25_shared_native/` is a later reusable RGB/depth selection with reference
splits; it is not a renamed replacement for the original training dataset.

Reconstruction orchestration: [mono_workflow.py](Reconstruction/mono_workflow.py)
provides `prepare`, `colmap`, `train` and `diagnose`. Detailed commands are in
[Reconstruction/README.md](Reconstruction/README.md).

## 3. What did COLMAP produce?

The complete remote results were copied into:

```text
data/scene_20261025_200034/training_reference/Workflow/Reconstruction/
  datasets/flight_20261025_native/
    colmap/
      database.db                image features and pairwise matches
      sparse/0/
        cameras.bin              lens parameters and camera models
        images.bin               registered poses and image observations
        points3D.bin             sparse landmarks and tracks
      model_analyzer.log
      provenance.json            actual COLMAP commands and environment
      ...logs
    undistorted/
      images/                    296 undistorted RGB training images
      sparse/0/                  corresponding PINHOLE camera/pose/point model
```

The undistorted tree also contains sparse-model files directly under `sparse/`;
these are retained loader-compatible copies. The simulator configuration points
to `undistorted/sparse/0/cameras.bin`.

Stored model analysis: **296/296 registered images, 32,507 sparse points,
0.788 px mean reprojection error**. Sparse points are not Gaussian primitives
or collision geometry. COLMAP used all registered views for pose estimation.

## 4. Which Gaussian model is the final scene?

The confirmed successful run is:

```text
.../training_reference/Workflow/Reconstruction/runs/
  flight_20261025_native_20261025T215743_160060Z/
    provenance.json
    training.log
    model/flight_20261025_native/undistorted-2510_175936/
      ckpt_last.pt               model actually loaded by our renderer
      export_last.ply            final 500,000-Gaussian export
      parsed.yaml                resolved external trainer configuration
      metrics.json               held-out appearance metrics
      ours_30000/renders/        37 saved held-out renders
      reconstruction_preview.mp4 output preview of the trained scene
```

Use these descriptive shortcuts:

- [final_scene_gaussians.ply](../data/scene_20261025_200034/organized/03_gaussian_scene/final_scene_gaussians.ply)
- [scene_renderer_checkpoint.pt](../data/scene_20261025_200034/organized/03_gaussian_scene/scene_renderer_checkpoint.pt)

The originally selected `reconstruction/scene_20261025_200034(1).ply` is
**byte-identical** to `export_last.ply`. Both were rechecked with SHA-256:

```text
9791f10202f8fe3a958efd2690543ba580ff6f1a6b4278a6a67809896d2ed88e
```

The native renderer loads **`ckpt_last.pt`**, not the PLY. The PLY is useful for
Gaussian viewers/export; it is not a mesh or a verified collision surface.
`rendering.checkpoint` and `rendering.camera_model` in the active simulation
configuration select the actual assets. NVIDIA 3DGRUT/3DGUT software remains
external at `~/tools/3dgrut`; this project contains orchestration and adapters.

The root-level `gaussians.ply`, `gaussians (1).ply`, October 13 MP4s,
`AnySplats- V1-result/`, and older reconstruction trials are legacy assets.
They are not the confirmed October 25 simulation model. Their presence does
not select them for integration.

## 5. Fidelity and coordinate alignment

| Topic | Code | Confirmed output / status |
| --- | --- | --- |
| Held-out RGB review | Training output + `evaluation/transferred_review.json` | 37 views; PSNR 30.18 dB, SSIM 0.911, LPIPS 0.375 |
| Gaussian depth vs recorded depth | [Evaluation/depth_check.py](Evaluation/depth_check.py) | `evaluation/depth_20261103T041136_791961Z/`; 25 views, median absolute Z difference 0.369 m |
| Vicon/scene/mount estimation | [Calibration/align_vicon.py](Calibration/align_vicon.py) | `calibration/alignment_20261103T220923_264755Z/alignment.json` |
| Alignment explanation figure | [Calibration/plot_alignment_workflow.py](Calibration/plot_alignment_workflow.py) | `diagnostics/figures/vicon_3dgs_alignment_equations.*` |

Paths in the output column are under `data/scene_20261025_200034/` unless a
different prefix is shown. Detailed limitations are in
[Calibration/README.md](Calibration/README.md), [Evaluation/README.md](Evaluation/README.md)
and [the scene review](../diagnostics/scene_20261025_200034_review.md).

The pinned mapping combines:

```text
Vicon marker/body pose
  → estimated fixed camera mounting transform
  → camera pose in Vicon
  → inverse metric scene alignment
  → optical camera pose in original 3DGS coordinates
  → rendered RGB
```

`gs_to_vicon` stores scale/rotation/translation; `camera_to_tracked_body`
stores the estimated optical-camera mounting transform. Scale is 0.754179
m/reconstruction unit. The fitted RGB/Vicon time offset is −6.260 ms.
Use the saved rotations for orientations; the scaled position matrix is not
itself a rotation matrix.

The alignment is **a candidate**, especially the mounting translation. It has
good recorded-trajectory consistency but the mount height varies by 6.8 cm
across fits. Marker-body axes are provisionally used as virtual drone axes;
their relationship to the real drone center/flight-controller frame still
needs independent measurement. Depth fidelity is not established for tight
collision clearance. These are separate issues from good tracking in a simulator.

## 6. Where are dynamics, trajectories, controllers and randomization?

| Component | Source | Purpose |
| --- | --- | --- |
| Custom app / CLI | [Simulation/run.py](Simulation/run.py), [run.sh](Simulation/run.sh) | Viewer, headless runs, recorder and renderer orchestration |
| Simulation loop | [figs/simulator.py](Simulation/figs/simulator.py) | Reset, current state, reference horizon, expert command, plant stepping |
| Simplified quadrotor | [figs/dynamics.py](Simulation/figs/dynamics.py) | Thrust/body-rate response approximation and shared state |
| Custom rigid body | [figs/rigid_body.py](Simulation/figs/rigid_body.py) | Rate controller → bounded torque → inertia/rotation/translation |
| Native Isaac rigid body | [isaac_sim/native_backend.py](Simulation/isaac_sim/native_backend.py) | PhysX forces/torques and floor/body contact |
| Ideal integrators | [figs/integrators.py](Simulation/figs/integrators.py) | Single/double integrator alternatives |
| Trajectories | [tasks/trajectory.py](Simulation/tasks/trajectory.py) | M, circle and Lissajous figure eight; constant/tangent-aligned yaw |
| Default expert | [control/quadrotor_mpc.py](Simulation/control/quadrotor_mpc.py) | NMPC predicting nominal rate/thrust response; outputs thrust + body rates |
| Linear MPC alternative | [control/mpc.py](Simulation/control/mpc.py) | Double-integrator acceleration + yaw-rate commands |
| Baseline tracker | [control/tracker.py](Simulation/control/tracker.py) | Position/attitude feedback; not the MPC teacher |
| Episode randomization | [randomization.py](Simulation/randomization.py) | Task, initial state and actual-plant sampling at reset |
| Native 3DGUT adapter | [rendering/threedgut.py](Simulation/rendering/threedgut.py) | Full achieved pose + mount/alignment → RGB/depth |
| Browser UI | [viewer.html](Simulation/viewer.html) | Interactive preview and route editing |
| Native collector | [isaac_sim/native_rollout.py](Simulation/isaac_sim/native_rollout.py) | Finite PhysX validation and synchronized image/action collection |
| Paired dataset writer | [dataset.py](Simulation/dataset.py) | Pre-action PNG + state/reference/expert label |
| Dataset checks | [audit_dataset.py](Simulation/audit_dataset.py) | Timestamps, actions, solver status, checksums and counts |

The active profile is [quadrotor_nmpc.json](Simulation/configs/quadrotor_nmpc.json).

| Configuration section | What to change |
| --- | --- |
| `vehicle` | Mass, inertia, rate/thrust time constants, force/rate/torque limits, custom rotation model |
| `task` | Shape, X/Y dimensions, altitude, duration, vertical drop, heading |
| `randomization` | Per-episode ranges; sampled physical parameters are hidden from the nominal expert model |
| `nmpc` | Horizon, costs and solver iteration limit |
| `rendering` | Checkpoint, camera model, candidate alignment, optional mount override, resolution |

Current defaults: 0.6×0.6 m horizontal figure eight, Z=1.2 m, duration **30 s**,
minimum duration **30 s**, duration randomization **30–33 s**. UI X/Y dimensions
can reach 2 m, altitude 8 m; those UI bounds are not verified reconstructed-room
coverage. `task.speed_m_s` controls the M route; figure-eight speed follows its
dimensions and duration. Local config changes must be transferred before a
remote run uses them.

Current randomization includes route translation/yaw/timing, initial errors,
mass, drag, rate/thrust gain and time constants, and acceleration bias.
Route-size sampling, appearance/noise/lighting augmentation, and scene-geometry
randomization remain unimplemented. More poses do not by themselves implement
appearance randomization. Room obstacles currently affect visuals, not collisions.

## 7. Collection, results and visualization

Use [Simulation/START_HERE.md](Simulation/START_HERE.md) for custom/native launches,
[native collection](Simulation/isaac_sim/NATIVE_BACKEND.md#run-native-isaac-collection)
for full PhysX demonstrations, and [Simulation/RESULTS.md](Simulation/RESULTS.md)
for SFTP, downloading and local visualization. Custom collection remains in
[DATA_COLLECTION.md](Simulation/DATA_COLLECTION.md).

Native output: `data/scene_20261025_200034/isaac_sim/<run>/episode_NNNN/`.
Custom output: `data/scene_20261025_200034/simulation/<run>/`.

```text
Current actual state + reference → MPC command
Current actual state → camera pose → saved RGB
Save (RGB_t, state_t, reference_t, thrust/body-rates_t)
Apply command through rate controller and plant → state_(t+1)
```

Keep `report.json`, effective config, `episode_parameters.json`, metadata,
`rgb_actions.jsonl`, `rgb/`, telemetry and audit results together. Split whole
episodes/seeds for training and evaluation. A browser session or old visualization
bridge alone is not the synchronized training collector.

Confirmed downloaded pilot: `isaac_sim/pilot_20261110T164842Z/`, seeds 1000/1001,
850 paired samples, native PhysX, passing audits, no failed solver labels.
This pilot predates the new 30-second default; retain its actual saved settings.

| Visualization | Script |
| --- | --- |
| Saved RGB + trajectory/attitude/actions video | [visualize_dataset.py](Simulation/visualize_dataset.py) |
| Actual collected randomized trajectories | [plot_collected_trajectories.py](Simulation/plot_collected_trajectories.py) |
| Sampled settings/reference preview, not flights | [plot_randomization.py](Simulation/plot_randomization.py) |
| Actual camera tilt diagnostics | [plot_camera_tilt.py](Simulation/plot_camera_tilt.py) |
| Conceptual architecture/camera diagrams | [plot_dynamics_comparison.py](Simulation/plot_dynamics_comparison.py) |
| Standalone synthetic A→B experiment | [animate_ab_flight.py](Simulation/animate_ab_flight.py) |

The A→B linear-MPC/geometric-controller experiment is separate from the main
dataset mode. `isaac_sim/live_bridge.py` is the older custom-viewer bridge;
`native_backend.py` / `native_rollout.py` own actual native physical motion.
Upstream FiGS/ACADOS is not installed or invoked by this project.

## 8. What is historical, and what remains?

`Workflow/Archive/` preserves the overview documentation before this cleanup;
its old defaults/planned-work statements are historical. Existing immutable run
provenance and scene manifests also describe their creation-time state. Current
launch/configuration instructions are linked above, not inferred from old runs.

Completed: recording extraction, COLMAP, confirmed Gaussian model, fidelity
evaluation, candidate alignment, custom/native simulation, thrust/body-rate
expert, physical/task randomization, paired collection and visualization.

Before claiming a complete sim-to-real training system: implement visual and
additional geometric randomization, build the student loader/network/training
loop, define splits and closed-loop evaluation, independently measure mount/
body frames and hardware dynamics, and validate the deployment command interface.
No trained student or real-drone command bridge exists yet.
