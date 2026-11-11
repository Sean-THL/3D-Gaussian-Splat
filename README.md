# Gaussian scene reconstruction and drone imitation learning

ROS RGB-D/Vicon recording → COLMAP → native NVIDIA 3DGUT scene → aligned
quadrotor simulation → synchronized RGB/thrust-body-rate demonstrations.

## Find things

| Start here | Contents |
| --- | --- |
| [Project map](Workflow/PROJECT_MAP.md) | Exact recording, COLMAP inputs/outputs, final Gaussian model, Vicon alignment, code ownership and current status |
| [Named scene assets](data/scene_20261025_200034/organized/README.md) | Stage-based shortcuts to the original data, final PLY/checkpoint and result folders |
| [Run the simulators](Workflow/Simulation/START_HERE.md) | Local/custom browser and native Isaac launches |
| [Collect with native Isaac](Workflow/Simulation/isaac_sim/NATIVE_BACKEND.md#run-native-isaac-collection) | Full rigid-body demonstrations and physics checks |
| [Browse/download results](Workflow/Simulation/RESULTS.md) | Remote SFTP, transfer and offline videos/plots |

## Confirmed scene identity

- Recording: `flight_rgbd_20261025_024629`.
- Scene/model: `scene_20261025_200034`.
- COLMAP input: **296 PNG frames extracted directly from the rosbag**, not MP4.
- Final Gaussian export: `organized/03_gaussian_scene/final_scene_gaussians.ply`
  under the scene directory.
- Actual rendering model: the matching `scene_renderer_checkpoint.pt`.
- Calibration: pinned candidate Vicon/3DGS alignment, with estimated camera mount.

The selected downloaded PLY and trainer's final PLY are byte-identical.
The source paths and evidence are in [scene_oct25_assets.json](Workflow/scene_oct25_assets.json).
A Gaussian PLY is not a verified collision mesh, and the current mount/hardware
parameters remain provisional.

## Layout

```text
Workflow/
  Data/             rosbag extraction and frame selection
  Reconstruction/   COLMAP and external 3DGUT training orchestration
  Calibration/      Vicon/3DGS, camera mount and timestamp alignment
  Evaluation/       Gaussian depth and recorded-image fidelity
  Simulation/       plants, controllers, routes, randomization, rendering, data
  Archive/          previous overview documents; historical statements only
  PROJECT_MAP.md    current end-to-end index
  scene_oct25_assets.json
  organize_scene.py

data/scene_20261025_200034/  generated data, models, run records and named shortcuts
AnySplat/                   separate reconstruction baseline submodule
diagnostics/               review notes, figures and demonstration videos
```

## Rebuild the named asset view

```bash
python3 Workflow/organize_scene.py --create-links --verify-hashes
```

This creates relative symlinks; it neither moves source files nor copies large
assets. Missing files are reported on each machine. The scripts and registry are
tracked; raw recordings, images, models, videos and generated links are ignored
by Git and must be copied separately. Do not copy Python environments between
machines. [Data setup](Workflow/Data/README.md) and
[export/select/resize workflow](Workflow/DATASET_WORKFLOW.md) remain available.

For a fresh code clone including the AnySplat baseline:

```bash
git clone --recurse-submodules https://github.com/Sean-THL/3D-Gaussian-Splat.git
```

The current system collects demonstrations; a student network/trainer,
visual appearance randomization and validated real-flight deployment remain
future work. See the project map for precise implementation boundaries.
