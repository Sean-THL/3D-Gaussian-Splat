# Mono RGB baseline: COLMAP → 3DGUT

Keep our workflow code in this repository and NVIDIA's software separately at `~/tools/3dgrut`. Datasets, caches, logs, and training runs are excluded from Git. `mono_workflow.py` provides `prepare`, `colmap`, and `train`; it does not install software or modify NVIDIA's code.

## 1. Get this project on the remote computer

In your authenticated `tk@ROB-DASC-06U` terminal, clone the repository if not already cloned:

```bash
git clone --recurse-submodules https://github.com/Sean-THL/3D-Gaussian-Splat.git \
  ~/Seans_Project/DASC/3D-Gaussian-Splat
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
```

For an existing clone, enter it and run `git pull --ff-only` instead. An earlier transferred folder without `.git` is not a clone; leave it intact and clone into the separate directory above. Run the preparation/COLMAP commands below from this project's root.

## 2. Install preparation tools

```bash
sudo apt update
sudo apt install -y colmap ffmpeg python3-venv build-essential gcc-11 g++-11 libgl1-mesa-dev
/usr/bin/python3 -m venv Workflow/Data/.venv
Workflow/Data/.venv/bin/python -m pip install -r Workflow/Data/requirements.txt
```

## 3. Prepare pictures from the existing bag

```bash
python3 Workflow/Reconstruction/mono_workflow.py prepare \
  --bag ~/Seans_Project/DASC/Gaussian_Splat/rosbag \
  --name flight_33_70 \
  --start 33 --end 70 --rate 2
```

This reads the bag from its existing location; another bag copy is unnecessary. It creates `Workflow/Reconstruction/datasets/flight_33_70/` with `images/`, an intact `source_export/` including timestamps and manifest, `extraction.log`, and `preparation.json`.

### Calibrated retry for this RealSense bag

The original baseline estimates intrinsics. If it fails, compare a separate run using the actual cropped-image calibration, keeping the failed run intact:

```bash
python3 Workflow/Reconstruction/mono_workflow.py prepare \
  --bag ~/Seans_Project/DASC/Gaussian_Splat/rosbag \
  --name flight_33_70_calibrated \
  --start 33 --end 70 --rate 2 --calibrated

python3 Workflow/Reconstruction/mono_workflow.py colmap \
  --dataset Workflow/Reconstruction/datasets/flight_33_70_calibrated
```

`--calibrated` exports camera_info and matches its topic to the selected RGB stream. It requires constant, valid calibration matching the PNG dimensions and writes `calibration.json`. COLMAP then uses the recorded intrinsics/distortion and disables intrinsic refinement. ROS/OpenCV principal points are shifted by +0.5 pixels for COLMAP's pixel convention. FULL_OPENCV retains all five plumb_bob coefficients (including k3), with unused denominator terms zero; rational_polynomial with eight coefficients is also supported. Binning/ROI, changing calibration, and unsupported distortion models are rejected rather than silently approximated. Crop-adjusted camera_info from this recording already has a zero ROI.

The included recording has approximately fx=385.97, fy=385.55 pixels at 448×448. The calibrated retry still estimates camera poses from RGB; it does not use Vicon or measured depth, nor guarantee reconstruction success. Initialization also needs valid geometry and sufficient viewpoint overlap. Calibration is an additional input and should be identified when comparing methods.

To inspect a failed run without changing it:

```bash
python3 Workflow/Reconstruction/mono_workflow.py diagnose \
  --dataset Workflow/Reconstruction/datasets/flight_33_70
```

This reports stored camera parameters, connected-component sizes (including isolated images) for verified-pair thresholds of 15, 30, and 100 matches, numeric two-view configuration counts, and recent mapper events. Large pairwise match counts alone do not establish that all views connect or have enough parallax. Thresholds here are diagnostic summaries, not changes to COLMAP's mapping acceptance criteria.

Review `images/` for blur and glare; remove rejected pictures from that dataset copy only. No automatic quality selection is performed. This interval is a comparison baseline, not a guarantee of suitable coverage. Times are relative to the first bag record; `--rate` is the maximum samples/second, with zero exporting every RGB frame. Use another `--name` for each new trial; existing directories are refused. The default extractor Python is `Workflow/Data/.venv/bin/python`; override with `--extractor-python PATH` if needed.

## 4. Run COLMAP

```bash
python3 Workflow/Reconstruction/mono_workflow.py colmap \
  --dataset Workflow/Reconstruction/datasets/flight_33_70
```

CPU feature extraction/matching is the default for SSH use. Add `--gpu` if your COLMAP installation supports it. The helper detects older SiftExtraction/SiftMatching and newer FeatureExtraction/FeatureMatching flag names.

It runs feature extraction, exhaustive matching, sparse mapping, model analysis, and image undistortion. Camera estimation uses OPENCV distortion; training receives undistorted images. It stops if the reconstruction has multiple disconnected models or fewer than 70% of input images register. `--min-registered-fraction` changes that screening threshold, but investigate failures rather than simply lowering it.

Inspect the sparse model under `datasets/flight_33_70/colmap/sparse/` and `colmap/model_analyzer.log` before training. Registration count alone does not validate geometry. Commands, counts, and completion status are recorded in `colmap/provenance.json`. Successful training input is `datasets/flight_33_70/undistorted/`.

The COLMAP stage refuses an existing `colmap/` directory to prevent stale state from being reused. After a failure, fix the inputs and prepare a new dataset name. Automatic resume/model selection is not implemented.

## 5. Install NVIDIA's tool separately

In a terminal with Conda initialized, run once:

```bash
mkdir -p ~/tools
git clone --recursive https://github.com/nv-tlabs/3dgrut.git ~/tools/3dgrut
cd ~/tools/3dgrut

CUDA_VERSION=11.8 bash scripts/create_conda.sh 3dgrut
conda activate 3dgrut
python -m pip install uv
MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 bash install_env_uv.sh

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The current upstream uses these scripts rather than the older tutorial's `install_env.sh`. CUDA 11.8 is supported; the host driver must support that runtime. Stop on installation errors. Build limits reduce RAM pressure without guaranteeing fit on every machine. This environment is separate from AnySplat.

If the checkout already exists, inspect it instead of cloning over it or updating its version mid-experiment.

## 6. Train with provenance

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
conda activate 3dgrut

python Workflow/Reconstruction/mono_workflow.py train \
  --dataset Workflow/Reconstruction/datasets/flight_33_70 \
  --tool-dir ~/tools/3dgrut \
  --max-gaussians 500000 \
  --workers 2
```

Training uses the active Python environment. The helper checks CUDA access and successful COLMAP completion, then launches `apps/colmap_3dgut_mcmc.yaml` with PLY export. Each run creates a new timestamped folder under `Workflow/Reconstruction/runs/` containing:

- `provenance.json`: exact command, external Git commit/submodule revisions and worktree status, project revision/status, available package versions, Python, GPU/driver, CUDA compiler, and COLMAP provenance.
- `training.log`: console output.
- `model/`: NVIDIA's outputs, configuration, and checkpoints in its experiment subdirectory.

Keep the trainer's effective configuration with provenance. A dirty external checkout is recorded but not copied; commit external modifications separately before relying on the SHA for reproducibility. Run AnySplat, Gaussian training, and Isaac Sim separately. Monitor memory with `watch -n 1 nvidia-smi`; limits are starting choices, not an out-of-memory guarantee.

## Scope and checks

This is an RGB-only baseline. Metric scale, depth/Vicon synchronization, camera-to-body calibration, collision geometry, and version-specific Isaac export remain later work.

Both uncalibrated and calibrated preparation were exercised locally on the real bag. The calibrated interval contained 805 identical RGB calibration messages matching its 71 images. COLMAP and GPU training require remote tools and have not been run locally. Compatibility/error checks:

```bash
python3 -m unittest discover -s Workflow/Reconstruction -p 'test_*.py' -v
python3 Workflow/Reconstruction/mono_workflow.py --help
```

References: [NVIDIA mono guide](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html), [COLMAP CLI](https://colmap.github.io/cli.html), [COLMAP calibration conventions](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools), [NVIDIA installation](https://github.com/nv-tlabs/3dgrut#-1-dependencies-and-installation), [MCMC configuration](https://github.com/nv-tlabs/3dgrut/blob/main/configs/strategy/mcmc.yaml).
