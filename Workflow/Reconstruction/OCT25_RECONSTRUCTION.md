# October 25 recording: COLMAP and 3DGUT

This experiment ends at a trained Gaussian checkpoint and PLY reconstruction. The source is `flight_rgbd_20261025_024629`, duration about 133.67 s. RGB is **1280×720**, on `/camera_full/color/image_raw/compressed`. The matching full-resolution calibration is `/camera/camera/color/camera_info`; `/camera_448/...` calibration belongs to a different crop and must not be used here.

The complete bag has been copied locally into `Workflow/Data/rosbags/flight_rgbd_20261025_024629/` and verified byte-identical. A preview was generated at `Workflow/Data/exports/flight_rgbd_20261025_024629/preview/`. The local first dataset uses 12–130 seconds, up to three images per second, and fixed recorded intrinsics. This includes most of the motion and removes the initial stationary portion. Sampling and this interval are starting choices, not automatic quality/coverage certification.

## Transfer to the 4090 computer

Inside your authenticated **remote terminal**:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
git pull --ff-only
mkdir -p Workflow/Data/rosbags
```

In a **new terminal on the laptop containing the recording**, outside SSH:

```bash
rsync -avh --progress \
  /home/robin/Seans_Project/DASC/Gaussian_Splat/Workflow/Data/rosbags/flight_rgbd_20261025_024629 \
  tk@141.212.76.61:~/Seans_Project/DASC/3D-Gaussian-Splat/Workflow/Data/rosbags/
```

The source path is on the laptop. Running that command inside the remote SSH session will look for the source on the wrong computer. Do not transfer `.venv`, previous runs, or the complete extracted dataset; reconstruct those on the remote host.

## Prepare the native RGB dataset

Back in the **remote terminal**, at the repository root:

```bash
Workflow/Data/.venv/bin/python -c "import cv2, rosbags; print('Extractor ready')"

python3 Workflow/Reconstruction/mono_workflow.py prepare \
  --bag Workflow/Data/rosbags/flight_rgbd_20261025_024629 \
  --name flight_20261025_native \
  --start 12 --end 130 --rate 3 \
  --calibrated \
  --rgb-topic /camera_full/color/image_raw/compressed \
  --camera-info-topic /camera/camera/color/camera_info
```

If the extractor environment is missing, follow the setup in [Data/README.md](../Data/README.md). Existing environments from the previous recording can be reused. Review the dataset's `images/` before reconstruction and remove obviously blurred frames from that working copy only. Use a new `--name` for another trial; existing datasets are refused.

`calibration.json` must report width 1280, height 720, fx approximately 643.64, and fy approximately 642.93 pixels. It preserves all five distortion coefficients and converts the principal-point convention for COLMAP. The full-resolution and 448 camera_info messages share a frame ID; dimensions and the explicit topic selection distinguish the correct calibration.

## Run COLMAP

```bash
python3 Workflow/Reconstruction/mono_workflow.py colmap \
  --dataset Workflow/Reconstruction/datasets/flight_20261025_native
```

CPU feature processing is the default; the 4090 is used later by 3DGUT. Wait for the final result. The helper stops on disconnected or poorly registered models. If it stops, retain the outputs and inspect them rather than repeatedly rerunning or reducing quality thresholds:

```bash
python3 Workflow/Reconstruction/mono_workflow.py diagnose \
  --dataset Workflow/Reconstruction/datasets/flight_20261025_native
```

## Train the Gaussian reconstruction

After COLMAP succeeds, inspect its sparse geometry. Install `~/tools/3dgrut` and its separate environment using [Reconstruction/README.md, step 5](README.md#5-install-nvidias-tool-separately) if not already installed. Then, from this project's root:

```bash
conda activate 3dgrut

python Workflow/Reconstruction/mono_workflow.py train \
  --dataset Workflow/Reconstruction/datasets/flight_20261025_native \
  --tool-dir ~/tools/3dgrut \
  --max-gaussians 500000 --workers 2
```

Monitor GPU memory from another remote terminal with `watch -n 1 nvidia-smi`. Training outputs are under `Workflow/Reconstruction/runs/flight_20261025_native_TIMESTAMP/`; `provenance.json` records exact software revisions and commands, and `training.log` captures progress. NVIDIA's model subdirectory contains its checkpoint, configuration, and enabled PLY export.

Training this RGB baseline does not establish metric scale. Measured depth/Vicon are preserved in the original bag and can support later geometry/pose validation. Full-room success still depends on the new capture's coverage and view consistency.
