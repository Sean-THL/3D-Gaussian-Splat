# Gaussian Splat workflow

1. **Data/** — source rosbags, reusable exporter, extracted images/video/depth/telemetry.
2. **Calibration/** — timestamp association, intrinsics, camera-to-body extrinsics, and frame conventions.
3. **Reconstruction/** — AnySplat baselines and later optimized Gaussian reconstruction.
4. **Evaluation/** — held-out-view image quality, geometry/pose checks, runtime, memory, and comparisons.
5. **IsaacSim/** — version-compatible visual USD assets, aligned collision geometry, robot/sensor validation.

Only step 1 is implemented here. Each later folder contains its intended inputs and outputs.

Start with [Data/README.md](Data/README.md). Source recordings remain immutable; each extraction creates a new output directory with a manifest.
