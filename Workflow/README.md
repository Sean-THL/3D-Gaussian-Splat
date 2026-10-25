# Gaussian Splat workflow

1. **Data/** — source rosbags, reusable exporter, extracted images/video/depth/telemetry.
2. **Calibration/** — timestamp association, intrinsics, camera-to-body extrinsics, and frame conventions.
3. **Reconstruction/** — AnySplat baselines and later optimized Gaussian reconstruction.
4. **Evaluation/** — held-out-view image quality, geometry/pose checks, runtime, memory, and comparisons.
5. **IsaacSim/** — version-compatible visual USD assets, aligned collision geometry, robot/sensor validation.

Step 1 is implemented. Step 3 has a mono RGB dataset preparation helper and launchers for externally installed COLMAP and NVIDIA 3DGRUT. Calibration, evaluation, and Isaac Sim integration remain planned. See [Reconstruction/README.md](Reconstruction/README.md) for setup and current validation scope.

Start with [Data/README.md](Data/README.md). Source recordings remain immutable; each extraction creates a new output directory with a manifest.

Use [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md) to export a recording once, select shared views, and make method-specific crop/resize variants without decoding the bag again. Intrinsic transformation formulas, resolution examples, and the current split-adapter limitations are documented there. Existing reconstruction attempts remain valid and untouched.
