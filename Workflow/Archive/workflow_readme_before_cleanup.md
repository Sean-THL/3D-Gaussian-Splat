> Historical snapshot saved before the November 10 organization pass. Use the current root README and PROJECT_MAP.md for current behavior.

# Gaussian Splat workflow

1. **Data/** — source rosbags, reusable exporter, extracted images/video/depth/telemetry.
2. **Calibration/** — timestamp association, intrinsics, camera-to-body extrinsics, and frame conventions.
3. **Reconstruction/** — AnySplat baselines and later optimized Gaussian reconstruction.
4. **Evaluation/** — held-out-view image quality, geometry/pose checks, runtime, memory, and comparisons.
5. **[Simulation/](Simulation/README.md)** — runnable FiGS-style prototype with native 3DGUT rendering, M-route tracking, browser controls, and video/telemetry export. MPC integration remains planned.

Data extraction and dataset selection are implemented. Reconstruction has a mono RGB dataset preparation helper and launchers for externally installed COLMAP and NVIDIA 3DGRUT. Evaluation includes a native 3DGUT depth comparison for the October 25 scene. Calibration includes a candidate COLMAP-to-Vicon trajectory alignment with estimated fixed mounting transform and timing offset; independent physical validation remains outstanding. The simulator prototype is runnable with `Workflow/Simulation/run.sh`; it uses our lightweight model and baseline controller rather than the full upstream FiGS/ACADOS stack. See the workflow guides for usage and limitations.

Start with [Data/README.md](Data/README.md). Source recordings remain immutable; each extraction creates a new output directory with a manifest.

Use [DATASET_WORKFLOW.md](DATASET_WORKFLOW.md) to export a recording once, select shared views, and make method-specific crop/resize variants without decoding the bag again. Intrinsic transformation formulas, resolution examples, and the current split-adapter limitations are documented there. Existing reconstruction attempts remain valid and untouched.
