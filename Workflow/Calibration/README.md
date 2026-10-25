# Calibration and synchronization

Planned inputs: Data exports, RGB/depth camera_info, TF, image timestamps, Vicon tracked-body transforms, and camera-to-body calibration.

Planned outputs: time associations with measured residuals, validated camera intrinsics/depth units, camera extrinsics, world-frame conventions, and metric camera poses. Recorded Vicon/PX4 body poses must not be treated as camera optical poses without the appropriate calibration.

Image-plane calibration transformations are now implemented by `Data/dataset_tools.py variant`: crop/resize adjusts focal lengths and principal point, preserves normalized distortion coefficients, and applies the same image operation to already-aligned depth. Read [the workflow and intrinsics explanation](../DATASET_WORKFLOW.md#intrinsics-when-changing-resolution).

Nearest-time association of aligned depth is available during dataset selection, with explicit clock and tolerance. It reports gaps; it does not estimate clock offsets, geometrically align depth, interpolate poses, or calibrate camera-to-body extrinsics. The remaining pose/sensor synchronization work above is not implemented yet.
