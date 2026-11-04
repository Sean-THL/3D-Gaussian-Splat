# Calibration and synchronization

Planned inputs: Data exports, RGB/depth camera_info, TF, image timestamps, Vicon tracked-body transforms, and camera-to-body calibration.

Planned outputs: time associations with measured residuals, validated camera intrinsics/depth units, camera extrinsics, world-frame conventions, and metric camera poses. Recorded Vicon/PX4 body poses must not be treated as camera optical poses without the appropriate calibration.

Image-plane calibration transformations are now implemented by `Data/dataset_tools.py variant`: crop/resize adjusts focal lengths and principal point, preserves normalized distortion coefficients, and applies the same image operation to already-aligned depth. Read [the workflow and intrinsics explanation](../DATASET_WORKFLOW.md#intrinsics-when-changing-resolution).

Nearest-time association of aligned depth is available during dataset selection, with explicit clock and tolerance. It reports gaps; it does not estimate clock offsets, geometrically align depth, interpolate poses, or calibrate camera-to-body extrinsics. The scene-specific trajectory alignment below implements RGB/Vicon pose association and an estimated fixed mounting transform. General sensor calibration remains planned.

## October 25 scene alignment

`align_vicon.py` estimates a fixed similarity transform between the original COLMAP/3DGUT world and `vicon/world`. It jointly estimates the camera-to-Vicon-marker mounting transform and one constant sensor-header timestamp offset. The user confirmed that the RealSense stayed rigidly mounted for this recording.

The recorded `CameraInfo` messages contain lens calibration. Recorded TF contains the RealSense's internal links and the Vicon marker-body pose, but no link between the marker body and `camera_link`. Therefore the mounting transform below is **estimated from motion**, not loaded from the bag. Its parent is `vicon/px4_201/px4_201` and its child is `camera_color_optical_frame`. Do not assume the marker body's axes are the same as PX4's body axes.

Run from the repository root with NumPy, SciPy, and Matplotlib installed; CUDA is unnecessary:

```bash
python3 Workflow/Calibration/align_vicon.py
python3 -m unittest Workflow.Calibration.test_align_vicon -v
```

Defaults reference the scene's parsed COLMAP camera-pose JSON, original RGB timestamp CSV, and full-export Vicon JSONL. Command-line options override scene, timestamps, Vicon, or output paths. Source files and original Gaussians are never modified. Each run writes a new directory under `data/scene_20261025_200034/calibration/`; `latest_alignment.json` points to the latest candidate.

### Coordinate mapping

Let G denote the original Gaussian world, V the recorded Vicon world, B the tracked marker body, and C the RGB optical camera. Rotation `R_AB` maps coordinates from B into A. Translation and tracked-body positions below are in metres.

```text
p_VC = p_VB + R_VB t_BC
R_VC = R_VB R_BC

p_VC = s R_VG p_GC + t_VG

p_GC = (1/s) R_VG^T (p_VB + R_VB t_BC - t_VG)
R_GC = R_VG^T R_VB R_BC
```

The saved `gs_to_vicon.position_matrix_4x4` and inverse matrix transform **positions**, including scale. Use the separately saved rotation matrices for orientations; do not treat the scaled 3×3 block as a rotation. `camera_to_tracked_body` contains `R_BC` and `t_BC`.

The helper `vicon_body_to_gs(body_position, body_rotation, calibration)` returns optical-camera position and orientation in the original Gaussian world. `body_rotation` is a SciPy `Rotation` representing the recorded marker-body-to-Vicon-world orientation. SciPy quaternion order is `[x, y, z, w]`. For example, after loading the saved alignment JSON:

```python
from scipy.spatial.transform import Rotation
from Workflow.Calibration.align_vicon import vicon_body_to_gs

camera_position_gs, camera_rotation_gs = vicon_body_to_gs(
    body_position=position_xyz_metres,
    body_rotation=Rotation.from_quat(vicon_quaternion_xyzw),
    calibration=alignment,
)
```

This maps the **camera** pose required by rendering. A vehicle reference point used by a task planner may need a separate marker-to-vehicle transform.

### Timing and validation

RGB timestamps come from `flight_20261025_native/source_export/rgb/timestamps.csv`, matched by image name to the sorted COLMAP poses. Vicon positions are linearly interpolated and orientations use SLERP. The fitted convention is `t_vicon = t_rgb_sensor_header + time_offset_s`. No extrapolation is allowed. The offset is specific to the recording; verify clocks and delays again for online operation.

The script tolerates and sorts latency-corrected Vicon timestamp reversals smaller than 1 ms, counts duplicate headers, and stops for larger reversals or gaps over 100 ms. This recording has six reversals no larger than 0.103 ms, and a maximum gap of approximately 71 ms.

Position residuals are weighted at 0.02 m and orientation residuals at 1 degree, using a soft-L1 loss. Three fits hold out alternating 10-second blocks (block index modulo three). Each view receives a held-out residual. A final fit uses all 296 poses to save the candidate transform. Validation uses the same capture, and COLMAP originally used all images for pose estimation.

### Current result and limitations

Completed run: `data/scene_20261025_200034/calibration/alignment_20261103T220923_264755Z/`.

| Quantity | Result |
| --- | --- |
| Scale | 0.754179 metres per reconstruction unit |
| RGB-to-Vicon header-time offset | -0.006260 s |
| Held-out median camera-position difference | 0.00443 m |
| Held-out 90th-percentile position difference | 0.00907 m |
| Held-out median orientation difference | 0.367 degrees |
| Estimated camera origin in marker-body coordinates | approximately (0.0440, -0.0049, 0.0974) m |
| Vertical mount translation spread across three fits | 0.0682 m |

The scale agrees closely with the independently computed training-depth estimate of 0.754573 m/unit. The small trajectory residuals support a common coordinate mapping for this recording. They **do not** imply millimetre-accurate Gaussian surfaces or a physically validated mounting transform. Predominantly vertical motion and yaw leave mounting height coupled to world translation; independent mount/room measurements are still needed. The vertical mount spread is a sensitivity diagnostic, not a formal uncertainty interval.

Artifacts: `alignment.json` contains forward/inverse position matrices, rotations, mounting estimate, timing convention, source hashes, assumptions, and fold results; `heldout_errors.csv` contains per-view validation errors; `alignment_overview.png` overlays trajectories and plots held-out errors. Treat the JSON status `candidate_alignment_mount_estimated` as deliberate until independently checked.
