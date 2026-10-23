# Review of the 33–70 second reconstruction

Inspected the original RGB recording, rosbag Vicon transforms, local demo source, and the supplied outputs in `AnySplats- V1-result` (`rgb.mp4`, `depth.mp4`, and the PLY header). No new reconstruction was run.

## Observations

- `input_33_70.jpg` samples original timestamps 33, 37, 41, 45, 49, 53, 57, 61, 65, 69 seconds. `reconstruction_rgb_depth.jpg` samples the output videos at 5%, 25%, 50%, 75%, and 95%. Rendered-video timestamps are not original recording timestamps.
- The output contains doubled/transparent room structures, stretched wall features, and irregular surfaces. These are consistent with inaccurate geometry or alignment; images alone cannot establish whether camera-pose error or depth error dominates.
- The camera repeatedly changes its view across walls, windows, floor, and upper room. Vicon tracked-object Z rises from approximately 0.78 m at 33 s to 8.33 m at 69 s. Over 33–70 s, XYZ ranges are approximately 0.95, 0.90, 7.67 m. These are tracked-object positions, not calibrated camera poses. This is substantial translation, mostly vertical; it is not a stationary-camera sequence. Many surfaces receive limited repeated coverage.
- Around 53 s, strong direct-window glare obscures scene detail. The 49–57 s neighborhood contains glare, relatively featureless walls, and thin cables/netting. Window reflections and repetitive blocks/netting provide ambiguous correspondence cues.
- Input RGB is already 448×448. The model preprocessing also uses 448×448. More GPU memory cannot recover detail absent from the recording. Video re-encoding adds compression, but the observed glare and coverage problems are already in the source footage.
- The demo samples video at 1 frame/second and uses all sampled frames. It has no blur, glare, overlap, or viewpoint quality selection.
- The demo predicts camera poses and Gaussians from RGB through `model.inference`. It does not ingest bag depth, camera_info, or Vicon transforms, and it does not run scene-specific post-optimization. The output depth video is rendered from the reconstructed Gaussians; it is not the recorded RealSense depth and is not independent validation of geometry.
- The RGB output interpolates predicted camera extrinsics between sampled views. Artifacts between observations can therefore be worse than reconstructions at actual input viewpoints. Evaluate both before attributing every issue to the PLY export or viewer.

## Recommended next experiment

Use 8–12 clear, overlapping frames of one small area from a short continuous part of the flight. Extract original compressed RGB messages directly to PNG to avoid additional video encoding. Select for shared scene content and a change of viewpoint; reject glare, strong blur, and near-duplicates. The objective is a local reconstruction sanity check, not full room coverage. No specific interval is guaranteed to work without reviewing its individual frames.

If the local test is successful, reconstruct overlapping sections or use a pipeline that supports scene optimization and calibrated camera poses. Integrating Vicon requires timestamp association and camera-to-body extrinsic calibration. Integrating recorded depth requires depth decoding, alignment, units, and invalid-pixel handling; the existing demo does not provide these steps automatically.

The supplied images support this diagnosis but do not prove the relative contribution of each factor. Confirming pose or scale errors requires exported predicted poses or an instrumented rerun and comparison with calibrated reference poses.
