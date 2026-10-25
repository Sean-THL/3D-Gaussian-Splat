# Extract data from a rosbag

For multiple reconstruction methods, use [the reusable dataset workflow](../DATASET_WORKFLOW.md): export native data once, then run `dataset_tools.py select` and `dataset_tools.py variant`. The new tool adjusts intrinsics with resizing/cropping, optionally associates aligned measured depth with explicit timestamp tolerances, and records frame IDs/reference splits. It uses this folder's existing `.venv` and requirements.

This folder contains the copied flight bag, `extract_rosbag.py`, and the message definitions needed to decode its PX4 topics. The original `Gaussian_Splat/rosbag` is unchanged.

The extractor streams messages on the CPU. No ROS installation, GPU, ROS replay, or AnySplat environment is required. It reads ROS 2 SQLite/MCAP bags supported by Rosbags; provide the complete bag directory, including metadata and all split files. This recording's SQLite format was tested.

The included bag has already been fully exported to `exports/flight_rgbd_20261013_202629/full_export/` (about 3.4 GB): **2,993 RGB images, 2,992 depth frames, a 134.75-second RGB video, and every recorded non-image stream**. `validation.json` records the successful checks: all topic counts match, CSV/JSONL counts agree, first/middle/last image and depth samples match an independent ROS decoder, depth units/header timestamps are preserved, and the full video decodes successfully. The copied bag was verified byte-identical to the original.

## Setup

On this machine the `.venv` is already installed and tested. From this directory:

```bash
cd /home/robin/Seans_Project/DASC/Gaussian_Splat/Workflow/Data
source .venv/bin/activate
python extract_rosbag.py --help
```

On another computer, create a fresh environment rather than transferring `.venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

MP4 export additionally requires `ffmpeg` with the `libx264` encoder on PATH. On Ubuntu it is supplied by the `ffmpeg` package. PNG, depth, and telemetry export do not need FFmpeg.

## Choose what to export

```bash
# Inventory of every recorded topic, type, and message count
python extract_rosbag.py --list-topics

# All RGB pictures, retaining original recorded resolution
python extract_rosbag.py --RGB

# RGB pictures sampled at up to 1 per second
python extract_rosbag.py --RGB --image-rate 1

# Full-rate RGB video, independent of PNG sampling
python extract_rosbag.py --MP4

# Metric measured depth plus native uint16 depth PNGs
python extract_rosbag.py --depth

# Vicon poses and PX4 estimated/external-odometry positions
python extract_rosbag.py --vicon --position

# Commands, calibration, and transforms
python extract_rosbag.py --trajectory --camera-info --tf

# All non-image messages, including status, logs, calibration, and transforms
python extract_rosbag.py --telemetry

# All data; colorized depth previews are optional
python extract_rosbag.py --all --depth-preview

# Small trial, times in seconds relative to first BAG record
python extract_rosbag.py --all --start 33 --end 43 --image-rate 2

# Additional non-image topics, e.g. a future IMU
python extract_rosbag.py --topic /imu/data
```

`--RGB` / `--rgb` and `--MP4` / `--mp4` are aliases. `--position`, `--positions`, and `--PX4` are aliases. Flags can be combined. With no export flag, the script shows an error and usage rather than exporting unexpectedly.

## New recordings

Copy the **whole bag directory** into `rosbags/`, or pass its existing path directly:

```bash
python extract_rosbag.py --bag rosbags/NEW_RECORDING --RGB --MP4 --depth --vicon --position
```

If there is exactly one bag in `rosbags/`, `--bag` is optional. If multiple bags exist, choose one explicitly. Default output is `exports/BAG_NAME/UTC_RUN_TIMESTAMP/`; the existing bag uses `flight_rgbd_20261013_202629` as its name.

To choose an output folder:

```bash
python extract_rosbag.py --bag rosbags/NEW_RECORDING --all --output exports/my_trial
```

An existing output directory is refused to prevent mixing recordings or overwriting work. Choose a new directory for each run. A failed run retains its files and a manifest with `status: failed`; these outputs should not be treated as a complete dataset.

RGB and depth streams are selected from topic names/types, preferring color/RGB and aligned depth. If several exist, alternatives are printed. Inspect the inventory and override as needed:

```bash
python extract_rosbag.py --bag /path/to/bag --RGB --depth \
  --rgb-topic /camera/color/image_raw/compressed \
  --depth-topic /camera/aligned_depth_to_color/image_raw/compressedDepth
```

Only one selected RGB stream and one selected depth stream are extracted per run. `--all` means these image streams plus every non-image topic; it does not extract every camera in a multi-camera recording. Choose other streams in separate runs.

For other custom message packages, place matching definitions at `message_definitions/PACKAGE/msg/Type.msg` or pass `--msg-dir /another/definitions_root`. Extra directories override bundled definitions. Use the message version that recorded the bag, especially for PX4; a mismatched schema can fail or decode misleading values. Definitions embedded in newer bag formats are handled by Rosbags.

## Output contents

| Directory/file | Contents |
| --- | --- |
| `manifest.json` | Source path, full topic inventory, options, chosen streams, counts, file index, conventions, completion status |
| `source_metadata.yaml` | Original bag metadata copy |
| `rgb/*.png` | Decoded RGB frames without additional lossy image encoding |
| `rgb/timestamps.csv` | Filenames, bag/header timestamps, relative time, dimensions |
| `video/rgb.mp4` | RGB-only H.264 MP4; default 20 FPS |
| `video/frames.csv` | Video frame time and the actual source image bag/header timestamps |
| `depth/*.npy` | Float32 depth in **metres**, NaN invalid |
| `depth/*.png` | Native uint16 depth, when source is 16UC1; zero invalid |
| `depth/timestamps.csv` | File associations, timestamps, source encoding, depth scale, valid fraction |
| `depth/preview/*.png` | Optional colorized display images, **not metric depth** |
| `vicon/` | Recorded tracked-body transforms/poses |
| `position/` | PX4 estimated local position and external visual odometry |
| `trajectory/` | Commanded trajectory/setpoint data |
| `camera_info/` | All recorded calibration updates, including K/D/R/P, crop/binning, frame ID |
| `tf/` | Recorded dynamic and static transform messages |
| `telemetry/` | Other recorded topics such as status and ROS logs |

Non-image topics receive both CSV and JSONL. JSONL retains the complete message hierarchy; CSV flattens nested objects and keeps arrays as JSON cells. File names escape topic punctuation so distinct topic names cannot collide; use `manifest.json -> files` to map names to topics. Columns such as `transform.translation.x` and `transform.rotation.w` are original ROS fields. PX4 local-position CSV contains original `x`, `y`, `z`, velocities, validity flags, estimator-reset counters, and other available fields. These are **recorded estimates**, not a live telemetry stream.

## Units, timestamps, and limitations

- `bag_timestamp_ns` is the recording timestamp, exactly preserved as an integer. `bag_elapsed_s` is relative to the first record in the bag, not the first RGB frame.
- `header_timestamp_ns` preserves the sensor/ROS header clock separately; a header value of zero is retained. PX4 `timestamp` and `timestamp_sample` are kept unchanged (typically microseconds since boot). Do not treat these clocks as interchangeable without checking synchronization.
- Vicon poses describe the tracked body. PX4 local position uses its recorded local convention (normally NED). No camera extrinsic calibration, frame conversion, camera-pose generation, or coordinate alignment is applied.
- RGB, depth, calibration, and poses are exported independently. The timestamps enable later synchronization, but this step does **not** silently pair or interpolate them. A time-window export includes only messages inside that window; it may omit earlier `/tf_static` and calibration messages. Export full `--tf --camera-info` separately when needed.
- uint16 depth uses `--depth-scale 0.001` metres per unit by default, matching this RealSense setup. Verify the scale for a new sensor. 32FC1 depth is in metres. Supported compressed depth is PNG (16UC1 or inverse-depth 32FC1); RVL is rejected explicitly. Raw depth supports 16UC1 and 32FC1, including padded rows and either byte order.
- Depth previews use a fixed 0–10 m display range by default (`--preview-max-depth`). Display saturation does not clip metric arrays.
- `--image-rate` limits RGB PNG and depth sampling independently; it never changes video sampling or telemetry rates. Zero means all recorded image messages. The MP4 uses bag timing and holds the previous frame across gaps; it does not shorten pauses. Compression can slightly change colors; use RGB PNGs for reconstruction.
- No blur/glare filtering, resizing, denoising, pose estimation, or reconstruction is performed. These are separate downstream choices. Bad measurements remain visible in the raw export.
- Non-finite telemetry floats become JSON `null` / empty CSV cells. The original bag remains the lossless source. Empty topics and topics with no messages in the selected window are listed in the manifest rather than fabricated.
- Full-resolution/full-rate extraction can consume several GB. For a first look, use `--image-rate 1` or a short `--start/--end` window; retain the rosbag for future full-rate exports.

## Verify

```bash
python -m unittest -v test_extract_rosbag.py
```

Tests cover native and inverse compressed depth, invalid depth, raw image padding/endianness, RGB channel order, unsupported depth codecs, and timestamp-driven video resampling. The extractor was also exercised against the included real recording. This does not certify calibration or pose accuracy.

Implementation references: [Rosbags reader](https://ternaris.gitlab.io/rosbags/topics/highlevel.html), [custom definitions](https://ternaris.gitlab.io/rosbags/examples/register_types.html), [ROS compressed-depth codec](https://github.com/ros-perception/image_transport_plugins/blob/rolling/compressed_depth_image_transport/src/codec.cpp).
