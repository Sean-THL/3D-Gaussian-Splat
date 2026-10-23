# 3D Gaussian Splat

ROS bag data preparation, Gaussian reconstruction comparisons, and a workflow toward Isaac Sim integration.

## Project layout

- `Workflow/Data/`: reusable ROS bag extractor, custom message definitions, tests, and usage documentation.
- `Workflow/Calibration/`: planned synchronization and camera/body calibration.
- `Workflow/Reconstruction/`: planned baseline and optimized reconstruction experiments.
- `Workflow/Evaluation/`: planned image/geometry comparisons and resource measurements.
- `Workflow/IsaacSim/`: planned visual assets, collision geometry, and simulation validation.
- `AnySplat/`: existing AnySplat repository, tracked as a submodule pinned to its current commit.
- `diagnostics/`: reconstruction quality notes and small comparison figures.

Start with [Workflow/Data/README.md](Workflow/Data/README.md).

## Clone with AnySplat

```bash
git clone --recurse-submodules https://github.com/Sean-THL/3D-Gaussian-Splat.git
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Commit code changes made inside `AnySplat/` to that repository first, then commit the updated submodule pointer here. The submodule keeps its original history and remote.

## Data and environments

Git tracks scripts, configuration, documentation, message definitions, and small diagnostic figures. ROS bags, extracted datasets, videos, Gaussian PLY files, model weights, and Python environments remain local and are excluded by `.gitignore`. A Git clone does not contain those assets. Copy the required recording separately into `Workflow/Data/rosbags/BAG_NAME/`, then run the extractor.

Create a new Python environment on each computer; do not copy `.venv` between machines. The Data extractor has its own requirements and does not require ROS or a GPU.

## Extract a rosbag

### Setup

From the repository root, create the extractor's Python environment once:

```bash
cd Workflow/Data
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On later runs, activate the existing environment with `source .venv/bin/activate` from `Workflow/Data`. MP4 export also requires `ffmpeg` with the `libx264` encoder on your PATH. On Ubuntu, install it with `sudo apt install ffmpeg` if needed.

Copy the complete ROS 2 bag directory, including `metadata.yaml` and all `.db3` or `.mcap` files, into `Workflow/Data/rosbags/NEW_RECORDING/`. Recordings are excluded from Git, so they must be copied separately after cloning.

### Examples

Run these commands from `Workflow/Data` with its environment activated:

```bash
# Inspect available topics before choosing exports
python extract_rosbag.py --bag rosbags/NEW_RECORDING --list-topics

# Export all RGB pictures
python extract_rosbag.py --bag rosbags/NEW_RECORDING --RGB

# Export RGB video, measured depth, and recorded positions
python extract_rosbag.py --bag rosbags/NEW_RECORDING --MP4 --depth --vicon --position

# Export everything
python extract_rosbag.py --bag rosbags/NEW_RECORDING --all

# Export RGB pictures from 33 to 70 seconds, at up to one image per second
python extract_rosbag.py --bag rosbags/NEW_RECORDING --RGB \
  --start 33 --end 70 --image-rate 1

# Choose a new output directory
python extract_rosbag.py --bag /path/to/another_bag --all --output exports/my_trial

# Show built-in help
python extract_rosbag.py --help
```

`--bag` is optional when exactly one recording exists in `rosbags/`. Combine export flags to request multiple outputs. At least one export flag or `--list-topics` is required.

### Export flags

| Argument | Output |
| --- | --- |
| `--RGB`, `--rgb` | RGB PNG images with a timestamp CSV |
| `--MP4`, `--mp4` | RGB-only H.264 video and a source-frame timestamp CSV |
| `--depth` | Float32 depth arrays in metres (`.npy`), plus native uint16 PNGs when available |
| `--vicon` | Recorded Vicon transforms/poses as CSV and JSONL |
| `--position`, `--positions`, `--PX4` | Recorded PX4 local position, odometry, and external visual odometry |
| `--trajectory` | Commanded trajectories/setpoints |
| `--camera-info` | Recorded camera calibration messages |
| `--tf` | Recorded dynamic and static transforms |
| `--telemetry` | Every non-image topic, including calibration, transforms, status, and ROS logs |
| `--all` | Selected RGB stream as pictures and video, selected depth stream, and every non-image topic |
| `--depth-preview` | Also export colorized depth PNGs; implies `--depth`. Previews are for display, not measurement. |
| `--topic TOPIC` | An additional non-image topic; repeat for multiple topics |
| `--list-topics` | Print topic names, types, and message counts without exporting |

### Input, timing, and format options

| Argument | Meaning / default |
| --- | --- |
| `--bag PATH` | Source bag directory; otherwise use the sole bag in `rosbags/` |
| `-o PATH`, `--output PATH` | New output directory; existing directories are refused |
| `--start SECONDS` | Inclusive start relative to the first bag record; default `0` |
| `--end SECONDS` | Exclusive end relative to the first bag record; default is the recording's end |
| `--image-rate HZ` | Maximum RGB PNG and depth samples per second per stream; default `0` exports all. Does not affect MP4 or telemetry. |
| `--fps FPS` | MP4 output frame rate; default `20`. Bag timing is preserved by holding frames across gaps. |
| `--rgb-topic TOPIC` | Override automatic RGB stream selection |
| `--depth-topic TOPIC` | Override automatic depth stream selection |
| `--depth-scale METRES` | Metres per native uint16 depth unit; default `0.001` for this RealSense setup. Verify for other sensors. |
| `--preview-max-depth METRES` | Colorized depth-preview range; default `10`. Does not clip metric data. |
| `--msg-dir ROOT` | Additional custom definitions under `ROOT/PACKAGE/msg/*.msg`; repeatable, overrides bundled definitions |
| `-h`, `--help` | Show command-line usage |

### Output folders and interpretation

By default, each run creates `Workflow/Data/exports/BAG_NAME/UTC_RUN_TIMESTAMP/` containing only the requested exports:

```text
manifest.json          # Options, topic inventory, file index, counts, completion status
source_metadata.yaml   # Original recording metadata, when available
rgb/                   # PNG pictures and timestamps.csv
video/                 # rgb.mp4 and frames.csv
depth/                 # Metric .npy arrays, native PNGs, timestamps.csv, optional preview/
vicon/                 # Recorded tracked-body poses
position/              # Recorded estimated/external-odometry positions
trajectory/            # Recorded commands
camera_info/           # Calibration messages
tf/                    # Transform messages
telemetry/             # Other non-image messages
```

Non-image topics receive both CSV and JSONL; `manifest.json` maps topic names to filenames. Bag timestamps and sensor-header timestamps are preserved separately. Invalid metric depth is `NaN`; native uint16 PNG depth uses zero for invalid pixels.

RGB, depth, and poses are exported independently: this step does not synchronize streams or convert Vicon body poses into camera poses. A time-window export may omit `/tf_static` or calibration messages recorded before its start. `--all` selects one RGB and one depth stream; use topic overrides and separate runs for additional cameras. Full-rate exports can take several GB; use `--image-rate` or a shorter time window for a small trial.

See [the detailed extractor guide](Workflow/Data/README.md) for depth formats, clock/frame conventions, custom message definitions, and validation instructions.
