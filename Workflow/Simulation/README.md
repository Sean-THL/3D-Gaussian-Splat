# Gaussian flight simulator

A runnable local simulator couples a FiGS-style lightweight quadrotor model to the existing native 3DGUT renderer. It provides a browser camera view, RGB/depth toggle, M-route tracking, telemetry, route settings, hover/nudge controls, and a disturbance button. Measured camera offsets can replace the candidate mount later.

This is our prototype implementation of the lightweight approach. It does not install or invoke upstream FiGS/ACADOS. The first controller is a baseline geometric feedback tracker; MPC and student training remain future work. Isaac Sim is not required.

A native Isaac Sim backend is the next requested direction. Its [host inspection and step-by-step build guide](isaac_sim/README.md) are separate from this already runnable prototype. Remote Isaac Sim installation/version are not yet verified.

## Start here

From the repository root:

```bash
Workflow/Simulation/run.sh
```

Open **http://127.0.0.1:8765**. The simulator starts paused. Press **Play** to run the reference, or **Fly M route** to restart and run it. **Pause** freezes simulation time. **Reset** returns to the initial paused state. **Hold position** switches to hover; the XYZ nudge buttons move that hover target by 10 cm. **Wind pulse** applies a repeatable velocity disturbance. **Download telemetry** exports the current episode's state/reference/control CSV.

The default M route is 0.6×0.6 m at Z=1.2 m, with fixed yaw 0 degrees and peak speed 0.3 m/s. It is a provisional small route near the capture trajectory, not a surveyed flight task. Route settings can be changed in the browser; **Apply & reset** starts a new paused episode. The route slows to zero speed at each corner using a quintic trajectory with continuous position, velocity, and acceleration.

The script activates `~/tools/3dgrut/.venv` when present, which supplies CUDA and Slang compiler paths. On the training laptop it falls back to `conda run -n 3dgrut`. Native 3DGUT may compile its cached extension on the first run. In environments where the sandbox hides NVIDIA devices or prevents localhost binding, launch the command with host execution. Ctrl+C stops the server and saves current telemetry. `--port 8766` chooses another port. `SIMULATION_ENV=/path/to/environment` overrides the environment location.

## Run and export a complete rollout

```bash
Workflow/Simulation/run.sh --headless --record
```

This runs the full route plus two seconds of settling and writes a camera/map MP4, telemetry CSV, effective configuration, provenance, last camera frame, and tracking summary. `--duration 5` selects a shorter rollout. Output folders are timestamped under `data/scene_20261025_200034/simulation/`; nothing is copied from the large input datasets. `--output PATH` selects a new output folder, and existing output folders are refused.

For CPU-only dynamics, with NumPy, SciPy, and Pillow available:

```bash
python3 Workflow/Simulation/run.py --no-render --headless
```

`--no-render` disables Gaussian rendering explicitly. It can also run the browser viewer for dynamics/route controls, with a labeled placeholder instead of a scene image. FFmpeg is needed only for `--record`.

## Where to find the code

| File | Purpose |
| --- | --- |
| `run.py` | CLI, local HTTP viewer, simulation loop, telemetry and video export |
| `viewer.html` | Live camera, route map, controls, and settings |
| `configs/demo.json` | Scene paths, alignment, vehicle, camera, timing, controller, and M-route defaults |
| `figs/dynamics.py` | Prototype 10-state thrust/body-rate dynamics |
| `figs/simulator.py` | Deterministic stepping, episode control, references, and telemetry |
| `tasks/trajectory.py` | Quintic M-route trajectory |
| `control/tracker.py` | Baseline position/attitude feedback controller |
| `rendering/threedgut.py` | Native renderer adapter for moving camera poses |
| `test_simulation.py` | Physics, trajectory limits, tracking recovery, and control-state tests |

Upstream 3DGUT remains at `~/tools/3dgrut`, separate from our adapters. The model's recorded upstream revision is `e6f4552b408af020a96de09ab5757f9b57430aaa`. Upstream FiGS may be integrated later with a pinned revision; the SOUS VIDE reference uses FiGS revision `11ad36ccf294c8423b54e4e7263a03e6a246f1b1`.

## Dynamics, coordinates, and rendering

The virtual world uses Vicon's recorded Z-up frame. The virtual body is right-handed, with X forward, Y left, and Z up. State consists of world position, world velocity, and a body-to-world quaternion (10 scalar values). Collective thrust is in newtons along body +Z; angular-rate commands are in body rad/s. The model integrates translation and commanded rotation directly, following the lightweight FiGS approach rather than motor/rotor dynamics.

The baseline controller converts reference acceleration, position/velocity errors, and gravity compensation into collective thrust and body rates. Defaults are 100 Hz physics, 20 Hz control, and 10 Hz rendering. Vehicle mass (0.85 kg), drag, rate limits, and feedback gains are prototype values, not identified hardware parameters.

The renderer transforms each virtual body pose into a camera pose using the configured candidate marker-to-camera mounting transform and Vicon-to-Gaussian alignment. For the prototype, virtual body axes are assumed to coincide with the recorded marker-body axes; measured marker/body/camera transforms can replace this assumption. The renderer uses a 640×360 pinhole camera derived from the undistorted COLMAP calibration, with matching focal lengths and principal point. Depth display is opacity-normalized expected camera-Z depth in metres on a fixed 0–8 m color scale; gray indicates missing/invalid values.

`configs/demo.json` pins the current alignment JSON rather than silently loading a changing latest result. Update `rendering.alignment` to select another calibration. To replace only the mount, set `rendering.mount_override` to an object with:

```json
{
  "rotation_matrix": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
  "translation_m": [0, 0, 0]
}
```

Those numbers illustrate the **format and canonical forward optical-camera orientation only**; they are not your measured calibration. Rotation maps camera optical axes into the virtual body, and translation is the camera origin expressed in that body frame. Follow the [calibration guide](../Calibration/README.md) when replacing the current estimate. Input PLYs/checkpoints are never transformed or modified.

## Assets and outputs

- Source bags and full reference exports stay in [`../Data/`](../Data/README.md).
- Reconstruction images, intrinsics, and timestamp lists stay in `../Reconstruction/datasets/`.
- The selected PLY and transferred checkpoints remain under [`../../data/scene_20261025_200034/`](../../data/scene_20261025_200034/).
- Scene depth fidelity results remain in [`../Evaluation/`](../Evaluation/README.md) and the scene's `evaluation/` folder.
- Simulator runs go only under the scene's `simulation/` folder by default. These large/generated assets are ignored by Git.

Git synchronizes code only. The remote training laptop already has the original COLMAP/checkpoint tree, so the renderer resolves the configured copied `training_reference/Workflow/...` path back to the same `Workflow/...` run when necessary. Transfer the small fitted alignment JSON separately; see the Isaac Sim guide's synchronization commands. Run/output provenance still records the configured paths; inspect the adapter's resolved checkpoint path when auditing host-specific asset layouts.

Each run records initial configuration, source-code hashes, alignment hash, Git revision, and limitations. Interactive runs log commands/settings to `events.jsonl` and save final effective configuration on exit. Telemetry records state **after** the integration interval, with `decision_time_s` identifying when the associated control was selected. It is diagnostic telemetry, not a prepared imitation-learning dataset. Reset starts a new episode and clears the current telemetry; download before resetting to retain that episode.

## Validation and current limits

```bash
python3 -m unittest Workflow.Simulation.test_simulation -v
```

Validated on the local RTX 4060 Laptop GPU with the October 25 checkpoint: the full task plus settling simulated 17.7 seconds in approximately 5.6 seconds headlessly, exporting a 960×360 camera/map MP4 at 10 fps. Tracking RMS was 0.00392 m and maximum 0.00820 m **within the prototype's own model**. These numbers are not real-flight, geometry, or calibration accuracy. Live RGB/depth streaming, play/pause, reset, hover nudges, and browser layout were also exercised. A disturbance-recovery test passes.

The first complete recording is `data/scene_20261025_200034/simulation/20261104T170951_575410Z/simulation.mp4`. `simulation/latest_demo.json` points to the completed demo and live session artifacts.

Only an assumed floor plane at Z=0 is enforced. Walls, furniture, cables, and other scene objects are visual, not collidable. The current Gaussian depth and mounting estimates do not establish flight clearance. There is no ROS/PX4 command interface and no learned policy. The prototype is ready for task/rendering iteration while independently measured calibration and collision boundaries are added.

Next integration stages: load measured mount/body transforms; define a surveyed route and collision proxies; identify vehicle response/delays; add MPC demonstrations and then student training; validate the eventual deployment interface through PX4 simulation.

References: [SOUS VIDE / FiGS source](https://github.com/StanfordMSL/SousVide), [FiGS source](https://github.com/StanfordMSL/FiGS), [reference paper](https://arxiv.org/html/2412.16346v2).
