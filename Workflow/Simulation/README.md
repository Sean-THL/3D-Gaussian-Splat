# Gaussian flight simulator

A runnable local simulator couples selectable single-integrator, double-integrator, and FiGS-style quadrotor models to the existing native 3DGUT renderer. The default is now the double integrator. It provides a browser camera view, RGB/depth toggle, M-route tracking, telemetry, route settings, hover/nudge controls, and a disturbance button. Measured camera offsets can replace the candidate mount later.

This is our prototype implementation of the lightweight approach. It does not install or invoke upstream FiGS/ACADOS. The controllers are baseline reference trackers; MPC and student training remain future work. Isaac Sim is not required.

A native Isaac Sim backend is the next requested direction. Its [host inspection and step-by-step build guide](isaac_sim/README.md) are separate from this already runnable prototype. Remote Isaac Sim installation/version are not yet verified.

## Start here

From the repository root:

```bash
Workflow/Simulation/run.sh
```

Open **http://127.0.0.1:8765**. The simulator starts paused. Press **Play** to run the reference, or **Run reference** to restart and run it. **Pause** freezes simulation time. **Reset** returns to the initial paused state. **Hold position** switches to hover; the XYZ nudge buttons move that hover target by 10 cm. **Disturbance** applies a repeatable velocity impulse for the double integrator and quadrotor, or a position displacement for the single integrator (which has no velocity memory). **Download telemetry** exports the current episode's state/reference/control CSV.

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

## Select the dynamics

No build is needed. Restart with one of:

```bash
Workflow/Simulation/run.sh --dynamics single_integrator --trajectory lissajous_figure8
Workflow/Simulation/run.sh --dynamics double_integrator --trajectory lissajous_figure8
Workflow/Simulation/run.sh --dynamics quadrotor --trajectory lissajous_figure8
```

For a CPU-only rollout, use `python3 Workflow/Simulation/run.py --no-render --headless` with the same options. Alternatively set the top-level `dynamics_model` in the JSON config. An older config without that key retains the original quadrotor behavior.

| Model | Translational equation | Command | Tracking baseline |
| --- | --- | --- | --- |
| `single_integrator` | `p_dot = u` | World XYZ velocity (m/s) | Reference velocity + proportional position correction |
| `double_integrator` | `p_dot = v`, `v_dot = u` | World XYZ acceleration (m/s²) | Reference acceleration + position/velocity feedback |
| `quadrotor` | Thrust direction, gravity, linear drag | Collective thrust (N) + body rates (rad/s) | Geometric tracking controller |

The integrators have independent yaw dynamics `yaw_dot = yaw_rate`, with wrapped yaw-error feedback. They retain zero roll/pitch, so changing `task.yaw_deg` changes the camera heading without rotating the route. They model no mass, gravity, motor response, or aerodynamic coupling. Their camera images therefore also lack the banking of a physical drone.

In `vehicle`, `max_speed_m_s` limits the single-integrator command norm; `max_acceleration_m_s2` limits the double-integrator command norm. The double integrator has no hard speed cap. `max_body_rate_rad_s` limits integrator yaw rate; `floor_z_m` supplies the same assumed floor constraint as before. Mass, drag, gravity, and thrust limits apply only to `quadrotor`. Single-integrator velocity is diagnostic output, not an independently evolving state.

Telemetry records `dynamics_model`, command units, and XYZ integrator commands. Thrust is null/blank for integrators. These remain baseline controllers, not MPC experts; an eventual expert and student must use the selected model's action interface. An integrator action is not a direct thrust/body-rate command for deployment.

## Analytic references

All three references expose `trajectory_function(t)`, returning
`[px, py, pz, vx, vy, vz, ax, ay, az, yaw]` in metres, seconds, and radians.
Select a shape in the browser or from the command line:

```bash
Workflow/Simulation/run.sh --trajectory lissajous_figure8
Workflow/Simulation/run.sh --trajectory horizontal_circle
```

The analytic references implement the supplied circle and Lissajous equations. The default remains the M route. This interface is connected to the lightweight prototype; the native Isaac drone backend is still pending.

- **Circle:** `radius_m` defaults to half `width_m`. `max_angular_speed_rad_s` defaults to 0.75 rad/s and `angular_ramp_rate_per_s` to 0.1 per second. Its center is one radius left of its starting position. Motion continues until paused; `duration_s` controls the preview and default recording horizon. Asymptotic speed is radius times angular speed. The name `a_const` in the supplied snippet represents angular speed, not acceleration.
- **Figure eight:** `a_m` defaults to half `width_m`; `b_m` defaults to `height_m`, giving total XY extents `2a` by `b`. `duration_s` defaults to 20 seconds for one cycle, then the reference holds at its starting point with zero velocity and acceleration. `vertical_drop_m` defaults to 0.5 m in the Z-up frame; set it to zero for a horizontal route.
- Both use `start_position_m` when supplied; otherwise they start at `[center_xy_m[0], center_xy_m[1], altitude_m]`. Yaw stays at `yaw_deg`.

To reproduce the supplied dimensions exactly, set `radius_m: 1.0` for the circle, or `a_m: 1.0`, `b_m: 1.0`, `vertical_drop_m: 0.5`, and `duration_s: 20.0` for the figure eight in a custom JSON config passed with `--config`. Explicit radius/a/b/start overrides take precedence over the browser's dimension fields.

The M route's speed/acceleration settings apply to the M route. Analytic speeds and accelerations follow from their dimensions and timing; the circle/figure eight do not automatically enforce those M-route limits. Longer figure-eight durations reduce speed in proportion to `1/T` and acceleration to `1/T²`.

## Where to find the code

| File | Purpose |
| --- | --- |
| `run.py` | CLI, local HTTP viewer, simulation loop, telemetry and video export |
| `viewer.html` | Live camera, route map, controls, and settings |
| `configs/demo.json` | Scene paths, alignment, vehicle, camera, timing, controller, and M-route defaults |
| `figs/dynamics.py` | Prototype 10-state thrust/body-rate dynamics |
| `figs/integrators.py` | Ideal velocity/acceleration models and tracking controllers |
| `figs/simulator.py` | Deterministic stepping, episode control, references, and telemetry |
| `tasks/trajectory.py` | Common reference interface, quintic M route, analytic circle and figure eight |
| `control/tracker.py` | Baseline position/attitude feedback controller |
| `rendering/threedgut.py` | Native renderer adapter for moving camera poses |
| `test_simulation.py` | Physics, trajectory limits, tracking recovery, and control-state tests |

Upstream 3DGUT remains at `~/tools/3dgrut`, separate from our adapters. The model's recorded upstream revision is `e6f4552b408af020a96de09ab5757f9b57430aaa`. Upstream FiGS may be integrated later with a pinned revision; the SOUS VIDE reference uses FiGS revision `11ad36ccf294c8423b54e4e7263a03e6a246f1b1`.

## Dynamics, coordinates, and rendering

The virtual world uses Vicon's recorded Z-up frame. The virtual body is right-handed, with X forward, Y left, and Z up. The common state container holds world position, world velocity, and a body-to-world quaternion (10 scalar values). For the quadrotor model, collective thrust is in newtons along body +Z; angular-rate commands are in body rad/s. The model integrates translation and commanded rotation directly, following the lightweight FiGS approach rather than motor/rotor dynamics.

For the quadrotor model, the baseline controller converts reference acceleration, position/velocity errors, and gravity compensation into collective thrust and body rates. Defaults are 100 Hz physics, 20 Hz control, and 10 Hz rendering. Vehicle mass (0.85 kg), drag, rate limits, and feedback gains are prototype values, not identified hardware parameters.

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

The original quadrotor model was validated on the local RTX 4060 Laptop GPU with the October 25 checkpoint: the full task plus settling simulated 17.7 seconds in approximately 5.6 seconds headlessly, exporting a 960×360 camera/map MP4 at 10 fps. Tracking RMS was 0.00392 m and maximum 0.00820 m **within the prototype's own model**. These numbers are not real-flight, geometry, or calibration accuracy. Live RGB/depth streaming, play/pause, reset, hover nudges, and browser layout were also exercised. Regression tests cover the original quadrotor, exact constant-input integrator solutions, command limits, independent yaw, and all three routes with disturbance recovery.

The first complete recording is `data/scene_20261025_200034/simulation/20261104T170951_575410Z/simulation.mp4`. `simulation/latest_demo.json` points to the completed demo and live session artifacts.

Only an assumed floor plane at Z=0 is enforced. Walls, furniture, cables, and other scene objects are visual, not collidable. The current Gaussian depth and mounting estimates do not establish flight clearance. There is no ROS/PX4 command interface and no learned policy. The prototype is ready for task/rendering iteration while independently measured calibration and collision boundaries are added.

Next integration stages: load measured mount/body transforms; define a surveyed route and collision proxies; identify vehicle response/delays; add MPC demonstrations and then student training; validate the eventual deployment interface through PX4 simulation.

References: [SOUS VIDE / FiGS source](https://github.com/StanfordMSL/SousVide), [FiGS source](https://github.com/StanfordMSL/FiGS), [reference paper](https://arxiv.org/html/2412.16346v2).
