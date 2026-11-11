# Drone simulation and paired demonstration collection

The default action is **collective thrust (N) + XYZ body rates (rad/s)**.
The current configuration uses a 0.6×0.6 m horizontal figure eight at Z=1.2 m,
30-second nominal/minimum duration, and velocity-aligned heading. Physical/task
randomization is implemented; visual appearance and scene-geometry
randomization and student training remain future work.

## Choose the workflow

| Task | Guide |
| --- | --- |
| Run custom browser / native Isaac viewer | [START_HERE.md](START_HERE.md) |
| Collect full PhysX demonstrations | [isaac_sim/NATIVE_BACKEND.md](isaac_sim/NATIVE_BACKEND.md#run-native-isaac-collection) |
| Collect with the custom simulator | [DATA_COLLECTION.md](DATA_COLLECTION.md) |
| Browse, download, audit and visualize results | [RESULTS.md](RESULTS.md) |
| Find exact scene, calibration and source modules | [Project map](../PROJECT_MAP.md#6-where-are-dynamics-trajectories-controllers-and-randomization) |

## Launch locally

From the repository root, with the local 3DGUT environment and scene assets:

```bash
bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/quadrotor_nmpc.json \
  --quadrotor-plant rigid_body
```

Open http://127.0.0.1:8765, click Play, and use Apply & reset for route changes.
Ctrl+C stops the process. Use `--yaw-mode constant` when isolating translation
and tilt; add `--randomize --seed 42` for randomized episodes.

The full path from state to camera uses the candidate mount and Vicon/3DGS
alignment, including achieved roll and pitch. The custom simulator does not
install or invoke upstream FiGS/ACADOS. Native Isaac uses the same 3DGUT camera
renderer in a separate process.

## Available plants and controllers

| Selection | Plant | Expert/control |
| --- | --- | --- |
| Default `quadrotor_nmpc.json` | Custom numerical force/torque rigid body | Thrust/body-rate NMPC |
| `--quadrotor-plant rate_response` | Earlier simplified quadrotor response | Same thrust/body-rate interface |
| `isaac_sim/run_native.sh` | Native PhysX rigid body and contact | Same NMPC; bounded rate-controller torques |
| `--config Workflow/Simulation/configs/demo.json` | Ideal double integrator + yaw | Linear MPC, acceleration/yaw-rate outputs |
| `--controller baseline --dynamics single_integrator` | Ideal position integrator | Baseline tracker |

The MPC predicts nominal response; the actual rigid-body plant evolves through
forces and torques. Hardware mass, inertia, motor/rate response and camera mount
remain provisional. Scene objects are visual; only an assumed floor is physical.

## Edit configuration

[configs/quadrotor_nmpc.json](configs/quadrotor_nmpc.json) contains:

- `vehicle`: mass, inertia, torque limits, response constants and custom plant.
- `task`: shape, width/X span, height/Y span, altitude, timing and yaw.
- `randomization`: per-episode task/initial/actual-plant sampling ranges.
- `nmpc`: horizon, costs and solver settings.
- `rendering`: checkpoint, intrinsics, candidate alignment and resolution.

Restart after file edits. Browser route changes apply through reset. Figure-eight
speed is set by its dimensions and duration, not `speed_m_s` (the M-route setting).
The 30-second minimum is enforced in configuration, browser requests and sampling.
The current randomized duration range is 30–33 seconds. Historical saved runs,
including the 20-second pilot, retain their original settings.

## Inspect existing demonstrations

```bash
python3 Workflow/Simulation/visualize_dataset.py PATH_TO_COMPLETED_RUN
python3 Workflow/Simulation/plot_collected_trajectories.py PATH_TO_COMPLETED_RUN
```

These audit episodes and use saved synchronized observations; they do not rerun
physics or GPU rendering. See RESULTS.md for output files and dependencies.
`--headless --dataset` is required for custom paired collection; interactive
viewers and the older `live_bridge.py` are not paired training collectors.

The synthetic A→B linear-MPC/geometric-controller example is retained in
`animate_ab_flight.py`. Conceptual diagrams, sampled-reference previews and
actual collected-run visualizations are distinguished in the project map.

Previous overview text is preserved in
[Archive/simulation_readme_before_cleanup.md](../Archive/simulation_readme_before_cleanup.md).
