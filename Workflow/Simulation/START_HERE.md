# Run the two simulators

To browse or download completed runs and generate result videos/trajectory plots,
see [RESULTS.md](RESULTS.md).

Both workflows run on `tk@141.212.76.61` in:

```text
/home/tk/Seans_Project/DASC/3D-Gaussian-Splat
```

The commands below use the same thrust/body-rate NMPC profile. Use one workflow
at a time while learning the setup. No manual build or colcon build is required.

| Workflow | Vehicle motion owner | Where you see it |
| --- | --- | --- |
| Custom simulator | Our quadrotor integration and response model | Local browser, via SSH forwarding |
| Native Isaac simulator | PhysX rigid body, forces, torques and rate loop | Local WebRTC client |

Both render the recorded scene with the same CUDA 3DGUT model. Native Isaac's
room collision geometry is not implemented; only its floor/body contact is
physical. The Gaussian image is shown in a panel, not imported as native room
collision geometry.

## A. Custom simulator: one remote terminal + local browser

On the local laptop:

```bash
ssh -L 8767:127.0.0.1:8765 tk@141.212.76.61
```

Then, in that remote shell:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
PATH="$HOME/anaconda3/bin:$PATH" SIMULATION_ENV="$HOME/anaconda3/envs/3dgrut" \
  bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/quadrotor_nmpc.json \
  --randomize --seed 42
```

Wait for `SIMULATOR_URL`, then open **http://127.0.0.1:8767** on the local laptop.
Click **Play** or **Run reference**. The viewer shows Gaussian RGB/depth,
reference/actual paths and telemetry. Keep the SSH terminal open.

To stop: Ctrl+C in the remote shell, then `exit`. This closes browser forwarding.

## B. Native Isaac: two remote terminals + local WebRTC client

Open two local terminal windows. Connect each with:

```bash
ssh tk@141.212.76.61
```

### Remote terminal 1: Gaussian pose renderer

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
PATH="$HOME/anaconda3/bin:$PATH" SIMULATION_ENV="$HOME/anaconda3/envs/3dgrut" \
  bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/quadrotor_nmpc.json \
  --pose-render-service --port 8766
```

Wait for `POSE_RENDER_URL=http://127.0.0.1:8766`. Leave it running.
This service renders images at the poses supplied by Isaac; it does not run a
second dynamics simulation.

### Remote terminal 2: native physics, expert and streamed UI

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
conda deactivate
bash Workflow/Simulation/isaac_sim/run_native.sh \
  --stream --randomize --seed 42 \
  --render-url http://127.0.0.1:8766
```

Deactivate further if a Conda environment remains active. Wait for
`NATIVE_VIEWER_READY`. Isaac uses its bundled Python; the renderer uses the
project Conda environment.

### On the local laptop: WebRTC

```bash
~/Downloads/isaacsim-webrtc-streaming-client-1.0.6-linux-x64.AppImage
```

Connect to **141.212.76.61**. Find **DASC Native PhysX: thrust/body-rate NMPC**.
Click **Play / resume** in that panel. The native viewer starts paused.

- Pause freezes the vehicle; Play resumes it.
- Restart route starts a fresh randomized episode immediately.
- Reset starts a fresh episode, paused.
- Hold maintains the current position and heading.
- A figure eight or M ends and holds. The elapsed clock can keep counting.
- The regular Isaac toolbar timeline Play is not the NMPC start button.

To stop: click **Quit viewer** or Ctrl+C in terminal 2, then Ctrl+C in terminal 1.
Type `exit` in both SSH shells and close the WebRTC client.

Do not run `run_live_bridge.sh` for native physics: that is the older visualizer
for the custom server. It shares WebRTC port 49100 with the native viewer.

## Configuration and collection

For approximately a million pairs with split indexes and resumable native batches,
see [LARGE_COLLECTION.md](LARGE_COLLECTION.md).
Before starting that campaign, use [DATASET_VERSIONS.md](DATASET_VERSIONS.md)
to freeze a version and inspect its first three episodes separately.

The default custom-simulator profile is now `configs/quadrotor_nmpc.json`:
custom rigid-body dynamics with thrust/body-rate NMPC. Use
`--quadrotor-plant rate_response` for the earlier simplified custom model.
The double-integrator profile
remains selectable with `--config Workflow/Simulation/configs/demo.json`.
The A-to-B linear-MPC/geometric-controller experiment remains a separate
demonstration in `animate_ab_flight.py`; it is not a production dataset mode.

For the custom simulator's pilot, batch collection, auditing and seed splits,
follow [DATA_COLLECTION.md](DATA_COLLECTION.md).

Edit `Workflow/Simulation/configs/quadrotor_nmpc.json` to change reference,
vehicle/actuator parameters, nonlinear MPC weights, and randomization bounds.
File edits require restarting the affected process. Interactive modes run until
you stop them; they are different from finite dataset collection.

For synchronized RGB/expert-action batches and native validation checks, follow
[NATIVE_BACKEND.md](isaac_sim/NATIVE_BACKEND.md). In particular, omit `--stream`
and use `--checks --episodes N --render-url ...` for native collection. The live
native viewer saves telemetry/actions, not an audited synchronized RGB dataset.

## Troubleshooting

- Address already in use: reuse or stop the existing service; don't launch a duplicate.
- Blank camera/error: check the CUDA renderer terminal and the `--render-url`.
- No motion: check PAUSED/HOLDING/ROUTE COMPLETE in the DASC panel.
- Property-window `scroll_y_max` exception: our launchers install a process-local
  guard for Isaac 4.5. It skips scroll saving only while the inspector's frame is
  absent and preserves normal/reset behavior. Installed Isaac files are unchanged.
  Restart through the updated launcher to load it.

Both workflows are development simulators. Hardware calibration, identified
vehicle response, collision geometry, and real flight-controller mapping remain
required before hardware deployment.
