# Run the remote CUDA simulator and Isaac viewer

This page describes the older custom-simulator visualization bridge. For the **actual native PhysX viewer**, use `run_native.sh --stream` and follow [NATIVE_BACKEND.md](NATIVE_BACKEND.md#see-the-actual-native-drone-through-webrtc).

The remote host is `tk@141.212.76.61`, repository
`/home/tk/Seans_Project/DASC/3D-Gaussian-Splat`.
The local laptop only runs a browser, SSH, and the WebRTC client.

```text
Remote RTX 4090:
  episode reset -> task/plant randomization -> trajectory + yaw references
                                                   |
                  current state -> MPC -> acceleration + yaw rate
                                      |               |
                                 demonstrations   custom dynamics
                                                      |
                                             updated drone pose
                                               /            \
                              native 3DGUT camera        Isaac live bridge
                                   RGB/depth        drone/path/point proxy + camera UI
                                                              |
Local laptop:                              WebRTC client at 141.212.76.61
  Browser at localhost:8767 --SSH forward--> CUDA server localhost:8765
```

There is one vehicle-state owner: the external simulator. Isaac's timeline is
not this simulator's clock. Use the **DASC panel** controls to run the vehicle.
The bridge has no native rigid-body flight or obstacle collision response.

## Start from a stopped system

Check that the previous project processes have stopped before launching more.
The deployed applications may already be running; then just connect to them.
Logs/PIDs are under `data/scene_20261025_200034/remote_launch/`. Do not stop other
projects or launch a second service on an occupied port.

### Terminal A on the local laptop: SSH, browser forwarding, CUDA server

```bash
ssh -L 8767:127.0.0.1:8765 tk@141.212.76.61
```

Enter the password only in that terminal. Once connected, run **remotely**:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
PATH="$HOME/anaconda3/bin:$PATH" SIMULATION_ENV="$HOME/anaconda3/envs/3dgrut" \
  bash Workflow/Simulation/run.sh \
  --controller mpc --dynamics double_integrator \
  --trajectory lissajous_figure8 --yaw-mode velocity_aligned \
  --randomize --seed 42
```

Wait for `Renderer ready: NVIDIA GeForce RTX 4090` and
`SIMULATOR_URL=http://127.0.0.1:8765`. Keep the terminal open.
Open **http://127.0.0.1:8767** locally to view/control it independently of Isaac.
The Gaussian viewer must use this CUDA environment, not Isaac's bundled Python.

### Terminal B on the local laptop: start Isaac remotely

```bash
ssh tk@141.212.76.61
```

Then **remotely**:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
conda deactivate  # Repeat if another Conda environment remains active.
bash Workflow/Simulation/isaac_sim/run_live_bridge.sh
```

Wait for `DASC_ISAAC_BRIDGE_READY`. Keep this terminal open too.
Isaac uses `/home/tk/isaacsim/isaac-sim.streaming.sh`, separate from 3DGUT.
Do not install either project's packages into the other's environment.

### Terminal C on the local laptop: open the streamed window

```bash
chmod +x ~/Downloads/isaacsim-webrtc-streaming-client-1.0.6-linux-x64.AppImage
~/Downloads/isaacsim-webrtc-streaming-client-1.0.6-linux-x64.AppImage
```

Connect to **141.212.76.61**. If FUSE prevents launching the client, add
`--appimage-extract-and-run` to the AppImage command.
Find **DASC: CUDA camera + MPC controls** in the streamed editor.

## Run, pause, and repeat a trajectory

- **Play / resume:** run or resume the current episode.
- **Pause:** freeze the external simulator.
- **Restart route:** start a new episode and run immediately; randomization
  samples new conditions if enabled.
- **Reset / resample:** start a new episode, paused. Then press Play.
- **Hold:** maintain the current position/heading; leaves route tracking.
- **Constant yaw / Velocity yaw:** change heading mode and reset the episode.
- **RGB / Depth:** choose which Gaussian camera image appears in the panel.

**A figure eight is one finite cycle, not an endless loop.** Once its route
clock reaches its duration (nominally 20 seconds, varied by randomization), it
holds its endpoint. Overall elapsed simulation time and frame counters keep
increasing while the image is stationary. The panel shows **ROUTE COMPLETE -
HOLDING**. Click **Restart route**, or Reset followed by Play, to fly again.
The M route also ends and holds. The horizontal circle keeps circling.

The DASC panel shows both route time and overall elapsed simulation time. The
normal Isaac timeline can count even when the external vehicle is paused.

## Configuration and recording

Edit `Workflow/Simulation/configs/demo.json` on the remote machine, then restart
the CUDA server for file changes. Changing modes in the browser/panel applies
through the server API and starts a new episode without a rebuild.

- `task`: dimensions, altitude, duration, yaw, and trajectory selection.
- `mpc`: horizon, tracking/input/smoothness weights, iteration limit.
- `randomization`: per-episode task, initial-state, and plant ranges.
- `vehicle`: command limits and nominal plant parameters.

The current MPC supports only double-integrator position plus yaw-rate control.
For the simplified quadrotor, select `--controller baseline --dynamics quadrotor`;
a quadrotor MPC is not implemented yet.

Each CUDA launch prints its output directory under
`data/scene_20261025_200034/simulation/`. It saves episode parameters,
state/reference/action demonstrations, and telemetry. Completed interactive
records are archived before a reset. On a foreground launch, use Ctrl+C to stop
cleanly and save the current episode.

For CPU/GPU batch collection, without the Isaac live bridge:

```bash
# Run in the remote repository, using the same CUDA launch environment above.
PATH="$HOME/anaconda3/bin:$PATH" SIMULATION_ENV="$HOME/anaconda3/envs/3dgrut" \
  bash Workflow/Simulation/run.sh --headless --record \
  --controller mpc --dynamics double_integrator \
  --trajectory lissajous_figure8 --yaw-mode velocity_aligned \
  --randomize --episodes 10 --seed 42
```

A headless batch does not serve live state to Isaac. Exported state/action
records are not yet a synchronized RGB imitation-learning dataset.

## If motion appears stopped

Check the DASC status first:

1. `PAUSED`: click its Play button.
2. `ROUTE COMPLETE - HOLDING`: click Restart route.
3. `HOLDING POSITION`: click Restart route to return to trajectory tracking.
4. `Bridge error`: inspect the server terminal/log and the runtime report
   `data/scene_20261025_200034/isaac_sim/live_bridge_status.json`.
5. A changing frame counter alone does not prove motion: a stationary camera
   produces identical pixels. Compare position and route time too.

If no DASC panel appears, open Isaac's Script Editor and run:

```python
import sys
sys.path.insert(0, '/home/tk/Seans_Project/DASC/3D-Gaussian-Splat')
from Workflow.Simulation.isaac_sim import start_live_bridge
```
