# Isaac Sim backend: inspection and build sequence

This directory is for a **native Isaac Sim backend** alongside the existing lightweight prototype. The inspection script and a finite native physics smoke test are implemented. The smoke test has not yet executed on the remote machine; the drone application is not implemented yet. Noninteractive SSH authentication from the local agent currently fails, so remote execution currently uses the user's SSH terminal.

## Confirmed remote inventory

The user supplied the inspection output on November 4, 2026:

| Item | Reported result |
| --- | --- |
| Host | `ROB-DASC-06U`, Ubuntu 22.04.5 |
| Standalone installation | `/home/tk/isaacsim` |
| Version | `4.5.0-rc.36+release.19112.f59b3005.gl` |
| GPU | RTX 4090, 24,564 MiB total, 24,108 MiB free |
| Driver | 555.42.06 |
| RAM / free disk | About 59 GiB available / 606 GiB free |
| Display in SSH session | No DISPLAY or WAYLAND_DISPLAY |
| Other available Docker images | Isaac Sim 5.1.0, Isaac Lab 2.3.2, and an existing lab image; none running |

Start with the standalone **4.5.0** installation and headless physics. Do not reinstall Isaac Sim, update the driver, or alter the other project's Docker images before testing the existing installation. Inventory confirms files and resources, not driver/runtime compatibility; the smoke test establishes whether this release actually starts and simulates.

## 1. Sync code through Git

On the local development laptop, commit and push the reviewed workflow code. On the training laptop:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
git status --short
git pull --ff-only origin main
```

If local changes or branch divergence prevent the pull, inspect and preserve them rather than resetting the remote checkout. The earlier GitHub push permission error for `tkkim-robot` concerns writes; it does not by itself establish that reading/pulling this repository will fail.

Git transfers scripts/configuration/documentation. It does **not** transfer ignored recordings, checkpoints, exported scenes, or fitted calibration JSON. The remote laptop already contains the October 25 original COLMAP and training run. Our renderer can resolve those original paths when the local copied `training_reference` layout is absent.

To run the existing prototype remotely, first copy its small fitted calibration file from the local repository root:

```bash
rsync -avR \
  data/scene_20261025_200034/calibration/alignment_20261103T220923_264755Z/alignment.json \
  tk@141.212.76.61:/home/tk/Seans_Project/DASC/3D-Gaussian-Splat/
```

`Workflow/Simulation/run.sh` now selects `~/tools/3dgrut/.venv` if present, otherwise the `3dgrut` Conda environment. Isaac Sim will use its own supported environment; do not install it into the 3DGUT environment merely to reuse this launcher.

## 2. Inspect the remote installation

From the local development laptop, run the following in your own terminal so SSH can request its password:

```bash
cd /home/robin/Seans_Project/DASC/Gaussian_Splat
ssh tk@141.212.76.61 'bash -s' < Workflow/Simulation/isaac_sim/inspect_host.sh
```

Alternatively, after pulling on the remote laptop:

```bash
bash Workflow/Simulation/isaac_sim/inspect_host.sh
```

The script is read-only. It inventories OS, GPU/driver/free VRAM, RAM/disk, standalone launchers, installed Python distribution metadata, and Isaac-related Docker images/containers. It does not launch or install Isaac Sim. Searches cover the home directory, `/opt`, and `/usr/local`, to a bounded depth with large dataset/build/cache trees excluded. A missing match is not proof that no custom installation exists. The script has a 20-second timeout per external inventory command.

## 3. Select the installed release and verify native physics

Record the installation root, exact Isaac Sim version, its Python entry point, and matching driver requirements. Reuse an existing suitable installation. If missing or incompatible, choose an isolated version-compatible installation after reviewing this inventory, using that release's NVIDIA instructions. Do not upgrade an unrelated environment or GPU driver blindly: the machine may support other projects.

First milestone: launch a finite **headless native Isaac Sim** test with a stage in metres, Z up, a physics scene, a floor collider, and one falling rigid object. Verify gravity and contact; save the version, run log, and final state. This milestone deliberately does not require the drone, Gaussian model, ROS, or PX4. The launch command and API imports target the discovered standalone 4.5 release.

The first test is now written for the discovered 4.5 API. On the remote machine, after pulling the latest code:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
# If the prompt shows (base) or another Conda environment:
conda deactivate
bash Workflow/Simulation/isaac_sim/run_smoke_test.sh
```

If another Conda environment is still active, deactivate it as well. The wrapper deliberately uses `/home/tk/isaacsim/python.sh`; it does not modify Conda or install packages. An alternative installation root can be selected with `ISAAC_SIM_ROOT=/path/to/isaacsim`.

The test creates only procedural USD primitives, so no room assets or dataset transfers are needed. It simulates 240 physics steps (4 seconds) and checks that a 20 cm cube drops from Z=1.5 m, rests with its centre near Z=0.1 m on the floor, and settles without passing through the floor. It prints `ISAAC_SMOKE_RESULT=...` and saves `report.json`, `trajectory.csv`, and `smoke_scene.usda` under `data/scene_20261025_200034/isaac_sim/smoke_TIMESTAMP/`. First application startup can take longer than the four simulated seconds. Return the final result or the terminal error if startup fails; do not treat syntax checks as a successful native runtime test.

The user subsequently reported that Isaac 4.5 reaches `app ready`. The excerpt contains startup warnings but no `PHYSICS` samples or final result, so the physics test is still unverified. The code already uses modern imports (`isaacsim`, `isaacsim.core.api`, and `isaacsim.core.api.objects`). It does not import the deprecated `omni.isaac.asset_browser` extension. That extension can be loaded transitively by the installed application's startup dependencies; switching an import in our script cannot guarantee removal of a vendor dependency warning.

The launcher now explicitly selects `apps/isaacsim.exp.base.python.kit` rather than leaving experience selection to the default compatibility search. A custom location can be supplied with `--experience /path/to/isaacsim.exp.base.python.kit`. Progress is printed and saved in `report.json` as `ISAAC_SMOKE_PHASE=starting_kit`, `app_ready`, `creating_world`, `creating_bodies`, `resetting_physics`, `stepping_physics`, and `finished`. These markers identify where startup/physics is taking time. Warnings are not suppressed and installed extension files are not edited. The reported TLAS warning sequence itself ends with a valid adjusted limit; it is not evidence of a failed contact simulation.

To retain all startup output while preserving the script's failure status:

```bash
set -o pipefail
bash Workflow/Simulation/isaac_sim/run_smoke_test.sh 2>&1 | tee /tmp/dasc_isaac_smoke.log
```

If it does not finish, provide the last `ISAAC_SMOKE_PHASE` and the subsequent error/traceback, not just an earlier deprecation notice. The installed 4.5 extension renaming is documented in [NVIDIA's migration guide](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_4_5/extensions_renaming.html).

`--gui` is optional only from an actual desktop/display session. Keep the default headless mode over plain SSH. Once this test passes, move to the native drone stage below.

### See the falling cube in the existing WebRTC editor

The standalone smoke test and the full streaming editor are separate application processes. Starting `~/isaacsim/isaac-sim.streaming.sh` opens an editor; it does not automatically load our test scene. The standalone test exits when its checks finish, so it is not the application to use for a persistent streamed demonstration.

`drop_demo.py` is an editor-native alternative. It creates a new demo stage, sets the viewport camera, adds a procedural floor and dynamic cube, and starts native physics while leaving the streaming application open. Save an existing scene before running it because the demo replaces the current stage. It uses current `isaacsim.core.utils` and USD physics APIs and does not instantiate another `SimulationApp`.

1. Pull the latest code in a separate terminal on the remote laptop, leaving the streaming application running.
2. In the streamed Isaac editor, open **Window > Script Editor**.
3. Paste and run:

```python
exec(open('/home/tk/Seans_Project/DASC/3D-Gaussian-Splat/Workflow/Simulation/isaac_sim/drop_demo.py').read())
```

The view should frame the turquoise cube above a dark floor, then start dropping it after a short delay. Click **Stop**, then **Play** in the main timeline to repeat. The code is syntax-checked locally; visual/physics execution in the remote editor still requires confirmation. This interactive demonstration is separate from the automated smoke test's pass/fail report.

## 4. Build a drone scene with native rigid-body dynamics

Create a simple drone rigid body with mass, inertia, four rotor locations, and an attached camera. Begin with explicit provisional dimensions/offsets in configuration, replaceable with measured values. Add a known floor and simple room/obstacle proxy shapes as separate collision geometry. Keep assumed geometry labeled until surveyed.

Apply forces and torques through Isaac Sim's physics interface. Do not drive the drone by overwriting its pose each frame. Verify hover and angular-rate response before following a route. A real drone's thrust/body-rate command interface requires a simulated rate/actuator response; the current prototype's instantaneous body-rate integration is not sufficient for this backend.

## 5. Bring in the reconstructed scene

First verify the installed release's supported Gaussian/NuRec/USD pathway and relevant extensions. NVIDIA documents neural-volume scenes referenced into USD, but a Gaussian PLY is not automatically a collider or a universally supported scene asset. Export a compatible visual asset from the existing 3DGUT checkpoint and smoke-test its loading before adding the drone.

Keep the original reconstruction unchanged. Apply the calibrated scale/rotation/translation to a scene parent transform, with correctly transformed camera poses. Validate a few recorded viewpoints against RGB and known dimensions. Keep collision proxies separate from visual Gaussians. The fitted camera-to-marker mounting transform remains a candidate pending independent measurement.

If the installed release cannot render our Gaussian export, explicitly choose between a compatible release/extension and an external 3DGUT camera renderer driven by Isaac Sim poses. Do not silently substitute the existing browser prototype and call it Isaac Sim.

## 6. Add the M-route and synchronized sensors

Reuse the backend-independent quintic M-route definition. Use a feedback controller first, with a rate loop that produces native body torques/rotor forces. Verify tracking, simulation/control/render frequencies, camera intrinsics, frame conventions, timestamps, and episode reset. Save RGB, optional depth, state, commands, and tracking metrics. Validate RGB and depth separately: neural visuals may not participate in every sensor/geometry pathway in the same way.

Next add MPC only after the dynamics/sensor loop works. ROS 2/PX4 and student training are later integration stages, not prerequisites for a visible native simulator.

## Planned code and output locations

| Path | Status / role |
| --- | --- |
| `inspect_host.sh` | Implemented read-only host inventory |
| `smoke_test.py`, `run_smoke_test.sh` | Implemented for 4.5; native execution awaiting remote test |
| `drop_demo.py` | Implemented editor-native falling-cube demo for the already-running stream; remote visualization awaiting confirmation |
| `build_scene.py` | Planned: native stage, drone/camera, collision proxies, visual scene |
| `run_quadrotor.py` | Planned: dynamics/controller/sensor loop |
| `../configs/isaac_*.json` | Planned: version, assets, dynamics, and mount settings |
| `data/scene_20261025_200034/isaac_sim/` | Ignored generated assets, logs, recordings, and USD scenes |

## Primary references

- [Isaac Sim installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html): use the version selector to match the discovered release.
- [System requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html): driver/VRAM requirements vary by release; the remote 4090 must still be checked at runtime.
- [Neural volume rendering](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_nurec.html): supported neural-scene workflow and limitations; use a regular USD root stage referencing packaged scene assets.

The current `latest` documentation was consulted on November 4, 2026; it is not evidence of what is installed remotely.
