# Isaac Sim backend: inspection and build sequence

This directory is for a **native Isaac Sim backend** alongside the existing lightweight prototype. The inspection script is implemented; the Isaac scene and drone application are not implemented or tested yet. The installed Isaac Sim version on `tk@141.212.76.61` must be identified first. Noninteractive SSH authentication from the local agent currently fails, so remote installation status remains unverified.

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

First milestone: launch a finite **headless native Isaac Sim** test with a stage in metres, Z up, a physics scene, a floor collider, and one falling rigid object. Verify gravity and contact; save the version, run log, and final state. This milestone deliberately does not require the drone, Gaussian model, ROS, or PX4. The precise launch command and API imports will be written against the discovered release, not guessed from another version's documentation.

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
| `smoke_test.py` | Planned after version discovery: finite physics/contact test |
| `build_scene.py` | Planned: native stage, drone/camera, collision proxies, visual scene |
| `run_quadrotor.py` | Planned: dynamics/controller/sensor loop |
| `../configs/isaac_*.json` | Planned: version, assets, dynamics, and mount settings |
| `data/scene_20261025_200034/isaac_sim/` | Ignored generated assets, logs, recordings, and USD scenes |

## Primary references

- [Isaac Sim installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html): use the version selector to match the discovered release.
- [System requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html): driver/VRAM requirements vary by release; the remote 4090 must still be checked at runtime.
- [Neural volume rendering](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_nurec.html): supported neural-scene workflow and limitations; use a regular USD root stage referencing packaged scene assets.

The current `latest` documentation was consulted on November 4, 2026; it is not evidence of what is installed remotely.
