# Thrust/body-rate expert and native Isaac backend

This implementation adds two usable data-generation paths with one action contract:
`[collective thrust in N, body omega_x, omega_y, omega_z in rad/s]`.
The previous double-integrator linear MPC remains available independently.

## Which simulator should generate the data?

| Aspect | Custom quadrotor + 3DGUT | Native Isaac PhysX + 3DGUT |
| --- | --- | --- |
| Dynamics | Numerical force/torque rigid body by default; earlier rate-response model remains selectable | Native rigid-body integration with mass, inertia, force, torque, and contact |
| Rate command response | Torque-limited rate loop; `--quadrotor-plant rate_response` selects the earlier approximation | Torque-limited body-rate feedback loop on the physical rigid body |
| Scene image | Native CUDA 3DGUT | Same CUDA 3DGUT in a separate pose-render process |
| Contacts | Assumed floor clamp | PhysX floor/body contact |
| Room obstacles | Visual only | Visual only; no verified room collision model yet |
| Environment | Project 3DGUT Python | Isaac bundled Python plus project-private solver dependency |
| Best current role | Simple, inspectable collection/ablation backend | Independent physical-response validation and native pilot collection |

Isaac is not automatically a better Gaussian camera renderer or a calibrated
real drone. The two backends intentionally share the renderer. A Gaussian PLY
is not imported as a native collider. PhysX runs on CPU for this single-body
prototype; 3DGUT renders on the RTX 4090. Reuse both backends to separate visual
issues from physical-model issues. The SOUS VIDE reference itself uses lightweight
thrust/body-rate dynamics for Gaussian dataset synthesis:
[paper](https://arxiv.org/html/2412.16346v2).

The existing WebRTC `live_bridge.py` remains a **visualization bridge** to the
custom server. `native_rollout.py` is a separate, finite **native physics/data
runner**. Starting the old bridge does not activate this new physical backend.
The native runner supports both finite headless collection and an interactive native WebRTC viewer with `--stream`. The new viewer uses native PhysX directly and does not start the old bridge.

## Expert implementation

`control/quadrotor_mpc.py` performs bounded nonlinear receding-horizon
optimization. CasADi computes exact derivatives of the shooting model; SciPy
SLSQP solves the local nonlinear program. Controls are thrust and body rates,
not a post-hoc conversion of double-integrator commands.

The nominal prediction includes position, velocity, orientation quaternion,
three achieved body rates, and achieved thrust: 14 internal states. The four
actuator states account for nominal rate/thrust response delay. The public pose
remains position/velocity/quaternion. The expert has privileged access to the
simulated achieved actuator state; a future student does not need that privileged
input. Actual body rates are recorded as an observable gyro-like quantity.

Cost terms penalize position/velocity error, sign-invariant quaternion attitude
error, feedforward-input error, and command changes, with terminal weighting.
Thrust is scaled during numerical optimization; logged commands stay in N.
Bounds are collective thrust and body rates. There are no predicted obstacle,
clearance, hard tilt, or hard velocity constraints. Failed optimization raises an
error; fallback-controller labels are never silently mixed into the dataset.

The randomized actual plant varies mass, drag, thrust/rate gain, rate/thrust time
constants, and acceleration bias. The expert uses nominal parameters and current
state. Route shifts, heading offsets, initial errors, and timing variation are
sampled as before. Inertia and torque limits are explicit provisional values,
not identified hardware parameters. Mass changes preserve absolute motor thrust
capacity instead of increasing motor capacity with payload mass.

## Native force/torque implementation

`native_backend.py` creates a procedural floor and a drone rigid body. It applies:

```text
F_world = R * [0,0,achieved_thrust] + m*(disturbance - drag*v)
tau_body = clip(I*(desired_body_rate - measured_body_rate)/rate_tau
                + omega cross (I*omega), body_torque_limits)
tau_world = R * tau_body
```

PhysX applies gravity and evolves the body. A first-order thrust response is
included. Position/orientation setters are used only at reset, not for flight.
Measured mass and inertia are checked against the authored values after reset.
The default geometry/inertia describes a provisional proxy drone, not a CAD model
or an identified rotor/motor model. The inner rate loop is proportional with
gyroscopic compensation; a real flight-controller PID is not reproduced exactly.

`native_supervisor.py` requires both a successful process exit and a passing
`report.json`. This protects against Kit shutdown masking failed validation.
Baseline hover/rate/contact checks run with nominal parameters; randomized
conditions are then applied to flight episodes.

## Remote environment setup (already done on ROB-DASC-06U)

No Isaac installation, driver upgrade, or unrelated environment was changed.
CasADi 3.6.6 was added to the project 3DGUT environment. For Isaac Python it was
installed without dependencies into this ignored project-private directory:

```text
data/scene_20261025_200034/isaac_dependencies/
```

The native launcher prepends that directory to PYTHONPATH. Pinned optional
requirements are in `control/requirements-nmpc.txt`. The remote profile is
`configs/quadrotor_nmpc.json`; the old default config was preserved.

## Run custom-backend collection

On the remote host:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
PATH="$HOME/anaconda3/bin:$PATH" SIMULATION_ENV="$HOME/anaconda3/envs/3dgrut" \
  bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/quadrotor_nmpc.json \
  --headless --dataset --randomize --episodes 2 --seed 301
```

Omit `--headless --dataset` to use the existing browser viewer with the new
quadrotor expert. Do not use the old `--controller mpc` flag for thrust/body-rate
labels; the new profile selects `nmpc` and `quadrotor` together.

## See the actual native drone through WebRTC

Start the CUDA pose renderer in remote terminal A using the command below.
In remote terminal B, deactivate Conda and run:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
conda deactivate
bash Workflow/Simulation/isaac_sim/run_native.sh \
  --stream --randomize --seed 42 \
  --render-url http://127.0.0.1:8766
```

Wait for `NATIVE_VIEWER_READY`. On the local laptop, open the existing Isaac
WebRTC client and connect to `141.212.76.61`. Find **DASC Native PhysX:
thrust/body-rate NMPC**, then click **Play / resume**. It starts paused so you
can connect before motion begins. **Restart route** starts a new randomized
episode; **Pause**, **Reset**, **Hold**, **Disturbance**, and yaw-mode controls
are also provided. **Quit viewer** closes the native application cleanly.

The viewport shows the actual rigid body with visual arms/rotors, the floor,
and reference/actual paths. The panel shows Gaussian RGB rendered from the
physical drone pose. Room collision geometry is still not implemented.
This command replaces `run_live_bridge.sh` when you want native physics.
Do not start both Isaac viewers: they share WebRTC port 49100. A localhost
status/control endpoint at 8768 is reserved for native viewer diagnostics.

`World.render()` refreshes UI/graphics without stepping physics. Native episode
time is measured relative to the authored initial state, after Isaac warm-up
steps, so clocks remain consistent through pause and reset. `--viewer-seconds N`
is an optional finite test duration; omit it for normal interactive use.

The viewer saves per-episode telemetry and state/action demonstrations on reset
and exit. It does not save synchronized RGB training pairs: use finite collection
below for the audited image/action dataset. Stop the CUDA renderer afterward.

## Run native Isaac collection

Use two remote terminals. No WebRTC client is required for headless collection.

Terminal A, CUDA pose renderer (leave open):

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
PATH="$HOME/anaconda3/bin:$PATH" SIMULATION_ENV="$HOME/anaconda3/envs/3dgrut" \
  bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/quadrotor_nmpc.json \
  --pose-render-service --port 8766
```

Wait for `POSE_RENDER_URL=http://127.0.0.1:8766`.

Terminal B, native physics/expert/recorder:

```bash
cd ~/Seans_Project/DASC/3D-Gaussian-Splat
conda deactivate
bash Workflow/Simulation/isaac_sim/run_native.sh \
  --checks --randomize --episodes 2 --seed 301 \
  --render-url http://127.0.0.1:8766
```

It exits after the finite batch and prints `NATIVE_VALIDATION_PASSED` only after
all gates pass. Omit `--render-url` for physics-only validation. `--duration 3`
is a short test; omit it for complete trajectories. Stop terminal A with Ctrl+C
when collection finishes. Exit both SSH shells when finished.

## Data contract and audit

Both collection paths save one PNG per successful decision at 20 Hz simulation
time, plus `rgb_actions.jsonl`. Image capture occurs **before** selecting and
applying that step's action. The physics clock is not advanced while rendering.
Wall-clock execution may be slower than real time; simulation timestamps still
advance at the configured control interval.

Each row has state, observed body rates, reference horizon, thrust/body-rate
label, solver diagnostics, image path, image timestamp, and SHA256. Metadata
records control dt, image dimensions, command limits, backend, and camera/model
provenance. Current renderer metadata includes checkpoint/calibration hashes,
intrinsics, and the candidate camera/body transform. Native reports include
software versions and source hashes. Older pilot metadata can have fewer fields;
its calibration hash and exact source manifest remain recorded.

Audit an episode with the project environment:

```bash
/home/tk/anaconda3/envs/3dgrut/bin/python Workflow/Simulation/audit_dataset.py \
  data/scene_20261025_200034/isaac_sim/native_TIMESTAMP/episode_0001
```

The audit checks image/action time equality and regular spacing, finite states,
quaternion norms, action type/bounds, solver status, unique image paths, image
checksums, RGB dimensions, and sample counts. It rejects a failed native run or
an episode with a recorded execution failure. Passing is a structural-data check,
not a test of image realism, clearance, or real-flight safety.

## Current validation evidence and remaining gates

A first randomized native test failed with torque saturation. Adding nominal
actuator response to MPC prediction resolved the same test conditions without
relaxing the 0.2 m development maximum-position-error threshold. This threshold
is a regression gate, not a safe clearance recommendation.

Evidence is summarized in `diagnostics/quadrotor_native_validation.md` and the
saved report/dataset audit JSONs. Do not call this real-flight validated:

- Native hover, three-axis rate response, inertia/mass consistency, and floor
  contact have been tested; full randomized flights and RGB pilots were checked.
- Mass, inertia, actuator response and torque bounds still need physical
  measurement/identification against the real vehicle/controller.
- Camera mounting and Vicon alignment remain candidates; scene depth errors and
  room collision geometry remain unresolved.
- Real thrust-to-flight-controller command mapping, latency, and frame conventions
  need validation before connecting a learned policy to hardware.
- No student has been trained. Separate train/validation/test seeds and compare
  closed-loop policy performance, rather than trusting dataset size alone.

CasADi model/derivative support: [official documentation](https://web.casadi.org/docs/).
The actual installed Isaac 4.5 force/torque API was inspected before use; native
physics runtime results, rather than import/syntax checks alone, establish the
reported development status.

## Isaac 4.5 Properties-window scroll exception

If selection changes print `AttributeError: NoneType has no attribute scroll_y_max`
from `omni.kit.window.property`, the failing callback belongs to the built-in
Properties inspector. The traceback alone does not establish an MPC/physics
failure. Opening/rebuilding the Property window may restore the live UI frame:

```python
import omni.kit.window.property as property_ui
window = property_ui.get_window()
if window is not None:
    window.set_visible(True)
    window.request_rebuild()
```

The project now has `ui_compat.py`, which wraps only `save_scroll_pos` and skips
the call while `properties_frame` is None. It leaves valid-frame behavior and
unrelated exceptions unchanged and modifies no installed extension files. Native
streaming and the older bridge bootstrap install it after Kit startup. For a
running session after deploying that module, execute in Isaac's Script Editor:

```python
from Workflow.Simulation.isaac_sim.ui_compat import install_property_scroll_guard
print(install_property_scroll_guard())
```

Two pure-Python regression tests cover the absent-frame lifecycle, preserved
valid calls, idempotency, and propagation of unrelated errors. An isolated installed-runtime probe reproduced the original exception on
ROB-DASC-06U and verified the missing-frame guard, preserved reset behavior, and
normal scroll saving. No installed extension files were edited. API reference:
[Kit PropertyWindow](https://docs.omniverse.nvidia.com/kit/docs/omni.kit.window.property/1.11.1/omni.kit.window.property.window/omni.kit.window.property.window.PropertyWindow.html).
