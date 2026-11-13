# Full motor dynamics with simplified MPC

The supplied [dev-quad-control-pipeline](https://github.com/tkkim-robot/dev-quad-control-pipeline)
is vendored at commit `010185660dba317c2b471bd47d1d6bf94f7cebf6` under
`third_party/dev_quad_control/`, with its MIT license and deployed-model reference
fixtures. The source equations and RK4 integrator are unchanged.

## Signal flow

```text
Reference position/velocity/acceleration/yaw horizon
           + actual position/velocity/yaw
                        ↓
Simplified MPC: ideal double integrator for XYZ, integrator for yaw
                        ↓
Desired world acceleration + yaw rate
           + actual attitude and body rates
                        ↓
Upstream geometric controller → desired thrust and moment
                        ↓
Thrust/body-rate interface → inner rate loop → nominal motor allocator
                        ↓
Full plant: rotor lag, rigid-body rotation/translation, drag, rotor inertia
                        ↓
Actual position, velocity, attitude, body rates and rotor speeds
           ├─ feedback to next control decision
           └─ camera mounting + scene alignment → Gaussian RGB
```

MPC predicts seven states `[x,y,z,vx,vy,vz,yaw]`, with inputs
`[ax,ay,az,yaw_rate]`. It keeps nominal parameters when the plant is randomized.
The actual plant has 22 stored coordinates: position 3, velocity 3, rotation
matrix 9, body rates 3, rotor speeds 4. The rotation matrix has constraints;
22 is the storage size, not the number of independent degrees of freedom.

The upstream geometric controller receives MPC acceleration. Its reference
position and velocity equal the current values, so their feedback terms vanish;
MPC owns that feedback. Desired thrust direction comes from `m*(a + g*e3)`.
Current yaw supplies the heading and MPC yaw rate supplies heading feedforward.
Tilt-rate feedforward is omitted; geometric feedback tracks changing tilt.

For the geometric moment `M`, nominal inertia `J`, and rate gain `kOmega`,
the adapter forms `Omega_cmd = Omega + (M - Omega × J*Omega)/kOmega`.
The inner loop reconstructs this moment at the decision state before command
clipping. It recomputes its rate feedback as the plant evolves. This preserves
thrust/body-rate labels while allowing the plant to pitch/roll before translating.
These body rates are measured about the body XYZ axes, not Euler-angle rates.

## Run locally

From the repository root, launch the CUDA viewer:

```bash
bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/motor_mpc.json
```

Open the printed `SIMULATOR_URL`, then click Play. To change heading behavior,
add `--yaw-mode velocity_aligned`. For randomized conditions, add
`--randomize --seed 53`. Use `--port 8768` if another viewer owns the default port.

For three complete, synchronized dataset episodes:

```bash
bash Workflow/Simulation/run.sh \
  --config Workflow/Simulation/configs/motor_mpc.json \
  --headless --dataset --randomize --seed 53 --episodes 3 \
  --output data/scene_20261025_200034/simulation/motor_mpc_pilot_53
```

Choose a new output folder per attempt. Each episode saves RGB, pre-action full
pose/velocity/body rates/rotor speeds, the reference horizon, thrust N/body-rate
rad/s labels, outer acceleration/yaw-rate commands, solver status and sampled
parameters. The outer acceleration is diagnostic; the student target remains
thrust/body rates. `audit_dataset.py EPISODE_FOLDER` validates synchronized data.

Physics uses 200 Hz, MPC 20 Hz. Dataset capture renders every 20 Hz decision,
regardless of the browser's 10 Hz display refresh. No colcon build is required.

## Parameters and randomization

Edit `configs/motor_mpc.json` before creating a new dataset version:

| Setting | Where | Units/default |
| --- | --- | --- |
| Mass | `vehicle.mass_kg` | 0.680 kg |
| Inertia | `vehicle.inertia_kg_m2` | [0.005, 0.005, 0.003] kg m² |
| Motor time constant | `vehicle.motor_time_constant_s` | 0.020 s |
| Rotor inertia | `vehicle.rotor_inertia_kg_m2` | 6.62e-6 kg m² |
| Arm dimensions | `vehicle.motor_arm_x_m`, `motor_arm_y_m` | 0.08 m each |
| Rotor coefficients | `motor_thrust_coefficient`, `motor_torque_coefficient` | SI, speeds in rad/s |
| Route and duration | `task` | 0.6 m × 0.6 m, altitude 1.2 m, 30 s |
| MPC tuning | `mpc` | prediction horizon and tracking/input weights |
| Geometric/rate gains | `geometric_controller` | upstream gains |
| Sampling ranges | `randomization` | sampled once per episode |

The new profile varies route offset/altitude, yaw, duration, initial position,
velocity and roll/pitch, mass, inertia, motor time constant, rotor coefficients,
quadratic drag, wind and acceleration disturbance. It does not automatically
enable visual augmentation. Use the existing image-augmentation settings when
creating a deliberately reviewed appearance profile.

The source defaults are inherited simulation values, not measurements of your
drone. The new motor time constant is distinct from the older surrogate
`rate_time_constant_s` and `thrust_time_constant_s`. Quadratic force drag is also
distinct from the old velocity damping coefficient.

## Isaac and collection limits

This profile uses the custom NumPy motor/RK4 plant and CUDA Gaussian renderer.
Native Isaac collection still uses its PhysX plant; it rejects this profile
rather than substituting different dynamics. An equivalent native Isaac port
would require implementing and validating the same rotor forces/moments, motor
states and integration behavior. The current large-campaign collector remains
native-only; use the custom runner's `--episodes` for this profile.

There is no source contact solver or room collision mesh. Crossing the assumed
floor aborts the motor run. Existing geometry/mount alignment uncertainty still
applies. The simplified MPC has no disturbance estimator or integral action:
random mass/motor changes can cause steady tracking offsets. Review the
validation report and pilots before bulk collection.
