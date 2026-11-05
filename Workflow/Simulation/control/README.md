# Linear MPC expert

`mpc.py` is a nominal linear MPC for the existing **double-integrator position plus single-integrator yaw** plant. It does not control the simplified thrust/body-rate quadrotor. Select that model with `--controller baseline`, or implement a separate quadrotor MPC later; acceleration labels cannot be treated as thrust/body-rate labels.

## Model and optimization

State is `x=[px,py,pz,vx,vy,vz,yaw]`, input is `u=[ax,ay,az,yaw_rate]`. Inputs are in world coordinates. The exact nominal discrete model for a held command is:

```text
p[k+1]   = p[k] + dt*v[k] + 0.5*dt²*a[k]
v[k+1]   = v[k] + dt*a[k]
yaw[k+1] = yaw[k] + dt*yaw_rate[k]
```

At every control step, MPC samples `N+1` trajectory references from now through `now+N*dt`. It minimizes:

```text
sum(k=0..N-1) [
    ||x[k+1] - x_ref[k+1]||²_Q[k]
  + input_weight * ||u[k] - u_ref[k]||²
  + input_change_weight * ||u[k] - u[k-1]||²
]
```

`Q` has separate position, velocity, and yaw weights. Its final stage is multiplied by `terminal_multiplier`; `u[-1]` is the previous applied command, initially zero. `u_ref` contains reference acceleration and heading rate. The current yaw is lifted onto the reference's angle branch, avoiding a full revolution when crossing ±π. Route yaw references are already continuous/unwrapped.

The optimization has hard command bounds:

```text
|ax|, |ay|, |az| <= nominal_max_acceleration / sqrt(3)
|yaw_rate| <= nominal_max_body_rate
```

The inscribed acceleration box conservatively guarantees the nominal command norm limit; it cannot use the full norm limit along a single axis. There are no hard speed, predicted floor, or obstacle constraints. The plant still enforces its assumed floor as before. Randomized response scaling and bias can make realized acceleration exceed the nominal command bound.

Linear dynamics are condensed into a strictly convex quadratic objective. SciPy's L-BFGS-B solves it with analytic gradients and a shifted previous-plan warm start. Only the first action is applied. It is a real horizon optimization, not a renamed PD controller. A failed solve raises an error; live simulation pauses and headless execution records a failure and stops. Solver status, iterations, and wall time are logged. This is a prototype optimizer with measured timing, not a hard real-time guarantee.

Configuration lives in the top-level `mpc` section of `configs/demo.json`. Default `N=20`, `dt=0.05` seconds, position weight 30, velocity weight 3, yaw weight 8, input weight 0.08, command-change weight 0.1, and terminal multiplier 4. PD gain settings do not tune MPC.

The numerical solver uses the existing SciPy dependency and works in the current system and 3DGUT Python environments. See [SciPy's L-BFGS-B documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

## Randomized actual plant

The actual double-integrator plant can evolve as:

```text
p_dot   = v
v_dot   = acceleration_scale * limited_command + acceleration_bias - damping*v
yaw_dot = limited_yaw_rate + yaw_rate_bias
```

The expert continues to predict ideal unit-gain integrators. Sampled plant parameters and disturbances are not passed to its optimization. Initial state and task references are available to the expert. This produces feedback corrections under model mismatch; it does not guarantee robustness for arbitrary randomization ranges.

`randomization.py` samples bounded independent uniform values once at episode reset. The task's position offset, heading offset, and duration are visible through the reference; actual plant changes remain hidden. It uses a local seeded NumPy generator. `episode_parameters.json` records every effective task/plant value and initial-state offset. Route collision clearance is unresolved; only the assumed floor is checked.

## Demonstration timing and schema

Each line of `demonstrations.jsonl` is one successful expert decision:

- `time_s`, `episode`: simulation decision time and episode number.
- `position_m`, `velocity_m_s`, `quaternion_xyzw`: **pre-action** true simulated state, in Vicon/world coordinates; quaternion maps body to world.
- `reference_horizon`: `N+1` rows, each `[p3,v3,a3,yaw,yaw_rate]` (11 values), beginning at the decision time, sampled every `horizon_dt_s`.
- `action_acceleration_m_s2`, `action_yaw_rate_rad_s`: the first optimized command, held over the next control interval.
- `solver`: solve status, iterations, message, and solve time.

Telemetry remains **post-action** state with `decision_time_s` identifying its action time. Do not pair a post-action camera frame with a pre-action label as though they were simultaneous. Current exports are state/reference/action demonstrations, not a synchronized RGB student dataset. No student is trained yet.

For later Isaac integration, keep the reference generator and nominal MPC and replace the plant/sensor adapter. If Isaac owns rigid-body physics, add the acceleration/yaw-to-actuator controller and validate the resulting response; do not also integrate this plant or overwrite the physical drone pose every step.

## Verification

```bash
python3 -m unittest Workflow.Simulation.test_simulation Workflow.Simulation.test_mpc -v
```

Tests check condensed versus direct state predictions, agreement with an analytical unconstrained QP solution, anticipation of a future target, input limits, angle wrapping, explicit solver failure, seeded sampling/replay, pre-action data timing, and randomized tracking of all three routes with both yaw modes and an injected impulse.

CPU validation on November 5, 2026: 19 combined tests passed. Three randomized figure-eight episodes (seeds 42–44, velocity-aligned yaw) completed with no failed solves, position RMS 0.00734–0.00829 m, and average optimization time 1.19–1.25 ms. These are simulated tracking results, not physical-flight accuracy. Artifacts: `data/scene_20261025_200034/simulation/20261105T215011_212825Z/`. Live dynamics-only play/pause, yaw switching, hover, and episode archival were exercised. A native Gaussian-rendered verification was attempted but CUDA reported a busy/unavailable device; no new rendered rollout was validated in this change.
