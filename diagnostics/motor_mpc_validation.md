# Motor plant / simplified MPC validation — 2026-11-12

Implemented source: `tkkim-robot/dev-quad-control-pipeline` commit
`010185660dba317c2b471bd47d1d6bf94f7cebf6`. The vendored module keeps its
MIT license and reference fixtures from deployed source commit
`dc78ff9a14736644c068013e0a9166e3d9acd2d7`.

Actual state evolves through the source 22-coordinate motor/rigid-body RK4
model. The outer MPC predicts seven p/v/yaw states, producing desired world
acceleration and yaw rate. Upstream geometric conversion, rate feedback and
nominal motor allocation connect these interfaces. RGB uses the achieved
attitude, never a level orientation inferred from the reference path.

Validation performed locally:

- Seven independent deployed-reference cases match geometric wrench, motor
  allocation, state derivatives and RK4 next state to rtol/atol 1e-12.
- Tests check unchanged source hashes, hover, motor lag, rotation orthogonality,
  exact wrench/rate factorization before clipping, nominal MPC model isolation,
  floor-crossing failure, randomized full-state/pre-action dataset timing,
  invalid rotor-data rejection, and native-backend substitution rejection.
- 39 simulation tests and four Isaac bridge/UI compatibility tests pass.
- RTX 4060 CUDA rendered 20 synchronized image/action samples in a one-second
  smoke test. Dataset audit passed. This short run checks wiring; it is not a
  completed training episode or an appearance-fidelity evaluation.
- A full 30 s velocity-aligned figure eight, plus two settling seconds, had
  0.26 mm RMS and 0.59 mm maximum position error with nominal parameters.
- A separate complete constant-yaw flight had 0.26 mm RMS and 0.58 mm maximum
  position error, with zero failed solves.
- Three complete randomized flights (34–44 s including settling) had RMS errors
  1.22, 9.45 and 8.02 cm; maximum errors were 4.00, 9.53 and 8.11 cm.
  All had zero failed MPC solves. These are three selected development seeds,
  not statistical coverage of the randomization ranges.

The randomized offsets are material. Mass, thrust coefficient and wind vary
while the ideal MPC remains nominal; it has no integral action or disturbance
estimator. Successful numerical execution does not make these labels suitable
for the final dataset. Review acceptance tolerances and address persistent
offsets before scaling collection. Do not compensate by hiding the tracking
error or changing recorded reference coordinates.

Raw local outputs are under
`data/scene_20261025_200034/simulation/motor_model_validation_20261112/`.
See [validation plot](figures/motor_mpc_validation.png) and
[machine-readable metrics](figures/motor_mpc_validation.json).

Source defaults are inherited model values, not measured drone parameters.
No room/contact solver, physical mount measurement, hardware comparison or new
scene-fidelity validation was added. This is the custom motor/RK4 integration;
native Isaac still uses its separate PhysX plant. No remote deployment or bulk
collection was performed for this change. Earlier campaign_v1 data are intact.
