# Quadrotor expert and native Isaac validation — 2026-11-06

The thrust/body-rate nonlinear MPC and native PhysX data backend are implemented and deployed on ROB-DASC-06U. This is a validated development pipeline for the tested simulated cases, not a validated real-flight system.

## Implementation

- Shared expert: collective thrust in N and body angular rates in rad/s. Quaternion prediction includes nominal rate/thrust response, with analytic derivatives from CasADi and bounded SciPy SLSQP optimization.
- Custom backend: direct quadrotor integration with configurable response and disturbances.
- Native backend: Isaac 4.5 PhysX rigid body, explicit mass/inertia, torque-limited inner rate feedback, thrust response, force/torque application, and native floor contact. Pose setters are used only at reset.
- Recorder: one RGB image at the pre-action state per successful expert decision. Timestamp/state/action consistency is checked before writing; an independent audit verifies image integrity, finite values, bounds, references, and complete sample counts.
- Failure handling: no silent fallback labels; failed native reports and failed episodes are rejected. A parent supervisor verifies the report and process exit status.

## Native flight results

All listed episodes passed the development maximum-position-error gate of 0.2 m. The gate is not a clearance recommendation.

| Route | Seed | Steps | Position RMS (m) | Maximum error (m) | Mean solve (ms) | Maximum solve (ms) | Torque-limit steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lissajous_figure8 | 201 | 455 | 0.00680 | 0.03296 | 19.75 | 28.12 | 0 |
| lissajous_figure8 | 202 | 425 | 0.01436 | 0.01948 | 20.68 | 28.46 | 0 |
| lissajous_figure8 | 203 | 426 | 0.02967 | 0.03536 | 21.53 | 33.39 | 0 |
| lissajous_figure8 | 301 | 471 | 0.03534 | 0.03616 | 22.01 | 28.09 | 0 |
| lissajous_figure8 | 302 | 416 | 0.02502 | 0.02586 | 21.34 | 28.00 | 0 |
| m | 401 | 376 | 0.00914 | 0.02813 | 20.65 | 36.09 | 46 |
| horizontal_circle | 501 | 440 | 0.02868 | 0.02936 | 18.77 | 27.73 | 0 |

No failed solves in these 7 complete randomized native episodes. The M route briefly engaged the torque limit (46 physics steps), while remaining within the tracking gate. These seven samples do not establish statistical robustness across the entire randomization range or real-time guarantees.

## Physics and data checks

- Native hover drift: 0.0 m in the nominal 2-second check. Three-axis rate-response errors about 1.3e-6 rad/s; final floor-contact body-center Z approximately 0.020 m for the 0.040 m high body. Actual inertia matched the configured diagonal tensor.
- 24 local automated tests passed: existing dynamics/trajectory/yaw/control tests, linear MPC, nonlinear model/gradient agreement, command labels, response mismatch, RGB pre-action timing, corrupt-image rejection, and bridge protocol checks.
- Native RGB pilot: seeds 301 and 302 produced 471 and 416 pairs (887 total), all passing structural audits.
- Custom RGB pilot: 40 pairs passed the same audit, including checkpoint and camera-transform provenance.
- Representative native RGB frames across both full trajectories were visually inspected. This is a coverage review, not a replacement for the earlier quantitative fidelity analysis.

## Important failure found and resolved

The initial instantaneous-rate prediction failed a randomized native case (seed 201), reaching 0.693 m maximum tracking error with heavy torque saturation. The same seed also degraded in the custom backend with actuator lag. Adding nominal actuator states/response to prediction reduced the native maximum error to 0.033 m without relaxing the acceptance threshold. Failed runs remain marked failed. Baseline physics tests are separated from deliberately biased randomized plants.

## Saved evidence

- Native physics diagnostics: `data/scene_20261025_200034/isaac_sim/native_20261106T070047_523702Z/report.json` (remote).
- Native three-seed figure-eight regression: `native_20261106T070247_232620Z/` (remote).
- Native RGB pilot and audits: `data/scene_20261025_200034/isaac_sim/native_20261106T070548_764152Z/` (reports/contact sheet copied locally; full images remain remote).
- Native M and circle regressions: `native_20261106T071302_675038Z/`, `native_20261106T071321_605075Z/` (remote).
- Custom RGB pilot: `data/scene_20261025_200034/simulation/20261106T070934_640153Z/` (remote).
- Compact summary: `data/scene_20261025_200034/isaac_sim/quadrotor_validation_summary.json` (local and remote).

## Remaining work before hardware or large-scale training claims

- Identify physical mass/inertia, motor/thrust and rate response, torque limits, transport latency, and the real flight-controller command mapping.
- Independently measure camera/marker/drone-center transforms; resolve metric scene-depth errors and establish collision geometry. Current room objects are not colliders.
- Define student inputs and deployment interface; create episode-level train/validation/test splits and evaluate learned policies in closed loop. No student is trained yet.
- Native data collection remains a finite headless mode. The new `run_native.sh --stream` mode displays actual native PhysX dynamics with controls and Gaussian RGB. The older `run_live_bridge.sh` remains a custom-simulator visualization bridge.
- Native and custom records share the same renderer; moving to Isaac does not itself improve Gaussian geometric accuracy.

Exact launch commands and model/data details: [NATIVE_BACKEND.md](../Workflow/Simulation/isaac_sim/NATIVE_BACKEND.md).

## Native WebRTC viewer verification

The native streamed viewer was tested separately after implementation. It starts paused, uses the actual PhysX body, and refreshes rendering without additional physics stepping. An automated localhost control test verified moving pose, changed Gaussian RGB pixels, pause stability, and a successful episode restart. Episode-relative physics/control clock discrepancy was approximately 4.4e-8 s over the tested segment. The native status/control endpoint is local port 8768; WebRTC is 49100. The test viewer exited cleanly. This is runtime verification of controls/data updates; human WebRTC client appearance remains to be inspected on the user's display.

## Properties-inspector UI compatibility verification

The installed `omni.kit.window.property` 1.11.5 implementation was inspected. Its `save_scroll_pos(reset=False)` accesses a missing scrolling frame after a lifecycle change; `reset=True` only clears cached state. A process-local project wrapper now skips only the absent-frame save and preserves reset behavior. In an isolated Isaac runtime, the unwrapped method reproduced the user's `scroll_y_max` exception, the guard handled that case, and both reset and valid-frame scroll saving were preserved. Installed Isaac extension files were not changed. This fix is loaded by the native streaming and older bridge launchers.
