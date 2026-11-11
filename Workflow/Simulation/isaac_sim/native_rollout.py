#!/usr/bin/env python3
"""Finite native PhysX drone validation/rollout, with optional synchronized RGB.

Run through run_native.sh. No real-drone connection or external asset downloads.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'Workflow/Simulation/configs/quadrotor_nmpc.json')
    parser.add_argument('--duration', type=float, default=0., help='0 runs full route plus two seconds')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--trajectory', choices=['m', 'horizontal_circle', 'lissajous_figure8'])
    parser.add_argument('--yaw-mode', choices=['constant', 'velocity_aligned'])
    parser.add_argument('--checks', action='store_true', help='Verify hover, rate response and native floor contact first')
    parser.add_argument('--stream', action='store_true', help='Interactive native PhysX viewer through WebRTC')
    parser.add_argument('--viewer-seconds', type=float, default=0., help='Optional finite viewer test duration; 0 stays open')
    parser.add_argument('--control-port', type=int, default=8768, help='Localhost native viewer status/control port')
    parser.add_argument('--render-url', help='Optional localhost 3DGUT pose-render service')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stream:
        if not 1 <= args.control_port <= 65535:
            parser.error('Invalid native viewer control port')
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                probe.bind(('127.0.0.1',args.control_port))
        except OSError as error:
            parser.error(f'Native viewer control port unavailable: {error}. Stop the existing viewer first.')
        with socket.socket() as probe:
            probe.settimeout(1.)
            if probe.connect_ex(('127.0.0.1',49100))==0:
                parser.error('WebRTC port 49100 is already in use. Stop the old Isaac viewer before launching native streaming.')
    if not math.isfinite(args.viewer_seconds) or args.viewer_seconds<0:
        parser.error('--viewer-seconds must be finite and nonnegative')
    if not math.isfinite(args.duration) or args.duration < 0 or args.episodes < 1 or args.seed < 0:
        parser.error('Duration must be nonnegative and episodes positive')
    output = args.output or ROOT/'data/scene_20261025_200034/isaac_sim'/('native_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    if config.get('dynamics_model') != 'quadrotor' or config.get('controller', {}).get('type') != 'nmpc':
        parser.error('Native expert rollout requires quadrotor dynamics and nmpc controller')
    config['randomization']['enabled'] = args.randomize
    if args.trajectory:
        config['task']['trajectory_type'] = args.trajectory
    if args.yaw_mode:
        config['task']['yaw_mode'] = args.yaw_mode
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    report = dict(status='starting', physics_backend='isaac_physx_forces_and_torques',
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), tests={}, episodes=[])
    report['max_position_error_gate_m'] = config.get('validation', {}).get('max_position_error_m', .2)
    report['effective_config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'Workflow/Simulation').rglob('*.py'))}
    path = output/'report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print('NATIVE_OUTPUT='+str(output), flush=True)
    application = None
    world = None
    try:
        from isaacsim import SimulationApp
        isaac_root = Path(os.environ.get('ISAAC_SIM_ROOT', str(Path.home()/'isaacsim')))
        experience = isaac_root/('apps/isaacsim.exp.full.streaming.kit' if args.stream else 'apps/isaacsim.exp.base.python.kit')
        application = SimulationApp({'headless': True, 'width': 1280 if args.stream else 640, 'height': 720 if args.stream else 360},
            experience=str(experience))
        if args.stream:
            from Workflow.Simulation.isaac_sim.ui_compat import install_property_scroll_guard
            report['property_scroll_guard'] = install_property_scroll_guard()
            print('NATIVE_UI_COMPAT='+json.dumps(report['property_scroll_guard']), flush=True)
        import numpy as np
        import scipy
        import casadi
        from scipy.spatial.transform import Rotation
        from isaacsim.core.api import World
        from pxr import UsdGeom
        from Workflow.Simulation.isaac_sim.native_backend import NativeSimulator
        from Workflow.Simulation.run import write_telemetry
        from Workflow.Simulation.figs.dynamics import State
        from Workflow.Simulation.dataset import DemonstrationRecorder, PoseRenderClient
        world = World(stage_units_in_meters=1., physics_dt=1/config['physics_hz'],
                      rendering_dt=1/config['render_hz'], backend='numpy')
        UsdGeom.SetStageUpAxis(world.stage, UsdGeom.Tokens.z)
        world.get_physics_context().set_gravity(-config['vehicle'].get('gravity_m_s2', 9.81))
        report['software'] = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, casadi=casadi.__version__,
                                  isaac=(isaac_root/'VERSION').read_text().strip(), physics_compute='CPU PhysX')
        check_config = copy.deepcopy(config)
        check_config['randomization']['enabled'] = False
        sim = NativeSimulator(check_config, world)
        if args.checks:
            vehicle = sim.vehicle
            h = 1/config['physics_hz']
            initial = State(np.array([0., 0., 1.5]), np.zeros(3), Rotation.identity())
            vehicle.initialize_state(initial)
            state = initial
            for _ in range(round(2/h)):
                state = vehicle.step(state, vehicle.mass*vehicle.gravity, [0., 0., 0.], h)
            report['tests']['hover_drift_m'] = float(np.linalg.norm(state.position-initial.position))
            report['tests']['configured_inertia'] = vehicle.inertia.tolist()
            report['tests']['actual_physx_inertia'] = vehicle.physics_inertia.tolist()
            axis_errors = []
            for axis in range(3):
                vehicle.initialize_state(initial)
                state = initial
                rates = np.zeros(3); rates[axis] = .5
                for _ in range(round(.5/h)):
                    state = vehicle.step(state, vehicle.mass*vehicle.gravity, rates, h)
                axis_errors.append(float(np.linalg.norm(vehicle.actual_rates-rates)))
            report['tests']['body_rate_axis_errors_rad_s'] = axis_errors
            vehicle.initialize_state(initial)
            state = initial
            for _ in range(round(1/h)):
                state = vehicle.step(state, vehicle.mass*vehicle.gravity, [0., 0., .5], h)
            report['tests']['yaw_rate_error_rad_s'] = float(abs(vehicle.actual_rates[2]-.5))
            report['tests']['yaw_change_rad'] = float(state.orientation.as_euler('xyz')[2])
            vehicle.initialize_state(initial)
            state = initial
            minimum_z = state.position[2]
            for _ in range(round(4/h)):
                state = vehicle.step(state, 0., [0., 0., 0.], h)
                minimum_z = min(minimum_z, state.position[2])
            report['tests']['floor_final_center_z_m'] = float(state.position[2])
            report['tests']['floor_min_center_z_m'] = float(minimum_z)
            passed = (report['tests']['hover_drift_m'] < .02 and max(axis_errors) < .03 and report['tests']['yaw_rate_error_rad_s'] < .03
                      and report['tests']['yaw_change_rad'] > .3 and .005 < state.position[2] < .05 and minimum_z > -.01)
            report['tests']['passed'] = bool(passed)
            print('NATIVE_CHECKS='+json.dumps(report['tests']), flush=True)
            if not passed:
                raise RuntimeError('Native hover/rate/floor checks failed; no dataset collected')
        render_client = PoseRenderClient(args.render_url) if args.render_url else None
        sim.config['randomization']['enabled'] = args.randomize
        if args.stream:
            from Workflow.Simulation.isaac_sim.native_viewer import NativeViewer
            sim.config['seed'] = args.seed
            sim.rng = np.random.default_rng(args.seed)
            sim.reset()
            viewer = NativeViewer(sim, application, output, render_client, args.control_port)
            report['interactive'] = viewer.run(args.viewer_seconds)
            report['status'] = 'interactive_closed' if report['interactive']['error'] is None else 'failed'
            world.stage.GetRootLayer().Export(str(output/'native_scene.usda'))
            return
        for episode in range(args.episodes):
            sim.config['seed'] = args.seed+episode
            sim.rng = np.random.default_rng(args.seed+episode)
            sim.reset()
            episode_output = output/f'episode_{episode+1:04d}'
            episode_output.mkdir()
            (episode_output/'episode_parameters.json').write_text(json.dumps(sim.episode_parameters, indent=2)+'\n')
            duration = args.duration or sim.route.duration+2.
            recorder = DemonstrationRecorder(episode_output, render_client) if render_client else None
            sim.paused = False
            before = time.perf_counter()
            try:
                for step in range(int(np.ceil(duration/sim.control_dt))):
                    if not application.is_running():
                        raise RuntimeError('Isaac closed during rollout')
                    observation = recorder.capture(sim) if recorder else None
                    sim.advance()
                    if recorder:
                        recorder.write(sim.demonstrations[-1], observation)
                    if step % 100 == 0:
                        print(f'NATIVE_STEP episode={episode+1} t={sim.time:.2f} error={sim.snapshot()["tracking_error_m"]:.4f}', flush=True)
            finally:
                if recorder:
                    recorder.close()
                write_telemetry(episode_output/'telemetry.csv', sim.telemetry)
                with (episode_output/'demonstrations.jsonl').open('w') as f:
                    for row in sim.demonstrations:
                        f.write(json.dumps(row)+'\n')
            errors = np.array([row['error_m'] for row in sim.telemetry])
            solves = np.array([row['mpc_solve_ms'] for row in sim.telemetry])
            result = dict(seed=args.seed+episode, steps=len(errors), position_rms_m=float(np.sqrt(np.mean(errors**2))),
                position_max_m=float(errors.max()), solve_mean_ms=float(solves.mean()), solve_p95_ms=float(np.percentile(solves, 95)),
                solve_max_ms=float(solves.max()), failed_solves=0, wall_seconds=time.perf_counter()-before,
                torque_limit_steps=sim.vehicle.torque_clips,
                passed=bool(np.isfinite(errors).all() and errors.max() < report['max_position_error_gate_m']))
            report['episodes'].append(result)
            print('NATIVE_EPISODE='+json.dumps(result), flush=True)
            if not result['passed']:
                raise RuntimeError('Native tracking exceeded development acceptance limit')
        report['status'] = 'passed'
        world.stage.GetRootLayer().Export(str(output/'native_scene.usda'))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        path.write_text(json.dumps(report, indent=2)+'\n')
        print('NATIVE_REPORT='+json.dumps({k:v for k,v in report.items() if k != 'source_sha256'}), flush=True)
        if world is not None:
            world.stop()
        if application is not None:
            application.close()
    print('NATIVE_REPORT='+json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
