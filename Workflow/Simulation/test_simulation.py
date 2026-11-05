import json
import unittest
import tempfile
import copy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.figs.dynamics import Quadrotor, State
from Workflow.Simulation.figs.integrators import Integrator
from Workflow.Simulation.figs.simulator import Simulator
from Workflow.Simulation.tasks.trajectory import MRoute, HorizontalCircle, FigureEight, make_trajectory
from Workflow.Simulation.rendering.assets import resolve_training_asset


CONFIG = json.loads((Path(__file__).parent/'configs/demo.json').read_text())
CONFIG['dynamics_model'] = 'quadrotor'  # Keep regression coverage for the original model.


class SimulationTests(unittest.TestCase):
    def test_integrator_constant_inputs(self):
        state = State(np.array([0., 0., 2.]), np.array([.1, 0., 0.]), Rotation.identity())
        for model in ['single_integrator', 'double_integrator']:
            vehicle = Integrator(CONFIG['vehicle'], model)
            result = state
            for _ in range(100):
                result = vehicle.step(result, [.2, 0., 0.], .3, .01)
            expected_x = .2 if model == 'single_integrator' else .1+.5*.2
            expected_v = .2 if model == 'single_integrator' else .1+.2
            self.assertAlmostEqual(result.position[0], expected_x)
            self.assertAlmostEqual(result.velocity[0], expected_v)
            self.assertAlmostEqual(result.orientation.as_euler('xyz')[2], .3)
            self.assertAlmostEqual(result.position[2], 2.)  # No gravity in ideal integrators.
            limited = vehicle.step(state, [100., 0., 0.], 100., .1)
            if model == 'single_integrator':
                self.assertAlmostEqual(limited.velocity[0], vehicle.speed_limit)
            else:
                self.assertAlmostEqual(limited.velocity[0], .1+vehicle.acceleration_limit*.1)
            with self.assertRaises(ValueError):
                vehicle.step(state, [float('nan'), 0, 0], 0, .1)

    def test_integrator_tracking_all_shapes_and_disturbance(self):
        for model in ['single_integrator', 'double_integrator']:
            for shape in ['m', 'horizontal_circle', 'lissajous_figure8']:
                config = copy.deepcopy(CONFIG)
                config['dynamics_model'] = model
                config['task'].update(trajectory_type=shape, yaw_deg=30.)
                sim = Simulator(config)
                sim.command({'action': 'play'})
                for i in range(int((sim.route.duration+2)/sim.control_dt)):
                    if i == 100:
                        sim.command({'action': 'disturbance'})
                    sim.advance()
                self.assertLess(max(r['error_m'] for r in sim.telemetry), .15)
                self.assertLess(np.linalg.norm(sim.state.position-sim.reference()[0]), .03)
                self.assertAlmostEqual(sim.state.orientation.as_euler('xyz')[2], np.pi/6)
                self.assertIsNone(sim.snapshot()['thrust_n'])
                self.assertEqual(sim.snapshot()['dynamics_model'], model)

    def test_switch_reference_and_reject_floor_crossing(self):
        sim = Simulator(copy.deepcopy(CONFIG))
        sim.command({'action': 'configure', 'task': {
            'trajectory_type': 'lissajous_figure8', 'vertical_drop_m': 0.}})
        self.assertEqual(sim.snapshot()['trajectory_type'], 'lissajous_figure8')
        self.assertTrue(sim.paused)
        previous = copy.deepcopy(sim.config)
        with self.assertRaises(ValueError):
            sim.command({'action': 'configure', 'task': {'vertical_drop_m': 2.}})
        self.assertEqual(sim.config, previous)
        sim.command({'action': 'configure', 'task': {'trajectory_type': 'horizontal_circle'}})
        self.assertTrue(sim.snapshot()['continuous'])

    def test_analytic_derivatives_agree_with_finite_differences(self):
        task = {**CONFIG['task'], 'radius_m': 1., 'a_m': 1., 'b_m': 1., 'vertical_drop_m': .5}
        for trajectory in [HorizontalCircle(task), FigureEight(task)]:
            for t in [1., 3.2, 7.1, 12.3, 18.]:
                dt = 1e-4
                previous = trajectory.trajectory_function(t-dt)
                current = trajectory.trajectory_function(t)
                following = trajectory.trajectory_function(t+dt)
                np.testing.assert_allclose((following[:3]-previous[:3])/(2*dt), current[3:6], atol=1e-7)
                np.testing.assert_allclose((following[3:6]-previous[3:6])/(2*dt), current[6:9], atol=1e-7)

    def test_figure_eight_endpoints_and_vertical_motion(self):
        reference = FigureEight({**CONFIG['task'], 'a_m': 1., 'b_m': 1., 'vertical_drop_m': .5})
        for t in [-1., 0., reference.duration, reference.duration+5]:
            sample = reference.trajectory_function(t)
            np.testing.assert_allclose(sample[:3], reference.start, atol=1e-12)
            np.testing.assert_allclose(sample[3:9], 0., atol=1e-12)
        self.assertAlmostEqual(reference.trajectory_function(reference.duration/2)[2], reference.start[2]-.5)
        horizontal = FigureEight({**CONFIG['task'], 'vertical_drop_m': 0.})
        self.assertAlmostEqual(horizontal.trajectory_function(7)[2], horizontal.start[2])

    def test_circle_keeps_moving_and_is_stable_at_long_times(self):
        config = copy.deepcopy(CONFIG)
        config['task']['trajectory_type'] = 'horizontal_circle'
        sim = Simulator(config)
        sim.route_time = sim.route.duration+1
        sim.command({'action': 'play'})
        sim.advance()
        self.assertGreater(sim.route_time, sim.route.duration)
        self.assertFalse(sim.snapshot()['complete'])
        self.assertGreater(np.linalg.norm(sim.reference()[1]), .1)
        ref = sim.route.trajectory_function(1e6)
        self.assertTrue(np.isfinite(ref).all())
        self.assertAlmostEqual(np.linalg.norm(ref[3:6]), sim.route.radius*sim.route.omega_max)

    def test_common_reference_interface_and_analytic_tracking(self):
        for name in ['m', 'horizontal_circle', 'lissajous_figure8']:
            config = copy.deepcopy(CONFIG)
            config['task']['trajectory_type'] = name
            trajectory = make_trajectory(config['task'])
            self.assertEqual(trajectory.trajectory_function(1).shape, (10,))
            sim = Simulator(config)
            sim.command({'action': 'play'})
            for _ in range(int((trajectory.duration+2)/sim.control_dt)):
                sim.advance()
            self.assertLess(max(r['error_m'] for r in sim.telemetry), .05)

    def test_transferred_asset_resolves_original_remote_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root/'Workflow/Reconstruction/runs/exact_run/ckpt_last.pt'
            original.parent.mkdir(parents=True)
            original.write_bytes(b'test')
            requested = Path('data/scene/training_reference/Workflow/Reconstruction/runs/exact_run/ckpt_last.pt')
            self.assertEqual(resolve_training_asset(root, requested), original)
            copied = root/requested
            copied.parent.mkdir(parents=True)
            copied.write_bytes(b'copied')
            self.assertEqual(resolve_training_asset(root, requested), copied)
            with self.assertRaises(FileNotFoundError):
                resolve_training_asset(root, 'data/another_run.pt')

    def test_hover_and_body_rate_integration(self):
        vehicle = Quadrotor(CONFIG['vehicle'])
        state = State(np.array([0., 0., 2.]), np.zeros(3), Rotation.identity())
        for _ in range(100):
            state = vehicle.step(state, vehicle.mass*vehicle.gravity, [0, 0, 1.], .01)
        np.testing.assert_allclose(state.position, [0, 0, 2], atol=1e-10)
        self.assertAlmostEqual(state.orientation.as_euler('xyz')[2], 1., places=8)

    def test_quintic_route_respects_speed_and_acceleration(self):
        route = MRoute(CONFIG['task'])
        samples = [route.sample(t) for t in np.linspace(0, route.duration, 2000)]
        self.assertLessEqual(max(np.linalg.norm(s[1]) for s in samples), route.speed+1e-8)
        self.assertLessEqual(max(np.linalg.norm(s[2]) for s in samples), route.acceleration+1e-8)
        np.testing.assert_allclose(route.sample(-1)[0], route.points[0])
        np.testing.assert_allclose(route.sample(route.duration+1)[0], route.points[-1])
        for edge in route.edges:
            self.assertLess(np.linalg.norm(route.sample(route.hold+edge)[1]), 1e-8)

    def test_complete_closed_loop_route_and_disturbance_recovery(self):
        sim = Simulator(CONFIG)
        sim.command({'action': 'play'})
        for i in range(int((sim.route.duration+4)/sim.control_dt)):
            if i == 100:
                sim.command({'action': 'disturbance'})
            sim.advance()
        self.assertLess(np.linalg.norm(sim.state.position-sim.route.points[-1]), .01)
        self.assertLess(max(row['error_m'] for row in sim.telemetry), .15)
        np.testing.assert_allclose(np.linalg.norm(sim.state.orientation.as_quat()), 1., atol=1e-10)

    def test_pause_reset_and_invalid_settings(self):
        sim = Simulator(CONFIG)
        sim.advance()
        self.assertEqual(sim.time, 0.)
        sim.command({'action': 'nudge', 'delta': [0, 0, .1]})
        self.assertEqual(sim.mode, 'hover')
        sim.advance()
        self.assertGreater(sim.time, 0.)
        with self.assertRaises(ValueError):
            sim.command({'action': 'configure', 'task': {'speed_m_s': float('nan')}})
        sim.command({'action': 'reset'})
        self.assertTrue(sim.paused)
        self.assertEqual(sim.time, 0.)
        np.testing.assert_allclose(sim.state.position, sim.route.points[0])


if __name__ == '__main__':
    unittest.main()
