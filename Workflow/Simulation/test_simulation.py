import json
import unittest
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.figs.dynamics import Quadrotor, State
from Workflow.Simulation.figs.simulator import Simulator
from Workflow.Simulation.tasks.trajectory import MRoute
from Workflow.Simulation.rendering.assets import resolve_training_asset


CONFIG = json.loads((Path(__file__).parent/'configs/demo.json').read_text())


class SimulationTests(unittest.TestCase):
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
