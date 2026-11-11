"""Physical response and synchronized dataset integration for the custom plant."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.figs.dynamics import State
from Workflow.Simulation.figs.rigid_body import RigidBodyQuadrotor
from Workflow.Simulation.figs.simulator import Simulator
from Workflow.Simulation.dataset import DemonstrationRecorder
from Workflow.Simulation.audit_dataset import audit_episode
from Workflow.Simulation.randomization import sample_episode
from Workflow.Simulation.tasks.trajectory import make_trajectory

CONFIG = json.loads((Path(__file__).parent/'configs/quadrotor_nmpc.json').read_text())


def initial():
    return State(np.array([0., 0., 1.2]), np.zeros(3), Rotation.identity(), np.zeros(3))


class RigidBodyTests(unittest.TestCase):
    def test_collection_duration_minimum(self):
        self.assertEqual(CONFIG['task']['duration_s'], 30.)
        config = copy.deepcopy(CONFIG)
        config['randomization']['enabled'] = True
        for seed in range(20):
            task, _, _ = sample_episode(config, np.random.default_rng(seed))
            self.assertGreaterEqual(make_trajectory(task).duration, 30.)
        with self.assertRaisesRegex(ValueError, 'at least 30'):
            make_trajectory({**CONFIG['task'], 'duration_s': 29.})
        config['randomization']['duration_scale'] = [.9, .9]
        with self.assertRaisesRegex(ValueError, 'below task.minimum_duration_s'):
            sample_episode(config, np.random.default_rng(0))

    def test_hover_and_tilted_thrust(self):
        plant = RigidBodyQuadrotor(CONFIG['vehicle'])
        state = initial()
        for _ in range(200):
            state = plant.step(state, plant.mass*plant.gravity, np.zeros(3), .01)
        np.testing.assert_allclose(state.position, [0, 0, 1.2], atol=1e-10)
        tilted = initial()
        tilted.orientation = Rotation.from_euler('y', .2)
        result = plant.step(tilted, plant.mass*plant.gravity, np.zeros(3), .01)
        self.assertGreater(result.velocity[0], 0.)
        self.assertLess(result.velocity[2], 0.)

    def test_inertia_changes_torque_limited_response(self):
        states = []
        for multiplier in [1., 2.]:
            config = copy.deepcopy(CONFIG['vehicle'])
            config['inertia_kg_m2'] = (np.array(config['inertia_kg_m2'])*multiplier).tolist()
            config['max_body_torque_nm'] = [.001]*3
            plant = RigidBodyQuadrotor(config)
            result = plant.step(initial(), plant.mass*plant.gravity, [4., 0., 0.], .01)
            self.assertEqual(plant.torque_clips, 1)
            self.assertLessEqual(max(abs(plant.last_torque_nm)), .001)
            states.append(result)
        self.assertAlmostEqual(states[0].body_rates[0]/states[1].body_rates[0], 2., places=8)
        self.assertGreater(states[0].orientation.as_rotvec()[0], states[1].orientation.as_rotvec()[0])

    def test_floor_and_invalid_parameters(self):
        plant = RigidBodyQuadrotor(CONFIG['vehicle'])
        state = initial()
        for _ in range(200):
            state = plant.step(state, 0., np.zeros(3), .01)
        self.assertEqual(state.position[2], plant.floor)
        self.assertEqual(state.velocity[2], 0.)
        for override in [{'inertia_kg_m2': [0., 1., 1.]}, {'rate_time_constant_s': 0.},
                         {'rigid_body_max_step_s': float('nan')}]:
            with self.assertRaises(ValueError):
                RigidBodyQuadrotor({**CONFIG['vehicle'], **override})

    @unittest.skipUnless(importlib.util.find_spec('casadi'), 'NMPC dependency required')
    def test_actual_pose_and_nmpc_labels_are_synchronized(self):
        class Renderer:
            metadata = {'kind': 'test only: pose-coded pixels, no 3DGS'}
            def render(self, state):
                value = int(np.clip(100+state.orientation.as_euler('xyz', degrees=True)[1], 0, 255))
                return np.full((6, 8, 3), value, dtype=np.uint8), None
        config = copy.deepcopy(CONFIG)
        config['randomization']['enabled'] = True
        config['seed'] = 1000
        sim = Simulator(config)
        sim.paused = False
        self.assertIsInstance(sim.vehicle, RigidBodyQuadrotor)
        with tempfile.TemporaryDirectory() as directory:
            recorder = DemonstrationRecorder(directory, Renderer())
            for _ in range(12):
                observation = recorder.capture(sim)
                sim.advance()
                recorder.write(sim.demonstrations[-1], observation)
            recorder.close()
            audit = audit_episode(directory)
            self.assertEqual(audit['samples'], 12)
            self.assertEqual(audit['physics_backend'], RigidBodyQuadrotor.physics_backend)
            self.assertGreater(np.linalg.norm(sim.state.body_rates), 0.)
            self.assertLess(max(row['error_m'] for row in sim.telemetry), .2)


if __name__ == '__main__':
    unittest.main()
