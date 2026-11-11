"""Nonlinear model gradients, action interface, recovery, and RGB timing gates."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.control.quadrotor_mpc import QuadrotorMPC
from Workflow.Simulation.figs.dynamics import Quadrotor, State
from Workflow.Simulation.figs.simulator import Simulator
from Workflow.Simulation.dataset import DemonstrationRecorder
from Workflow.Simulation.audit_dataset import audit_episode

CONFIG = json.loads((Path(__file__).parent/'configs/quadrotor_nmpc.json').read_text())


@unittest.skipUnless(importlib.util.find_spec('casadi'), 'Install requirements-nmpc.txt for nonlinear MPC tests')
class QuadrotorMPCTests(unittest.TestCase):
    def test_symbolic_dynamics_and_gradients_match_numeric(self):
        mpc = QuadrotorMPC({'horizon_steps': 6}, CONFIG['vehicle'], .05, .01)
        vehicle = Quadrotor(CONFIG['vehicle'])
        state = State(np.array([.1, -.2, 1.2]), np.array([.2, .1, -.1]), Rotation.from_euler('xyz', [.1, -.1, .3]))
        action = np.array([9., .1, -.2, .3])
        expected = vehicle.step(state, action[0], action[1:], .01)
        initial = np.r_[state.position, state.velocity, state.orientation.as_quat(), np.zeros(3), vehicle.mass*vehicle.gravity]
        actual = np.asarray(mpc.physics_step(initial, action)).ravel()
        np.testing.assert_allclose(actual, np.r_[expected.position, expected.velocity, expected.orientation.as_quat(), expected.body_rates, expected.actuator_thrust_n], atol=1e-12)
        refs = np.zeros((7, 11));refs[:, 2] = 1.2
        desired, ff = mpc.reference_targets(refs)
        x0 = initial
        inputs = mpc.plan.ravel()
        _, gradient = mpc.objective(inputs, x0, desired, ff, mpc.last_action)
        for index in [0, 1, 2, 3, 10, 20]:
            plus, minus = inputs.copy(), inputs.copy()
            plus[index] += 1e-5;minus[index] -= 1e-5
            a, _ = mpc.objective(plus, x0, desired, ff, mpc.last_action)
            b, _ = mpc.objective(minus, x0, desired, ff, mpc.last_action)
            self.assertAlmostEqual(float((a-b)/2e-5), float(gradient[index]), places=5)

    def test_hover_tracking_response_mismatch_and_action_contract(self):
        config = copy.deepcopy(CONFIG)
        config['randomization']['enabled'] = True
        sim = Simulator(config);sim.paused = False
        for _ in range(70):sim.advance()
        self.assertLess(max(r['error_m'] for r in sim.telemetry), .15)
        self.assertLessEqual(sim.thrust, sim.controller.max_thrust+1e-9)
        self.assertLessEqual(max(abs(sim.rates)), sim.controller.max_rate+1e-9)
        row = sim.demonstrations[-1]
        self.assertEqual(row['action_type'], 'thrust_body_rates')
        self.assertNotIn('action_acceleration_m_s2', row)
        self.assertNotEqual(sim.vehicle.mass, sim.controller.mass)
        self.assertEqual(row['solver']['status'], 'solved')

    def test_rgb_labels_are_pre_action_and_corruption_is_rejected(self):
        class Renderer:
            def render(self, state):
                return np.full((6, 8, 3), 100, dtype=np.uint8), None
        sim = Simulator(copy.deepcopy(CONFIG));sim.paused = False
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            recorder = DemonstrationRecorder(folder, Renderer())
            for _ in range(3):
                observation = recorder.capture(sim)
                sim.advance()
                recorder.write(sim.demonstrations[-1], observation)
            recorder.close()
            result = audit_episode(folder)
            self.assertEqual(result['samples'], 3)
            rows = [json.loads(line) for line in (folder/'rgb_actions.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['image_time_s'], 0.)
            self.assertEqual(sim.telemetry[0]['time_s'], .05)
            (folder/rows[0]['image']).write_bytes(b'corrupted image')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                audit_episode(folder)


if __name__ == '__main__':
    unittest.main()
