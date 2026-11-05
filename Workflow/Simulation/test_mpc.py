"""Expert optimality, anticipation, robustness, reproducibility, and data timing."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.control.mpc import LinearMPC
from Workflow.Simulation.figs.dynamics import State
from Workflow.Simulation.figs.simulator import Simulator


CONFIG = json.loads((Path(__file__).parent/'configs/demo.json').read_text())


class MPCTests(unittest.TestCase):
    def config(self, randomize=False):
        config = copy.deepcopy(CONFIG)
        config['dynamics_model'] = 'double_integrator'
        config['controller']['type'] = 'mpc'
        config['randomization']['enabled'] = randomize
        config['task'].update(trajectory_type='lissajous_figure8', yaw_mode='velocity_aligned')
        return config

    def test_condensed_dynamics_match_direct_integration(self):
        mpc = LinearMPC({}, CONFIG['vehicle'], .05)
        rng = np.random.default_rng(3)
        x = rng.normal(size=7)
        inputs = rng.normal(size=(mpc.horizon, 4))
        expected = []
        current = x.copy()
        for action in inputs:
            current = mpc.A@current+mpc.B@action
            expected.extend(current)
        np.testing.assert_allclose(mpc.Sx@x+mpc.Su@inputs.ravel(), expected, atol=1e-12)

    def test_unconstrained_optimum_and_future_anticipation(self):
        mpc = LinearMPC({'horizon_steps': 12}, CONFIG['vehicle'], .05)
        state = State(np.array([0., 0., 2.]), np.zeros(3), Rotation.identity())
        refs = np.zeros((13, 11)); refs[:, 2] = 2.
        refs[4:, 0] = .02  # Future target changes; current target/error is zero.
        action, rate = mpc.command_horizon(state, refs)
        self.assertGreater(action[0], .01)
        target = np.c_[refs[1:, :6], refs[1:, 9]].ravel()
        linear = mpc.Su.T@(mpc.weights*(mpc.Sx@np.r_[state.position, state.velocity, 0.]-target))
        expected = -np.linalg.solve(mpc.H, linear)
        self.assertLess(np.max(np.abs(expected)), min(mpc.limits))
        np.testing.assert_allclose(mpc.plan.ravel(), expected, atol=2e-5)
        self.assertAlmostEqual(rate, 0.)

    def test_bounds_yaw_wrap_and_solver_failure(self):
        mpc = LinearMPC({}, CONFIG['vehicle'], .05)
        refs = np.zeros((21, 11)); refs[:, :3] = [100., 100., 2.]
        refs[:, 9] = np.deg2rad(-179.)
        state = State(np.array([0., 0., 2.]), np.zeros(3), Rotation.from_euler('z', 179., degrees=True))
        action, rate = mpc.command_horizon(state, refs)
        self.assertLessEqual(np.linalg.norm(action), CONFIG['vehicle']['max_acceleration_m_s2']+1e-9)
        self.assertGreater(rate, 0.)
        self.assertLess(rate, .5)  # Correct short rotation across the ±pi boundary.
        self.assertTrue((np.abs(mpc.plan.ravel()) <= mpc.limits+1e-9).all())
        with patch('Workflow.Simulation.control.mpc.minimize', return_value=SimpleNamespace(success=False, message='forced failure', nit=1)):
            with self.assertRaisesRegex(RuntimeError, 'MPC solve failed'):
                mpc.command_horizon(state, refs)
        self.assertEqual(mpc.diagnostics['status'], 'failed')

    def test_randomized_tracking_and_demonstration_timing(self):
        for shape in ['m', 'lissajous_figure8', 'horizontal_circle']:
            for yaw_mode in ['constant', 'velocity_aligned']:
                config = self.config(True)
                config['task'].update(trajectory_type=shape, yaw_mode=yaw_mode)
                sim = Simulator(config)
                before = sim.state.position.copy()
                sim.command({'action': 'play'})
                sim.advance()
                np.testing.assert_array_equal(sim.demonstrations[0]['position_m'], before)
                self.assertEqual(sim.demonstrations[0]['time_s'], 0.)
                self.assertEqual(sim.telemetry[0]['time_s'], sim.control_dt)
                self.assertEqual(sim.demonstrations[0]['reference_horizon'][0][9], sim.route.yaw_reference(0.)[0])
                for i in range(int((sim.route.duration+2)/sim.control_dt)-1):
                    if i == 100:
                        sim.command({'action': 'disturbance'})
                    sim.advance()
                self.assertLess(max(r['error_m'] for r in sim.telemetry), .15)
                self.assertTrue(all(row['solver']['status'] == 'solved' for row in sim.demonstrations))
                self.assertEqual(len(sim.demonstrations), len(sim.telemetry))
                sim.command({'action': 'hover'})
                horizon = sim.reference_horizon(sim.controller.horizon)
                np.testing.assert_allclose(horizon, np.tile(horizon[0], (len(horizon), 1)))

    def test_seeded_sampling_and_nominal_expert(self):
        config = self.config(True)
        a, b = Simulator(config), Simulator(config)
        self.assertEqual(a.episode_parameters, b.episode_parameters)
        self.assertNotEqual(a.vehicle.acceleration_scale, 1.)
        self.assertEqual(a.controller.a_limit, config['vehicle']['max_acceleration_m_s2']/np.sqrt(3.))
        for sim in (a, b):
            sim.command({'action': 'play'})
            for _ in range(10):
                sim.advance()
        np.testing.assert_array_equal(a.state.position, b.state.position)
        first = copy.deepcopy(a.episode_parameters)
        a.reset(); b.reset()
        self.assertEqual(a.episode_parameters, b.episode_parameters)
        self.assertNotEqual(first['plant'], a.episode_parameters['plant'])
        invalid = self.config(); invalid['dynamics_model'] = 'quadrotor'
        with self.assertRaisesRegex(ValueError, 'requires'):
            Simulator(invalid)
        invalid = self.config(True); invalid['randomization']['z_offset_m'] = [-3., -3.]
        with self.assertRaisesRegex(ValueError, 'floor'):
            Simulator(invalid)


if __name__ == '__main__':
    unittest.main()
