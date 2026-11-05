"""Constrained linear MPC: world acceleration and yaw rate, nominal integrators.

Condenses linear dynamics into a strictly convex, box-constrained quadratic
program, solved with SciPy L-BFGS-B using an analytic gradient and warm start.
No obstacle, state, or motor constraints are implied.
"""
import time

import numpy as np
from scipy.optimize import minimize


class LinearMPC:
    def __init__(self, config, vehicle_config, dt):
        self.dt = float(dt)
        self.horizon = int(config.get('horizon_steps', 20))
        if self.horizon < 2 or self.horizon > 100 or self.dt <= 0:
            raise ValueError('MPC needs 2..100 horizon steps and positive dt')
        self.max_iterations = int(config.get('max_iterations', 150))
        self.a_limit = float(vehicle_config.get('max_acceleration_m_s2', 2.))/np.sqrt(3.)
        self.r_limit = float(vehicle_config.get('max_body_rate_rad_s', 4.))
        self.limits = np.tile([self.a_limit]*3+[self.r_limit], self.horizon)
        weights = np.array([config.get('position_weight', 30.)]*3+
                           [config.get('velocity_weight', 3.)]*3+[config.get('yaw_weight', 8.)], dtype=float)
        input_weight = float(config.get('input_weight', .08))
        smooth_weight = float(config.get('input_change_weight', .1))
        terminal = float(config.get('terminal_multiplier', 4.))
        if not np.isfinite(np.r_[weights, input_weight, smooth_weight, terminal, self.limits]).all() or min(weights) <= 0 or input_weight <= 0 or smooth_weight < 0 or terminal <= 0:
            raise ValueError('Invalid MPC weights or input limits')
        if min(self.limits) <= 0 or self.max_iterations < 1:
            raise ValueError('MPC limits and iteration count must be positive')
        A = np.eye(7)
        A[:3, 3:6] = self.dt*np.eye(3)
        B = np.zeros((7, 4))
        B[:3, :3] = .5*self.dt**2*np.eye(3)
        B[3:6, :3] = self.dt*np.eye(3)
        B[6, 3] = self.dt
        self.A, self.B = A, B
        self.Sx = np.vstack([np.linalg.matrix_power(A, k+1) for k in range(self.horizon)])
        self.Su = np.zeros((7*self.horizon, 4*self.horizon))
        for i in range(self.horizon):
            for j in range(i+1):
                self.Su[7*i:7*i+7, 4*j:4*j+4] = np.linalg.matrix_power(A, i-j)@B
        self.weights = np.tile(weights, self.horizon)
        self.weights[-7:] *= terminal
        self.D = np.eye(4*self.horizon)-np.eye(4*self.horizon, k=-4)
        self.input_weight, self.smooth_weight = input_weight, smooth_weight
        self.H = self.Su.T@(self.weights[:, None]*self.Su)+input_weight*np.eye(4*self.horizon)+smooth_weight*self.D.T@self.D
        self.last_action = np.zeros(4)
        self.plan = np.zeros((self.horizon, 4))
        self.diagnostics = dict(status='not_run', iterations=0, solve_ms=0.)

    def command_horizon(self, state, references):
        """references: (N+1, 11) rows [p3, v3, a3, yaw, yaw_rate], now..future."""
        refs = np.asarray(references, dtype=float)
        if refs.shape != (self.horizon+1, 11) or not np.isfinite(refs).all():
            raise ValueError('MPC reference horizon has invalid shape or nonfinite values')
        actual_yaw = state.orientation.as_euler('xyz')[2]
        yaw = refs[0, 9]+np.arctan2(np.sin(actual_yaw-refs[0, 9]), np.cos(actual_yaw-refs[0, 9]))
        x0 = np.r_[state.position, state.velocity, yaw]
        desired = np.c_[refs[1:, :6], refs[1:, 9]].ravel()
        nominal_input = np.c_[refs[:-1, 6:9], refs[:-1, 10]].ravel()
        delta_target = np.r_[self.last_action, np.zeros(4*(self.horizon-1))]
        residual = self.Sx@x0-desired
        linear = self.Su.T@(self.weights*residual)-self.input_weight*nominal_input-self.smooth_weight*self.D.T@delta_target
        initial = np.vstack([self.plan[1:], self.plan[-1:]]).ravel()
        start = time.perf_counter()
        def objective(u):
            Hu = self.H@u
            return float(u@Hu+2*linear@u), 2*(Hu+linear)
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          bounds=list(zip(-self.limits, self.limits)),
                          options=dict(maxiter=self.max_iterations, ftol=1e-12, gtol=1e-6, maxls=40))
        self.diagnostics = dict(status='solved' if result.success else 'failed',
                                message=str(result.message), iterations=int(result.nit),
                                solve_ms=1000*(time.perf_counter()-start))
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f'MPC solve failed: {result.message}')
        self.plan = np.clip(result.x, -self.limits, self.limits).reshape(self.horizon, 4)
        self.last_action = self.plan[0].copy()
        return self.last_action[:3].copy(), float(self.last_action[3])
