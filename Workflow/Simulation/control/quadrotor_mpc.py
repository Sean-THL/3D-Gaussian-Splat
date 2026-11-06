"""Nonlinear receding-horizon expert with thrust (N) and body-rate (rad/s) actions.

CasADi differentiates the quaternion shooting model; SciPy SLSQP solves the
bounded nonlinear program. This is local optimization, with explicit failure
reporting, not a globally optimal or collision-certified flight controller.
"""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


class QuadrotorMPC:
    def __init__(self, config, vehicle, dt, physics_dt):
        try:
            import casadi as ca
        except ImportError as error:
            raise RuntimeError('Quadrotor NMPC requires CasADi. Install control/requirements-nmpc.txt into the project environment.') from error
        self.ca = ca
        self.dt = float(dt)
        self.horizon = int(config.get('horizon_steps', 20))
        self.substeps = round(self.dt/physics_dt)
        if not 2 <= self.horizon <= 60 or self.substeps < 1 or not np.isclose(self.substeps*physics_dt, self.dt):
            raise ValueError('Invalid NMPC horizon or physics/control time steps')
        self.mass = float(vehicle['mass_kg'])
        self.gravity = float(vehicle.get('gravity_m_s2', 9.81))
        self.drag = float(vehicle.get('linear_drag_per_s', .15))
        self.rate_tau = float(vehicle.get('rate_time_constant_s', .04))
        self.thrust_tau = float(vehicle.get('thrust_time_constant_s', .03))
        self.max_thrust = self.mass*self.gravity*float(vehicle.get('max_thrust_to_weight', 2.5))
        self.max_rate = float(vehicle.get('max_body_rate_rad_s', 4.))
        self.max_iterations = int(config.get('max_iterations', 150))
        self.weights = dict(position=float(config.get('position_weight', 40.)),
                            velocity=float(config.get('velocity_weight', 5.)),
                            attitude=float(config.get('attitude_weight', 4.)),
                            input=float(config.get('input_weight', .04)),
                            smooth=float(config.get('input_change_weight', .08)),
                            terminal=float(config.get('terminal_multiplier', 4.)))
        values = np.r_[list(self.weights.values()), self.mass, self.gravity, self.max_thrust, self.max_rate]
        if not np.isfinite(values).all() or min(values) <= 0 or self.drag < 0 or self.max_iterations < 1:
            raise ValueError('Invalid NMPC physical parameters or weights')
        self.last_action = np.array([self.mass*self.gravity, 0., 0., 0.])
        self.plan = np.tile(self.last_action, (self.horizon, 1))
        self.bounds = [(0., self.max_thrust), *[(-self.max_rate, self.max_rate)]*3]*self.horizon
        self.decision_scale = np.tile([self.mass*self.gravity, 1., 1., 1.], self.horizon)
        self.normalized_bounds = [(low/scale, high/scale) for (low, high), scale in zip(self.bounds, self.decision_scale)]
        self.diagnostics = dict(status='not_run', solve_ms=0., iterations=0)
        self._build_model(float(physics_dt))

    def _build_model(self, h):
        ca = self.ca
        def multiply(a, b):
            return ca.vertcat(a[3]*b[:3]+b[3]*a[:3]+ca.cross(a[:3], b[:3]), a[3]*b[3]-ca.dot(a[:3], b[:3]))
        def increment(rates, duration):
            norm = ca.sqrt(ca.dot(rates, rates)+1e-16)
            return ca.vertcat(rates*(ca.sin(norm*duration/2)/norm), ca.cos(norm*duration/2))
        # Four internal actuator states let MPC anticipate finite response.
        # Output commands remain thrust/body rates, not torques or rotor speeds.
        x, u = ca.SX.sym('x', 14), ca.SX.sym('u', 4)
        def response(previous, target, tau):
            if tau <= 0:
                return target, target
            decay = np.exp(-h/tau)
            return target+(previous-target)*decay, target+(previous-target)*(tau/h)*(1-decay)
        next_rates, average_rates = response(x[10:13], u[1:4], self.rate_tau)
        next_thrust, average_thrust = response(x[13], u[0], self.thrust_tau)
        q = x[6:10]/ca.norm_2(x[6:10])
        mid = multiply(q, increment(average_rates, h/2))
        e3 = ca.DM([0., 0., 1.])
        direction = e3+2*mid[3]*ca.cross(mid[:3], e3)+2*ca.cross(mid[:3], ca.cross(mid[:3], e3))
        acceleration = direction*(average_thrust/self.mass)-self.gravity*e3-self.drag*x[3:6]
        next_q = multiply(q, increment(average_rates, h))
        nxt = ca.vertcat(x[:3]+h*x[3:6]+.5*h*h*acceleration, x[3:6]+h*acceleration, next_q/ca.norm_2(next_q), next_rates, next_thrust)
        self.physics_step = ca.Function('quadrotor_step', [x, u], [nxt])
        actions = ca.MX.sym('actions', 4*self.horizon)
        initial = ca.MX.sym('initial', 14)
        desired = ca.MX.sym('desired', 10*self.horizon)
        feedforward = ca.MX.sym('feedforward', 4*self.horizon)
        previous = ca.MX.sym('previous', 4)
        state, old, cost = initial, previous, 0
        input_scale = ca.DM([self.mass*self.gravity, self.max_rate, self.max_rate, self.max_rate])
        for k in range(self.horizon):
            control = actions[4*k:4*k+4]
            for _ in range(self.substeps):
                state = self.physics_step(state, control)
            target = desired[10*k:10*k+10]
            tracking = (self.weights['position']*ca.sumsqr(state[:3]-target[:3])+
                        self.weights['velocity']*ca.sumsqr(state[3:6]-target[3:6])+
                        self.weights['attitude']*(1-ca.dot(state[6:10], target[6:10])**2))
            cost += (self.weights['terminal'] if k == self.horizon-1 else 1.)*tracking
            cost += self.weights['input']*ca.sumsqr((control-feedforward[4*k:4*k+4])/input_scale)
            cost += self.weights['smooth']*ca.sumsqr((control-old)/input_scale)
            old = control
        self.objective = ca.Function('quadrotor_cost_gradient', [actions, initial, desired, feedforward, previous],
                                     [cost, ca.gradient(cost, actions)])

    def reference_targets(self, references):
        targets, thrusts, rotations = [], [], []
        for row in references:
            force = row[6:9]+[0., 0., self.gravity]+self.drag*row[3:6]
            if np.linalg.norm(force) < 1e-6:
                raise ValueError('Reference requires undefined thrust direction')
            z = force/np.linalg.norm(force)
            heading = np.array([np.cos(row[9]), np.sin(row[9]), 0.])
            y = np.cross(z, heading)
            if np.linalg.norm(y) < 1e-6:
                raise ValueError('Reference heading parallel to thrust direction')
            y /= np.linalg.norm(y)
            rotation = Rotation.from_matrix(np.column_stack([np.cross(y, z), y, z]))
            rotations.append(rotation)
            targets.append(np.r_[row[:6], rotation.as_quat()])
            thrusts.append(self.mass*np.linalg.norm(force))
        feedforward = []
        for k in range(self.horizon):
            rate = (rotations[k].inv()*rotations[k+1]).as_rotvec()/self.dt
            feedforward.append(np.r_[np.clip(thrusts[k], 0., self.max_thrust), np.clip(rate, -self.max_rate, self.max_rate)])
        return np.asarray(targets[1:]).ravel(), np.asarray(feedforward).ravel()

    def command_horizon(self, state, references):
        refs = np.asarray(references, dtype=float)
        if refs.shape != (self.horizon+1, 11) or not np.isfinite(refs).all():
            raise ValueError('Invalid quadrotor MPC reference horizon')
        actual_rates = state.body_rates if state.body_rates is not None else np.zeros(3)
        actual_thrust = state.actuator_thrust_n if state.actuator_thrust_n is not None else self.mass*self.gravity
        initial_state = np.r_[state.position, state.velocity, state.orientation.as_quat(), actual_rates, actual_thrust]
        desired, feedforward = self.reference_targets(refs)
        initial_plan = np.vstack([self.plan[1:], self.plan[-1:]]).ravel()/self.decision_scale
        def objective(normalized_actions):
            actions = normalized_actions*self.decision_scale
            cost, gradient = self.objective(actions, initial_state, desired, feedforward, self.last_action)
            return float(cost), np.asarray(gradient).ravel()*self.decision_scale
        before = time.perf_counter()
        result = minimize(objective, initial_plan, jac=True, method='SLSQP', bounds=self.normalized_bounds,
                          options=dict(maxiter=self.max_iterations, ftol=1e-7))
        self.diagnostics = dict(status='solved' if result.success else 'failed', message=str(result.message),
                                solve_ms=1000*(time.perf_counter()-before), iterations=int(result.nit))
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f'Quadrotor NMPC failed: {result.message}')
        actions = result.x*self.decision_scale
        self.plan = np.clip(actions, [b[0] for b in self.bounds], [b[1] for b in self.bounds]).reshape(self.horizon, 4)
        self.last_action = self.plan[0].copy()
        return float(self.last_action[0]), self.last_action[1:].copy()
