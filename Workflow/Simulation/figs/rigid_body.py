"""Custom rigid-body plant with thrust/body-rate interface, no Isaac dependency.

The bounded inner rate loop generates body torques. Euler's rigid-body equation
and quaternion kinematics evolve actual angular velocity and attitude. This is
a provisional diagonal-inertia drone model, not identified PX4/motor dynamics.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.figs.dynamics import Quadrotor, State


class RigidBodyQuadrotor(Quadrotor):
    physics_backend = 'custom_rigid_body_forces_and_torques'

    def __init__(self, config):
        super().__init__(config)
        self.inertia = np.asarray(config.get('inertia_kg_m2', [.0049, .0049, .0088]), dtype=float)
        self.torque_limit = np.asarray(config.get('max_body_torque_nm', [.3, .3, .15]), dtype=float)
        self.integration_dt = float(config.get('rigid_body_max_step_s', .001))
        if (self.inertia.shape != (3,) or self.torque_limit.shape != (3,) or
                not np.isfinite(np.r_[self.inertia, self.torque_limit, self.integration_dt]).all() or
                min(self.inertia) <= 0 or min(self.torque_limit) <= 0 or
                not 0 < self.integration_dt <= .01 or self.rate_tau <= 0):
            raise ValueError('Rigid-body inertia, torque limits, integration step and rate time constant must be positive')
        self.torque_clips = 0
        self.last_torque_nm = np.zeros(3)

    def step(self, state, thrust_n, body_rates, dt):
        rates = np.asarray(body_rates, dtype=float)
        if rates.shape != (3,) or not np.isfinite(np.r_[rates, thrust_n, dt]).all() or dt <= 0:
            raise ValueError('Expected finite thrust, XYZ body rates and positive dt')
        count = int(np.ceil(dt/self.integration_dt))
        h = dt/count
        p, v, rotation = state.position.copy(), state.velocity.copy(), state.orientation
        omega = np.zeros(3) if state.body_rates is None else state.body_rates.copy()
        target_rates = self.rate_scale*np.clip(rates, -self.rate_limit, self.rate_limit)
        target_thrust = self.thrust_scale*np.clip(float(thrust_n), 0., self.thrust_limit)
        clipped_this_step = False
        for _ in range(count):
            gyroscopic = np.cross(omega, self.inertia*omega)
            torque = self.inertia*(target_rates-omega)/self.rate_tau+gyroscopic
            self.last_torque_nm = np.clip(torque, -self.torque_limit, self.torque_limit)
            clipped_this_step |= bool(np.any(np.abs(torque-self.last_torque_nm) > 1e-9))
            alpha = (self.last_torque_nm-gyroscopic)/self.inertia
            midpoint_rates = omega+.5*h*alpha
            midpoint_rotation = rotation*Rotation.from_rotvec(midpoint_rates*h*.5)
            if self.thrust_tau > 0:
                decay = np.exp(-h/self.thrust_tau)
                average = target_thrust+(self.actual_thrust-target_thrust)*(self.thrust_tau/h)*(1-decay)
                self.actual_thrust = target_thrust+(self.actual_thrust-target_thrust)*decay
            else:
                self.actual_thrust = average = target_thrust
            acceleration = midpoint_rotation.apply([0., 0., average/self.mass])+[0., 0., -self.gravity]
            acceleration += self.wind-self.drag*v
            p += h*v+.5*h*h*acceleration
            v += h*acceleration
            rotation = rotation*Rotation.from_rotvec(midpoint_rates*h)
            omega += h*alpha
            # Same assumed floor as the original custom plant, not a contact solver.
            if p[2] < self.floor:
                p[2] = self.floor
                v[2] = max(v[2], 0.)
        self.torque_clips += int(clipped_this_step)
        self.actual_rates = omega.copy()
        return State(p, v, rotation, omega, float(self.actual_thrust))
