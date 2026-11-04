"""FiGS-style semi-kinematic quadrotor model in a right-handed Z-up world.

This is our small prototype model, not the upstream FiGS/ACADOS implementation.
State: world position/velocity and body-to-world quaternion (10 scalars).
Input: collective thrust in newtons and body angular rate in rad/s.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class State:
    position: np.ndarray
    velocity: np.ndarray
    orientation: Rotation


class Quadrotor:
    def __init__(self, config):
        self.mass = float(config['mass_kg'])
        self.gravity = float(config.get('gravity_m_s2', 9.81))
        self.drag = float(config.get('linear_drag_per_s', .15))
        self.thrust_limit = float(config.get('max_thrust_to_weight', 2.5))*self.mass*self.gravity
        self.rate_limit = float(config.get('max_body_rate_rad_s', 4.))
        self.floor = float(config.get('floor_z_m', 0.))
        if self.mass <= 0 or self.gravity <= 0 or self.drag < 0 or self.rate_limit <= 0:
            raise ValueError('Invalid vehicle parameters')
        self.wind = np.zeros(3)

    def step(self, state, thrust_n, body_rates, dt):
        if dt <= 0:
            raise ValueError('Time step must be positive')
        rates = np.clip(np.asarray(body_rates, dtype=float), -self.rate_limit, self.rate_limit)
        thrust = np.clip(float(thrust_n), 0., self.thrust_limit)
        mid = state.orientation*Rotation.from_rotvec(rates*dt*.5)
        acceleration = mid.apply([0., 0., thrust/self.mass]) + [0., 0., -self.gravity]
        acceleration += self.wind-self.drag*state.velocity
        position = state.position+dt*state.velocity+.5*dt*dt*acceleration
        velocity = state.velocity+dt*acceleration
        if position[2] < self.floor:
            position[2] = self.floor
            velocity[2] = max(velocity[2], 0.)
        return State(position, velocity, state.orientation*Rotation.from_rotvec(rates*dt))
