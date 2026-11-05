"""Ideal world-frame velocity/acceleration models, with separate yaw-rate input."""
import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.figs.dynamics import State


def limit_norm(vector, limit):
    vector = np.asarray(vector, dtype=float)
    return vector * min(1., limit / max(float(np.linalg.norm(vector)), 1e-12))


class Integrator:
    def __init__(self, config, model):
        self.model = model
        self.floor = float(config.get('floor_z_m', 0.))
        self.speed_limit = float(config.get('max_speed_m_s', 1.))
        self.acceleration_limit = float(config.get('max_acceleration_m_s2', 2.))
        self.rate_limit = float(config.get('max_body_rate_rad_s', 4.))
        if not np.isfinite([self.floor, self.speed_limit, self.acceleration_limit, self.rate_limit]).all() or min(self.speed_limit, self.acceleration_limit, self.rate_limit) <= 0:
            raise ValueError('Integrator limits must be finite and positive')

    def step(self, state, command, yaw_rate, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Time step must be finite and positive')
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all() or not np.isfinite(yaw_rate):
            raise ValueError('Integrator command must be finite XYZ and yaw rate')
        if self.model == 'single_integrator':
            velocity = limit_norm(command, self.speed_limit)
            position = state.position + velocity*dt
        else:
            acceleration = limit_norm(command, self.acceleration_limit)
            velocity = state.velocity + acceleration*dt
            position = state.position + state.velocity*dt + .5*acceleration*dt*dt
        if position[2] < self.floor:
            position[2] = self.floor
            velocity[2] = max(velocity[2], 0.)
        yaw = state.orientation.as_euler('xyz')[2]
        yaw += float(np.clip(yaw_rate, -self.rate_limit, self.rate_limit))*dt
        return State(position, velocity, Rotation.from_euler('z', yaw))


class IntegratorTracker:
    def __init__(self, config):
        self.kp = float(config.get('position_gain', 4.))
        self.kd = float(config.get('velocity_gain', 3.2))
        self.yaw_gain = float(config.get('attitude_gain', 12.))

    def command(self, state, reference, yaw, vehicle):
        position, velocity, acceleration = reference
        if vehicle.model == 'single_integrator':
            action = limit_norm(velocity+self.kp*(position-state.position), vehicle.speed_limit)
        else:
            action = limit_norm(acceleration+self.kp*(position-state.position)+self.kd*(velocity-state.velocity), vehicle.acceleration_limit)
        error = yaw-state.orientation.as_euler('xyz')[2]
        error = np.arctan2(np.sin(error), np.cos(error))
        return action, float(np.clip(self.yaw_gain*error, -vehicle.rate_limit, vehicle.rate_limit))
