"""Baseline position/attitude feedback controller. This is not an MPC expert."""
import numpy as np
from scipy.spatial.transform import Rotation


class TrackingController:
    def __init__(self, config):
        self.kp = float(config.get('position_gain', 4.))
        self.kd = float(config.get('velocity_gain', 3.2))
        self.attitude_gain = float(config.get('attitude_gain', 12.))
        self.max_acceleration = float(config.get('max_acceleration_m_s2', 2.))

    def command(self, state, reference, yaw, vehicle, yaw_rate=0.):
        position, velocity, acceleration = reference
        a = acceleration+self.kp*(position-state.position)+self.kd*(velocity-state.velocity)
        magnitude = np.linalg.norm(a)
        if magnitude > self.max_acceleration:
            a *= self.max_acceleration/magnitude
        specific_force = a+np.array([0., 0., vehicle.gravity])+vehicle.drag*state.velocity
        z = specific_force/np.linalg.norm(specific_force)
        def desired_rotation(heading_angle):
            heading = np.array([np.cos(heading_angle), np.sin(heading_angle), 0.])
            y = np.cross(z, heading)
            y /= np.linalg.norm(y)
            x = np.cross(y, z)
            return Rotation.from_matrix(np.column_stack([x, y, z]))
        desired = desired_rotation(yaw)
        rates = self.attitude_gain*(state.orientation.inv()*desired).as_rotvec()
        if yaw_rate:
            # Heading feedforward at fixed thrust direction, converted to body rates.
            eps = 1e-5
            world_rate = (desired_rotation(yaw+eps)*desired_rotation(yaw-eps).inv()).as_rotvec()/(2*eps)*yaw_rate
            rates += state.orientation.inv().apply(world_rate)
        rates = np.clip(rates, -vehicle.rate_limit, vehicle.rate_limit)
        thrust = vehicle.mass*np.dot(specific_force, state.orientation.apply([0., 0., 1.]))
        return float(np.clip(thrust, 0, vehicle.thrust_limit)), rates
