"""Deterministic dynamics/controller loop, independent of GPU and viewer."""
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.control.tracker import TrackingController
from Workflow.Simulation.figs.dynamics import Quadrotor, State
from Workflow.Simulation.tasks.trajectory import MRoute


class Simulator:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.vehicle = Quadrotor(config['vehicle'])
        self.controller = TrackingController(config['controller'])
        self.physics_dt = 1/float(config['physics_hz'])
        self.control_dt = 1/float(config['control_hz'])
        self.substeps = round(self.control_dt/self.physics_dt)
        if self.substeps < 1 or not np.isclose(self.substeps*self.physics_dt, self.control_dt):
            raise ValueError('physics_hz must be an integer multiple of control_hz')
        self.rng = np.random.default_rng(config.get('seed', 42))
        self.episode = 0
        self.reset()

    def reset(self):
        self.route = MRoute(self.config['task'])
        self.state = State(self.route.points[0].copy(), np.zeros(3),
                           Rotation.from_euler('z', self.route.yaw))
        self.time, self.route_time = 0., 0.
        self.mode, self.paused = 'route', True
        self.hover_target = self.state.position.copy()
        self.vehicle.wind[:] = 0.
        self.thrust, self.rates = self.vehicle.mass*self.vehicle.gravity, np.zeros(3)
        self.history, self.telemetry = [self.state.position.tolist()], []
        self.episode += 1

    def reference(self):
        if self.mode == 'hover':
            return self.hover_target.copy(), np.zeros(3), np.zeros(3)
        return self.route.sample(self.route_time)

    def advance(self):
        if self.paused:
            return
        decision_time = self.time
        self.thrust, self.rates = self.controller.command(
            self.state, self.reference(), self.route.yaw, self.vehicle)
        for _ in range(self.substeps):
            self.state = self.vehicle.step(self.state, self.thrust, self.rates, self.physics_dt)
        self.time += self.control_dt
        if self.mode == 'route':
            self.route_time = min(self.route_time+self.control_dt, self.route.duration)
        reference = self.reference()[0]
        p, v = self.state.position, self.state.velocity
        q = self.state.orientation.as_quat()
        self.history.append(p.tolist())
        self.history = self.history[-2500:]
        self.telemetry.append(dict(time_s=self.time, decision_time_s=decision_time,
            route_time_s=self.route_time, x_m=p[0], y_m=p[1], z_m=p[2],
            vx_m_s=v[0], vy_m_s=v[1], vz_m_s=v[2],
            qx=q[0], qy=q[1], qz=q[2], qw=q[3],
            reference_x_m=reference[0], reference_y_m=reference[1], reference_z_m=reference[2],
            error_m=float(np.linalg.norm(p-reference)), thrust_n=self.thrust,
            body_rate_x_rad_s=self.rates[0], body_rate_y_rad_s=self.rates[1], body_rate_z_rad_s=self.rates[2]))
        self.telemetry = self.telemetry[-20000:]

    def command(self, command):
        name = command.get('action')
        if name == 'play':
            self.paused = False
        elif name == 'pause':
            self.paused = True
        elif name == 'reset':
            self.reset()
        elif name == 'route':
            self.reset()
            self.paused = False
        elif name == 'hover':
            self.hover_target = self.state.position.copy()
            self.mode, self.paused = 'hover', False
        elif name == 'nudge':
            delta = np.asarray(command['delta'], dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all() or np.max(np.abs(delta)) > .5:
                raise ValueError('Nudge requires three finite offsets no greater than 0.5 m')
            if self.mode != 'hover':
                self.hover_target = self.state.position.copy()
            self.hover_target += delta
            self.hover_target[2] = max(self.hover_target[2], self.vehicle.floor+.1)
            self.mode, self.paused = 'hover', False
        elif name == 'disturbance':
            self.state.velocity += np.array([.2, -.15, .05])
        elif name == 'configure':
            ranges = dict(width_m=(.1, 2.), height_m=(.1, 2.), altitude_m=(.2, 8.),
                          speed_m_s=(.05, 1.), yaw_deg=(-180., 180.))
            values = command.get('task', {})
            update = {}
            for key, value in values.items():
                if key not in ranges:
                    raise ValueError(f'Unsupported route setting: {key}')
                value = float(value)
                low, high = ranges[key]
                if not np.isfinite(value) or not low <= value <= high:
                    raise ValueError(f'{key} must be in [{low}, {high}]')
                update[key] = value
            self.config['task'].update(update)
            self.reset()
        else:
            raise ValueError(f'Unknown command: {name}')

    def snapshot(self):
        reference = self.reference()[0]
        return dict(time_s=self.time, route_time_s=self.route_time, duration_s=self.route.duration,
                    paused=self.paused, mode=self.mode, episode=self.episode,
                    complete=self.route_time >= self.route.duration,
                    position=self.state.position.tolist(), velocity=self.state.velocity.tolist(),
                    quaternion_xyzw=self.state.orientation.as_quat().tolist(),
                    reference=reference.tolist(), speed_m_s=float(np.linalg.norm(self.state.velocity)),
                    tracking_error_m=float(np.linalg.norm(self.state.position-reference)),
                    thrust_n=self.thrust, body_rates_rad_s=self.rates.tolist(),
                    task=dict(self.config['task']), route=self.route.polyline(), history=[p[:] for p in self.history])
