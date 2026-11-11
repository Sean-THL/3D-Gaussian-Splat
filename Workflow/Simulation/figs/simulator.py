"""Deterministic dynamics/controller loop, independent of GPU and viewer."""
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.control.tracker import TrackingController
from Workflow.Simulation.figs.dynamics import Quadrotor, State
from Workflow.Simulation.figs.integrators import Integrator, IntegratorTracker
from Workflow.Simulation.tasks.trajectory import make_trajectory, TRAJECTORIES
from Workflow.Simulation.control.mpc import LinearMPC
from Workflow.Simulation.randomization import sample_episode


class Simulator:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.dynamics_model = config.get('dynamics_model', 'quadrotor')
        if self.dynamics_model not in ('quadrotor', 'single_integrator', 'double_integrator'):
            raise ValueError(f'Unknown dynamics_model: {self.dynamics_model}')
        self.is_integrator = self.dynamics_model != 'quadrotor'
        self.controller_type = config.get('controller', {}).get('type', 'baseline')
        if self.controller_type not in ('baseline', 'mpc', 'nmpc'):
            raise ValueError('controller.type must be baseline, mpc, or nmpc')
        self.is_expert = self.controller_type in ('mpc', 'nmpc')
        if self.controller_type == 'mpc' and self.dynamics_model != 'double_integrator':
            raise ValueError('MPC currently requires --dynamics double_integrator; use --controller baseline for other models')
        if self.controller_type == 'nmpc' and self.dynamics_model != 'quadrotor':
            raise ValueError('NMPC requires --dynamics quadrotor')
        self.physics_dt = 1/float(config['physics_hz'])
        self.control_dt = 1/float(config['control_hz'])
        self.substeps = round(self.control_dt/self.physics_dt)
        if self.substeps < 1 or not np.isclose(self.substeps*self.physics_dt, self.control_dt):
            raise ValueError('physics_hz must be an integer multiple of control_hz')
        self.rng = np.random.default_rng(config.get('seed', 42))
        self.episode = 0
        self.reset()

    def reset(self):
        task, plant, sampled = sample_episode(self.config, self.rng)
        self.route = make_trajectory(task)
        self.vehicle = self.make_vehicle(plant)
        initial_position = self.route.trajectory_function(0.)[:3]+sampled['initial_position_offset_m']
        if initial_position[2] <= self.vehicle.floor or min(p[2] for p in self.route.polyline(401)) <= self.vehicle.floor:
            raise ValueError('Sampled route/initial position crosses the assumed floor; revise randomization bounds')
        initial_yaw = self.route.yaw_reference(0.)[0]+np.deg2rad(sampled['initial_yaw_offset_deg'])
        self.state = State(initial_position, np.asarray(sampled['initial_velocity_m_s']), Rotation.from_euler('z', initial_yaw))
        self.controller = (LinearMPC(self.config.get('mpc', {}), self.config['vehicle'], self.control_dt)
                           if self.controller_type == 'mpc' else
                           IntegratorTracker(self.config['controller']) if self.is_integrator else TrackingController(self.config['controller']))
        if self.controller_type == 'nmpc':
            from Workflow.Simulation.control.quadrotor_mpc import QuadrotorMPC
            self.controller = QuadrotorMPC(self.config.get('nmpc', {}), self.config['vehicle'], self.control_dt, self.physics_dt)
        self.episode_parameters = dict(episode=self.episode+1, seed=self.config.get('seed', 42),
                                       task=task, plant=plant, randomization=sampled,
                                       expert_model={'mpc': 'nominal ideal double integrator', 'nmpc': 'nominal thrust/body-rate quadrotor', 'baseline': 'baseline feedback'}[self.controller_type])
        self.episode_parameters['physics_backend'] = getattr(self.vehicle, 'physics_backend',
            'custom_'+self.dynamics_model)
        self.time, self.route_time = 0., 0.
        self.mode, self.paused = 'route', True
        self.hover_target = self.state.position.copy()
        self.hover_yaw = self.route.yaw_reference(0.)[0]
        self.thrust = None if self.is_integrator else self.vehicle.mass*self.vehicle.gravity
        self.rates = np.zeros(3)
        self.translation_command, self.yaw_rate = np.zeros(3), 0.
        self.history, self.telemetry, self.demonstrations = [self.state.position.tolist()], [], []
        self.episode += 1

    def reference(self):
        if self.mode == 'hover':
            return self.hover_target.copy(), np.zeros(3), np.zeros(3)
        ref = self.route.trajectory_function(self.route_time)
        return ref[:3], ref[3:6], ref[6:9]

    def advance(self):
        if self.paused:
            return
        decision_time = self.time
        yaw, yaw_rate = self.yaw_reference()
        if self.is_expert:
            refs = self.reference_horizon(self.controller.horizon)
            action, rotation_action = self.controller.command_horizon(self.state, refs)
            demonstration = dict(time_s=self.time, episode=self.episode,
                position_m=self.state.position.tolist(), velocity_m_s=self.state.velocity.tolist(),
                quaternion_xyzw=self.state.orientation.as_quat().tolist(),
                reference_horizon=refs.tolist(), horizon_dt_s=self.control_dt,
                solver=dict(self.controller.diagnostics))
            demonstration['observed_body_rates_rad_s'] = (self.state.body_rates.tolist() if self.state.body_rates is not None else [0., 0., 0.])
            if self.controller_type == 'nmpc':
                demonstration.update(action_type='thrust_body_rates', action_thrust_n=action,
                                     action_body_rates_rad_s=rotation_action.tolist())
            else:
                demonstration.update(action_type='acceleration_yaw_rate', action_acceleration_m_s2=action.tolist(),
                                     action_yaw_rate_rad_s=rotation_action)
            self.demonstrations.append(demonstration)
        else:
            action, rotation_action = self.controller.command(
                self.state, self.reference(), yaw, self.vehicle, yaw_rate=yaw_rate)
        if self.is_integrator:
            self.translation_command, self.yaw_rate = action, rotation_action
            self.rates = np.array([0., 0., self.yaw_rate])
        else:
            self.thrust, self.rates = action, rotation_action
        for _ in range(self.substeps):
            self.state = self.vehicle.step(self.state, action, rotation_action, self.physics_dt)
        self.time += self.control_dt
        if self.mode == 'route':
            self.route_time += self.control_dt
            if not self.route.continuous:
                self.route_time = min(self.route_time, self.route.duration)
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
            reference_yaw_rad=yaw, reference_yaw_rate_rad_s=yaw_rate,
            yaw_reference_time_s=decision_time,
            error_m=float(np.linalg.norm(p-reference)), thrust_n=self.thrust,
            body_rate_x_rad_s=self.rates[0], body_rate_y_rad_s=self.rates[1], body_rate_z_rad_s=self.rates[2],
            dynamics_model=self.dynamics_model,
            command_units=self.command_units,
            command_x=self.translation_command[0] if self.is_integrator else None,
            command_y=self.translation_command[1] if self.is_integrator else None,
            command_z=self.translation_command[2] if self.is_integrator else None))
        self.telemetry[-1].update(controller=self.controller_type,
            mpc_solve_ms=self.controller.diagnostics['solve_ms'] if self.is_expert else None,
            mpc_status=self.controller.diagnostics['status'] if self.is_expert else None)
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
            self.hover_yaw = self.state.orientation.as_euler('xyz')[2]
            self.mode, self.paused = 'hover', False
        elif name == 'nudge':
            delta = np.asarray(command['delta'], dtype=float)
            if delta.shape != (3,) or not np.isfinite(delta).all() or np.max(np.abs(delta)) > .5:
                raise ValueError('Nudge requires three finite offsets no greater than 0.5 m')
            if self.mode != 'hover':
                self.hover_target = self.state.position.copy()
                self.hover_yaw = self.state.orientation.as_euler('xyz')[2]
            self.hover_target += delta
            self.hover_target[2] = max(self.hover_target[2], self.vehicle.floor+.1)
            self.mode, self.paused = 'hover', False
        elif name == 'disturbance':
            if self.dynamics_model == 'single_integrator':
                self.state.position += np.array([.1, -.075, .025])
            else:
                self.state.velocity += np.array([.2, -.15, .05])
        elif name == 'configure':
            ranges = dict(width_m=(.1, 2.), height_m=(.1, 2.), altitude_m=(.2, 8.),
                          speed_m_s=(.05, 1.), yaw_deg=(-180., 180.), duration_s=(5., 120.),
                          vertical_drop_m=(0., 2.), max_angular_speed_rad_s=(.05, 2.))
            ranges['duration_s'] = (max(5., self.config['task'].get('minimum_duration_s', 0.)), 120.)
            values = command.get('task', {})
            update = {}
            for key, value in values.items():
                if key in ('trajectory_type', 'yaw_mode'):
                    choices = TRAJECTORIES if key == 'trajectory_type' else ('constant', 'velocity_aligned')
                    if value not in choices:
                        raise ValueError(f'Unknown {key}: {value}')
                    update[key] = value
                    continue
                if key not in ranges:
                    raise ValueError(f'Unsupported route setting: {key}')
                value = float(value)
                low, high = ranges[key]
                if not np.isfinite(value) or not low <= value <= high:
                    raise ValueError(f'{key} must be in [{low}, {high}]')
                update[key] = value
            candidate = {**self.config['task'], **update}
            route = make_trajectory(candidate)
            if hasattr(route, 'drop') and route.start[2]-route.drop <= self.vehicle.floor:
                raise ValueError('Figure-eight vertical drop must stay above the prototype floor')
            self.config['task'] = candidate
            self.reset()
        else:
            raise ValueError(f'Unknown command: {name}')

    def snapshot(self):
        reference = self.reference()[0]
        return dict(time_s=self.time, route_time_s=self.route_time, duration_s=self.route.duration,
                    paused=self.paused, mode=self.mode, episode=self.episode,
                    complete=not self.route.continuous and self.route_time >= self.route.duration,
                    continuous=self.route.continuous, trajectory_type=self.route.trajectory_type,
                    trajectory_label=self.route.label,
                    position=self.state.position.tolist(), velocity=self.state.velocity.tolist(),
                    quaternion_xyzw=self.state.orientation.as_quat().tolist(),
                    reference=reference.tolist(), speed_m_s=float(np.linalg.norm(self.state.velocity)),
                    tracking_error_m=float(np.linalg.norm(self.state.position-reference)),
                    thrust_n=self.thrust, body_rates_rad_s=self.rates.tolist(),
                    dynamics_model=self.dynamics_model, command_units=self.command_units,
                    physics_backend=self.episode_parameters['physics_backend'],
                    translation_command=self.translation_command.tolist() if self.is_integrator else None,
                    reference_yaw_rad=self.yaw_reference()[0], reference_yaw_rate_rad_s=self.yaw_reference()[1],
                    controller=self.controller_type,
                    solver=dict(self.controller.diagnostics) if self.is_expert else None,
                    randomization_enabled=self.episode_parameters['randomization']['enabled'],
                    episode_parameters=copy.deepcopy(self.episode_parameters), floor_z_m=self.vehicle.floor,
                    task=dict(self.route.config), route=self.route.polyline(), history=[p[:] for p in self.history])

    @property
    def command_units(self):
        return {'quadrotor': 'N and rad/s', 'single_integrator': 'm/s', 'double_integrator': 'm/s^2'}[self.dynamics_model]

    def yaw_reference(self):
        if self.mode == 'hover':
            return self.hover_yaw, 0.
        return self.route.yaw_reference(self.route_time)

    def reference_horizon(self, steps):
        rows = []
        for k in range(steps+1):
            if self.mode == 'hover':
                rows.append(np.r_[self.hover_target, np.zeros(6), self.hover_yaw, 0.])
            else:
                t = self.route_time+k*self.control_dt
                rows.append(np.r_[self.route.trajectory_function(t), self.route.yaw_reference(t)[1]])
        return np.asarray(rows)

    def make_vehicle(self, plant):
        """Backend hook: native Isaac overrides the plant, not the MPC/reference."""
        if self.is_integrator:
            return Integrator(plant, self.dynamics_model)
        model = plant.get('rotation_model', 'rate_response')
        if model == 'rigid_body':
            from Workflow.Simulation.figs.rigid_body import RigidBodyQuadrotor
            return RigidBodyQuadrotor(plant)
        if model != 'rate_response':
            raise ValueError('vehicle.rotation_model must be rate_response or rigid_body')
        return Quadrotor(plant)
