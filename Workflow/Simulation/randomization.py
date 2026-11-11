"""Choose the conditions for one simulation episode.

Called by Simulator.reset(), once per episode, not at every physics step.
Ranges come from the "randomization" section in configs/demo.json.
For example, "x_offset_m": [-0.1, 0.1] means a uniform draw in that range.

Three things can change:
  1. Task: where the route is, desired heading, and traversal time.
  2. Actual drone response: acceleration gain, disturbance, damping, yaw drift.
  3. Initial state: position, velocity, and heading errors.

The MPC receives the changed task reference and current state. It continues
using nominal dynamics; it is not told the sampled drone-response parameters.
"""
import copy

import numpy as np


def sample_uniform(settings, rng, name, default_range, size=None):
    """Read [minimum, maximum] from config and draw one value, or an XYZ vector.

    rng is the seeded random-number generator owned by the simulator.
    size=3 draws an independent value for each of X, Y, and Z.
    """
    bounds = np.asarray(settings.get(name, default_range), dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all():
        raise ValueError(f'{name} must be a finite [minimum, maximum] pair')

    minimum, maximum = bounds
    if minimum > maximum:
        raise ValueError(f'{name} must be a finite [minimum, maximum] pair')

    return rng.uniform(minimum, maximum, size=size)


def randomize_task(task, settings, rng):
    """Move the route, change drone heading, and change traversal time.

    Modifies this episode's task copy. Returns the sampled route offset and
    duration scale so they can also be saved in the episode metadata.
    """
    x_offset = sample_uniform(settings, rng, 'x_offset_m', [-0.1, 0.1])
    y_offset = sample_uniform(settings, rng, 'y_offset_m', [-0.1, 0.1])
    z_offset = sample_uniform(settings, rng, 'z_offset_m', [-0.1, 0.1])
    route_offset = np.array([x_offset, y_offset, z_offset])

    # Some analytic-route configs specify a start directly instead of a center.
    if 'start_position_m' in task:
        start_position = np.asarray(task['start_position_m'])
        task['start_position_m'] = (start_position + route_offset).tolist()

    center_xy = np.asarray(task.get('center_xy_m', [0.0, 0.0]))
    task['center_xy_m'] = (center_xy + route_offset[:2]).tolist()
    task['altitude_m'] += float(z_offset)

    # Changes the drone's heading, not the orientation of the position route.
    # In velocity-aligned mode, yaw_deg is an offset from the path tangent.
    yaw_offset = float(sample_uniform(settings, rng, 'yaw_offset_deg', [-15.0, 15.0]))
    task['yaw_deg'] = task.get('yaw_deg', 0.0) + yaw_offset

    duration_scale = float(sample_uniform(settings, rng, 'duration_scale', [0.9, 1.1]))
    if duration_scale <= 0:
        raise ValueError('duration_scale must be positive')

    # A scale > 1 makes traversal slower. Each route expresses timing differently.
    trajectory_type = task.get('trajectory_type', 'm')
    if trajectory_type == 'm':
        task['speed_m_s'] /= duration_scale
        task['max_acceleration_m_s2'] = task.get('max_acceleration_m_s2', 0.5) / duration_scale**2
    elif trajectory_type == 'horizontal_circle':
        task['max_angular_speed_rad_s'] = task.get('max_angular_speed_rad_s', 0.75) / duration_scale
        task['angular_ramp_rate_per_s'] = task.get('angular_ramp_rate_per_s', 0.1) / duration_scale
    else:  # Figure eight: duration is specified directly in seconds.
        task['duration_s'] = task.get('duration_s', 20.0) * duration_scale
    if trajectory_type != 'm' and task['duration_s'] < task.get('minimum_duration_s', 0.):
        raise ValueError('Randomized duration is below task.minimum_duration_s; revise duration_scale')

    return route_offset.tolist(), duration_scale


def randomize_dynamics(vehicle, settings, rng):
    """Change the actual double-integrator response; leave MPC's model alone.

    Actual acceleration = gain * command + bias - damping * velocity.
    Actual yaw rate = command + yaw-rate bias.
    These values stay fixed throughout this episode.
    """
    vehicle['acceleration_scale'] = float(
        sample_uniform(settings, rng, 'acceleration_scale', [0.8, 1.2])
    )
    vehicle['acceleration_disturbance_m_s2'] = sample_uniform(
        settings, rng, 'acceleration_bias_m_s2', [-0.1, 0.1], size=3
    ).tolist()
    vehicle['velocity_damping_per_s'] = float(
        sample_uniform(settings, rng, 'velocity_damping_per_s', [0.0, 0.15])
    )
    vehicle['yaw_rate_bias_rad_s'] = float(
        sample_uniform(settings, rng, 'yaw_rate_bias_rad_s', [-0.02, 0.02])
    )


def randomize_initial_state(metadata, settings, rng):
    """Choose starting errors; Simulator.reset() applies them to the state."""
    metadata['initial_position_offset_m'] = sample_uniform(
        settings, rng, 'initial_position_offset_m', [-0.03, 0.03], size=3
    ).tolist()
    metadata['initial_velocity_m_s'] = sample_uniform(
        settings, rng, 'initial_velocity_m_s', [-0.03, 0.03], size=3
    ).tolist()
    metadata['initial_yaw_offset_deg'] = float(
        sample_uniform(settings, rng, 'initial_yaw_offset_deg', [-5.0, 5.0])
    )


def randomize_quadrotor_dynamics(vehicle, settings, rng):
    """Actual mass/drag/response vary; commands remain thrust N and body rad/s."""
    nominal_mass = vehicle['mass_kg']
    vehicle['max_thrust_n'] = vehicle.get('max_thrust_n', nominal_mass*vehicle.get('gravity_m_s2', 9.81)*vehicle.get('max_thrust_to_weight', 2.5))
    vehicle['mass_kg'] *= float(sample_uniform(settings, rng, 'mass_scale', [.85, 1.15]))
    vehicle['linear_drag_per_s'] = float(sample_uniform(settings, rng, 'linear_drag_per_s', [.05, .25]))
    vehicle['thrust_scale'] = float(sample_uniform(settings, rng, 'thrust_scale', [.9, 1.1]))
    vehicle['rate_scale'] = float(sample_uniform(settings, rng, 'rate_scale', [.9, 1.1]))
    vehicle['rate_time_constant_s'] = float(sample_uniform(settings, rng, 'rate_time_constant_s', [.02, .06]))
    vehicle['thrust_time_constant_s'] = float(sample_uniform(settings, rng, 'thrust_time_constant_s', [.02, .05]))
    vehicle['acceleration_disturbance_m_s2'] = sample_uniform(settings, rng, 'acceleration_bias_m_s2', [-.1, .1], size=3).tolist()


def sample_episode(config, rng):
    """Return (episode task, actual vehicle parameters, episode metadata).

    Copies keep the nominal config unchanged. With randomization disabled,
    the copies are unchanged and the initial-state offsets are zero.
    """
    task = copy.deepcopy(config['task'])
    vehicle = copy.deepcopy(config['vehicle'])
    settings = config.get('randomization', {})
    enabled = bool(settings.get('enabled', False))

    metadata = {
        'enabled': enabled,
        'initial_position_offset_m': [0.0, 0.0, 0.0],
        'initial_velocity_m_s': [0.0, 0.0, 0.0],
        'initial_yaw_offset_deg': 0.0,
        'initial_roll_offset_deg': 0.0,
        'initial_pitch_offset_deg': 0.0,
    }

    if not enabled:
        return task, vehicle, metadata

    model = config.get('dynamics_model', 'quadrotor')
    if model not in ('double_integrator', 'quadrotor'):
        raise ValueError('Episode randomization supports double_integrator or quadrotor')

    # Sample in this order to preserve the results associated with existing seeds.
    route_offset, duration_scale = randomize_task(task, settings, rng)
    if model == 'quadrotor':
        randomize_quadrotor_dynamics(vehicle, settings, rng)
    else:
        randomize_dynamics(vehicle, settings, rng)
    metadata['route_offset_m'] = route_offset
    metadata['duration_scale'] = duration_scale
    randomize_initial_state(metadata, settings, rng)

    # Optional uniform spatial scaling, sampled last to preserve older seeds.
    # This changes route dimensions, not reconstructed surfaces or obstacles.
    if 'route_size_scale' in settings:
        size_scale = float(sample_uniform(settings, rng, 'route_size_scale', [1., 1.]))
        if size_scale <= 0:
            raise ValueError('route_size_scale must be positive')
        for key in ('width_m', 'height_m', 'a_m', 'b_m', 'radius_m'):
            if key in task:
                task[key] *= size_scale
        metadata['route_size_scale'] = size_scale

    if model == 'quadrotor':
        for axis in ('roll', 'pitch'):
            name = f'initial_{axis}_offset_deg'
            if name in settings:
                metadata[name] = float(sample_uniform(settings, rng, name, [0., 0.]))

    return task, vehicle, metadata
