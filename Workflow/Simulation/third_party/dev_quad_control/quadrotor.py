"""A standalone NumPy quadrotor control pipeline, in SI units.

PointNavigator -> Reference -> GeometricController -> wrench [N, N m]
              -> ControlAllocator -> motor commands [rad/s]
              -> QuadrotorDynamics -> State

Equations/defaults are extracted from lab-experiment-sim-starter at dc78ff9,
using its deployed ROS model (including the rotor-momentum sign correction).
The original Julia source is kalebbennaveed/meSch. See PROVENANCE.md and LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return array.copy()


@dataclass(frozen=True)
class VehicleParameters:
    """Inherited model defaults; these are not identified airframe measurements."""

    mass: float = 0.680
    inertia: tuple[float, float, float] = (0.005, 0.005, 0.003)
    gravity: float = 9.81
    arm_x: float = 0.08
    arm_y: float = 0.08
    motor_time_constant: float = 0.02
    rotor_inertia: float = 6.62e-6
    thrust_coefficient: float = 1.91e-6
    torque_coefficient: float = 2.7e-7
    linear_drag: float = 0.1
    angular_drag: float = 0.003
    max_motor_speed: float = 2520.0

    def __post_init__(self):
        positive = ("mass", "gravity", "arm_x", "arm_y", "motor_time_constant",
                    "thrust_coefficient", "torque_coefficient", "max_motor_speed")
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("rotor_inertia", "linear_drag", "angular_drag"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        inertia = _vector(self.inertia, 3, "inertia")
        if np.min(inertia) <= 0:
            raise ValueError("inertia entries must be positive")
        object.__setattr__(self, "inertia", tuple(inertia))

    @property
    def hover_motor_speed(self):
        return float(np.sqrt(self.mass * self.gravity / (4 * self.thrust_coefficient)))


@dataclass(frozen=True)
class ControllerGains:
    kx: float = 3.0
    kv: float = 5.0
    kR: float = 0.90
    kOmega: float = 0.120

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class State:
    """p(3), v(3), R(3x3, body -> world), Omega(3), rotor speeds(4).

    pack() uses NumPy row-major order: [p, v, R.ravel(), Omega, omega].
    R occupies nine coordinates subject to the SO(3) constraint.
    """

    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    angular_velocity: np.ndarray
    motor_speed: np.ndarray

    def pack(self):
        return np.concatenate((self.position, self.velocity, self.rotation.ravel(),
                               self.angular_velocity, self.motor_speed))

    @classmethod
    def unpack(cls, packed):
        x = np.asarray(packed, dtype=float)
        if x.shape != (22,):
            raise ValueError("packed state must have shape (22,)")
        return cls(x[:3], x[3:6], x[6:15].reshape(3, 3), x[15:18], x[18:22])

    @classmethod
    def hover(cls, vehicle=None, position=(0., 0., 1.), yaw=0.):
        """Initialize at hover, including steady rotor speeds, in a z-up world."""
        vehicle = VehicleParameters() if vehicle is None else vehicle
        if not np.isfinite(yaw):
            raise ValueError("yaw must be finite")
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        return cls(_vector(position, 3, "position"), np.zeros(3), rotation,
                   np.zeros(3), np.full(4, vehicle.hover_motor_speed))


@dataclass
class Reference:
    """World-frame p/v/a, yaw, and optional desired-body-frame rate feedforward.

    The default zero angular feedforward is sufficient for this fixed-yaw
    point example. A trajectory tracker may supply consistent nonzero values.
    heading optionally specifies the desired body x-axis in world coordinates.
    """

    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading: np.ndarray | None = None

    def __post_init__(self):
        for name in ("position", "velocity", "acceleration", "angular_velocity", "angular_acceleration"):
            setattr(self, name, _vector(getattr(self, name), 3, name))
        if not np.isfinite(self.yaw):
            raise ValueError("yaw must be finite")
        if self.heading is not None:
            self.heading = _vector(self.heading, 3, "heading")
            if np.linalg.norm(self.heading) < 1e-9:
                raise ValueError("heading must be nonzero")


def hat(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def desired_rotation(force, yaw, heading=None):
    magnitude = np.linalg.norm(force)
    if magnitude < 1e-9:
        raise ValueError("Desired force is zero; attitude is undefined")
    b3 = force / magnitude
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.]) if heading is None else np.asarray(heading)
    b2 = np.cross(b3, heading)
    alternative = np.cross(b3, np.eye(3)[np.argmin(np.abs(b3))])
    b2 = np.where(np.sum(b2 * b2) < 1e-18, alternative, b2)
    b2 = b2 / np.linalg.norm(b2)
    return np.column_stack((np.cross(b2, b3), b2, b3))


class PointNavigator:
    """Supply a fixed point and yaw; position/velocity feedback lives in the tracker.

    Replace __call__ with your own policy returning a Reference. A new goal
    takes effect on the next call. This example has no obstacle avoidance.
    """

    def __init__(self, goal, yaw=0.):
        if not np.isfinite(yaw):
            raise ValueError("yaw must be finite")
        self.yaw = float(yaw)
        self.set_goal(goal)

    def set_goal(self, goal):
        self.goal = _vector(goal, 3, "goal")

    def __call__(self, state):
        return Reference(position=self.goal, yaw=self.yaw)

    def reached(self, state, position_tolerance=0.05, speed_tolerance=0.05):
        return bool(np.linalg.norm(state.position - self.goal) <= position_tolerance
                    and np.linalg.norm(state.velocity) <= speed_tolerance)


class GeometricController:
    """State + Reference -> requested wrench [thrust, Mx, My, Mz].

    Thrust is in N along body +z; moments are in N m in the body frame.
    Allocation is intentionally a separate block in this interface.
    """

    def __init__(self, vehicle=None, gains=None):
        self.vehicle = VehicleParameters() if vehicle is None else vehicle
        self.gains = ControllerGains() if gains is None else gains

    def __call__(self, state, reference):
        p, v, R, omega = (state.position, state.velocity, state.rotation, state.angular_velocity)
        vehicle, gains = self.vehicle, self.gains
        e3 = np.array([0., 0., 1.])
        inertia = np.diag(vehicle.inertia)
        force = (-gains.kx * (p - reference.position)
                 - gains.kv * (v - reference.velocity)
                 + vehicle.mass * (vehicle.gravity * e3 + reference.acceleration))
        rd = desired_rotation(force, reference.yaw, reference.heading)
        e_rotation = 0.5 * vee(rd.T @ R - R.T @ rd)
        e_omega = omega - R.T @ rd @ reference.angular_velocity
        thrust = force @ R[:, 2]
        moment = (-gains.kR * e_rotation - gains.kOmega * e_omega
                  + np.cross(omega, inertia @ omega)
                  - inertia @ (hat(omega) @ R.T @ rd @ reference.angular_velocity
                               - R.T @ rd @ reference.angular_acceleration))
        return np.concatenate(([thrust], moment))


def allocation_matrix(vehicle):
    """Motor order (+x,-y), (-x,+y), (+x,+y), (-x,-y); spin [1,1,-1,-1]."""
    kf, km = vehicle.thrust_coefficient, vehicle.torque_coefficient
    lx, ly = vehicle.arm_x, vehicle.arm_y
    return np.array([[kf, kf, kf, kf],
                     [-ly * kf, ly * kf, ly * kf, -ly * kf],
                     [-lx * kf, lx * kf, -lx * kf, lx * kf],
                     [-km, -km, km, km]])


class ControlAllocator:
    """Requested wrench -> four desired motor speeds in rad/s.

    Invert the allocation matrix, clip squared speeds to physical limits,
    and take the square root. Clipping can change the achieved wrench.
    """

    def __init__(self, vehicle=None):
        self.vehicle = VehicleParameters() if vehicle is None else vehicle
        self.matrix = allocation_matrix(self.vehicle)
        self.inverse = np.linalg.inv(self.matrix)

    def __call__(self, wrench):
        squared_speed = self.inverse @ _vector(wrench, 4, "wrench")
        return np.sqrt(np.clip(squared_speed, 0., self.vehicle.max_motor_speed ** 2))

    def wrench(self, motor_speed):
        """Instantaneous rotor aerodynamic wrench, excluding drag/rotor inertia."""
        speed = _vector(motor_speed, 4, "motor_speed")
        return self.matrix @ (speed * np.abs(speed))


def rigid_body_derivative(packed, command, vehicle, allocation, wind, external_force):
    """Continuous-time 22-coordinate model from the deployed source.

    command is desired rotor speed; the actual rotor speed is part of packed.
    Rotation storage is row-major. Wind and external force are world-frame.
    """
    s, p = State.unpack(packed), vehicle
    inertia = np.asarray(p.inertia)
    direction = np.array([1., 1., -1., -1.])
    e3 = np.array([0., 0., 1.])
    motor_dot = (np.clip(command, 0., p.max_motor_speed) - s.motor_speed) / p.motor_time_constant
    thrust, mx, my, mz = allocation @ (s.motor_speed * np.abs(s.motor_speed))
    air_velocity = s.velocity - wind
    speed = np.sqrt(np.maximum(np.sum(air_velocity ** 2), 1e-24))
    spin = np.sqrt(np.maximum(np.sum(s.angular_velocity ** 2), 1e-24))
    force = (s.rotation[:, 2] * thrust - p.linear_drag * speed * air_velocity
             + external_force - p.mass * p.gravity * e3)
    moment = np.array([mx, my, mz]) - p.rotor_inertia * (direction @ motor_dot) * e3
    moment = moment - p.angular_drag * spin * s.angular_velocity
    # Preserve the deployed model's corrected PLUS sign for rotor momentum.
    momentum = inertia * s.angular_velocity + p.rotor_inertia * (direction @ s.motor_speed) * e3
    omega_dot = (moment - np.cross(s.angular_velocity, momentum)) / inertia
    return np.concatenate((s.velocity, force / p.mass,
                           (s.rotation @ hat(s.angular_velocity)).ravel(), omega_dot, motor_dot))


class QuadrotorDynamics:
    """Motor commands -> nonlinear state derivative / RK4 State update."""

    def __init__(self, vehicle=None, wind=(0., 0., 0.), external_force=(0., 0., 0.)):
        self.vehicle = VehicleParameters() if vehicle is None else vehicle
        self.allocation = allocation_matrix(self.vehicle)
        self.wind = _vector(wind, 3, "wind")
        self.external_force = _vector(external_force, 3, "external_force")

    def derivative(self, packed, command):
        return rigid_body_derivative(packed, _vector(command, 4, "command"), self.vehicle,
                                     self.allocation, self.wind, self.external_force)

    def step(self, state, command, dt):
        """Hold command for dt, integrate with RK4, then project R onto SO(3)."""
        if not np.isfinite(dt) or not 0 < dt <= min(0.01, self.vehicle.motor_time_constant / 2):
            raise ValueError("dt must be positive and <= min(0.01, motor_time_constant / 2)")
        x = state.pack()
        k1 = self.derivative(x, command)
        k2 = self.derivative(x + dt * k1 / 2, command)
        k3 = self.derivative(x + dt * k2 / 2, command)
        k4 = self.derivative(x + dt * k3, command)
        result = State.unpack(x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        u, _, vt = np.linalg.svd(result.rotation)
        result.rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        return result

