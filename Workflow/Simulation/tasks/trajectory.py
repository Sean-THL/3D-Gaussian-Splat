"""References share trajectory_function(t) -> [p(3), v(3), a(3), yaw].

The M route is piecewise quintic. Circle and figure-eight references are
analytic and follow the user-provided equations, with configurable dimensions.
"""
import numpy as np


class TimeTrajectory:
    continuous = False

    def sample(self, time):
        reference = self.trajectory_function(time)
        return reference[:3], reference[3:6], reference[6:9]

    def polyline(self, count=160):
        return [self.sample(t)[0].tolist() for t in np.linspace(0, self.duration, count)]


class MRoute(TimeTrajectory):
    trajectory_type = 'm'
    label = 'M route'
    def __init__(self, config):
        self.config = dict(config)
        self.width = float(config['width_m'])
        self.height = float(config['height_m'])
        self.altitude = float(config['altitude_m'])
        self.speed = float(config['speed_m_s'])
        self.acceleration = float(config.get('max_acceleration_m_s2', .5))
        self.yaw = np.deg2rad(float(config.get('yaw_deg', 0.)))
        self.hold = float(config.get('hover_seconds', 1.))
        if min(self.width, self.height, self.speed, self.acceleration) <= 0 or self.hold < 0:
            raise ValueError('Route dimensions, speed and acceleration must be positive')
        center = np.asarray(config.get('center_xy_m', [0., 0.]), dtype=float)
        xy = np.array([[-.5, -.5], [-.5, .5], [0., -.15], [.5, .5], [.5, -.5]])
        xy = xy*[self.width, self.height]+center
        self.points = np.column_stack([xy, np.full(5, self.altitude)])
        lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        # Quintic smoothstep: max velocity 1.875, max acceleration 10/sqrt(3).
        self.durations = np.maximum(1.875*lengths/self.speed,
                                    np.sqrt((10/np.sqrt(3))*lengths/self.acceleration))
        self.edges = np.r_[0., np.cumsum(self.durations)]
        self.duration = float(self.edges[-1]+2*self.hold)

    def sample(self, time):
        local = float(time)-self.hold
        if local <= 0:
            return self.points[0].copy(), np.zeros(3), np.zeros(3)
        if local >= self.edges[-1]:
            return self.points[-1].copy(), np.zeros(3), np.zeros(3)
        index = min(int(np.searchsorted(self.edges, local, side='right')-1), 3)
        duration = self.durations[index]
        u = (local-self.edges[index])/duration
        direction = self.points[index+1]-self.points[index]
        s = 10*u**3-15*u**4+6*u**5
        ds = (30*u**2-60*u**3+30*u**4)/duration
        dds = (60*u-180*u**2+120*u**3)/duration**2
        return self.points[index]+s*direction, ds*direction, dds*direction

    def trajectory_function(self, time):
        position, velocity, acceleration = self.sample(time)
        return np.r_[position, velocity, acceleration, self.yaw]


class AnalyticTrajectory(TimeTrajectory):
    def __init__(self, config):
        self.config = dict(config)
        self.yaw = np.deg2rad(float(config.get('yaw_deg', 0.)))
        xy = config.get('center_xy_m', [0., 0.])
        self.start = np.asarray(config.get('start_position_m', [*xy, config['altitude_m']]), dtype=float)
        self.duration = float(config.get('duration_s', 20.))
        if self.start.shape != (3,) or not np.isfinite(self.start).all() or not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError('Analytic reference needs a finite XYZ start and positive duration')


class HorizontalCircle(AnalyticTrajectory):
    trajectory_type = 'horizontal_circle'
    label = 'Horizontal circle'
    continuous = True

    def __init__(self, config):
        super().__init__(config)
        self.radius = float(config.get('radius_m', config['width_m']/2))
        self.omega_max = float(config.get('max_angular_speed_rad_s', .75))
        self.beta = float(config.get('angular_ramp_rate_per_s', .1))
        if not np.isfinite([self.radius, self.omega_max, self.beta]).all() or min(self.radius, self.omega_max, self.beta) <= 0:
            raise ValueError('Circle radius, angular speed and ramp rate must be finite and positive')

    def trajectory_function(self, time):
        if time < 0:
            return np.r_[self.start, np.zeros(6), self.yaw]
        bt = self.beta*float(time)
        # log(cosh(bt)) = logaddexp(bt, -bt) - log(2), without overflow.
        theta = self.omega_max/self.beta*(np.logaddexp(bt, -bt)-np.log(2.))
        omega = self.omega_max*np.tanh(bt)
        alpha = self.omega_max*self.beta*(1-np.tanh(bt)**2)
        sine, cosine = np.sin(theta), np.cos(theta)
        position = self.start+self.radius*np.array([cosine-1., sine, 0.])
        velocity = self.radius*np.array([-omega*sine, omega*cosine, 0.])
        acceleration = self.radius*np.array([-alpha*sine-omega**2*cosine,
                                             alpha*cosine-omega**2*sine, 0.])
        return np.r_[position, velocity, acceleration, self.yaw]


class FigureEight(AnalyticTrajectory):
    trajectory_type = 'lissajous_figure8'
    label = 'Figure eight'

    def __init__(self, config):
        super().__init__(config)
        self.a = float(config.get('a_m', config['width_m']/2))
        self.b = float(config.get('b_m', config['height_m']))
        self.drop = float(config.get('vertical_drop_m', .5))
        if not np.isfinite([self.a, self.b, self.drop]).all() or min(self.a, self.b) <= 0 or self.drop < 0:
            raise ValueError('Figure-eight dimensions must be positive and vertical drop nonnegative')

    def trajectory_function(self, time):
        u = float(np.clip(time/self.duration, 0., 1.))
        q = 10*u**3-15*u**4+6*u**5
        q_dot = (30*u**2-60*u**3+30*u**4)/self.duration
        q_ddot = (60*u-180*u**2+120*u**3)/self.duration**2
        theta = 2*np.pi*q
        a, b, h = self.a, self.b, self.drop
        position = self.start+np.array([a*np.sin(theta), .5*b*np.sin(2*theta),
                                        -.5*h*(1-np.cos(theta))])
        velocity = np.array([2*np.pi*a*np.cos(theta)*q_dot,
                             2*np.pi*b*np.cos(2*theta)*q_dot,
                             -np.pi*h*np.sin(theta)*q_dot])
        acceleration = np.array([
            -(2*np.pi)**2*a*np.sin(theta)*q_dot**2+2*np.pi*a*np.cos(theta)*q_ddot,
            -8*np.pi**2*b*np.sin(2*theta)*q_dot**2+2*np.pi*b*np.cos(2*theta)*q_ddot,
            -2*np.pi**2*h*np.cos(theta)*q_dot**2-np.pi*h*np.sin(theta)*q_ddot])
        return np.r_[position, velocity, acceleration, self.yaw]


TRAJECTORIES = {'m': MRoute, 'horizontal_circle': HorizontalCircle, 'lissajous_figure8': FigureEight}


def make_trajectory(config):
    name = config.get('trajectory_type', 'm')
    if name not in TRAJECTORIES:
        raise ValueError(f'Unknown trajectory_type: {name}')
    return TRAJECTORIES[name](config)
