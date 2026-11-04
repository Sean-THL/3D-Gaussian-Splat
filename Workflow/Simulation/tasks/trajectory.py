"""Smooth M-route reference with a hover at either end."""
import numpy as np


class MRoute:
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

    def polyline(self, count=160):
        return [self.sample(t)[0].tolist() for t in np.linspace(0, self.duration, count)]
