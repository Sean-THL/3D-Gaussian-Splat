"""Ideal double-integrator MPC + upstream geometric wrench conversion.

Linear MPC observes only p/v/yaw. Full attitude/body rates are used by the
geometric conversion and inner rate loop, not by MPC's prediction model.
Teacher labels remain thrust N + body rates rad/s. The wrench-to-rate
factorization is exact before public thrust/rate clipping.
"""
import numpy as np

from Workflow.Simulation.control.mpc import LinearMPC
from Workflow.Simulation.figs.motor_model import motor_parameters
from Workflow.Simulation.third_party.dev_quad_control import quadrotor as upstream


class LinearMPCGeometric:
    def __init__(self, mpc_config, vehicle, gains, dt):
        self.outer=LinearMPC(mpc_config,vehicle,dt)
        self.vehicle=motor_parameters(vehicle)
        self.gains=upstream.ControllerGains(**gains)
        if self.gains.kOmega<=0:
            raise ValueError('Geometric rate gain kOmega must be positive')
        self.tracker=upstream.GeometricController(self.vehicle,self.gains)
        self.horizon=self.outer.horizon
        self.max_rate=float(vehicle.get('max_body_rate_rad_s',4.))
        self.max_thrust=float(vehicle.get('max_thrust_n',4*self.vehicle.thrust_coefficient*self.vehicle.max_motor_speed**2))
        self.diagnostics=dict(self.outer.diagnostics)
        self.acceleration_command=np.zeros(3);self.yaw_rate_command=0.
        self.requested_wrench=np.array([self.vehicle.mass*self.vehicle.gravity,0.,0.,0.])

    def convert(self,state,acceleration,yaw_rate):
        # Position/velocity feedback belongs to MPC; do not add a second p/v loop.
        yaw=state.orientation.as_euler('xyz')[2]
        force=self.vehicle.mass*(np.asarray(acceleration)+[0.,0.,self.vehicle.gravity])
        desired=upstream.desired_rotation(force,yaw)
        epsilon=1e-5
        derivative=(upstream.desired_rotation(force,yaw+epsilon)-upstream.desired_rotation(force,yaw-epsilon))/(2*epsilon)
        # Heading feedforward at fixed desired thrust direction. Tilt feedforward
        # is left to geometric feedback; this is not a full flatness controller.
        omega_d=upstream.vee(desired.T@derivative)*yaw_rate
        reference=upstream.Reference(position=state.position,velocity=state.velocity,
            acceleration=acceleration,yaw=yaw,angular_velocity=omega_d)
        omega=np.zeros(3) if state.body_rates is None else state.body_rates
        full=upstream.State(state.position,state.velocity,state.orientation.as_matrix(),omega,np.zeros(4))
        wrench=self.tracker(full,reference)
        gyroscopic=np.cross(omega,np.asarray(self.vehicle.inertia)*omega)
        rate_command=omega+(wrench[1:]-gyroscopic)/self.gains.kOmega
        return float(np.clip(wrench[0],0.,self.max_thrust)),np.clip(rate_command,-self.max_rate,self.max_rate),wrench

    def command_horizon(self,state,references):
        acceleration,yaw_rate=self.outer.command_horizon(state,references)
        thrust,rates,self.requested_wrench=self.convert(state,acceleration,yaw_rate)
        self.acceleration_command=acceleration.copy();self.yaw_rate_command=float(yaw_rate)
        self.diagnostics=dict(self.outer.diagnostics)
        return thrust,rates
