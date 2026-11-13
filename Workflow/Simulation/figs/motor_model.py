"""Adapter for the pinned dev-quad-control-pipeline motor/RK4 plant.

The external equations and integration are unmodified. A fixed nominal inner
rate loop and allocator turn the public thrust/body-rate input into four desired
rotor speeds; actual rotor speeds, quadratic drag and rotor inertia evolve in
the upstream plant. No additional first-order body-rate/thrust lag is inserted.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.figs.dynamics import State
from Workflow.Simulation.third_party.dev_quad_control import quadrotor as upstream


PARAMETER_KEYS = dict(mass='mass_kg',inertia='inertia_kg_m2',gravity='gravity_m_s2',
    arm_x='motor_arm_x_m',arm_y='motor_arm_y_m',motor_time_constant='motor_time_constant_s',
    rotor_inertia='rotor_inertia_kg_m2',thrust_coefficient='motor_thrust_coefficient',
    torque_coefficient='motor_torque_coefficient',linear_drag='quadratic_drag_kg_per_m',
    angular_drag='quadratic_angular_drag_nm_s2',max_motor_speed='max_motor_speed_rad_s')


def motor_parameters(config):
    """Map explicit SI config keys; old linear damping/time constants are not motor parameters."""
    return upstream.VehicleParameters(**{name:config[key] for name,key in PARAMETER_KEYS.items() if key in config})


class MotorQuadrotor:
    physics_backend = 'dev_quad_motor_rk4'

    def __init__(self, config, nominal_config, gains):
        self.parameters=motor_parameters(config)
        self.nominal_parameters=motor_parameters(nominal_config)
        self.mass=self.parameters.mass;self.gravity=self.parameters.gravity
        self.floor=float(config.get('floor_z_m',0.))
        self.rate_limit=float(nominal_config.get('max_body_rate_rad_s',4.))
        capacity=4*self.nominal_parameters.thrust_coefficient*self.nominal_parameters.max_motor_speed**2
        self.thrust_limit=float(nominal_config.get('max_thrust_n',capacity))
        self.integration_dt=float(config.get('motor_integration_max_step_s',.005))
        self.k_omega=float(gains.get('kOmega',.12))
        if not np.isfinite([self.floor,self.rate_limit,self.thrust_limit,self.integration_dt,self.k_omega]).all() or min(self.rate_limit,self.thrust_limit,self.integration_dt,self.k_omega)<=0:
            raise ValueError('Invalid motor plant/rate controller limits')
        if self.parameters.hover_motor_speed>self.parameters.max_motor_speed:
            raise ValueError('Sampled mass exceeds the actual motor hover capacity')
        wind=np.asarray(config.get('wind_velocity_m_s',[0.,0.,0.]),dtype=float)
        force=np.asarray(config.get('external_force_n',[0.,0.,0.]),dtype=float)
        force=force+self.mass*np.asarray(config.get('acceleration_disturbance_m_s2',[0.,0.,0.]),dtype=float)
        self.plant=upstream.QuadrotorDynamics(self.parameters,wind=wind,external_force=force)
        self.allocator=upstream.ControlAllocator(self.nominal_parameters)
        self.actual_allocator=upstream.ControlAllocator(self.parameters)
        self.nominal_inertia=np.asarray(self.nominal_parameters.inertia)
        self.last_motor_command=np.full(4,self.parameters.hover_motor_speed)
        self.last_requested_wrench=np.array([self.mass*self.gravity,0.,0.,0.])
        self.motor_saturation_steps=0

    def initialize_motor_state(self, state):
        motors=np.full(4,self.parameters.hover_motor_speed)
        return State(state.position.copy(),state.velocity.copy(),state.orientation,
            np.zeros(3) if state.body_rates is None else state.body_rates.copy(),self.mass*self.gravity,motors)

    def inner_wrench(self, state, thrust_n, rates):
        omega=np.zeros(3) if state.body_rates is None else state.body_rates
        moment=self.k_omega*(np.clip(rates,-self.rate_limit,self.rate_limit)-omega)
        moment+=np.cross(omega,self.nominal_inertia*omega)
        return np.r_[np.clip(thrust_n,0.,self.thrust_limit),moment]

    def step(self, state, thrust_n, body_rates, dt):
        rates=np.asarray(body_rates,dtype=float)
        if rates.shape!=(3,) or not np.isfinite(np.r_[rates,thrust_n,dt]).all() or dt<=0:
            raise ValueError('Expected finite thrust, XYZ body rates and positive dt')
        if state.motor_speeds_rad_s is None:state=self.initialize_motor_state(state)
        full=upstream.State(state.position.copy(),state.velocity.copy(),state.orientation.as_matrix(),
            np.zeros(3) if state.body_rates is None else state.body_rates.copy(),state.motor_speeds_rad_s.copy())
        maximum_step=min(.01,self.parameters.motor_time_constant/2,self.integration_dt)
        count=int(np.ceil(dt/maximum_step));h=dt/count;saturated=False
        for _ in range(count):
            public=State(full.position,full.velocity,Rotation.from_matrix(full.rotation),full.angular_velocity)
            self.last_requested_wrench=self.inner_wrench(public,thrust_n,rates)
            squared=self.allocator.inverse@self.last_requested_wrench
            saturated|=bool(np.any((squared<0)|(squared>self.nominal_parameters.max_motor_speed**2)))
            self.last_motor_command=self.allocator(self.last_requested_wrench)
            full=self.plant.step(full,self.last_motor_command,h)
            if full.position[2]<self.floor:
                raise RuntimeError('Motor plant crossed the assumed floor; upstream model has no contact solver')
        self.motor_saturation_steps+=int(saturated)
        actual_thrust=self.actual_allocator.wrench(full.motor_speed)[0]
        return State(full.position,full.velocity,Rotation.from_matrix(full.rotation),
            full.angular_velocity,float(actual_thrust),full.motor_speed)
