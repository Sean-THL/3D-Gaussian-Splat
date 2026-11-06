"""Native Isaac 4.5 rigid-body plant with thrust/body-rate command interface.

Position/orientation evolve only through PhysX forces and torques. Pose setters
are used at reset only. Import after SimulationApp starts.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.figs.dynamics import Quadrotor, State
from Workflow.Simulation.figs.simulator import Simulator


class NativeQuadrotor(Quadrotor):
    def __init__(self, config, world):
        super().__init__(config)
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.prims import RigidPrim
        from pxr import Gf, PhysxSchema, UsdPhysics
        self.world = world
        self.inertia = np.asarray(config.get('inertia_kg_m2', [.0049, .0049, .0088]), dtype=float)
        self.torque_limit = np.asarray(config.get('max_body_torque_nm', [.3, .3, .15]), dtype=float)
        if self.inertia.shape != (3,) or self.torque_limit.shape != (3,) or min(self.inertia) <= 0 or min(self.torque_limit) <= 0:
            raise ValueError('Native inertia and torque limits must be positive XYZ vectors')
        self.rate_tau = max(float(config.get('rate_time_constant_s', .04)), .01)
        self.thrust_tau = float(config.get('thrust_time_constant_s', .03))
        world.stop()
        if not world.scene.object_exists('native_floor'):
            world.scene.add(FixedCuboid(prim_path='/World/Floor', name='native_floor',
                position=np.array([0., 0., self.floor-.05]), scale=np.array([10., 10., .1]),
                size=1., color=np.array([.12, .16, .2])))
        if world.scene.object_exists('native_drone'):
            self.body = world.scene.get_object('native_drone')
            self.body.set_mass(self.mass)
            self.view = world.scene.get_object('native_forces')
        else:
            self.body = world.scene.add(DynamicCuboid(prim_path='/World/NativeDrone', name='native_drone',
                position=np.array([0., 0., 1.2]), scale=np.array([.16, .10, .04]),
                size=1., mass=self.mass, color=np.array([.1, .7, .8])))
            self.view = world.scene.add(RigidPrim(prim_paths_expr='/World/NativeDrone', name='native_forces'))
        mass_api = UsdPhysics.MassAPI.Apply(self.body.prim)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*self.inertia))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0., 0., 0.))
        physics = PhysxSchema.PhysxRigidBodyAPI.Apply(self.body.prim)
        physics.CreateLinearDampingAttr(0.)
        physics.CreateAngularDampingAttr(0.)
        physics.CreateEnableGyroscopicForcesAttr(True)
        physics.CreateSolverPositionIterationCountAttr(16)
        physics.CreateSolverVelocityIterationCountAttr(4)
        self.torque_clips = 0

    def initialize_state(self, state):
        self.world.reset()
        self.world.play()
        q = state.orientation.as_quat()
        self.body.set_world_pose(state.position, np.r_[q[3], q[:3]])
        self.body.set_linear_velocity(state.velocity)
        self.body.set_angular_velocity(np.zeros(3))
        self.physics_inertia = np.asarray(self.view.get_inertias()[0]).reshape(3, 3)
        if not np.allclose(self.physics_inertia, np.diag(self.inertia), atol=1e-8, rtol=1e-4):
            raise RuntimeError('PhysX inertia does not match the authored body inertia')
        if not np.isclose(self.body.get_mass(), self.mass, rtol=1e-5):
            raise RuntimeError('PhysX mass does not match the sampled body mass')

    def read_state(self):
        position, wxyz = self.body.get_world_pose()
        rotation = Rotation.from_quat(np.r_[wxyz[1:], wxyz[0]])
        rates = rotation.inv().apply(self.body.get_angular_velocity())
        return State(np.asarray(position, dtype=float), np.asarray(self.body.get_linear_velocity(), dtype=float), rotation,
                     rates, float(self.actual_thrust))

    def step(self, state, thrust_n, body_rates, dt):
        if not np.isclose(dt, self.world.get_physics_dt()):
            raise ValueError('Native plant dt must match the Isaac physics step')
        actual = self.read_state()
        rates = actual.orientation.inv().apply(self.body.get_angular_velocity())
        desired_rates = self.rate_scale*np.clip(np.asarray(body_rates), -self.rate_limit, self.rate_limit)
        torque = self.inertia*(desired_rates-rates)/self.rate_tau+np.cross(rates, self.inertia*rates)
        clipped = np.clip(torque, -self.torque_limit, self.torque_limit)
        self.torque_clips += int(np.any(np.abs(torque-clipped) > 1e-9))
        target = self.thrust_scale*np.clip(float(thrust_n), 0., self.thrust_limit)
        if self.thrust_tau > 0:
            decay = np.exp(-dt/self.thrust_tau)
            average = target+(self.actual_thrust-target)*(self.thrust_tau/dt)*(1-decay)
            self.actual_thrust = target+(self.actual_thrust-target)*decay
        else:
            self.actual_thrust = average = target
        force = actual.orientation.apply([0., 0., average])+self.mass*(self.wind-self.drag*actual.velocity)
        self.view.apply_forces_and_torques_at_pos(forces=np.asarray([force], dtype=np.float32),
            torques=np.asarray([actual.orientation.apply(clipped)], dtype=np.float32), is_global=True)
        self.world.step(render=False)
        result = self.read_state()
        self.actual_rates = result.orientation.inv().apply(self.body.get_angular_velocity())
        return result


class NativeSimulator(Simulator):
    def __init__(self, config, world):
        if config.get('dynamics_model') != 'quadrotor':
            raise ValueError('Native backend requires quadrotor dynamics')
        self.world = world
        super().__init__(config)

    def make_vehicle(self, plant):
        return NativeQuadrotor(plant, self.world)

    def reset(self):
        super().reset()
        self.vehicle.initialize_state(self.state)
        self.state = self.vehicle.read_state()
        # Isaac performs warm-up steps before the authored episode initial state.
        self.physics_epoch_time_s = float(self.world.current_time)
        self.episode_parameters['physics_backend'] = 'isaac_physx_forces_and_torques'
        self.episode_parameters['actual_inertia_kg_m2'] = self.vehicle.inertia.tolist()
        self.episode_parameters['actual_torque_limits_nm'] = self.vehicle.torque_limit.tolist()

    def command(self, command):
        super().command(command)
        if command['action'] == 'disturbance':
            self.vehicle.body.set_linear_velocity(self.state.velocity)

    def snapshot(self):
        snapshot = super().snapshot()
        snapshot['physics_backend'] = 'isaac_physx_forces_and_torques'
        snapshot['native_physics_time_s'] = float(self.world.current_time)-self.physics_epoch_time_s
        return snapshot
