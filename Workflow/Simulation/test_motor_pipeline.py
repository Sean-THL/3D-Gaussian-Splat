"""Independent deployed-model parity and full-plant/simplified-controller contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.third_party.dev_quad_control import quadrotor as source
from Workflow.Simulation.control.mpc_geometric import LinearMPCGeometric
from Workflow.Simulation.figs.dynamics import State
from Workflow.Simulation.figs.motor_model import MotorQuadrotor
from Workflow.Simulation.figs.simulator import Simulator
from Workflow.Simulation.dataset import DemonstrationRecorder
from Workflow.Simulation.audit_dataset import audit_episode

HERE=Path(__file__).parent
CONFIG=json.loads((HERE/'configs/motor_mpc.json').read_text())


class MotorPipelineTests(unittest.TestCase):
    def test_pinned_files_and_independent_deployed_reference(self):
        folder=HERE/'third_party/dev_quad_control'
        provenance=json.loads((folder/'UPSTREAM.json').read_text())
        for name,digest in provenance['sha256'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest,name)
        fixture=json.loads((folder/'fixtures/deployed_reference.json').read_text())
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                parameters=source.VehicleParameters(**case['vehicle'])
                controller=source.GeometricController(parameters,source.ControllerGains(**case['gains']))
                plant=source.QuadrotorDynamics(parameters,wind=case['wind'],external_force=case['external_force'])
                state=source.State.unpack(case['state'])
                wrench=controller(state,source.Reference(**case['reference']))
                np.testing.assert_allclose(wrench,case['wrench'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(source.ControlAllocator(parameters)(wrench),case['allocated_command'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(plant.derivative(state.pack(),case['plant_command']),case['derivative'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(plant.step(state,case['plant_command'],case['dt']).pack(),case['next_state'],rtol=1e-12,atol=1e-12)

    def test_geometric_wrench_is_preserved_by_rate_interface(self):
        plant=MotorQuadrotor(CONFIG['vehicle'],CONFIG['vehicle'],CONFIG['geometric_controller'])
        controller=LinearMPCGeometric(CONFIG['mpc'],CONFIG['vehicle'],CONFIG['geometric_controller'],.05)
        state=State(np.array([0.,0.,1.2]),np.zeros(3),Rotation.from_euler('xyz',[.01,-.02,.3]),np.array([.03,-.02,.04]))
        thrust,rates,wrench=controller.convert(state,np.array([.1,-.1,.05]),.1)
        self.assertLess(max(abs(rates)),plant.rate_limit)
        np.testing.assert_allclose(plant.inner_wrench(state,thrust,rates),wrench,atol=1e-12)
        self.assertEqual(controller.outer.A.shape,(7,7))
        changed=copy.deepcopy(CONFIG['vehicle']);changed['mass_kg']*=1.15
        changed['inertia_kg_m2']=[.02]*3
        other=LinearMPCGeometric(CONFIG['mpc'],changed,CONFIG['geometric_controller'],.05)
        np.testing.assert_array_equal(controller.outer.A,other.outer.A)
        np.testing.assert_array_equal(controller.outer.B,other.outer.B)

    def test_native_backend_rejects_silent_model_substitution(self):
        from Workflow.Simulation.isaac_sim.native_backend import NativeSimulator
        with self.assertRaisesRegex(ValueError,'run.sh'):
            NativeSimulator(CONFIG,None)

    def test_hover_motor_lag_and_floor_contract(self):
        plant=MotorQuadrotor(CONFIG['vehicle'],CONFIG['vehicle'],CONFIG['geometric_controller'])
        initial=State(np.array([0.,0.,1.2]),np.zeros(3),Rotation.identity(),np.zeros(3))
        state=plant.initialize_motor_state(initial)
        for _ in range(100):state=plant.step(state,plant.mass*plant.gravity,np.zeros(3),.01)
        np.testing.assert_allclose(state.position,initial.position,atol=1e-12)
        powered=plant.step(state,1.2*plant.mass*plant.gravity,np.zeros(3),.005)
        self.assertGreater(powered.motor_speeds_rad_s.mean(),state.motor_speeds_rad_s.mean())
        self.assertLess(powered.actuator_thrust_n,1.2*plant.mass*plant.gravity)
        np.testing.assert_allclose(powered.orientation.as_matrix().T@powered.orientation.as_matrix(),np.eye(3),atol=1e-12)
        state.position[2]=.00001;state.velocity[2]=-1.
        with self.assertRaisesRegex(RuntimeError,'no contact solver'):
            plant.step(state,0.,np.zeros(3),.01)

    def test_randomized_pose_action_motor_dataset_contract(self):
        class Renderer:
            metadata={'kind':'pose-coded test pixels'}
            def render(self,state):
                return np.full((6,8,3),int(100+state.orientation.as_euler('xyz',degrees=True)[1]),dtype=np.uint8),None
        config=copy.deepcopy(CONFIG);config['randomization']['enabled']=True;config['seed']=53
        sim=Simulator(config);sim.paused=False
        self.assertNotEqual(sim.vehicle.mass,sim.controller.vehicle.mass)
        with tempfile.TemporaryDirectory() as directory:
            recorder=DemonstrationRecorder(directory,Renderer())
            for _ in range(20):
                before=sim.state.orientation.as_quat().copy()
                observation=recorder.capture(sim);sim.advance()
                row=sim.demonstrations[-1]
                np.testing.assert_allclose(row['quaternion_xyzw'],before)
                self.assertEqual(len(row['observed_motor_speeds_rad_s']),4)
                self.assertIn('outer_acceleration_m_s2',row)
                recorder.write(row,observation)
            recorder.close()
            self.assertEqual(audit_episode(directory)['samples'],20)
            path=Path(directory)/'rgb_actions.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['observed_motor_speeds_rad_s'][0]=-1
            path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            with self.assertRaisesRegex(ValueError,'motor state'):audit_episode(directory)


if __name__=='__main__':unittest.main()
