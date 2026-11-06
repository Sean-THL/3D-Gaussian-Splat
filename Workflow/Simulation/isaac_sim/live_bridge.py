"""Isaac 4.5 visualization of the live CUDA simulator, including its camera.

Execute in the streamed editor, not system Python. The CUDA server owns MPC,
randomization, and dynamics. This stage contains visual objects, not a second
physics simulation. Existing scene objects outside /World/DASCFlight are kept.
"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.ui as ui
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom, UsdLux

from Workflow.Simulation.isaac_sim.bridge_client import SimulatorClient


class IsaacLiveBridge:
    def __init__(self, url='http://127.0.0.1:8765'):
        self.client = SimulatorClient(url)
        self.depth = False
        self.running = True
        self.window = None
        self.status = None
        self.route_key = None
        self.last_frame = -1
        self.status_path = Path(__file__).resolve().parents[3]/'data/scene_20261025_200034/isaac_sim/live_bridge_status.json'
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError('Open a stage in Isaac first')
        if not list(self.stage.Traverse()):
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(self.stage, 1.)
        if UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.z or not np.isclose(UsdGeom.GetStageMetersPerUnit(self.stage), 1.):
            raise RuntimeError('This bridge expects a Z-up stage in metres; open a new Isaac stage first')
        self.root = '/World/DASCFlight'
        if self.stage.GetPrimAtPath(self.root).IsValid():
            self.stage.RemovePrim(self.root)
        UsdGeom.Xform.Define(self.stage, self.root)
        self.drone = UsdGeom.Xform.Define(self.stage, self.root+'/Drone')
        self.translation = self.drone.AddTranslateOp()
        self.orientation = self.drone.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self.make_box('Body', [.16, .1, .04], [0., 0., 0.], [.15, .65, .8])
        self.make_box('ArmX', [.3, .025, .018], [0., 0., 0.], [.2, .25, .3])
        self.make_box('ArmY', [.025, .3, .018], [0., 0., 0.], [.2, .25, .3])
        self.make_box('HeadingNose', [.08, .025, .025], [.13, 0., .02], [1., .4, .08])
        for i, (x, y) in enumerate([(.15, 0.), (-.15, 0.), (0., .15), (0., -.15)]):
            rotor = UsdGeom.Cylinder.Define(self.stage, self.root+f'/Drone/Rotor{i}')
            rotor.CreateRadiusAttr(.045); rotor.CreateHeightAttr(.008)
            rotor.CreateAxisAttr(UsdGeom.Tokens.z)
            rotor.AddTranslateOp().Set(Gf.Vec3d(x, y, .02))
            rotor.CreateDisplayColorAttr([Gf.Vec3f(.7, .75, .8)])
        self.reference = self.make_curve('Reference', [.3, .55, .95])
        self.history = self.make_curve('ActualPath', [.2, .9, .6])
        light = UsdLux.DomeLight.Define(self.stage, self.root+'/Light')
        light.CreateIntensityAttr(700.)
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window('DASC: CUDA camera + MPC controls', width=740, height=690)
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label('Gaussian camera from the external CUDA simulator', height=22)
                ui.ImageWithProvider(self.provider, height=280)
                self.status = ui.Label('Connecting...', height=64, word_wrap=True)
                self.parameters = ui.Label('', height=55, word_wrap=True)
                with ui.HStack(height=28):
                    for label, action in [('Play / resume', 'play'), ('Pause', 'pause'), ('Restart route', 'route'), ('Reset / resample', 'reset'), ('Hold', 'hover'), ('Disturbance', 'disturbance')]:
                        ui.Button(label, clicked_fn=lambda action=action: self.send(action))
                with ui.HStack(height=28):
                    ui.Button('RGB', clicked_fn=lambda: self.select_depth(False))
                    ui.Button('Depth', clicked_fn=lambda: self.select_depth(True))
                    ui.Button('Constant yaw', clicked_fn=lambda: self.send('configure', task={'yaw_mode': 'constant'}))
                    ui.Button('Velocity yaw', clicked_fn=lambda: self.send('configure', task={'yaw_mode': 'velocity_aligned'}))
                ui.Label('Viewport: drone, route, Gaussian-center proxy. Camera: native 3DGUT image.\nDynamics remain in the CUDA server; scene objects are not colliders.', height=40, word_wrap=True)
        self.task = asyncio.ensure_future(self.run())

    def make_box(self, name, size, position, color):
        box = UsdGeom.Cube.Define(self.stage, self.root+'/Drone/'+name)
        box.CreateSizeAttr(1.)
        box.AddTranslateOp().Set(Gf.Vec3d(*position))
        box.AddScaleOp().Set(Gf.Vec3f(*size))
        box.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    def make_curve(self, name, color):
        # Thin mesh ribbons render consistently in Isaac 4.5 RTX viewports.
        curve = UsdGeom.Mesh.Define(self.stage, self.root+'/'+name)
        curve.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        curve.CreateDoubleSidedAttr(True)
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return curve

    def set_curve(self, curve, points):
        if len(points) < 2:
            points = [points[0], points[0]] if points else [[0., 0., 0.]]*2
        vertices = []
        for start, end in zip(np.asarray(points[:-1]), np.asarray(points[1:])):
            side = np.cross(end-start, [0., 0., 1.])
            norm = np.linalg.norm(side)
            side = side/norm*.006 if norm > 1e-9 else np.array([.006, 0., 0.])
            vertices.extend([start-side, start+side, end+side, end-side])
        curve.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
        curve.CreateFaceVertexCountsAttr([4]*(len(vertices)//4))
        curve.CreateFaceVertexIndicesAttr(list(range(len(vertices))))

    def report(self, **values):
        temporary = self.status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(values, indent=2)+'\n')
        temporary.replace(self.status_path)

    def select_depth(self, enabled):
        self.depth = enabled
        self.last_frame = -1

    def send(self, action, **values):
        async def request():
            try:
                await asyncio.to_thread(self.client.command, action, **values)
            except Exception as error:
                self.status.text = f'Command error: {error}'
        asyncio.ensure_future(request())

    def fetch(self):
        state = self.client.get_state()
        image = self.client.get_rgba(self.depth) if state['rendered_frames'] != self.last_frame else None
        return state, image

    async def run(self):
        try:
            scene = await asyncio.to_thread(self.client.get_scene)
            if scene['points_m']:
                points = UsdGeom.Points.Define(self.stage, self.root+'/GaussianCenters')
                points.CreatePointsAttr([Gf.Vec3f(*p) for p in scene['points_m']])
                points.CreateWidthsAttr([.018])
                points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
                points.CreateDisplayColorAttr([Gf.Vec3f(.35, .4, .48)])
            while self.running and self.window.visible:
                try:
                    state, image = await asyncio.to_thread(self.fetch)
                    if omni.usd.get_context().get_stage() != self.stage:
                        raise RuntimeError('Stage changed; rerun the bridge on the new stage')
                    self.translation.Set(Gf.Vec3d(*state['position']))
                    x, y, z, w = state['quaternion_xyzw']
                    self.orientation.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
                    key = json.dumps([state['episode'], state['task']], sort_keys=True)
                    if key != self.route_key:
                        self.set_curve(self.reference, state['route'])
                        center = np.mean(np.asarray(state['route']), axis=0)
                        set_camera_view(eye=center+np.array([2.5, -3., 2.]), target=center)
                        self.route_key = key
                    self.set_curve(self.history, state['history'])
                    if image:
                        width, height, pixels = image
                        self.provider.set_bytes_data(list(pixels), [width, height])
                        self.last_frame = state['rendered_frames']
                    if state['paused']:
                        phase = 'PAUSED'
                    elif state['mode'] == 'hover':
                        phase = 'HOLDING POSITION'
                    elif state['complete']:
                        phase = 'ROUTE COMPLETE - HOLDING: click Restart route'
                    else:
                        phase = 'FLYING REFERENCE'
                    route_clock = (f"route {state['route_time_s']:.1f}/{state['duration_s']:.1f}s"
                                   if not state['continuous'] else f"continuous route {state['route_time_s']:.1f}s")
                    self.status.text = (f"{phase} | {route_clock} | elapsed {state['time_s']:.1f}s\n"
                        f"{state['controller'].upper()} | {state['renderer']} | error {100*state['tracking_error_m']:.1f} cm | {state.get('error') or 'OK'}")
                    parameters = state.get('episode_parameters', {})
                    plant = parameters.get('plant', {})
                    self.parameters.text = (f"Randomization: {state.get('randomization_enabled')} | yaw: {state['task'].get('yaw_mode')}\n"
                        f"Acceleration gain: {plant.get('acceleration_scale', 1.)} | bias: {plant.get('acceleration_disturbance_m_s2', [0,0,0])}\n"
                        f"Altitude: {state['position'][2]:.2f}m | actual task yaw/offset: {state['task'].get('yaw_deg', 0.):.1f} deg")
                    self.report(status='running', simulation_time_s=state['time_s'], episode=state['episode'],
                                renderer=state['renderer'], gpu=state.get('gpu'), displayed_frame=self.last_frame,
                                position_m=state['position'], server_error=state.get('error'),
                                phase=phase, route_time_s=state['route_time_s'], route_duration_s=state['duration_s'])
                except Exception as error:
                    self.status.text = f'Bridge error: {error}. Check CUDA server terminal.'
                    self.report(status='error', error=str(error))
                await asyncio.sleep(.1)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self.status.text = f'Bridge startup error: {error}'
            self.report(status='error', error=str(error))

    def close(self):
        self.running = False
        self.task.cancel()
        if self.window:
            self.window.visible = False


previous = globals().get('_DASC_LIVE_BRIDGE')
if previous is not None:
    previous.close()
_DASC_LIVE_BRIDGE = IsaacLiveBridge(globals().get('DASC_SIMULATOR_URL', 'http://127.0.0.1:8765'))
