"""Run inside Isaac Sim 4.5's Script Editor, including a WebRTC session.

Creates a new demo stage (save any existing scene first), frames a falling cube,
and starts native physics. The application stays open. Stop then Play to repeat.
Do not run this file with system Python or create a second SimulationApp.
"""
import asyncio

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics


async def build_drop_demo():
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()
    await omni.usd.get_context().new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())

    physics = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
    physics.CreateGravityMagnitudeAttr(9.81)

    floor = UsdGeom.Cube.Define(stage, '/World/Ground')
    floor.CreateSizeAttr(1.)
    floor.AddTranslateOp().Set(Gf.Vec3d(0., 0., -.05))
    floor.AddScaleOp().Set(Gf.Vec3f(4., 4., .1))
    floor.CreateDisplayColorAttr([Gf.Vec3f(.15, .19, .25)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    cube = UsdGeom.Cube.Define(stage, '/World/FallingCube')
    cube.CreateSizeAttr(.2)
    cube.AddTranslateOp().Set(Gf.Vec3d(0., 0., 1.5))
    cube.CreateDisplayColorAttr([Gf.Vec3f(.2, .75, .65)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(.2)
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(800.)

    await omni.kit.app.get_app().next_update_async()
    set_camera_view(eye=np.array([3., 3., 2.5]), target=np.array([0., 0., .6]))
    timeline.set_current_time(0.)
    # Leave enough time for the camera view to become visible before the drop.
    await asyncio.sleep(1.)
    timeline.play()
    print('DASC_DROP_DEMO_READY: cube falling from 1.5m. Stop then Play to repeat. The stream stays open.')


previous_task = globals().get('_DASC_DROP_TASK')
if previous_task is not None and not previous_task.done():
    previous_task.cancel()
_DASC_DROP_TASK = asyncio.ensure_future(build_drop_demo())
