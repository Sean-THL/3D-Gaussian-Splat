"""Kit --exec entry point, or import from Isaac's Script Editor.

Uses the existing streaming editor. Never creates a nested SimulationApp.
"""
import asyncio
import importlib
import os
import sys
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import UsdGeom

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def start():
    app = omni.kit.app.get_app()
    for _ in range(20):
        await app.next_update_async()
    from Workflow.Simulation.isaac_sim.ui_compat import install_property_scroll_guard
    print('DASC_UI_COMPAT='+str(install_property_scroll_guard()), flush=True)
    context = omni.usd.get_context()
    if context.get_stage() is None:
        await context.new_stage_async()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
    module_name = 'Workflow.Simulation.isaac_sim.live_bridge'
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)
    print('DASC_ISAAC_BRIDGE_READY: CUDA camera and MPC controls attached.', flush=True)


_previous = globals().get('_DASC_BRIDGE_START_TASK')
if _previous is not None and not _previous.done():
    _previous.cancel()
_DASC_BRIDGE_START_TASK = asyncio.ensure_future(start())
