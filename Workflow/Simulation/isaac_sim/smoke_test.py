#!/usr/bin/env python3
"""Finite native Isaac Sim 4.5 test: gravity, rigid body, and floor contact.

Run with the standalone installation's python.sh, not the 3DGUT environment.
No external scene assets, Gaussian model, ROS, or PX4 are required.
"""
import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui', action='store_true', help='Requires a desktop display; default is headless')
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 180:
        parser.error('Use at least 180 steps to allow floor contact to settle')
    if args.gui and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('--gui needs a desktop display; use headless mode over plain SSH')
    root = Path(__file__).resolve().parents[3]
    isaac_root = Path(os.environ.get('ISAAC_SIM_ROOT', str(Path.home()/'isaacsim')))
    version_path = isaac_root/'VERSION'
    version = version_path.read_text().strip() if version_path.is_file() else 'unknown'
    if version != 'unknown' and not version.startswith('4.5.'):
        parser.error(f'This first test targets the installed 4.5 release, found {version}')
    output = args.output or root/'data/scene_20261025_200034/isaac_sim'/(
        'smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'report.json'
    report = dict(status='starting', isaac_version=version, isaac_root=str(isaac_root),
                  platform=platform.platform(), python=sys.version, headless=not args.gui,
                  steps_requested=args.steps, physics_dt_s=1/60, external_assets_required=False)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'ISAAC_SMOKE_OUTPUT={output}', flush=True)
    application = None
    world = None
    try:
        from isaacsim import SimulationApp
        application = SimulationApp({'headless': not args.gui, 'width': 640, 'height': 480})

        # Kit-dependent modules must be imported after SimulationApp is created.
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from pxr import UsdGeom, UsdLux

        world = World(stage_units_in_meters=1., physics_dt=1/60, rendering_dt=1/60, backend='numpy')
        UsdGeom.SetStageUpAxis(world.stage, UsdGeom.Tokens.z)
        world.get_physics_context().set_gravity(-9.81)
        # Procedural primitives avoid Nucleus/remote asset downloads.
        world.scene.add(FixedCuboid(prim_path='/World/Ground', name='floor',
            position=np.array([0., 0., -.05]), scale=np.array([4., 4., .1]),
            size=1., color=np.array([.15, .19, .25])))
        cube = world.scene.add(DynamicCuboid(prim_path='/World/TestCube', name='falling_cube',
            position=np.array([0., 0., 1.5]), scale=np.array([.2, .2, .2]),
            size=1., mass=.2, color=np.array([.2, .75, .65])))
        UsdLux.DomeLight.Define(world.stage, '/World/Light').CreateIntensityAttr(800.)
        world.reset()
        world.play()
        initial_position = np.asarray(cube.get_world_pose()[0]).copy()
        samples = []
        for step in range(args.steps):
            if not application.is_running():
                raise RuntimeError('Isaac Sim stopped before the test completed')
            world.step(render=args.gui)
            position = np.asarray(cube.get_world_pose()[0])
            velocity = np.asarray(cube.get_linear_velocity())
            if not np.isfinite(position).all() or not np.isfinite(velocity).all():
                raise RuntimeError('Physics returned nonfinite state')
            samples.append(dict(step=step+1, time_s=(step+1)/60,
                                x_m=float(position[0]), y_m=float(position[1]), z_m=float(position[2]),
                                vz_m_s=float(velocity[2])))
            if (step+1) % 60 == 0:
                print(f"PHYSICS step={step+1} z={position[2]:.4f}m vz={velocity[2]:.4f}m/s", flush=True)
        final_position = np.asarray(cube.get_world_pose()[0])
        final_velocity = np.asarray(cube.get_linear_velocity())
        checks = dict(gravity_moves_object_down=bool(initial_position[2]-final_position[2] > 1.),
                      floor_contact_height_correct=bool(abs(final_position[2]-.1) < .03),
                      object_settled=bool(np.linalg.norm(final_velocity) < .05),
                      floor_not_passed_through=bool(min(r['z_m'] for r in samples) > .04))
        with (output/'trajectory.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        # Keep the authored stage as a reproducible native Isaac scene.
        world.stage.GetRootLayer().Export(str(output/'smoke_scene.usda'))
        report.update(status='passed' if all(checks.values()) else 'failed', checks=checks,
                      initial_position_m=initial_position.tolist(), final_position_m=final_position.tolist(),
                      final_velocity_m_s=final_velocity.tolist(), simulated_seconds=args.steps/60,
                      stage_up_axis=str(UsdGeom.GetStageUpAxis(world.stage)),
                      stage_metres_per_unit=float(UsdGeom.GetStageMetersPerUnit(world.stage)))
        print('ISAAC_SMOKE_RESULT='+json.dumps(report), flush=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        report_path.write_text(json.dumps(report, indent=2)+'\n')
        if world is not None:
            world.stop()
        if application is not None:
            application.close()
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
