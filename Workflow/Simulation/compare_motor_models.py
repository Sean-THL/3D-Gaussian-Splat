#!/usr/bin/env python3
"""Paired plant comparison: common ideal MPC, reference, initial state and timing.

The ideal integrator receives acceleration/yaw rate directly. The approximate
and full quadrotors receive thrust/body rates through the same geometric
conversion. These are closed-loop comparisons, not identical open-loop inputs.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/dasc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Workflow.Simulation.control.mpc import LinearMPC
from Workflow.Simulation.control.mpc_geometric import LinearMPCGeometric
from Workflow.Simulation.figs.dynamics import State,Quadrotor
from Workflow.Simulation.figs.integrators import Integrator
from Workflow.Simulation.figs.motor_model import MotorQuadrotor
from Workflow.Simulation.tasks.trajectory import make_trajectory

KINDS=['ideal_integrator','approximate_quadrotor','full_motor_quadrotor']
LABELS=['A · Ideal double integrator','B · Approximate thrust/body rates','C · Full motor dynamics']
COLORS=['#2979bf','#db8c35','#119c87']


def reference(t,task,scenario):
    if scenario=='figure8':return np.r_[make_trajectory(task).trajectory_function(t),0.]
    # A mechanical test only: 3 m in 4 s, with hold periods. No room-coverage claim.
    u=np.clip((t-1.)/4.,0.,1.)
    return np.array([3*(10*u**3-15*u**4+6*u**5),0,1.2,
        3*(30*u**2-60*u**3+30*u**4)/4.,0,0,
        3*(60*u-180*u**2+120*u**3)/16.,0,0,0,0])


def simulate(config,kind,scenario,mass_scale):
    nominal=copy.deepcopy(config['vehicle']);actual=copy.deepcopy(nominal)
    actual['mass_kg']*=mass_scale
    # Same nominal limits. Legacy approximation keeps its documented response law.
    actual.update(rate_time_constant_s=.04,thrust_time_constant_s=.03,linear_drag_per_s=.15,
        max_thrust_n=4*nominal['motor_thrust_coefficient']*nominal['max_motor_speed_rad_s']**2)
    if kind=='ideal_integrator':
        plant=Integrator(actual,'double_integrator')
        controller=LinearMPC(config['mpc'],nominal,1/config['control_hz'])
    else:
        plant=Quadrotor(actual) if kind=='approximate_quadrotor' else MotorQuadrotor(actual,nominal,config['geometric_controller'])
        controller=LinearMPCGeometric(config['mpc'],nominal,config['geometric_controller'],1/config['control_hz'])
    state=State(reference(0,config['task'],scenario)[:3],np.zeros(3),Rotation.identity(),np.zeros(3))
    if hasattr(plant,'initialize_motor_state'):state=plant.initialize_motor_state(state)
    dt=1/config['control_hz'];h=1/config['physics_hz'];total=32. if scenario=='figure8' else 8.
    rows=[]
    for k in range(round(total/dt)+1):
        t=k*dt;refs=np.array([reference(t+j*dt,config['task'],scenario) for j in range(controller.horizon+1)])
        action,rotation_action=controller.command_horizon(state,refs)
        if kind=='ideal_integrator':acc=action;yaw_rate=rotation_action;thrust=None;rates=None
        else:acc=controller.acceleration_command;yaw_rate=controller.yaw_rate_command;thrust=action;rates=rotation_action.tolist()
        rows.append(dict(time_s=t,position_m=state.position.tolist(),velocity_m_s=state.velocity.tolist(),
            quaternion_xyzw=state.orientation.as_quat().tolist(),reference=refs[0].tolist(),
            acceleration_command_m_s2=acc.tolist(),yaw_rate_command_rad_s=yaw_rate,
            thrust_n=thrust,body_rate_command_rad_s=rates,
            actual_body_rates_rad_s=None if state.body_rates is None else state.body_rates.tolist(),
            motor_speeds_rad_s=None if state.motor_speeds_rad_s is None else state.motor_speeds_rad_s.tolist(),
            solver=dict(controller.diagnostics)))
        if k<round(total/dt):
            for _ in range(round(dt/h)):state=plant.step(state,action,rotation_action,h)
    return rows


def metrics(rows):
    p=np.array([r['position_m'] for r in rows]);ref=np.array([r['reference'][:3] for r in rows])
    error=np.linalg.norm(p-ref,axis=1)
    angles=Rotation.from_quat([r['quaternion_xyzw'] for r in rows]).as_euler('xyz',degrees=True)
    return dict(rms_error_m=float(np.sqrt(np.mean(error**2))),max_error_m=float(error.max()),
        max_abs_roll_deg=float(abs(angles[:,0]).max()),max_abs_pitch_deg=float(abs(angles[:,1]).max()),
        failed_solves=sum(r['solver']['status']!='solved' for r in rows))


def plot(case,out,title):
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for kind,label,color in zip(KINDS,LABELS,COLORS):
        rows=case[kind];t=np.array([r['time_s'] for r in rows]);p=np.array([r['position_m'] for r in rows]);ref=np.array([r['reference'][:3] for r in rows])
        angles=Rotation.from_quat([r['quaternion_xyzw'] for r in rows]).as_euler('xyz',degrees=True)
        axes[0,0].plot(t,p[:,0],label=label,color=color)
        axes[0,1].plot(t,100*np.linalg.norm(p-ref,axis=1),color=color,label=label)
        axes[0,2].plot(t,angles[:,1],color=color,label=label)
        axes[1,0].plot(t,np.array([r['acceleration_command_m_s2'] for r in rows])[:,0],color=color,label=label)
        if kind!='ideal_integrator':
            axes[1,1].plot(t,[r['thrust_n'] for r in rows],color=color,label=label)
            axes[1,2].plot(t,np.array([r['body_rate_command_rad_s'] for r in rows])[:,1],color=color,label=label)
    axes[0,0].plot(t,ref[:,0],'k--',lw=1,label='Reference')
    for ax,y,title_ax in zip(axes.flat,['X (m)','Position error (cm)','Pitch (degrees)','World X acceleration (m/s²)','Collective thrust (N)','Body Y rate command (rad/s)'],['Position','Tracking error','Achieved pitch → camera tilt','Outer MPC command','Quadrotor training target','Quadrotor training target']):
        ax.set(xlabel='Time (s)',ylabel=y,title=title_ax);ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8);axes[1,1].legend(fontsize=8)
    fig.suptitle(title,fontsize=17,fontweight='bold')
    for extension in ['png','pdf']:fig.savefig(Path(str(out)+'.'+extension),dpi=170)
    plt.close(fig)


def video(case,config,out,fps):
    from Workflow.Simulation.rendering.threedgut import GaussianRenderer
    rendering={**config['rendering'],'width':480,'height':270}
    renderer=GaussianRenderer(rendering,ROOT)
    (out.parent/'renderer_metadata.json').write_text(json.dumps(renderer.metadata,indent=2)+'\n')
    process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1440x350','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    try:
        for frame in range(32*fps+1):
            t=frame/fps;index=round(t*config['control_hz']);canvas=Image.new('RGB',(1440,350),'white');draw=ImageDraw.Draw(canvas)
            for col,(kind,label) in enumerate(zip(KINDS,LABELS)):
                row=case[kind][index];state=State(np.array(row['position_m']),np.array(row['velocity_m_s']),Rotation.from_quat(row['quaternion_xyzw']))
                rgb,_=renderer.render(state);canvas.paste(Image.fromarray(rgb),(480*col,45))
                angles=state.orientation.as_euler('xyz',degrees=True)
                draw.text((480*col+10,10),label,fill=COLORS[col])
                draw.text((480*col+10,325),f't={t:.1f}s | roll={angles[0]:+.2f} | pitch={angles[1]:+.2f} deg',fill='#19334c')
            if frame==12*fps:canvas.save(out.with_suffix('.png'))
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Comparison video encoding failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/scene_20261025_200034/simulation/dynamics_comparison_20261112')
    parser.add_argument('--render-video',action='store_true');parser.add_argument('--fps',type=int,default=5)
    parser.add_argument('--video-only',action='store_true',help='Render previously saved comparison states without rerunning physics')
    args=parser.parse_args()
    if not 1<=args.fps<=20 or 20%args.fps:parser.error('FPS must divide 20 and be 1..20')
    if args.video_only:
        config=json.loads((args.output/'config.json').read_text())
        case=json.loads((args.output/'figure8_mass_1.json').read_text())
        video(case,config,args.output/'figure8_rgb_comparison.mp4',args.fps)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'Workflow/Simulation/configs/motor_mpc.json').read_text())
    config['task']['yaw_mode']='constant';config['randomization']['enabled']=False
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    result={};experiments={}
    for scenario,scale in [('figure8',1.),('a_to_b',1.),('figure8',1.15)]:
        name=f'{scenario}_mass_{scale:g}';case={kind:simulate(config,kind,scenario,scale) for kind in KINDS}
        experiments[name]=case;result[name]={kind:metrics(rows) for kind,rows in case.items()}
        (args.output/(name+'.json')).write_text(json.dumps(case)+'\n')
        plot(case,args.output/name,f'{scenario.replace("_"," ")} · plant mass ×{scale:g} · common simplified MPC')
        print(name,json.dumps(result[name]),flush=True)
    report=dict(metrics=result,method=__doc__,mass_note='Ideal acceleration dynamics have no mass parameter, so mass changes cannot affect A. B and C use the same nominal geometric conversion and a heavier actual plant.',approximate_plant=dict(rate_time_constant_s=.04,thrust_time_constant_s=.03,linear_drag_per_s=.15),scope='Numerical model comparison, not hardware validation. A-to-B is mechanical only; Gaussian video uses the small figure eight.',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'Workflow/Simulation/control/mpc.py',ROOT/'Workflow/Simulation/control/mpc_geometric.py',ROOT/'Workflow/Simulation/figs/dynamics.py',ROOT/'Workflow/Simulation/figs/motor_model.py']})
    (args.output/'comparison_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.render_video:video(experiments['figure8_mass_1'],config,args.output/'figure8_rgb_comparison.mp4',args.fps)


if __name__=='__main__':main()
