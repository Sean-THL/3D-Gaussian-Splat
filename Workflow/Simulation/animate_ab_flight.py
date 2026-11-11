#!/usr/bin/env python3
"""CPU A-to-B experiment: linear MPC + attitude conversion on two plants.

The rigid-body reference is a numerical model, not PhysX. Camera panels show
a synthetic wireframe room, not the reconstructed 3DGS scene.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dasc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Workflow.Simulation.control.mpc import LinearMPC
from Workflow.Simulation.figs.dynamics import Quadrotor, State

INK='#19334c'; MUTED='#667e91'; COLORS=['#2979bf','#119c87']
DT=.01; CONTROL_DT=.05; TOTAL=8.; START=1.; DURATION=4.; DISTANCE=3.


def reference(t):
    u=np.clip((t-START)/DURATION,0.,1.)
    q=10*u**3-15*u**4+6*u**5
    dq=(30*u**2-60*u**3+30*u**4)/DURATION
    ddq=(60*u-180*u**2+120*u**3)/DURATION**2
    return np.array([DISTANCE*q,0,1.2,DISTANCE*dq,0,0,DISTANCE*ddq,0,0,0,0])


class RigidBodyReference(Quadrotor):
    """Same rate-to-torque law as native_backend, integrated without Isaac."""
    def __init__(self,config):
        super().__init__(config)
        self.inertia=np.asarray(config['inertia_kg_m2'])
        self.torque_limit=np.asarray(config['max_body_torque_nm'])

    def step(self,state,thrust_n,body_rates,dt):
        h=dt/10
        p=state.position.copy();v=state.velocity.copy();r=state.orientation
        omega=np.zeros(3) if state.body_rates is None else state.body_rates.copy()
        for _ in range(10):
            desired=self.rate_scale*np.clip(body_rates,-self.rate_limit,self.rate_limit)
            torque=self.inertia*(desired-omega)/self.rate_tau+np.cross(omega,self.inertia*omega)
            torque=np.clip(torque,-self.torque_limit,self.torque_limit)
            alpha=(torque-np.cross(omega,self.inertia*omega))/self.inertia
            mid=r*Rotation.from_rotvec((omega+.5*h*alpha)*h*.5)
            target=self.thrust_scale*np.clip(thrust_n,0,self.thrust_limit)
            decay=np.exp(-h/self.thrust_tau)
            avg=target+(self.actual_thrust-target)*(self.thrust_tau/h)*(1-decay)
            self.actual_thrust=target+(self.actual_thrust-target)*decay
            acceleration=mid.apply([0.,0.,avg/self.mass])+[0,0,-self.gravity]-self.drag*v+self.wind
            p+=h*v+.5*h*h*acceleration;v+=h*acceleration
            r=r*Rotation.from_rotvec((omega+.5*h*alpha)*h)
            omega+=h*alpha
        return State(p,v,r,omega,float(self.actual_thrust))


def attitude_conversion(state,acceleration,yaw,nominal):
    # Nominal parameters only; geometric construction of desired thrust axis.
    force=acceleration+[0,0,nominal.gravity]+nominal.drag*state.velocity
    z=force/np.linalg.norm(force)
    heading=np.array([np.cos(yaw),np.sin(yaw),0.])
    y=np.cross(z,heading);y/=np.linalg.norm(y)
    desired=Rotation.from_matrix(np.column_stack([np.cross(y,z),y,z]))
    rates=12.*(state.orientation.inv()*desired).as_rotvec()
    thrust=nominal.mass*np.dot(force,state.orientation.apply([0.,0.,1.]))
    return np.clip(thrust,0,nominal.thrust_limit),np.clip(rates,-nominal.rate_limit,nominal.rate_limit)


def simulate(config,kind):
    nominal=Quadrotor(config['vehicle'])
    plant=(Quadrotor if kind=='custom' else RigidBodyReference)(config['vehicle'])
    mpc=LinearMPC(config['mpc'],config['vehicle'],CONTROL_DT)
    state=State(np.array([0.,0.,1.2]),np.zeros(3),Rotation.identity(),np.zeros(3),nominal.mass*nominal.gravity)
    records=[];yaw_command=0.
    for k in range(round(TOTAL/DT)+1):
        t=k*DT
        if k%5==0:
            refs=np.array([reference(t+j*CONTROL_DT) for j in range(mpc.horizon+1)])
            acc,yaw_rate=mpc.command_horizon(state,refs)
            yaw_command=state.orientation.as_euler('xyz')[2]+CONTROL_DT*yaw_rate
        thrust,rates=attitude_conversion(state,acc,yaw_command,nominal)
        records.append(dict(time_s=t,position_m=state.position.tolist(),velocity_m_s=state.velocity.tolist(),
                            quaternion_xyzw=state.orientation.as_quat().tolist(),pitch_deg=float(state.orientation.as_euler('xyz',degrees=True)[1]),
                            thrust_n=float(thrust),body_rate_command_rad_s=rates.tolist(),acceleration_command_m_s2=acc.tolist()))
        if k<round(TOTAL/DT):state=plant.step(state,thrust,rates,DT)
    return records


def room_segments():
    lines=[]
    for x in np.arange(-1,9,.5):lines.append([[x,-3,0],[x,3,0]])
    for y in np.arange(-3,3.01,.5):lines.append([[-1,y,0],[8,y,0]])
    for y in (-3,3):
        for z in (0,1,2,3):lines.append([[-1,y,z],[8,y,z]])
        for x in np.arange(-1,9,1):lines.append([[x,y,0],[x,y,3]])
    for y in np.arange(-3,3.01,1):lines.append([[8,y,0],[8,y,3]])
    for z in (0,1,2,3):lines.append([[8,-3,z],[8,3,z]])
    return np.array(lines,dtype=float)


def project(lines,position,rotation):
    points=rotation.inv().apply((lines-position).reshape(-1,3)).reshape(-1,2,3)
    segments=[]
    for pair in points:
        a,b=pair.copy();near=.08
        if max(a[0],b[0])<near:continue
        if a[0]<near:a+=(b-a)*(near-a[0])/(b[0]-a[0])
        if b[0]<near:b+=(a-b)*(near-b[0])/(a[0]-b[0])
        segments.append([[-a[1]/a[0],a[2]/a[0]],[-b[1]/b[0],b[2]/b[0]]])
    return segments


def render(records,output,fps):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':MUTED})
    fig=plt.figure(figsize=(12.8,9),facecolor='white')
    fig.text(.05,.955,'A → B: the drone tilts, and the camera tilts with it',fontsize=20,weight='bold')
    fig.text(.05,.917,'Linear MPC → geometric attitude conversion → plant → actual state feedback',fontsize=12,color=MUTED)
    fig.text(.05,.885,'Blue: existing custom plant    |    Teal: numerical rigid-body reference (not Isaac / PhysX)',fontsize=10,color=MUTED)
    side=[];camera=[];artists=[];room=room_segments()
    for i,color in enumerate(COLORS):
        y=.56-i*.265
        ax=fig.add_axes([.06,y,.49,.205]);ax.set(xlim=(-.55,4.1),ylim=(.1,2.25),aspect='equal')
        ax.set_title(['Custom: body-rate response lag','Rigid body: torque + inertia'][i],loc='left',color=color,fontsize=12,weight='bold')
        ax.axhline(1.2,color='#dce5ec',ls='--',lw=1)
        ax.scatter([0,3],[1.2,1.2],marker='x',s=35,color=INK)
        for x,label in [(0,'A'),(3,'B')]:ax.text(x,1.52,label,ha='center',weight='bold')
        if i==1:ax.set_xlabel('Forward position / m')
        ax.set_ylabel('Height / m')
        ax.spines[['top','right']].set_visible(False);ax.tick_params(colors=MUTED,labelsize=8)
        body,=ax.plot([],[],color=color,lw=4);trail,=ax.plot([],[],color=color,alpha=.3,lw=1.5)
        look,=ax.plot([],[],color=color,lw=1.5);thrust,=ax.plot([],[],color='#db8c35',lw=2)
        cone=plt.Polygon(np.zeros((3,2)),color=color,alpha=.09);ax.add_patch(cone)
        info=ax.text(.02,.045,'',transform=ax.transAxes,fontsize=9,color=color)
        cam=fig.add_axes([.62,y,.32,.205]);cam.set(xlim=(-1,1),ylim=(-.65,.65),aspect='equal',facecolor='#fbfcfe')
        cam.set_title('Synthetic forward camera',fontsize=11,color=MUTED,loc='left')
        cam.set_xticks([]);cam.set_yticks([])
        for spine in cam.spines.values():spine.set_color('#dce5ec')
        collection=LineCollection([],colors=color,linewidths=.65,alpha=.48);cam.add_collection(collection)
        cam.axhline(0,color='#b8c4cf',ls=':',lw=.8);cam.axvline(0,color='#b8c4cf',ls=':',lw=.8)
        cam.text(.02,.94,'Floor grid + room walls',transform=cam.transAxes,va='top',fontsize=8,color=MUTED)
        side.append(ax);camera.append(cam);artists.append((body,trail,look,thrust,cone,info,collection))
    pitch=fig.add_axes([.065,.095,.875,.105])
    for data,color,style in zip(records,COLORS,['--','-']):pitch.plot([r['time_s'] for r in data],[r['pitch_deg'] for r in data],color=color,lw=1.7,ls=style)
    pitch.axhline(0,color='#dce5ec');pitch.set(xlim=(0,TOTAL),xlabel='Time / s',ylabel='Pitch / deg')
    pitch.spines[['top','right']].set_visible(False);pitch.tick_params(colors=MUTED,labelsize=8)
    cursor=pitch.axvline(0,color=INK,lw=1)
    clock=fig.text(.94,.843,'',ha='right',fontsize=12,weight='bold')
    fig.text(.05,.025,'CPU demonstration • 3 m straight path in 4 s + hover • Fixed forward camera mount • Synthetic scene, not 3DGS RGB',fontsize=9,color=MUTED)
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=3500,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    with writer.saving(fig,str(output.with_suffix('.mp4')),dpi=100):
        for frame,t in enumerate(np.linspace(0,TOTAL,round(TOTAL*fps)+1)):
            k=min(round(t/DT),len(records[0])-1)
            phase='REF: HOLD A' if t<1 else 'REF: MOVE A → B' if t<5 else 'REF: HOLD B'
            clock.set_text(f'{t:4.1f} s  |  {phase}')
            cursor.set_xdata([t,t])
            for data,color,items in zip(records,COLORS,artists):
                row=data[k];p=np.array(row['position_m']);r=Rotation.from_quat(row['quaternion_xyzw'])
                body,trail,look,thrust,cone,info,collection=items
                def point(local):return (p+r.apply(local))[[0,2]]
                a,b=point([-.22,0,0]),point([.22,0,0]);body.set_data([a[0],b[0]],[a[1],b[1]])
                trail.set_data([q['position_m'][0] for q in data[:k+1:3]],[q['position_m'][2] for q in data[:k+1:3]])
                c=point([.15,0,-.04]);end=point([.9,0,-.04]);look.set_data([c[0],end[0]],[c[1],end[1]])
                cone.set_xy([c,point([.9,0,.20]),point([.9,0,-.28])])
                end=point([0,0,.45]);thrust.set_data([p[0],end[0]],[p[2],end[1]])
                info.set_text(f"x = {p[0]:.2f} m    speed = {row['velocity_m_s'][0]:+.2f} m/s    pitch = {row['pitch_deg']:+.1f}°")
                camera_position=p+r.apply([.15,0,-.04])
                collection.set_segments(project(room,camera_position,r))
            if frame==round(2.*fps):fig.savefig(output.with_suffix('.png'),dpi=160,facecolor='white')
            writer.grab_frame(facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'diagnostics/videos/ab_dynamics_comparison')
    parser.add_argument('--fps',type=int,default=24)
    args=parser.parse_args()
    if not 1<=args.fps<=60:parser.error('FPS must be between 1 and 60')
    config=json.loads((ROOT/'Workflow/Simulation/configs/quadrotor_nmpc.json').read_text())
    data=[simulate(copy.deepcopy(config),kind) for kind in ['custom','rigid_reference']]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    record={'description':__doc__,'config':config,'time_step_s':DT,'controller_step_s':CONTROL_DT,
            'reference':{'distance_m':DISTANCE,'move_duration_s':DURATION,'start_s':START},
            'custom':data[0],'numerical_rigid_body':data[1]}
    args.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    render(data,args.output,args.fps)
    for kind,rows in zip(['custom','rigid_reference'],data):
        print(kind,'final_xyz_m=',rows[-1]['position_m'],'max_abs_pitch_deg=',max(abs(r['pitch_deg']) for r in rows))
    print(args.output.with_suffix('.mp4'))


if __name__=='__main__':main()
