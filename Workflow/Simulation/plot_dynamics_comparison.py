#!/usr/bin/env python3
"""Conceptual white-background diagrams; no simulated flight or RGB results."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dasc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'diagnostics/figures'
INK, MUTED = '#19334c', '#667e91'
BLUE, TEAL, ORANGE = '#2979bf', '#119c87', '#db8c35'


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'text.color':INK, 'figure.facecolor':'white',
                         'savefig.facecolor':'white', 'pdf.fonttype':42,
                         'svg.fonttype':'none'})


def box(ax, x, y, w, h, title, detail, color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.018',
                              fc='#f7fafc',ec='#dce5ec',lw=1.2))
    ax.text(x+.025,y+h*.69,title,fontsize=12,weight='bold',color=color,va='center')
    ax.text(x+.025,y+h*.29,detail,fontsize=10,color=MUTED,va='center')


def arrow(ax, start, end, color=MUTED):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,lw=1.6,color=color))


def export(fig, name):
    OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf','svg'):
        path=OUT/f'{name}.{suffix}'
        fig.savefig(path,dpi=220,facecolor='white')
        print(path)
    plt.close(fig)


def architecture():
    fig=plt.figure(figsize=(16,11))
    fig.text(.055,.95,'Two ways to simulate a tilting drone',fontsize=26,weight='bold')
    fig.text(.055,.91,'Both update full orientation. The difference is how rotation responds to a command.',fontsize=13,color=MUTED)
    for x,color,label,title,rows in [
        (.055,BLUE,'A  |  EXISTING CUSTOM PLANT','Simplified thrust / body-rate dynamics',[
            ('Quadrotor NMPC','Reference + current state → thrust T and body rates p, q, r'),
            ('Approximate actuator response','Body-rate command → achieved rates through a response lag'),
            ('Update orientation and translation','Integrate achieved rates; thrust points along body +Z'),
            ('Render from the resulting camera pose','Actual position + roll / pitch / yaw + fixed camera mount')]),
        (.535,TEAL,'B  |  EXISTING NATIVE ISAAC PLANT','Rigid-body dynamics with inertia',[
            ('Quadrotor NMPC','Reference + current state → thrust T and body rates p, q, r'),
            ('Body-rate tracking controller','Rate error → bounded torque; thrust also has response lag'),
            ('PhysX integrates forces and torques','Inertia determines rotation; tilted thrust changes translation'),
            ('Render from the resulting camera pose','Actual position + roll / pitch / yaw + fixed camera mount')])]:
        ax=fig.add_axes([x,.32,.41,.53]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        ax.text(0,1.07,label,color=color,fontsize=12,weight='bold')
        ax.text(0,1.01,title,fontsize=15,weight='bold')
        for i,(head,detail) in enumerate(rows):
            y=.76-i*.235
            box(ax,.015,y,.97,.17,head,detail,color)
            if i<3:arrow(ax,(.5,y-.008),(.5,y-.06),color)
        ax.text(.5,-.075,'Feed the resulting state back to the next MPC decision',ha='center',fontsize=10,color=color)
    fig.text(.055,.25,'YOUR PROPOSAL  |  change the planner, retain a quadrotor plant',fontsize=13,weight='bold',color=ORANGE)
    ax=fig.add_axes([.055,.11,.89,.105]);ax.set(xlim=(-.02,1.02),ylim=(0,1));ax.axis('off')
    items=[('Linear MPC','XYZ acceleration + yaw'),('Geometric controller','Desired tilt → thrust / rates'),('Quadrotor plant','Orientation + translation'),('Camera renderer','RGB from actual pose')]
    for i,(title,detail) in enumerate(items):
        x=i*.255
        box(ax,x+.005,.15,.225,.70,title,detail,ORANGE)
        if i<3:arrow(ax,(x+.235,.5),(x+.255,.5),ORANGE)
    fig.text(.055,.065,'Same principle in all cases: render the achieved camera pose, rather than a level pose placed on the reference path.',fontsize=11,color=INK)
    fig.text(.055,.032,'Conceptual architecture • The proposed linear-MPC / geometric-controller combination is not yet wired into our simulator.',fontsize=10,color=MUTED)
    export(fig,'quadrotor_methods_overview')


def drone(ax,x,z,pitch,color,thrust=False):
    # Side view: body +X forward, body +Z up. Positive pitch tips +X downward.
    theta=np.deg2rad(pitch)
    rotation=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    def point(a,b):return np.array([x,z])+rotation@np.array([a,b])
    rear,front=point(-.31,0),point(.31,0)
    ax.plot([rear[0],front[0]],[rear[1],front[1]],color=color,lw=5,solid_capstyle='round')
    for a in (-.27,.27):
        p=point(a,.05);q=rotation@np.array([.115,0])
        ax.plot([p[0]-q[0],p[0]+q[0]],[p[1]-q[1],p[1]+q[1]],color=color,lw=2)
    ax.add_patch(Circle((x,z),.055,fc=color,ec='white',zorder=4))
    camera=point(.20,-.065)
    ax.scatter(*camera,s=24,marker='s',color=INK,zorder=5)
    # Illustrative camera: optical axis aligned with body +X; NOT fitted mount.
    directions=[]
    for angle in (-18,18):
        a=np.deg2rad(angle);directions.append(camera+rotation@np.array([np.cos(a),np.sin(a)])*.8)
    ax.add_patch(Polygon([camera,*directions],fc=color,alpha=.09,ec='none'))
    arrow(ax,camera,camera+rotation@np.array([.8,0]),color)
    if thrust:arrow(ax,(x,z),point(0,.62),ORANGE)


def camera_motion():
    fig=plt.figure(figsize=(16,10),facecolor='white')
    fig.text(.055,.95,'What happens to the camera?',fontsize=27,weight='bold')
    fig.text(.055,.907,'A side-view example of forward acceleration, cruising, and braking',fontsize=14,color=MUTED)
    fig.text(.055,.865,'Shaded cone + colored arrow: camera view     •     Orange arrow: thrust direction',fontsize=11,color=MUTED)
    phases=[('1  HOVER','Level; no forward acceleration',0),
            ('2  ACCELERATE','Tilt forward; camera looks downward',16),
            ('3  CRUISE','Near level if drag is small',0),
            ('4  BRAKE','Tilt back; camera looks upward',-14)]
    for row,(name,color) in enumerate([('Custom quadrotor',BLUE),('Rigid-body quadrotor',TEAL)]):
        ax=fig.add_axes([.055,.49-row*.30,.89,.24]);ax.set(xlim=(-.6,8.3),ylim=(0,1.8));ax.axis('off')
        ax.text(0,1.02,name,transform=ax.transAxes,color=color,weight='bold',fontsize=14)
        ax.plot([-.5,8.2],[.12,.12],color='#dce5ec',lw=1)
        for j,(_,_,pitch) in enumerate(phases):
            drone(ax,.1+j*2.15,.78,pitch,color,thrust=j in (1,3))
            ax.text(.1+j*2.15,.19,f'{pitch:+d}°' if pitch else '0°',ha='center',fontsize=11,color=color)
        ax.text(.99,.02,'Forward →',transform=ax.transAxes,ha='right',fontsize=10,color=MUTED)
    for j,(title,desc,_) in enumerate(phases):
        x=.055+(.1+j*2.15+.6)/8.9*.89
        fig.text(x,.815,title,ha='center',fontsize=11,weight='bold')
        fig.text(x,.785,desc,ha='center',fontsize=9,color=MUTED)
    fig.text(.055,.125,'Both methods can produce this tilt sequence.',fontsize=16,weight='bold')
    fig.text(.055,.086,'Custom: approximate rate response.  Rigid body: torque + inertia determine the response and any tracking error.',fontsize=12,color=MUTED)
    fig.text(.055,.047,'Illustration only: angles, spacing and viewing cones are schematic; these are not simulation results or rendered 3DGS RGB.',fontsize=10,color=MUTED)
    fig.text(.055,.025,'Camera optical axis assumed along body +X here. The real mounting rotation changes the absolute viewing direction; drag can require tilt during cruise.',fontsize=9.5,color=MUTED)
    export(fig,'quadrotor_camera_motion')


if __name__=='__main__':
    setup()
    architecture()
    camera_motion()
