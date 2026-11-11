#!/usr/bin/env python3
"""Compare randomized references and achieved trajectories from collected data."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dasc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Workflow.Simulation.audit_dataset import audit_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    folders = sorted(p for p in args.run.glob('episode_*') if (p/'rgb_actions.jsonl').exists())
    if not folders:parser.error('No collected episodes found')
    data = []
    for folder in folders:
        audit_episode(folder)
        rows = [json.loads(line) for line in (folder/'rgb_actions.jsonl').read_text().splitlines()]
        parameters = json.loads((folder/'episode_parameters.json').read_text())
        data.append((parameters, np.array([r['time_s'] for r in rows]),
                     np.array([r['position_m'] for r in rows]),
                     np.array([r['reference_horizon'][0][:3] for r in rows]),
                     Rotation.from_quat([r['quaternion_xyzw'] for r in rows]),
                     np.array([r['reference_horizon'][0][9] for r in rows])))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#19334c',
                         'axes.labelcolor':'#667e91','pdf.fonttype':42,'svg.fonttype':'none'})
    fig = plt.figure(figsize=(15,10),facecolor='white')
    grid = fig.add_gridspec(2,2,left=.07,right=.95,bottom=.17,top=.81,hspace=.49,wspace=.29)
    ax3 = fig.add_subplot(grid[0,0],projection='3d')
    xy = fig.add_subplot(grid[0,1]);z = fig.add_subplot(grid[1,0]);yaw = fig.add_subplot(grid[1,1])
    fig.text(.07,.945,'Randomized routes: reference and actual flight',fontsize=23,weight='bold')
    fig.text(.07,.906,f'{len(data)} collected episodes • actual saved PhysX states • all coordinates in Vicon metres',fontsize=12,color='#667e91')
    palette = ['#2979bf','#119c87','#db8c35','#885cb6']
    handles = [Line2D([0],[0],color='#19334c',ls='--',lw=1.5,label='Reference'),
               Line2D([0],[0],color='#19334c',lw=2,label='Achieved')]
    for index,(parameters,t,p,ref,rotation,yref) in enumerate(data):
        color = palette[index%len(palette)]
        handles.append(Line2D([0],[0],color=color,lw=2,label=f"Seed {parameters['seed']}"))
        ax3.plot(*ref.T,color=color,lw=1.2,ls='--',alpha=.6)
        ax3.plot(*p.T,color=color,lw=1.8)
        xy.plot(*ref[:,:2].T,color=color,lw=1.3,ls='--',alpha=.65)
        xy.plot(*p[:,:2].T,color=color,lw=1.8)
        xy.scatter(*p[0,:2],color=color,s=28,edgecolors='white',zorder=5)
        z.plot(t,ref[:,2],color=color,lw=1.2,ls='--',alpha=.65)
        z.plot(t,p[:,2],color=color,lw=1.8)
        z.scatter(t[-1],p[-1,2],color=color,s=25,zorder=5)
        actual_yaw = np.unwrap(rotation.as_euler('xyz')[:,2])
        actual_yaw += 2*np.pi*round((yref[0]-actual_yaw[0])/(2*np.pi))
        yaw.plot(t,np.rad2deg(yref),color=color,lw=1.3,ls='--',alpha=.65)
        yaw.plot(t,np.rad2deg(actual_yaw),color=color,lw=1.8)
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.065,.873),ncol=min(len(handles),6),frameon=False,fontsize=10)
    for ax,title in [(ax3,'3D trajectories'),(xy,'Top view — dots mark actual starting positions'),
                     (z,'Altitude and traversal timing'),(yaw,'Heading (velocity aligned + sampled offset)')]:
        ax.set_title(title,loc='left',fontsize=12,weight='bold',pad=14)
    ax3.set_xlabel('X / m');ax3.set_ylabel('Y / m');ax3.set_zlabel('Z / m')
    ax3.set_box_aspect([1,1,.4]);ax3.view_init(elev=25,azim=-58)
    for axis in [ax3.xaxis,ax3.yaxis,ax3.zaxis]:axis.set_pane_color((1,1,1,1))
    xy.set(xlabel='X / m',ylabel='Y / m');xy.set_aspect('equal',adjustable='datalim')
    z.set(xlabel='Simulation time / s',ylabel='Altitude / m')
    yaw.set(xlabel='Simulation time / s',ylabel='Unwrapped yaw / deg')
    for ax in [xy,z,yaw]:
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#c9d4dc')
        ax.grid(color='#e6edf2',lw=.7);ax.tick_params(colors='#667e91',labelsize=9)
    for index,(parameters,*_) in enumerate(data[:3]):
        task = parameters['task'];plant = parameters['plant']
        fig.text(.07,.105-.025*index,
            f"Seed {parameters['seed']}: reference centre ({task['center_xy_m'][0]:+.3f}, {task['center_xy_m'][1]:+.3f}) m"
            f"  |  altitude {task['altitude_m']:.3f} m  |  duration {task['duration_s']:.2f} s"
            f"  |  yaw offset {task['yaw_deg']:+.1f}°  |  mass {plant['mass_kg']:.3f} kg",fontsize=9,color=palette[index%len(palette)])
    fig.text(.07,.025,'Randomization changes the reference and actual plant. Solid paths include tracking error; they are not identical to the dashed references.',fontsize=9,color='#667e91')
    output=args.run/'randomized_trajectories'
    for extension in ['png','pdf','svg']:
        fig.savefig(output.with_suffix('.'+extension),dpi=200,facecolor='white')
    plt.close(fig)
    print(output.with_suffix('.png'))


if __name__=='__main__':main()
