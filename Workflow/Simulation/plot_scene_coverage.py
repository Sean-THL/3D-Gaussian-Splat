#!/usr/bin/env python3
"""Diagnostic capture-pose coverage; not an accepted flight-volume estimator."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/dasc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Workflow.Simulation.tasks.trajectory import make_trajectory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,default=ROOT/'data/scene_20261025_200034')
    args=parser.parse_args()
    scene=args.scene
    poses=json.loads((scene/'evaluation/colmap_camera_poses.json').read_text())['poses']
    alignment=json.loads((scene/'calibration/alignment_20261103T220923_264755Z/alignment.json').read_text())
    world=alignment['gs_to_vicon'];mount=alignment['camera_to_tracked_body']
    rvg=Rotation.from_matrix(world['rotation_matrix']);rbc=Rotation.from_matrix(mount['rotation_matrix'])
    body=[];camera=[];names=[]
    for pose in poses:
        q=pose['qvec_wxyz'];rcg=Rotation.from_quat([q[1],q[2],q[3],q[0]])
        camera_v=world['scale_m_per_gs_unit']*rvg.apply(-rcg.inv().apply(pose['tvec']))+world['translation_m']
        rvb=rvg*rcg.inv()*rbc.inv()
        camera.append(camera_v);body.append(camera_v-rvb.apply(mount['translation_m']));names.append(pose['name'])
    body=np.array(body);camera=np.array(camera);lookup={n:i for i,n in enumerate(names)}
    rgb=list(csv.DictReader((scene/'evaluation/paired_rgb_metrics.csv').open()))
    depth=list(csv.DictReader((scene/'evaluation/depth_20261103T041136_791961Z/per_frame.csv').open()))
    rgb_indices=[lookup[row['image']] for row in rgb]
    depth_indices=[lookup[row['image']] for row in depth]
    config=json.loads((ROOT/'Workflow/Simulation/configs/quadrotor_nmpc.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#19334c','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(16,6),facecolor='white')
    fig.subplots_adjust(left=.055,right=.96,top=.76,bottom=.20,wspace=.37)
    fig.text(.055,.94,'Where did the reconstruction cameras actually travel?',fontsize=22,weight='bold')
    fig.text(.055,.875,'296 captured poses mapped to the tracked-body frame using the candidate calibration',fontsize=12,color='#667e91')
    for ax in axes:
        ax.grid(color='#e6edf2',lw=.7);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(colors='#667e91',labelsize=8)
    ax=axes[0]
    ax.set_title('Capture path versus proposed route size',loc='left',fontsize=11,weight='bold')
    ax.scatter(body[:,0],body[:,1],s=9,c='#9eb0bf',alpha=.65,label='All captured body poses')
    for size,color,label in [(.6,'#2979bf','0.6 m reference'),(2.,'#db8c35','2.0 m reference')]:
        route=make_trajectory({**config['task'],'width_m':size,'height_m':size})
        p=np.array(route.polyline(500))
        ax.plot(p[:,0],p[:,1],c=color,lw=1.8,label=label)
    ax.set(xlabel='Vicon X / m',ylabel='Vicon Y / m',aspect='equal')
    ax.legend(fontsize=7,frameon=False,loc='upper right')
    ax=axes[1]
    ax.set_title('Held-out RGB quality at capture poses',loc='left',fontsize=11,weight='bold')
    ax.scatter(body[:,0],body[:,1],s=9,c='#d4dde5',alpha=.6)
    points=ax.scatter(body[rgb_indices,0],body[rgb_indices,1],c=[float(r['psnr_logged']) for r in rgb],
                      cmap='viridis',s=40,edgecolors='white',linewidths=.4)
    ax.set(xlabel='Vicon X / m',ylabel='Vicon Y / m',aspect='equal')
    fig.colorbar(points,ax=ax,shrink=.65,pad=.03,label='RGB PSNR / dB (higher is better)')
    ax=axes[2]
    ax.set_title('Depth error versus capture altitude',loc='left',fontsize=11,weight='bold')
    ax.scatter(body[depth_indices,2],[float(row['median_abs_m']) for row in depth],s=40,c='#2979bf')
    ax.set(xlabel='Tracked-body Vicon Z / m',ylabel='Per-view median absolute depth error / m')
    fig.text(.055,.105,'Diagnostic only: a bounding box or high PSNR at one pose does not certify nearby positions, orientations, surfaces or collision clearance.',fontsize=10,color='#667e91')
    fig.text(.055,.055,'References are centred at (0, 0); randomized translations are not shown. RGB/depth metrics use different image grids/subsets and are not directly equivalent.',fontsize=9,color='#667e91')
    output=scene/'evaluation/capture_pose_coverage'
    for extension in ['png','pdf']:fig.savefig(output.with_suffix('.'+extension),dpi=180,facecolor='white')
    plt.close(fig)
    summary=dict(captured_camera_min_m=camera.min(axis=0).tolist(),captured_camera_max_m=camera.max(axis=0).tolist(),
        candidate_body_min_m=body.min(axis=0).tolist(),candidate_body_max_m=body.max(axis=0).tolist(),
        note='Capture-pose diagnostic only; not a validated navigable or visually reliable region.')
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(output.with_suffix('.png'))


if __name__=='__main__':main()
