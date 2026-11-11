#!/usr/bin/env python3
"""Plot achieved tilt and mounted camera elevation from saved pilot states."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dasc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    folders = sorted(args.run.glob('episode_*'))
    if not folders:parser.error('No episodes found')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#19334c','pdf.fonttype':42})
    fig, axes = plt.subplots(3, len(folders), figsize=(13, 9), squeeze=False, facecolor='white')
    fig.subplots_adjust(left=.08,right=.96,bottom=.10,top=.83,hspace=.48,wspace=.23)
    fig.text(.08,.95,'The recorded flight tilts only slightly',fontsize=23,weight='bold')
    fig.text(.08,.905,'Actual saved PhysX poses • camera elevation calculated using the recorded mounting rotation',fontsize=11,color='#667e91')
    summary = []
    for column, folder in enumerate(folders):
        rows = [json.loads(s) for s in (folder/'rgb_actions.jsonl').read_text().splitlines()]
        metadata = json.loads((folder/'dataset_metadata.json').read_text())
        seed = json.loads((folder/'episode_parameters.json').read_text())['seed']
        t = np.array([r['time_s'] for r in rows])
        v = np.array([r['velocity_m_s'] for r in rows])
        r = Rotation.from_quat([row['quaternion_xyzw'] for row in rows])
        up = r.apply(np.tile([0.,0.,1.],(len(rows),1)))
        tilt = np.rad2deg(np.arccos(np.clip(up[:,2],-1,1)))
        mount = np.array(metadata['renderer']['camera_to_body']['rotation_matrix'])
        camera_forward = r.apply(np.tile(mount[:,2],(len(rows),1)))
        elevation = np.rad2deg(np.arcsin(np.clip(camera_forward[:,2],-1,1)))
        euler = r.as_euler('xyz',degrees=True)
        speed = np.linalg.norm(v[:,:2],axis=1)
        axes[0,column].plot(t,tilt,color='#2979bf',lw=2,label='Total tilt from vertical')
        axes[0,column].plot(t,euler[:,1],color='#119c87',lw=1.3,label='Body pitch (signed)')
        axes[0,column].set_title(f'Seed {seed}  |  peak tilt {tilt.max():.2f}°',loc='left',weight='bold',pad=12)
        axes[0,column].set_ylabel('Tilt / deg')
        axes[0,column].legend(fontsize=8,frameon=False,loc='upper right')
        axes[1,column].plot(t,elevation,color='#db8c35',lw=1.7)
        axes[1,column].axhline(0,color='#9cb0bf',lw=1,ls='--')
        axes[1,column].set_ylabel('Camera elevation / deg')
        axes[1,column].set_title('Negative = looks below horizontal',loc='left',fontsize=10,color='#667e91')
        axes[2,column].plot(t,speed,color='#2979bf',lw=1.7)
        axes[2,column].scatter(t[0],speed[0],color='#db8c35',s=28,zorder=5)
        axes[2,column].set_ylabel('Horizontal speed / m/s')
        axes[2,column].set_xlabel('Simulation time / s')
        axes[2,column].set_title(f'Initial horizontal speed: {speed[0]:.3f} m/s (randomized)',loc='left',fontsize=10,color='#667e91')
        for ax in axes[:,column]:
            ax.set_xlim(t[0],t[-1]);ax.grid(color='#e6edf2',lw=.7)
            ax.spines[['top','right']].set_visible(False)
            ax.spines[['left','bottom']].set_color('#c9d4dc')
            ax.tick_params(colors='#667e91',labelsize=8)
        summary.append(dict(seed=seed,peak_tilt_deg=float(tilt.max()),camera_elevation_min_deg=float(elevation.min()),
            camera_elevation_max_deg=float(elevation.max()),initial_horizontal_speed_m_s=float(speed[0]),
            peak_horizontal_speed_m_s=float(speed.max())))
    fig.text(.08,.035,'A gentle figure eight is not a start-from-rest acceleration test. Yaw rotation is much larger than the small pitch/roll changes.',fontsize=10,color='#667e91')
    output = args.run/'camera_tilt_diagnostic'
    for extension in ['png','pdf']:fig.savefig(output.with_suffix('.'+extension),dpi=180,facecolor='white')
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
