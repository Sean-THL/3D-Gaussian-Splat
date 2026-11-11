#!/usr/bin/env python3
"""Audit native pilot data and compare clean rerenders with saved augmented RGB.

Coverage indicators are diagnostics, not geometry/real-world validation.
"""
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
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Workflow.Simulation.audit_dataset import audit_episode


def load_image(path):
    with Image.open(path) as image:return np.asarray(image.convert('RGB')).copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version',type=Path)
    parser.add_argument('--run-name',required=True)
    args=parser.parse_args()
    run=args.version/'pilot_test/runs'/args.run_name
    review=args.version/'pilot_test/reviews'/args.run_name
    clean=review/'clean_render'
    output=review/'analysis';output.mkdir(exist_ok=True)
    report=json.loads((run/'report.json').read_text())
    clean_index=json.loads((clean/'clean_review.json').read_text())['frames']
    coverage=json.loads((ROOT/'data/scene_20261025_200034/evaluation/capture_pose_coverage.json').read_text())
    lower=np.array(coverage['candidate_body_min_m']);upper=np.array(coverage['candidate_body_max_m'])
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#19334c','font.size':10,'pdf.fonttype':42})
    summaries=[]
    for episode,item in zip(sorted(run.glob('episode_*')),report['episodes']):
        audit=audit_episode(episode)
        (episode/'dataset_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        rows=[json.loads(line) for line in (episode/'rgb_actions.jsonl').read_text().splitlines()]
        parameters=json.loads((episode/'episode_parameters.json').read_text())
        metadata=json.loads((episode/'dataset_metadata.json').read_text())
        p=np.array([row['position_m'] for row in rows]);velocity=np.array([row['velocity_m_s'] for row in rows])
        rotations=Rotation.from_quat([row['quaternion_xyzw'] for row in rows])
        angles=rotations.as_euler('xyz',degrees=True)
        tilt=np.rad2deg(np.arccos(np.clip(rotations.apply(np.tile([0.,0.,1.],(len(rows),1)))[:,2],-1,1)))
        outside=np.any((p[:,:2]<lower[:2])|(p[:,:2]>upper[:2]),axis=1)
        rates=np.abs(np.array([row['action_body_rates_rad_s'] for row in rows]))
        summary=dict(episode=episode.name,seed=parameters['seed'],samples=audit['samples'],audit=audit['status'],
            width_m=parameters['task']['width_m'],reference_altitude_m=parameters['task']['altitude_m'],
            route_duration_s=parameters['task']['duration_s'],position_rms_m=item['position_rms_m'],position_max_m=item['position_max_m'],
            failed_solves=item['failed_solves'],torque_limit_steps=item['torque_limit_steps'],
            initial_roll_deg=angles[0,0],initial_pitch_deg=angles[0,1],maximum_tilt_deg=float(tilt.max()),
            maximum_speed_m_s=float(np.linalg.norm(velocity,axis=1).max()),maximum_commanded_rate_rad_s=float(rates.max()),
            outside_capture_xy_bbox_fraction=float(outside.mean()),
            jpeg_bytes=sum((episode/row['image']).stat().st_size for row in rows),
            augmentation=metadata['visual_randomization'])
        # Convert NumPy scalar types before JSON output.
        summary={key:float(value) if isinstance(value,np.floating) else value for key,value in summary.items()}
        summaries.append(summary)
        frames=[frame for frame in clean_index if frame['episode']==episode.name]
        selected=[frames[i] for i in np.linspace(0,len(frames)-1,4).astype(int)]
        fig,axes=plt.subplots(2,4,figsize=(15,5.8),facecolor='white')
        for column,frame in enumerate(selected):
            row=rows[frame['index']]
            axes[0,column].imshow(load_image(clean/frame['clean_image']))
            axes[1,column].imshow(load_image(episode/frame['original_image']))
            axes[0,column].set_title(f"t={frame['time_s']:.2f} s  |  XY=({row['position_m'][0]:+.2f}, {row['position_m'][1]:+.2f}) m",fontsize=9)
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f"Seed {summary['seed']} — clean synthetic RGB (top) / stored augmented JPEG (bottom)",fontsize=14,weight='bold')
        fig.text(.03,.025,'Same saved physical pose in each column. Neither row is real recorded RGB; differences isolate appearance augmentation and JPEG encoding.',fontsize=9,color='#667e91')
        fig.tight_layout(rect=[0,.05,1,.92])
        fig.savefig(output/f'{episode.name}_clean_vs_augmented.png',dpi=160,facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(3,3,figsize=(14,9),facecolor='white')
        for ax in axes.flat:ax.axis('off')
        for ax,frame in zip(axes.flat,frames):
            row=rows[frame['index']]
            ax.imshow(load_image(clean/frame['clean_image']))
            ax.set_title(f"t={frame['time_s']:.2f} s  |  X={row['position_m'][0]:+.2f}, Y={row['position_m'][1]:+.2f}, Z={row['position_m'][2]:.2f} m",fontsize=9)
        fig.suptitle(f"Seed {summary['seed']} — unaugmented synthetic views at actual pilot poses",fontsize=15,weight='bold')
        fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.91,hspace=.25,wspace=.03)
        fig.savefig(output/f'{episode.name}_clean_views.png',dpi=150,facecolor='white');plt.close(fig)
    original_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file() and 'visualization' not in p.relative_to(run).parts)
    result=dict(native_status=report['status'],total_samples=sum(item['samples'] for item in summaries),
        original_run_bytes=original_bytes,episodes=summaries,
        scope='Tracking/data integrity checked. Clean/augmented images are synthetic; no matched real-world RGB for these new poses.',
        coverage_note='Outside-capture bounding-box fraction is diagnostic only; it is not a verified bad-view or collision fraction.')
    (output/'pilot_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=['episode','seed','samples','audit','width_m','reference_altitude_m','route_duration_s','position_rms_m','position_max_m',
        'maximum_tilt_deg','maximum_speed_m_s','maximum_commanded_rate_rad_s','outside_capture_xy_bbox_fraction','jpeg_bytes']
    with (output/'per_episode.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
