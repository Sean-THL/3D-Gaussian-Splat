#!/usr/bin/env python3
"""Rerender selected saved poses without RGB augmentation for visual review.

No physics is rerun and original observations/labels are unchanged. These are
clean synthetic views, not real reference images or a geometry validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Workflow.Simulation.dataset import PoseRenderClient
from Workflow.Simulation.figs.dynamics import State


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--render-url',default='http://127.0.0.1:8766')
    parser.add_argument('--frames',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.frames<2:parser.error('Use at least two review frames')
    report=json.loads((args.run/'report.json').read_text())
    if report.get('status')!='passed':parser.error('Review only a completed passing run')
    client=PoseRenderClient(args.render_url)
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for episode in sorted(args.run.glob('episode_*')):
        metadata=json.loads((episode/'dataset_metadata.json').read_text())
        for key in ('checkpoint_sha256','alignment_sha256','camera_to_body','intrinsics'):
            if metadata['renderer'][key]!=client.metadata[key]:
                raise ValueError('Renderer identity differs from the collected data: '+key)
        rows=[json.loads(line) for line in (episode/'rgb_actions.jsonl').read_text().splitlines()]
        rotations=Rotation.from_quat([row['quaternion_xyzw'] for row in rows])
        tilt=np.arccos(np.clip(rotations.apply(np.tile([0.,0.,1.],(len(rows),1)))[:,2],-1,1))
        indices=sorted(set(np.linspace(0,len(rows)-1,args.frames).astype(int).tolist()+[int(np.argmax(tilt))]))
        destination=args.output/episode.name;destination.mkdir()
        for index in indices:
            row=rows[index]
            state=State(np.array(row['position_m']),np.array(row['velocity_m_s']),rotations[index])
            rgb,_=client.render(state)
            path=destination/f'{index:06d}_clean.png'
            Image.fromarray(rgb).save(path)
            results.append(dict(episode=episode.name,index=index,time_s=row['time_s'],
                original_image=row['image'],clean_image=str(path.relative_to(args.output)),
                position_m=row['position_m'],quaternion_xyzw=row['quaternion_xyzw'],
                clean_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output/'clean_review.json').write_text(json.dumps(dict(source_run=str(args.run),
        renderer=client.metadata,description=__doc__,frames=results),indent=2)+'\n')
    print('Clean matched-pose review frames:',len(results),flush=True)


if __name__=='__main__':main()
