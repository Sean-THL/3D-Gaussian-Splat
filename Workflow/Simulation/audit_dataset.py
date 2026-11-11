#!/usr/bin/env python3
"""Structural audit for synchronized thrust/body-rate RGB demonstrations.

Passing this audit establishes data consistency, not collision safety or
sim-to-real validity. Run on an episode directory containing rgb_actions.jsonl.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def audit_episode(folder):
    folder = Path(folder).resolve()
    if (folder/'failure.json').exists():
        raise ValueError('Episode has a recorded execution failure')
    native_report = folder.parent/'report.json'
    if native_report.exists() and json.loads(native_report.read_text()).get('status') != 'passed':
        raise ValueError('Native run did not pass its validation gates')
    metadata = json.loads((folder/'dataset_metadata.json').read_text())
    limits = metadata['action_limits']
    count = 0
    previous = None
    images = set()
    for line in (folder/'rgb_actions.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['action_type'] != 'thrust_body_rates' or row['solver']['status'] != 'solved':
            raise ValueError('Invalid expert action type or failed solve included')
        time = row['time_s']
        if time != row['image_time_s'] or (previous is not None and not np.isclose(time-previous, metadata['control_dt_s'], atol=1e-8)):
            raise ValueError('Image/action timestamp mismatch or missing decisions')
        previous = time
        vector = np.r_[row['position_m'], row['velocity_m_s'], row['quaternion_xyzw'], row['action_thrust_n'], row['action_body_rates_rad_s'], time]
        if not np.isfinite(vector).all() or abs(np.linalg.norm(row['quaternion_xyzw'])-1) > 1e-5:
            raise ValueError('Nonfinite data or invalid quaternion')
        references = np.asarray(row['reference_horizon'], dtype=float)
        if references.ndim != 2 or references.shape[1] != 11 or len(references) < 3 or not np.isfinite(references).all():
            raise ValueError('Invalid reference horizon')
        if not np.isclose(row['horizon_dt_s'], metadata['control_dt_s']):
            raise ValueError('Reference sampling interval differs from control interval')
        if not -1e-7 <= row['action_thrust_n'] <= limits['max_thrust_n']+1e-7:
            raise ValueError('Expert thrust exceeds nominal command bound')
        if max(abs(v) for v in row['action_body_rates_rad_s']) > limits['max_body_rate_rad_s']+1e-7:
            raise ValueError('Expert body rate exceeds bound')
        image = (folder/row['image']).resolve()
        if not image.is_relative_to(folder) or image in images:
            raise ValueError('Image outside episode or reused for multiple decisions')
        if hashlib.sha256(image.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('Image checksum mismatch')
        with Image.open(image) as decoded:
            if list(decoded.size) != metadata['image_resolution_wh'] or decoded.mode != 'RGB':
                raise ValueError('Unexpected RGB image resolution or format')
            decoded.verify()
        images.add(image)
        count += 1
    expected = json.loads((folder/'dataset_count.json').read_text())['samples']
    if count == 0 or count != expected:
        raise ValueError('Empty or incomplete dataset')
    return dict(status='passed', samples=count, physics_backend=metadata['physics_backend'],
                checks='timestamps, finite state/actions, quaternion norms, command bounds, unique images, checksums, RGB dimensions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    args = parser.parse_args()
    result = audit_episode(args.episode)
    (args.episode/'dataset_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
