#!/usr/bin/env python3
"""Estimate scene Sim(3), fixed optical-camera mount, and RGB/Vicon time offset.

Uses only recorded trajectories, with blocked temporal validation. Estimated
mount parameters are not a substitute for an independently measured mounting
transform. Original Gaussian and camera assets are never modified.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def fit_similarity(source, target):
    source, target = np.asarray(source), np.asarray(target)
    x, y = source-source.mean(0), target-target.mean(0)
    u, d, vh = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vh)
    rotation = u @ np.diag(sign) @ vh
    variance = np.mean(np.sum(x*x, axis=1))
    if variance <= 1e-12 or np.linalg.svd(x, compute_uv=False)[1] <= 1e-6:
        raise ValueError('Trajectory cannot constrain a similarity transform')
    scale = np.sum(d*sign)/variance
    translation = target.mean(0)-scale*rotation @ source.mean(0)
    return scale, Rotation.from_matrix(rotation), translation


def load_inputs(poses_path, timestamps_path, vicon_path):
    poses = sorted(json.loads(poses_path.read_text())['poses'], key=lambda p: p['name'])
    with timestamps_path.open() as f:
        rows = {r['filename']: r for r in csv.DictReader(f)}
    if any(not rows[p['name']]['header_timestamp_ns'] for p in poses):
        raise ValueError('Image sensor-header timestamps required')
    image_ns = np.array([int(rows[p['name']]['header_timestamp_ns']) for p in poses], dtype=np.int64)
    origin_ns = int(image_ns.min())
    time = (image_ns-origin_ns).astype(float)*1e-9
    world_from_camera = Rotation.from_quat([
        np.array(p['qvec_wxyz'])[[1, 2, 3, 0]] for p in poses]).inv()
    centers = -world_from_camera.apply([p['tvec'] for p in poses])
    vicon = [json.loads(line) for line in vicon_path.open()]
    header_ns = np.array([int(r['header_timestamp_ns']) for r in vicon], dtype=np.int64)
    reversals = int((np.diff(header_ns) < 0).sum())
    if np.any(np.diff(header_ns) < -1_000_000):
        raise ValueError('Vicon header timestamps reverse by over 1 ms; inspect clocks')
    # Latency-corrected headers occasionally arrive out of timestamp order.
    order = np.argsort(header_ns, kind='stable')
    header_ns = header_ns[order]
    vicon = [vicon[i] for i in order]
    keep = np.r_[True, np.diff(header_ns) > 0]
    records = [r for r, ok in zip(vicon, keep) if ok]
    vtime = (header_ns[keep]-origin_ns).astype(float)*1e-9
    transforms = [r['message']['transform'] for r in records]
    positions = np.array([[r['translation'][k] for k in ('x', 'y', 'z')] for r in transforms])
    rotations = Rotation.from_quat([[r['rotation'][k] for k in ('x', 'y', 'z', 'w')] for r in transforms])
    frames = {(r['message']['header']['frame_id'], r['message']['child_frame_id']) for r in records}
    if len(frames) != 1:
        raise ValueError('Expected one fixed Vicon parent/child frame pair')
    return dict(time=time, centers=centers, camera_rotation=world_from_camera,
                vtime=vtime, body_position=positions, body_rotation=rotations,
                names=[p['name'] for p in poses], origin_ns=origin_ns,
                vicon_frames=list(next(iter(frames))),
                vicon_duplicates_dropped=int((~keep).sum()),
                vicon_small_timestamp_reversals_sorted=reversals)


class Alignment:
    def __init__(self, data):
        self.data = data
        self.slerp = Slerp(data['vtime'], data['body_rotation'])

    def body_at(self, times):
        if np.any(times < self.data['vtime'][0]) or np.any(times > self.data['vtime'][-1]):
            raise ValueError('No extrapolation of Vicon poses is permitted')
        p = np.column_stack([np.interp(times, self.data['vtime'], self.data['body_position'][:, i])
                             for i in range(3)])
        return p, self.slerp(times)

    def errors(self, x, mask):
        d = self.data
        body_p, body_r = self.body_at(d['time'][mask]+x[13])
        rw, mount_r = Rotation.from_rotvec(x[1:4]), Rotation.from_rotvec(x[7:10])
        reconstructed_p = np.exp(x[0])*rw.apply(d['centers'][mask])+x[4:7]
        reference_p = body_p+body_r.apply(x[10:13])
        position_error = reconstructed_p-reference_p
        orientation_error = ((body_r*mount_r).inv()*(rw*d['camera_rotation'][mask])).as_rotvec()
        return position_error, orientation_error

    def initialize(self, mask):
        d = self.data
        body_p, body_r = self.body_at(d['time'][mask])
        scale, rw, translation = fit_similarity(d['centers'][mask], body_p)
        mount_r = (body_r.inv()*rw*d['camera_rotation'][mask]).mean()
        return np.r_[np.log(scale), rw.as_rotvec(), translation, mount_r.as_rotvec(), np.zeros(3), 0.]

    def fit(self, mask, initial=None, max_offset=.75):
        if mask.sum() < 20:
            raise ValueError('At least 20 poses required for a fit')
        d = self.data
        lo, hi = np.full(14, -np.inf), np.full(14, np.inf)
        lo[13] = max(-max_offset, d['vtime'][0]-d['time'].min())
        hi[13] = min(max_offset, d['vtime'][-1]-d['time'].max())
        def residual(x):
            ep, er = self.errors(x, mask)
            return np.r_[(ep/.02).ravel(), (er/np.deg2rad(1.)).ravel()]
        fit = least_squares(residual, self.initialize(mask) if initial is None else initial,
                            bounds=(lo, hi), loss='soft_l1', x_scale='jac', max_nfev=300)
        if not fit.success:
            raise RuntimeError(f'Alignment did not converge: {fit.message}')
        return fit


def statistics(position, orientation):
    p = np.linalg.norm(position, axis=1)
    a = np.rad2deg(np.linalg.norm(orientation, axis=1))
    return dict(poses=len(p), position_median_m=float(np.median(p)),
                position_p90_m=float(np.percentile(p, 90)), position_max_m=float(p.max()),
                orientation_median_deg=float(np.median(a)), orientation_p90_deg=float(np.percentile(a, 90)),
                orientation_max_deg=float(a.max()))


def transform_dict(x):
    scale, rw = float(np.exp(x[0])), Rotation.from_rotvec(x[1:4]).as_matrix()
    t = x[4:7]
    forward = np.eye(4)
    forward[:3, :3], forward[:3, 3] = scale*rw, t
    mount = np.eye(4)
    mount[:3, :3], mount[:3, 3] = Rotation.from_rotvec(x[7:10]).as_matrix(), x[10:13]
    return dict(gs_to_vicon=dict(scale_m_per_gs_unit=scale, rotation_matrix=rw.tolist(),
                                translation_m=t.tolist(), position_matrix_4x4=forward.tolist()),
                vicon_to_gs=dict(position_matrix_4x4=np.linalg.inv(forward).tolist()),
                camera_to_tracked_body=dict(rotation_matrix=mount[:3, :3].tolist(),
                                            translation_m=mount[:3, 3].tolist(), rigid_matrix_4x4=mount.tolist(),
                                            source='jointly_estimated_from_trajectories'),
                time_offset_s=float(x[13]), time_equation='t_vicon = t_rgb_sensor_header + time_offset_s')


def vicon_body_to_gs(body_position, body_rotation, calibration):
    """Map marker-body pose to optical-camera pose in original GS coordinates.

    body_rotation is a scipy Rotation mapping marker-body axes into Vicon world.
    Returns camera position and scipy Rotation mapping camera axes into GS world.
    Timestamps must already follow the separately documented time convention.
    """
    world = calibration['gs_to_vicon']
    mount = calibration['camera_to_tracked_body']
    rw = Rotation.from_matrix(world['rotation_matrix'])
    camera_p = np.asarray(body_position)+body_rotation.apply(mount['translation_m'])
    gs_p = rw.inv().apply(camera_p-np.array(world['translation_m']))/world['scale_m_per_gs_unit']
    gs_r = rw.inv()*body_rotation*Rotation.from_matrix(mount['rotation_matrix'])
    return gs_p, gs_r


def plot_result(output, data, problem, x, fold_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = data['time']
    body_p, body_r = problem.body_at(t+x[13])
    reference_p = body_p+body_r.apply(x[10:13])
    predicted_p = np.exp(x[0])*Rotation.from_rotvec(x[1:4]).apply(data['centers'])+x[4:7]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, pair, label in [(axes[0, 0], (0, 1), 'Top view'), (axes[0, 1], (0, 2), 'Side view')]:
        a, b = pair
        ax.plot(reference_p[:, a], reference_p[:, b], label='Vicon camera (estimated mount)', lw=2)
        ax.plot(predicted_p[:, a], predicted_p[:, b], '--', label='Aligned COLMAP camera', lw=1)
        ax.set(xlabel='XYZ'[a]+' (m)', ylabel='XYZ'[b]+' (m)', title=label)
        ax.axis('equal')
        ax.legend(fontsize=8)
    axes[1, 0].scatter([r['time_s'] for r in fold_rows], [r['position_error_m']*100 for r in fold_rows], s=8)
    axes[1, 0].set(xlabel='RGB time from first selected frame (s)', ylabel='Position difference (cm)',
                   title='Held-out 10-second blocks; three fits')
    axes[1, 1].scatter([r['time_s'] for r in fold_rows], [r['orientation_error_deg'] for r in fold_rows], s=8)
    axes[1, 1].set(xlabel='RGB time from first selected frame (s)', ylabel='Orientation difference (degrees)',
                   title='Held-out blocks; mounting transform estimated')
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output/'alignment_overview.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, default=Path('data/scene_20261025_200034'))
    p.add_argument('--timestamps', type=Path, default=Path('Workflow/Reconstruction/datasets/flight_20261025_native/source_export/rgb/timestamps.csv'))
    p.add_argument('--vicon', type=Path, default=Path('Workflow/Data/exports/flight_rgbd_20261025_024629/native/vicon/_2fvicon_2fpx4_5f201_2fpx4_5f201.jsonl'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    poses = args.scene/'evaluation/colmap_camera_poses.json'
    data = load_inputs(poses, args.timestamps, args.vicon)
    if np.diff(data['vtime']).max() > .1:
        raise ValueError('Vicon gaps exceed 100 ms; inspect synchronization before fitting')
    problem = Alignment(data)
    folds = np.floor(data['time']/10).astype(int) % 3
    all_mask = np.ones(len(folds), dtype=bool)
    initial = problem.initialize(all_mask)
    fold_results, heldout_rows = [], []
    for fold in range(3):
        train, validation = folds != fold, folds == fold
        fit = problem.fit(train)
        ep, er = problem.errors(fit.x, validation)
        metrics = statistics(ep, er)
        fold_results.append(dict(fold=fold, training_poses=int(train.sum()), validation=metrics,
                                 **transform_dict(fit.x)))
        for index, pe, re in zip(np.where(validation)[0], ep, er):
            heldout_rows.append(dict(image=data['names'][index], time_s=float(data['time'][index]), fold=fold,
                                     position_error_m=float(np.linalg.norm(pe)),
                                     orientation_error_deg=float(np.rad2deg(np.linalg.norm(re)))))
    fit = problem.fit(all_mask, initial)
    x = fit.x
    ep, er = problem.errors(x, all_mask)
    translation_spread = np.ptp([r['camera_to_tracked_body']['translation_m'] for r in fold_results], axis=0)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    report = dict(status='candidate_alignment_mount_estimated', **transform_dict(x),
                  frames=dict(vicon_world=data['vicon_frames'][0], tracked_body=data['vicon_frames'][1],
                              camera='camera_color_optical_frame', gs_world='original COLMAP / 3DGUT world'),
                  all_pose_fit=statistics(ep, er), blocked_validation=fold_results,
                  heldout_aggregate=dict(position_median_m=float(np.median([r['position_error_m'] for r in heldout_rows])),
                                         position_p90_m=float(np.percentile([r['position_error_m'] for r in heldout_rows], 90)),
                                         orientation_median_deg=float(np.median([r['orientation_error_deg'] for r in heldout_rows]))),
                  mount_translation_spread_across_folds_m=translation_spread.tolist(),
                  optimizer=dict(position_residual_scale_m=.02, orientation_residual_scale_deg=1., loss='soft_l1',
                                 jacobian_singular_values=singular.tolist(), evaluations=fit.nfev),
                  timestamp_origin_ns=data['origin_ns'], max_vicon_gap_s=float(np.diff(data['vtime']).max()),
                  vicon_duplicates_dropped=data['vicon_duplicates_dropped'],
                  vicon_small_timestamp_reversals_sorted=data['vicon_small_timestamp_reversals_sorted'],
                  inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [poses, args.timestamps, args.vicon]},
                  assumptions=['User confirmed a fixed camera mount throughout this recording.',
                               'Only internal RealSense transforms were found in recorded TF; marker-to-camera mount is jointly estimated.',
                               'One constant RGB-to-Vicon header-time offset; linear position interpolation and orientation SLERP.',
                               'No assumption that Vicon tracked axes equal PX4 body axes.'],
                  limitations=['Temporal validation uses the same recording, not a new capture.',
                               'COLMAP used all reconstruction images for its original pose estimation.',
                               'Small trajectory residuals do not establish Gaussian surface accuracy.',
                               'Mount translation, especially its vertical component, is weakly constrained; compare fold spread and measure independently.',
                               'Validate physical dimensions and independent camera poses before hardware use.'])
    output = args.output or args.scene/'calibration'/('alignment_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'alignment.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'heldout_errors.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(heldout_rows[0]))
        w.writeheader()
        w.writerows(sorted(heldout_rows, key=lambda r: r['time_s']))
    plot_result(output, data, problem, x, heldout_rows)
    (args.scene/'calibration/latest_alignment.json').write_text(json.dumps(dict(directory=str(output),
        calibration=str(output/'alignment.json')), indent=2)+'\n')
    print(json.dumps(dict(output=str(output), scale=report['gs_to_vicon']['scale_m_per_gs_unit'],
                          time_offset_s=report['time_offset_s'], heldout=report['heldout_aggregate'],
                          mount_translation_m=report['camera_to_tracked_body']['translation_m'],
                          mount_spread_m=report['mount_translation_spread_across_folds_m']), indent=2))


if __name__ == '__main__':
    main()
