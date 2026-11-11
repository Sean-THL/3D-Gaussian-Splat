#!/usr/bin/env python3
"""Audit and visualize saved RGB/expert demonstrations; never reruns physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dasc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Workflow.Simulation.audit_dataset import audit_episode

INK, MUTED = '#19334c', '#667e91'
COLORS = ['#2979bf', '#119c87', '#db8c35']


def clean(ax, title, ylabel):
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold', color=INK, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#c9d4dc')
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color='#e7edf2', linewidth=.65)
    ax.set_axisbelow(True)


def visualize(folder, fps, plots_only):
    audit = audit_episode(folder)
    rows = [json.loads(line) for line in (folder/'rgb_actions.jsonl').read_text().splitlines()]
    parameters = json.loads((folder/'episode_parameters.json').read_text())
    metadata = json.loads((folder/'dataset_metadata.json').read_text())
    t = np.array([r['time_s'] for r in rows])
    p = np.array([r['position_m'] for r in rows])
    refs = np.array([r['reference_horizon'][0][:3] for r in rows])
    rotation = Rotation.from_quat([r['quaternion_xyzw'] for r in rows])
    angles = rotation.as_euler('xyz')
    angles[:, 2] = np.unwrap(angles[:, 2])
    angles = np.rad2deg(angles)
    rates = np.array([r['action_body_rates_rad_s'] for r in rows])
    thrust = np.array([r['action_thrust_n'] for r in rows])
    error = np.linalg.norm(p-refs, axis=1)
    heading = rotation.apply(np.tile([1., 0., 0.], (len(rows), 1)))
    destination = folder/'visualization'
    destination.mkdir(exist_ok=True)
    (destination/'dataset_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'pdf.fonttype': 42, 'font.size': 10})
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    fig.text(.05, .958, f"Recorded Isaac rollout  |  seed {parameters['seed']}", fontsize=24, weight='bold')
    fig.text(.05, .918, f"{len(rows)} synchronized pairs  •  {audit['physics_backend']}  •  structural audit passed",
             fontsize=11, color=MUTED)
    clock = fig.text(.95, .875, '', ha='right', fontsize=11, weight='bold')
    camera = fig.add_axes([.05, .53, .49, .32])
    camera.set_title('Recorded 3DGS RGB at the expert decision pose', loc='left', fontsize=12, weight='bold')
    camera.axis('off')
    def read_image(index):
        with Image.open(folder/rows[index]['image']) as decoded:
            return np.asarray(decoded.convert('RGB')).copy()
    display = camera.imshow(read_image(0))
    map_ax = fig.add_axes([.64, .53, .29, .32])
    clean(map_ax, 'Reference and achieved trajectory', 'Vicon Y / m')
    map_ax.set_xlabel('Vicon X / m', color=MUTED, fontsize=9)
    map_ax.plot(refs[:, 0], refs[:, 1], '--', color=INK, lw=1.5, label='Reference')
    map_ax.plot(p[:, 0], p[:, 1], color=COLORS[0], alpha=.2, lw=1.5, label='Complete achieved path')
    trace, = map_ax.plot([], [], color=COLORS[0], lw=2)
    marker, = map_ax.plot([], [], 'o', color=COLORS[0], ms=5)
    orientation, = map_ax.plot([], [], color=COLORS[1], lw=2.5)
    map_ax.set_aspect('equal', adjustable='datalim')
    map_ax.margins(.18)
    map_ax.legend(loc='best', fontsize=8, frameon=False)
    labels = [('Actual roll and pitch', 'Angle / deg'), ('Actual yaw (unwrapped)', 'Angle / deg'),
              ('Synchronized position error', 'Error / cm'), ('Expert collective thrust', 'Thrust / N'),
              ('Expert body-rate commands', 'Body rates / rad/s'), ('Achieved body rates', 'Body rates / rad/s')]
    axes = []
    for i, (title, ylabel) in enumerate(labels):
        ax = fig.add_axes([.065+(i%3)*.315, .307-(i//3)*.19, .26, .115])
        clean(ax, title, ylabel)
        ax.set_xlim(t[0], t[-1] if len(t)>1 else t[0]+.05)
        if i>=3:ax.set_xlabel('Simulation time / s', color=MUTED, fontsize=9)
        axes.append(ax)
    for j, name in enumerate(['Roll', 'Pitch']):
        axes[0].plot(t, angles[:, j], color=COLORS[j], lw=1.3, label=name)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    axes[1].plot(t, angles[:, 2], color=COLORS[1], lw=1.3)
    axes[2].plot(t, 100*error, color=COLORS[2], lw=1.3)
    axes[3].plot(t, thrust, color=COLORS[0], lw=1.3)
    for j, name in enumerate(['p / X', 'q / Y', 'r / Z']):
        axes[4].plot(t, rates[:, j], color=COLORS[j], lw=1.2, label=name)
    axes[4].legend(fontsize=7, frameon=False, ncol=3)
    actual = np.array([r['observed_body_rates_rad_s'] for r in rows])
    for j in range(3):axes[5].plot(t, actual[:, j], color=COLORS[j], lw=1.2)
    cursors = [ax.axvline(t[0], color=INK, lw=.9, alpha=.75) for ax in axes]
    task = parameters['task']
    plant = parameters['plant']
    fig.text(.05, .064, f"Randomized episode: mass {plant['mass_kg']:.3f} kg  |  altitude {task['altitude_m']:.2f} m"
             f"  |  heading offset {task['yaw_deg']:+.1f}°  |  route duration {task['duration_s']:.2f} s", fontsize=10, color=MUTED)
    fig.text(.05, .034, f'Saved pre-action images + states + expert labels; no physics rerun.  Video preview: {fps} fps; dataset retains every {1/metadata["control_dt_s"]:g} Hz sample.',
             fontsize=9, color=MUTED)
    def update(index):
        display.set_data(read_image(index))
        trace.set_data(p[:index+1, 0], p[:index+1, 1])
        marker.set_data([p[index, 0]], [p[index, 1]])
        end = p[index, :2]+.055*heading[index, :2]
        orientation.set_data([p[index, 0], end[0]], [p[index, 1], end[1]])
        for cursor in cursors:cursor.set_xdata([t[index], t[index]])
        clock.set_text(f't = {t[index]:.2f} s  |  z = {p[index, 2]:.2f} m  |  error = {100*error[index]:.1f} cm')
    middle = len(rows)//3
    update(middle)
    for extension in ['png', 'pdf']:
        fig.savefig(destination/f'overview.{extension}', dpi=160, facecolor='white')
    # A contact sheet makes selected RGB views easy to inspect independently.
    sheet, tile_axes = plt.subplots(2, 4, figsize=(14, 4.8), facecolor='white')
    for ax, index in zip(tile_axes.flat, np.linspace(0, len(rows)-1, 8).astype(int)):
        ax.imshow(read_image(index));ax.axis('off')
        ax.set_title(f't = {t[index]:.2f} s  |  yaw {angles[index, 2]:+.0f}°', fontsize=9)
    sheet.suptitle(f"Recorded RGB • seed {parameters['seed']}", fontsize=15, weight='bold')
    sheet.tight_layout()
    sheet.savefig(destination/'rgb_contact_sheet.png', dpi=150, facecolor='white')
    plt.close(sheet)
    movie = destination/'rollout.mp4'
    if not plots_only:
        writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=3500,
            extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        duration = t[-1]-t[0]+metadata['control_dt_s']
        # Use the most recent recorded observation, without interpolating images/states.
        video_times = t[0]+np.arange(int(np.ceil(duration*fps)))/fps
        indices = np.clip(np.searchsorted(t, video_times, side='right')-1, 0, len(t)-1)
        with writer.saving(fig, str(movie), dpi=80):
            for index in indices:
                update(index)
                writer.grab_frame(facecolor='white')
    plt.close(fig)
    summary = dict(seed=parameters['seed'], samples=len(rows), audit=audit,
        source=str(folder.resolve()), source_jsonl_sha256=hashlib.sha256((folder/'rgb_actions.jsonl').read_bytes()).hexdigest(),
        synchronized_position_error_rms_m=float(np.sqrt(np.mean(error**2))),
        synchronized_position_error_max_m=float(error.max()),
        note='Errors computed at pre-action decision timestamps; report.json uses post-step telemetry.',
        visualization_fps=fps, source_control_dt_s=metadata['control_dt_s'])
    (destination/'visualization_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(seed=summary['seed'], samples=len(rows), output=str(destination), video=not plots_only)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Episode or completed native collection root')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if not 1<=args.fps<=60:parser.error('FPS must be between 1 and 60')
    folders = [args.folder] if (args.folder/'rgb_actions.jsonl').exists() else sorted(args.folder.glob('episode_*'))
    if not folders:parser.error('No episode folders found')
    for folder in folders:visualize(folder, args.fps, args.plots_only)


if __name__ == '__main__':main()
