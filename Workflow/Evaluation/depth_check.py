#!/usr/bin/env python3
"""Compare native 3DGUT expected depth with recorded aligned RealSense depth.

Run from the repository root with the installed 3dgrut Python environment.
No training or source asset modification is performed. References are matched
by original bag timestamp, not by the renamed dataset image prefix.
"""
import argparse
import csv
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_images(path):
    images = []
    with path.open('rb') as f:
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            data = struct.unpack('<i7di', f.read(64))
            name = bytearray()
            while (char := f.read(1)) != b'\0':
                if not char:
                    raise EOFError(path)
                name.extend(char)
            n = struct.unpack('<Q', f.read(8))[0]
            observations = np.frombuffer(f.read(n * 24), dtype=[
                ('x', '<f8'), ('y', '<f8'), ('id', '<i8')]).copy()
            images.append(dict(name=name.decode(), q=np.array(data[1:5]),
                               t=np.array(data[5:8]), points=observations))
        if f.read():
            raise ValueError('Unexpected trailing camera data')
    return sorted(images, key=lambda im: im['name'])


def rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def read_points(path):
    points = {}
    with path.open('rb') as f:
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            values = struct.unpack('<Q3d3BdQ', f.read(51))
            points[values[0]] = values[1:4]
            f.seek(values[-1] * 8, 1)
        if f.read():
            raise ValueError('Unexpected trailing point data')
    return points


def timestamp(name):
    return Path(name).stem.rsplit('_', 1)[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def fit_scale(images, points, references, depth_dir, max_dt_ms):
    """Equal weight per training frame; test views never contribute to scale."""
    estimates = []
    for index, image in enumerate(images):
        frame = references.get(timestamp(image['name']))
        if index % 8 == 0 or not frame or not frame['depth_filename']:
            continue
        if abs(float(frame['depth_dt_ms'])) > max_dt_ms:
            continue
        depth = np.load(depth_dir / frame['depth_filename'])
        obs = image['points']
        obs = obs[np.array([int(i) in points for i in obs['id']])]
        if not len(obs):
            continue
        xyz = np.array([points[int(i)] for i in obs['id']])
        z = (xyz @ rotation(image['q']).T + image['t'])[:, 2]
        # COLMAP coordinates use corner-origin +0.5 pixel centers.
        u = np.rint(obs['x'] - 0.5).astype(int)
        v = np.rint(obs['y'] - 0.5).astype(int)
        good = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0]) & (z > 0)
        z = z[good]
        measured = depth[v[good], u[good]]
        good = np.isfinite(measured) & (measured > 0)
        if good.sum() >= 20:
            estimates.append(dict(image=image['name'], samples=int(good.sum()),
                                  metres_per_unit=float(np.median(measured[good] / z[good]))))
    if not estimates:
        raise ValueError('No usable training observations for scale fitting')
    return float(np.median([r['metres_per_unit'] for r in estimates])), estimates


def expected_z(accumulated_range, opacity, ray_z, scale):
    """Native 3DGUT accumulates weighted ray hit distance, without normalization."""
    result = accumulated_range / np.maximum(opacity, 1e-8) * ray_z * scale
    return np.where((opacity > 1e-8) & np.isfinite(result), result, np.nan)


def summarize(error):
    if not len(error):
        return dict(pixels=0)
    absolute = np.abs(error)
    return dict(pixels=int(len(error)), mae_m=float(absolute.mean()),
                rmse_m=float(np.sqrt(np.mean(error**2))),
                median_abs_m=float(np.median(absolute)),
                p90_abs_m=float(np.percentile(absolute, 90)),
                signed_mean_m=float(error.mean()),
                within_10cm=float(np.mean(absolute <= .1)),
                within_25cm=float(np.mean(absolute <= .25)))


def colorize(values, maximum):
    # Fixed shared depth scale; gray marks invalid pixels.
    x = np.clip(values / maximum, 0, 1)
    valid = np.isfinite(x)
    x = np.nan_to_num(x)
    rgb = np.stack([np.clip(1.5-abs(4*x-3), 0, 1),
                    np.clip(1.5-abs(4*x-2), 0, 1),
                    np.clip(1.5-abs(4*x-1), 0, 1)], -1)
    rgb[~valid] = .25
    return Image.fromarray((rgb * 255).astype('uint8'))


def panel(path, rgb, measured, predicted, opacity, mask, depth_max, rendered_rgb):
    error = np.where(mask, np.abs(predicted-measured), np.nan)
    images = [Image.fromarray(rgb), colorize(measured, depth_max),
              colorize(predicted, depth_max), colorize(error, 1),
              Image.fromarray((np.clip(opacity, 0, 1)*255).astype('uint8')).convert('RGB'),
              Image.fromarray(rendered_rgb)]
    labels = ['Recorded RGB', f'Recorded depth 0-{depth_max:g}m',
              f'3DGUT expected Z 0-{depth_max:g}m',
              'Absolute error 0-1m (gray excluded)', 'Rendered opacity 0-1',
              'Native rendered RGB (camera sanity check)']
    canvas = Image.new('RGB', (640*3, 390*2), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        x, y = i % 3 * 640, i // 3 * 390
        im.thumbnail((632, 356))
        canvas.paste(im, (x, y+24))
        draw.text((x+4, y+5), label, fill='black')
    canvas.save(path, quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path('data/scene_20261025_200034'))
    parser.add_argument('--shared', type=Path, default=Path('Workflow/Reconstruction/datasets/oct25_shared_native'))
    parser.add_argument('--native', type=Path, default=Path('Workflow/Reconstruction/datasets/flight_20261025_native'))
    parser.add_argument('--tool-dir', type=Path, default=Path.home()/'tools/3dgrut')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, default=0, help='0 means all eligible test views')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--min-opacity', type=float, default=.95)
    parser.add_argument('--max-depth', type=float, default=8.)
    parser.add_argument('--max-dt-ms', type=float, default=40.)
    args = parser.parse_args()
    if not 0 < args.min_opacity <= 1 or args.max_depth <= 0 or args.max_dt_ms < 0 or args.limit < 0:
        parser.error('Invalid opacity, depth, timing, or limit')
    base = args.scene.resolve()/'training_reference/Workflow/Reconstruction'
    dataset = base/'datasets/flight_20261025_native'
    run = base/'runs/flight_20261025_native_20261025T215743_160060Z'
    checkpoint = next(run.rglob('ckpt_last.pt'))
    images = read_images(dataset/'colmap/sparse/0/images.bin')
    with (args.shared/'frames.csv').open() as f:
        references = {r['bag_timestamp_ns']: r for r in csv.DictReader(f)}
    scale, estimates = fit_scale(images, read_points(dataset/'colmap/sparse/0/points3D.bin'),
                                 references, args.shared/'depth', args.max_dt_ms)
    selected = [(i, im, references[timestamp(im['name'])]) for i, im in enumerate(images)
                if i % 8 == 0 and timestamp(im['name']) in references
                and references[timestamp(im['name'])]['depth_filename']
                and abs(float(references[timestamp(im['name'])]['depth_dt_ms'])) <= args.max_dt_ms]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError('No eligible evaluation views')
    output = args.output or args.scene/'evaluation'/('depth_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    info = dict(status='prepared', scale_m_per_unit=scale, scale_training_frames=len(estimates),
                scale_estimates=estimates, test_views=len(selected),
                checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                selection=[dict(image=im['name'], depth=fr['depth_filename'],
                                depth_dt_ms=float(fr['depth_dt_ms'])) for _, im, fr in selected],
                options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                limitations=['Depth scale assumes 0.001 m/native unit.',
                             'RGB/depth may differ by up to the configured timing tolerance; no motion compensation.',
                             'Scale fitted on training sparse points from the same capture; not independent metric calibration.',
                             'COLMAP used test images for pose estimation.',
                             'Expected Gaussian depth is opacity-weighted and is not a collision surface.',
                             'Invalid sensor pixels are unknown, not free space.'])
    report = output/'summary.json'
    report.write_text(json.dumps(info, indent=2)+'\n')
    print(f'OUTPUT={output.resolve()}\nScale={scale:.6f} m/unit; {len(estimates)} training frames; {len(selected)} test views', flush=True)
    if args.prepare_only:
        return

    sys.path.insert(0, str(args.tool_dir.resolve()))
    import torch
    from torch.utils.data import default_collate
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    from threedgrut.model.model import MixtureOfGaussians
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; prepared reference report is saved')
    info['runtime'] = dict(gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                           cuda_build=str(torch.version.cuda))
    ckpt = torch.load(checkpoint, map_location='cuda', weights_only=False)
    conf = ckpt['config']
    if conf.render.method != '3dgut' or conf.dataset.normalize_world_space:
        raise ValueError('This evaluator requires native 3DGUT without world normalization')
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    del ckpt
    model.build_acc()
    # Raw camera model and raw RGB share the aligned measured depth grid.
    stage = output/'native_camera_dataset'
    stage.mkdir()
    (stage/'sparse').symlink_to((dataset/'colmap/sparse').resolve(), target_is_directory=True)
    (stage/'images').symlink_to((args.native/'images').resolve(), target_is_directory=True)
    ds = ColmapDataset(str(stage.resolve()), test_split_interval=0, normalize_world_space=False)
    assert [e.name for e in ds.cam_extrinsics] == [im['name'] for im in images]
    rows, errors, relative, reference_pixels, evaluated_pixels = [], [], [], 0, 0
    for number, (index, im, frame) in enumerate(selected):
        batch = ds.get_gpu_batch_with_intrinsics(default_collate([ds[index]]))
        with torch.inference_mode():
            result = model(batch)
        rendered_rgb = result['pred_features'][0].detach().cpu().numpy().clip(0, 1)
        opacity = result['pred_opacity'][0].detach().cpu().numpy().squeeze(-1)
        dist = result['pred_dist'][0].detach().cpu().numpy().reshape(opacity.shape)
        ray_z = batch.rays_dir[0, ..., 2].detach().cpu().numpy()
        pred = expected_z(dist, opacity, ray_z, scale)
        measured = np.load(args.shared/'depth'/frame['depth_filename'])
        assert pred.shape == measured.shape, (pred.shape, measured.shape)
        valid_ref = np.isfinite(measured) & (measured > 0) & (measured <= args.max_depth)
        valid_pred = np.isfinite(pred) & (pred > 0) & (opacity >= args.min_opacity)
        mask = valid_ref & valid_pred
        error = (pred-measured)[mask]
        reference_pixels += int(valid_ref.sum())
        evaluated_pixels += int(mask.sum())
        errors.append(error)
        relative.append(np.abs(error)/measured[mask])
        row = dict(image=im['name'], depth_dt_ms=float(frame['depth_dt_ms']),
                   valid_reference_pixels=int(valid_ref.sum()),
                   coverage=float(mask.sum()/max(1, valid_ref.sum())), **summarize(error))
        rgb = np.asarray(Image.open(args.native/'images'/im['name']).convert('RGB'))
        row['native_rgb_psnr_db'] = float(-10*np.log10(max(1e-12, np.mean(
            (rendered_rgb.astype('float64')-rgb.astype('float64')/255)**2))))
        rows.append(row)
        np.savez_compressed(output/f'{index:03d}_depth.npz', measured_z_m=measured,
                            predicted_z_m=pred.astype('float32'), opacity=opacity,
                            evaluation_mask=mask)
        panel(output/f'{index:03d}_comparison.jpg', rgb, measured,
              np.where(valid_pred, pred, np.nan), opacity, mask, args.max_depth,
              (rendered_rgb*255).round().astype('uint8'))
        print(f'{number+1}/{len(selected)} {row}', flush=True)
        del result, batch
    joined = np.concatenate(errors)
    info.update(status='complete', per_frame=rows, aggregate=summarize(joined),
                valid_reference_pixels=reference_pixels,
                evaluated_pixel_coverage=evaluated_pixels/max(1, reference_pixels),
                abs_relative_mean=float(np.concatenate(relative).mean()) if len(joined) else None,
                depth_definition='sum(weight*ray_hit_distance)/sum(weight) * normalized_camera_ray_z * metric_scale')
    report.write_text(json.dumps(info, indent=2)+'\n')
    with (output/'per_frame.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(info['aggregate'], indent=2), flush=True)


if __name__ == '__main__':
    main()
