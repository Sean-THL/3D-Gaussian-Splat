#!/usr/bin/env python3
"""Create reusable selections and calibrated crop/resize variants from exports."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(f'{exc}; run with Workflow/Data/.venv/bin/python')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Reconstruction'))
from mono_workflow import new_dataset_name, now, read_export_calibration, save


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe indexed path: {relative}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Indexed file escapes its source: {relative}')
    return path


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def png_size(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image.shape[1], image.shape[0]


def choose_frames(rows, start, end, rate, excluded):
    if any(not math.isfinite(v) for v in [start, rate]) or start < 0 or rate < 0:
        raise ValueError('Start and rate must be finite and nonnegative')
    if end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError('End must be finite and greater than start')
    filenames = [r['filename'] for r in rows]
    if len(set(filenames)) != len(filenames):
        raise ValueError('Duplicate RGB filenames in export')
    if excluded - set(filenames):
        raise ValueError(f'Excluded files not found in export: {sorted(excluded - set(filenames))}')
    result, next_stamp = [], -1
    for row in sorted(rows, key=lambda r: int(r['bag_timestamp_ns'])):
        t, stamp = float(row['bag_elapsed_s']), int(row['bag_timestamp_ns'])
        if t < start or (end is not None and t >= end) or stamp < next_stamp or row['filename'] in excluded:
            continue
        result.append(row)
        next_stamp = stamp + round(1e9 / rate) if rate else stamp
    if len(result) < 4:
        raise ValueError(f'Only {len(result)} selected frames; select at least four')
    return result


def nearest_depth(stamp, stamps, rows, tolerance_ns):
    i = bisect_left(stamps, stamp)
    candidates = [j for j in [i - 1, i] if 0 <= j < len(stamps)]
    if not candidates:
        return None
    j = min(candidates, key=lambda j: abs(stamps[j] - stamp))
    return rows[j] if abs(stamps[j] - stamp) <= tolerance_ns else None


def write_indexes(out, frames, metadata, calibration):
    fields = ['frame_id', 'filename', 'bag_timestamp_ns', 'bag_elapsed_s', 'header_timestamp_ns',
              'split', 'depth_filename', 'depth_timestamp_ns', 'depth_dt_ms']
    with (out / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frames)
    splits = {kind: [f['filename'] for f in frames if f['split'] == kind] for kind in ['train', 'test']}
    splits['policy'] = 'Reference split only; each training/evaluation adapter must explicitly honor these lists.'
    save(out / 'splits.json', splits)
    save(out / 'calibration.json', calibration)
    metadata.update(status='complete', frame_count=len(frames), width=calibration['width'], height=calibration['height'])
    save(out / 'dataset.json', metadata)


def selection(args):
    export = args.export.expanduser().resolve()
    manifest_path = export / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Source export is not complete')
    if args.test_every != 0 and args.test_every < 2:
        raise ValueError('--test-every must be zero (all train) or at least two')
    if not math.isfinite(args.max_depth_dt) or args.max_depth_dt < 0:
        raise ValueError('Invalid depth association tolerance')
    excluded = set()
    if args.exclude_file:
        excluded = {x.strip() for x in args.exclude_file.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')}
    selected = choose_frames(csv_rows(export / 'rgb/timestamps.csv'), args.start, args.end, args.rate, excluded)
    calibration = read_export_calibration(export, args.camera_info_topic)
    for row in selected:
        if (int(row['width']), int(row['height'])) != (calibration['width'], calibration['height']):
            raise ValueError('Recorded calibration and RGB dimensions differ')
    depth_rows, depth_stamps = [], []
    clock = 'header_timestamp_ns' if args.depth_clock == 'header' else 'bag_timestamp_ns'
    if args.with_depth:
        if 'aligned_depth_to_color' not in (manifest.get('depth_topic') or ''):
            raise ValueError('--with-depth requires depth recorded aligned to RGB; alignment is not computed here')
        depth_rows = csv_rows(export / 'depth/timestamps.csv')
        depth_rows = sorted([r for r in depth_rows if r.get(clock) and int(r[clock]) > 0], key=lambda r: int(r[clock]))
        depth_stamps = [int(r[clock]) for r in depth_rows]
    associations = []
    for row in selected:
        stamp = int(row[clock]) if row.get(clock) else 0
        match = nearest_depth(stamp, depth_stamps, depth_rows, round(args.max_depth_dt * 1e9)) if stamp > 0 else None
        if args.require_depth and match is None:
            raise ValueError(f'No depth within tolerance for {row["filename"]}; inspect clocks/alignment')
        associations.append(match)
    if args.with_depth and not any(associations):
        raise ValueError('No matching depth; verify timestamp clock and tolerance')
    out = new_dataset_name(args.name)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    if args.with_depth:
        (out / 'depth').mkdir()
    metadata = {'schema_version': 1, 'kind': 'selection', 'status': 'running', 'created_utc': now(),
        'source_export': str(export), 'source_manifest_sha256': sha256(manifest_path),
        'parameters': {'start': args.start, 'end': args.end, 'rate': args.rate, 'excluded': sorted(excluded),
                       'test_every': args.test_every, 'with_depth': args.with_depth,
                       'depth_clock': args.depth_clock, 'max_depth_dt_s': args.max_depth_dt},
        'calibration_role': 'Recorded intrinsics; camera poses have not been estimated',
        'depth_units': 'metres; NaN invalid', 'poses': 'not generated or copied'}
    save(out / 'dataset.json', metadata)
    frames = []
    try:
        for i, (row, match) in enumerate(zip(selected, associations)):
            filename = row['filename']
            if Path(filename).name != filename or not filename.endswith('.png'):
                raise ValueError('Expected flat PNG filenames in RGB export')
            src = contained(export / 'rgb', filename)
            if png_size(src) != (calibration['width'], calibration['height']):
                raise ValueError(f'RGB dimensions disagree with calibration: {filename}')
            shutil.copy2(src, out / 'images' / filename)
            frame = {'frame_id': Path(filename).stem, 'filename': filename,
                'bag_timestamp_ns': row['bag_timestamp_ns'], 'bag_elapsed_s': row['bag_elapsed_s'],
                'header_timestamp_ns': row['header_timestamp_ns'],
                'split': 'test' if args.test_every and i % args.test_every == 0 else 'train',
                'depth_filename': '', 'depth_timestamp_ns': '', 'depth_dt_ms': ''}
            if match:
                source_depth = contained(export / 'depth', match['metres_npy'])
                depth = np.load(source_depth, allow_pickle=False)
                if depth.shape != (calibration['height'], calibration['width']) or depth.dtype != np.float32:
                    raise ValueError('Expected aligned float32 metric depth matching RGB dimensions')
                frame.update(depth_filename=Path(filename).stem + '.npy', depth_timestamp_ns=match[clock],
                             depth_dt_ms=(int(match[clock]) - int(row[clock])) / 1e6)
                shutil.copy2(source_depth, out / 'depth' / frame['depth_filename'])
            frames.append(frame)
        metadata['depth_matched_frames'] = sum(bool(f['depth_filename']) for f in frames)
        metadata['depth_unmatched_frames'] = len(frames) - metadata['depth_matched_frames'] if args.with_depth else 0
        write_indexes(out, frames, metadata, calibration)
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save(out / 'dataset.json', metadata)
        raise
    print(f'Selected {len(frames)} native frames without reading the rosbag: {out}')
    if args.with_depth:
        print(f'Depth matches: {metadata["depth_matched_frames"]}/{len(frames)} ({args.depth_clock} clock)')
    print('Reference splits are recorded; external trainers need a matching split adapter for fair evaluation.')


def validate_dataset(source):
    meta = json.loads((source / 'dataset.json').read_text())
    if meta.get('status') != 'complete':
        raise ValueError('Dataset is not complete')
    frames = csv_rows(source / 'frames.csv')
    names = [f['filename'] for f in frames]
    if len(set(names)) != len(names) or len(frames) != meta['frame_count']:
        raise ValueError('Frame index count or names are inconsistent')
    if set(names) != {p.name for p in (source / 'images').iterdir() if p.is_file()}:
        raise ValueError('Dataset images changed; create a new selection using --exclude-file')
    calibration = json.loads((source / 'calibration.json').read_text())
    if calibration.get('model') != 'FULL_OPENCV' or len(calibration.get('params', [])) != 12 or not all(
            math.isfinite(x) for x in calibration['params']) or min(calibration['params'][:2]) <= 0:
        raise ValueError('Invalid stored calibration model or parameters')
    if any(f['split'] not in ['train', 'test'] for f in frames):
        raise ValueError('Unknown frame split label')
    splits = json.loads((source / 'splits.json').read_text())
    for kind in ['train', 'test']:
        if splits[kind] != [f['filename'] for f in frames if f['split'] == kind]:
            raise ValueError('Frame index and split manifests disagree')
    for frame in frames:
        if png_size(contained(source / 'images', frame['filename'])) != (calibration['width'], calibration['height']):
            raise ValueError('Image/calibration dimension mismatch')
        if frame['depth_filename']:
            depth = np.load(contained(source / 'depth', frame['depth_filename']), allow_pickle=False)
            if depth.shape != (calibration['height'], calibration['width']) or depth.dtype != np.float32:
                raise ValueError('Depth/calibration dimension or dtype mismatch')
    return meta, frames, calibration


def image_transform(width, height, crop, target_width, target_height, allow_stretch=False):
    x, y, w, h = crop or [0, 0, width, height]
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError('Crop rectangle is outside the source image')
    if target_width is None and target_height is None:
        target_width, target_height = w, h
    elif target_width is None:
        target_width = round(w * target_height / h)
    elif target_height is None:
        target_height = round(h * target_width / w)
    if min(target_width, target_height) <= 0:
        raise ValueError('Output dimensions must be positive')
    # Permit only the rounding error of an aspect-preserving resize unless explicitly stretched.
    expected_height = h * target_width / w
    if not allow_stretch and abs(target_height - expected_height) > .51:
        raise ValueError('Resize would stretch the image. Crop to the desired aspect first or pass --allow-stretch')
    return {'crop_xywh': [x, y, w, h], 'width': target_width, 'height': target_height,
            'scale_x': target_width / w, 'scale_y': target_height / h}


def transform_calibration(source, transform):
    if source['model'] != 'FULL_OPENCV' or len(source['params']) != 12:
        raise ValueError('This variant tool supports the recorded FULL_OPENCV calibration schema')
    params = list(source['params'])
    x, y, _, _ = transform['crop_xywh']
    sx, sy = transform['scale_x'], transform['scale_y']
    params[:4] = [sx * params[0], sy * params[1], sx * (params[2] - x), sy * (params[3] - y)]
    result = {k: copy.deepcopy(v) for k, v in source.items() if k not in ['source_message', 'original_source_message']}
    result['original_source_message'] = copy.deepcopy(source.get('original_source_message', source.get('source_message')))
    result.update(width=transform['width'], height=transform['height'], params=params, transform=transform,
                  source_intrinsics={'width': source['width'], 'height': source['height'], 'params': source['params']},
                  pixel_convention='COLMAP half-integer centres; crop then scale, consistent with cv2.resize half-pixel sampling')
    return result


def variant(args):
    source = args.source.expanduser().resolve()
    parent_meta, frames, calibration = validate_dataset(source)
    crop = args.crop
    if args.center_crop:
        w, h = args.center_crop
        crop = [(calibration['width'] - w) // 2, (calibration['height'] - h) // 2, w, h]
    transform = image_transform(calibration['width'], calibration['height'], crop, args.width, args.height, args.allow_stretch)
    result_calibration = transform_calibration(calibration, transform)
    out = new_dataset_name(args.name)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    if any(f['depth_filename'] for f in frames):
        (out / 'depth').mkdir()
    metadata = {'schema_version': 1, 'kind': 'variant', 'status': 'running', 'created_utc': now(),
        'source_dataset': str(source), 'source_dataset_sha256': sha256(source / 'dataset.json'),
        'transform': transform, 'depth_units': 'metres; unchanged by resize/crop',
        'poses': 'not generated or copied; extrinsics physically unchanged, but no COLMAP pose-format adapter is applied',
        'reference_split_inherited': True}
    save(out / 'dataset.json', metadata)
    x, y, w, h = transform['crop_xywh']
    size = (transform['width'], transform['height'])
    try:
        for frame in frames:
            image = cv2.imread(str(contained(source / 'images', frame['filename'])), cv2.IMREAD_COLOR)
            cropped = image[y:y+h, x:x+w]
            if (w, h) != size:
                interpolation = cv2.INTER_AREA if size[0] <= w and size[1] <= h else cv2.INTER_LINEAR
                cropped = cv2.resize(cropped, size, interpolation=interpolation)
            if not cv2.imwrite(str(out / 'images' / frame['filename']), cropped):
                raise OSError('PNG write failed')
            if frame['depth_filename']:
                depth = np.load(contained(source / 'depth', frame['depth_filename']), allow_pickle=False)[y:y+h, x:x+w]
                if (w, h) != size:
                    depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST_EXACT)
                np.save(out / 'depth' / frame['depth_filename'], depth, allow_pickle=False)
        write_indexes(out, frames, metadata, result_calibration)
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save(out / 'dataset.json', metadata)
        raise
    print(f'Created {len(frames)} frames at {size[0]}x{size[1]}: {out}')
    print(f'COLMAP fx, fy, cx, cy: {result_calibration["params"][:4]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    p = commands.add_parser('select', help='Choose shared native frames from an existing export; no bag decoding')
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--end', type=float)
    p.add_argument('--rate', type=float, default=0)
    p.add_argument('--exclude-file', type=Path, help='Source PNG filenames to reject, one per line')
    p.add_argument('--camera-info-topic', help='Explicit full-resolution calibration topic')
    p.add_argument('--test-every', type=int, default=8, help='Reference test frame every N selected frames, starting at index zero; 0 means all train')
    p.add_argument('--with-depth', action='store_true', help='Associate already-aligned depth by nearest timestamp')
    p.add_argument('--require-depth', action='store_true', help='Fail if any selected RGB frame has no depth match')
    p.add_argument('--depth-clock', choices=['header', 'bag'], default='header')
    p.add_argument('--max-depth-dt', type=float, default=.04, help='Depth association tolerance, seconds')
    p.set_defaults(func=selection)
    p = commands.add_parser('variant', help='Crop/resize RGB and aligned depth while transforming intrinsics')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--name', required=True)
    crops = p.add_mutually_exclusive_group()
    crops.add_argument('--crop', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'))
    crops.add_argument('--center-crop', nargs=2, type=int, metavar=('W', 'H'))
    p.add_argument('--width', type=int)
    p.add_argument('--height', type=int)
    p.add_argument('--allow-stretch', action='store_true')
    p.set_defaults(func=variant)
    p = commands.add_parser('validate', help='Check complete dataset, frame identities, dimensions, splits, and depth')
    p.add_argument('--dataset', type=Path, required=True)
    p.set_defaults(func=lambda args: print(f'Validated {len(validate_dataset(args.dataset.expanduser().resolve())[1])} frames'))
    args = parser.parse_args()
    if getattr(args, 'require_depth', False):
        args.with_depth = True
    try:
        args.func(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
