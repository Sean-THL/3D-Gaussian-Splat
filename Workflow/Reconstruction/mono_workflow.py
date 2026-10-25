#!/usr/bin/env python3
"""Prepare RGB data, run COLMAP, and launch an external 3DGRUT checkout.

This launcher uses only the Python standard library. Dataset extraction uses
Workflow/Data/.venv; training uses the currently active 3dgrut environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def probe(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=20)
        return {'returncode': result.returncode, 'output': (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': None, 'output': str(exc)}


def environment(tool=None):
    info = {'recorded_utc': now(), 'python': sys.version, 'python_executable': sys.executable,
            'project_commit': probe(['git', 'rev-parse', 'HEAD'], PROJECT),
            'project_worktree': probe(['git', 'status', '--porcelain'], PROJECT),
            'gpu': probe(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']),
            'nvcc': probe(['nvcc', '--version'])}
    info['packages'] = {}
    for package in ['torch', 'numpy', 'rosbags', 'opencv-python-headless']:
        try:
            info['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if tool:
        info['3dgrut_path'] = str(tool)
        info['3dgrut_commit'] = probe(['git', 'rev-parse', 'HEAD'], tool)
        info['3dgrut_submodules'] = probe(['git', 'submodule', 'status', '--recursive'], tool)
        info['3dgrut_worktree'] = probe(['git', 'status', '--porcelain'], tool)
    return info


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def logged(command, log_path, cwd=None):
    command = [str(x) for x in command]
    print('Running:', ' '.join(command), flush=True)
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f'Command exited {code}; see {log_path}')
    return log_path.read_text()


def new_dataset_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Dataset name must use letters, digits, underscores, dots, or hyphens')
    return HERE / 'datasets' / name


def calibration_parameters(message):
    """Map ROS OpenCV calibration to COLMAP, retaining all distortion terms."""
    k, d = message['k'], message['d']
    width, height = int(message['width']), int(message['height'])
    if len(k) != 9 or width <= 0 or height <= 0:
        raise ValueError('Invalid camera matrix or image dimensions')
    if not all(math.isfinite(float(v)) for v in [*k, *d]) or k[0] <= 0 or k[4] <= 0:
        raise ValueError('Camera intrinsics contain invalid values')
    if not (0 <= k[2] < width and 0 <= k[5] < height):
        raise ValueError('Camera principal point is outside the image; verify cropping/calibration')
    if abs(k[1]) > 1e-9 or abs(k[3]) > 1e-9:
        raise ValueError('Skewed camera matrices are unsupported')
    roi = message.get('roi', {})
    if message.get('binning_x', 0) not in (0, 1) or message.get('binning_y', 0) not in (0, 1):
        raise ValueError('Binned calibration needs explicit adjustment before use')
    if any(roi.get(key, 0) for key in ['x_offset', 'y_offset', 'width', 'height']):
        raise ValueError('ROI calibration needs explicit adjustment before use')
    kind = message['distortion_model']
    if kind == 'plumb_bob' and len(d) == 5:
        distortion = [*d, 0., 0., 0.]
    elif kind == 'rational_polynomial' and len(d) == 8:
        distortion = list(d)
    else:
        raise ValueError(f'Unsupported calibrated distortion model: {kind} with {len(d)} coefficients')
    # ROS/OpenCV integer pixel centres -> COLMAP half-integer pixel centres.
    params = [k[0], k[4], k[2] + .5, k[5] + .5, *distortion]
    return {'model': 'FULL_OPENCV', 'width': width, 'height': height, 'params': params,
            'pixel_convention': 'ROS/OpenCV cx,cy shifted +0.5 for COLMAP', 'fixed': True}


def read_export_calibration(export, camera_info_topic=None):
    manifest = json.loads((export / 'manifest.json').read_text())
    rgb = manifest['rgb_topic'].removesuffix('/compressed')
    topic = camera_info_topic or rgb.rsplit('/', 1)[0] + '/camera_info'
    if topic not in manifest['files']:
        raise ValueError(f'Calibration topic {topic} not exported; use --camera-info-topic for a different namespace')
    path = export / manifest['files'][topic]['jsonl']
    first = None
    count = 0
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            candidate = calibration_parameters(record['message'])
            if first is None:
                first = candidate
                first['source_message'] = record['message']
            elif (candidate['width'], candidate['height'], candidate['model']) != (first['width'], first['height'], first['model']) or not all(
                    math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(candidate['params'], first['params'])):
                raise ValueError('Camera calibration varies inside the selected interval; split the recording')
            count += 1
    if first is None:
        raise ValueError('No camera calibration messages in the selected interval')
    first.update(source_topic=topic, messages_checked=count)
    return first


def export_calibration(dataset, camera_info_topic=None):
    calibration = read_export_calibration(dataset / 'source_export', camera_info_topic)
    save(dataset / 'calibration.json', calibration)
    return calibration


def check_calibrated_images(images, calibration):
    for image in images:
        with image.open('rb') as stream:
            header = stream.read(24)
        if header[:8] != b'\x89PNG\r\n\x1a\n' or len(header) != 24:
            raise ValueError(f'Invalid PNG: {image}')
        size = struct.unpack('>II', header[16:24])
        if size != (calibration['width'], calibration['height']):
            raise ValueError(f'Image dimensions {size} differ from recorded calibration: {image}')


def prepare(args):
    # Resolving a venv's interpreter symlink would bypass pyvenv.cfg and lose its packages.
    python = args.extractor_python.expanduser().absolute()
    if not python.is_file():
        raise ValueError('Set up Workflow/Data/.venv first, or pass --extractor-python')
    if args.export is not None:
        command = [str(python), str(PROJECT / 'Workflow/Data/dataset_tools.py'), 'select',
                   '--export', str(args.export.expanduser().resolve()), '--name', args.name,
                   '--start', str(args.start), '--rate', str(args.rate)]
        if args.end is not None:
            command += ['--end', str(args.end)]
        if args.camera_info_topic:
            command += ['--camera-info-topic', args.camera_info_topic]
        if args.rgb_topic:
            manifest = json.loads((args.export.expanduser() / 'manifest.json').read_text())
            if manifest.get('rgb_topic') != args.rgb_topic:
                raise ValueError('--rgb-topic differs from the stream in this export; choose the matching export')
        subprocess.run(command, check=True)
        return
    bag = args.bag.expanduser().resolve()
    if not (bag / 'metadata.yaml').is_file():
        raise ValueError(f'Expected complete ROS 2 bag directory: {bag}')
    if args.start < 0 or args.rate < 0 or (args.end is not None and args.end <= args.start):
        raise ValueError('Invalid time window or image rate')
    if args.camera_info_topic and not args.calibrated:
        raise ValueError('--camera-info-topic requires --calibrated')
    dataset = new_dataset_name(args.name)
    dataset.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'preparing', 'bag': str(bag), 'environment': environment()}
    manifest = dataset / 'preparation.json'
    save(manifest, metadata)
    command = [python, PROJECT / 'Workflow/Data/extract_rosbag.py', '--bag', bag,
               '--RGB', '--start', str(args.start), '--image-rate', str(args.rate),
               '--output', dataset / 'source_export']
    if args.end is not None:
        command += ['--end', str(args.end)]
    if args.calibrated:
        command += ['--camera-info']
    if args.rgb_topic:
        command += ['--rgb-topic', args.rgb_topic]
    metadata['command'] = [str(x) for x in command]
    try:
        logged(command, dataset / 'extraction.log')
        images = sorted((dataset / 'source_export/rgb').glob('*.png'))
        if len(images) < 4:
            raise ValueError(f'Only {len(images)} images; choose a longer window or higher rate')
        (dataset / 'images').mkdir()
        for image in images:
            shutil.copy2(image, dataset / 'images' / image.name)
        if args.calibrated:
            calibration = export_calibration(dataset, args.camera_info_topic)
            check_calibrated_images(images, calibration)
            metadata['calibration'] = calibration
        metadata.update(status='prepared', image_count=len(images))
    except BaseException as exc:
        metadata.update(status='failed', error=str(exc))
        save(manifest, metadata)
        raise
    save(manifest, metadata)
    print(f'Prepared {len(images)} images: {dataset}')
    print('Review images/ for blur/glare; remove rejected images there, keeping source_export intact.')
    print(f'Next: python3 {Path(__file__).name} colmap --dataset {dataset}')


def gpu_option(help_text, stage, enabled):
    candidates = (['--FeatureExtraction.use_gpu', '--SiftExtraction.use_gpu'] if stage == 'extract'
                  else ['--FeatureMatching.use_gpu', '--SiftMatching.use_gpu'])
    option = next((x for x in candidates if x in help_text), None)
    if option is None:
        raise ValueError('Cannot identify COLMAP GPU option; inspect installed COLMAP --help')
    return [option, '1' if enabled else '0']


def registration_count(text):
    match = re.search(r'(?:Registered images|Images)\s*:\s*(\d+)', text, re.IGNORECASE)
    if not match:
        raise ValueError('Cannot read registered-image count; inspect model_analyzer.log')
    return int(match.group(1))


def run_colmap(args):
    dataset = args.dataset.expanduser().resolve()
    images = sorted((dataset / 'images').glob('*.png'))
    if len(images) < 4:
        raise ValueError('Expected at least four PNG images in DATASET/images/')
    if not 0 < args.min_registered_fraction <= 1:
        raise ValueError('--min-registered-fraction must be in (0, 1]')
    colmap = shutil.which('colmap')
    if not colmap:
        raise ValueError('COLMAP is missing. On Ubuntu: sudo apt install colmap')
    extraction_help = probe([colmap, 'feature_extractor', '-h'])['output']
    matching_help = probe([colmap, 'exhaustive_matcher', '-h'])['output']
    calibration_path = dataset / 'calibration.json'
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    model = 'OPENCV'
    calibration_options, mapper_options = [], []
    if calibration:
        check_calibrated_images(images, calibration)
        model = calibration['model']
        calibration_options = ['--ImageReader.camera_params', ','.join(str(x) for x in calibration['params'])]
        mapper_help = probe([colmap, 'mapper', '-h'])['output']
        # Older COLMAP also derives absolute-pose refinement from these BA options.
        for flag in ['--Mapper.ba_refine_focal_length', '--Mapper.ba_refine_principal_point',
                     '--Mapper.ba_refine_extra_params']:
            if flag not in mapper_help:
                raise ValueError(f'Installed COLMAP lacks required fixed-calibration option: {flag}')
            mapper_options += [flag, '0']
        for flag in ['--Mapper.abs_pose_refine_focal_length', '--Mapper.abs_pose_refine_extra_params']:
            if flag in mapper_help:
                mapper_options += [flag, '0']
    if '--ImageReader.single_camera' in extraction_help:
        camera = ['--ImageReader.single_camera', '1']
    elif '--camera_mode' in extraction_help:
        camera = ['--camera_mode', '1']  # SINGLE
    else:
        raise ValueError('Cannot identify COLMAP single-camera setting; inspect feature_extractor -h')
    extraction_gpu = gpu_option(extraction_help, 'extract', args.gpu)
    matching_gpu = gpu_option(matching_help, 'match', args.gpu)
    folder = dataset / 'colmap'
    folder.mkdir(exist_ok=False)
    (folder / 'sparse').mkdir()
    metadata = {'status': 'running', 'environment': environment(),
                'colmap_version': probe([colmap, '-h']), 'image_count': len(images),
                'gpu_features': args.gpu, 'min_registered_fraction': args.min_registered_fraction,
                'calibration': calibration,
                'commands': []}
    manifest = folder / 'provenance.json'
    save(manifest, metadata)
    def run(stage, parameters):
        command = [colmap, stage, *parameters]
        metadata['commands'].append([str(x) for x in command])
        save(manifest, metadata)
        return logged(command, folder / f'{stage}.log')
    try:
        run('feature_extractor', ['--database_path', folder / 'database.db', '--image_path', dataset / 'images',
                                 *camera, '--ImageReader.camera_model', model, *calibration_options, *extraction_gpu])
        run('exhaustive_matcher', ['--database_path', folder / 'database.db', *matching_gpu])
        run('mapper', ['--database_path', folder / 'database.db', '--image_path', dataset / 'images',
                       '--output_path', folder / 'sparse', *mapper_options])
        models = sorted(p for p in (folder / 'sparse').iterdir() if p.is_dir())
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} disconnected models. Review coverage/matches before training; see {folder / "sparse"}')
        report = run('model_analyzer', ['--path', models[0]])
        count = registration_count(report)
        metadata.update(registered_images=count, registered_fraction=count / len(images))
        if count / len(images) < args.min_registered_fraction:
            raise ValueError(f'Only {count}/{len(images)} images registered; review images and COLMAP before training')
        # Camera estimation uses distortion; training receives undistorted PINHOLE images.
        run('image_undistorter', ['--image_path', dataset / 'images', '--input_path', models[0],
                                 '--output_path', dataset / 'undistorted', '--output_type', 'COLMAP'])
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=str(exc))
        save(manifest, metadata)
        raise
    save(manifest, metadata)
    print(f'COLMAP complete: {count}/{len(images)} images registered.')
    print('Registration count is a screening check, not proof of correct geometry. Inspect the sparse model.')
    print(f'Training data: {dataset / "undistorted"}')


def train(args):
    dataset = args.dataset.expanduser().resolve()
    tool = args.tool_dir.expanduser().resolve()
    if not (tool / 'train.py').is_file():
        raise ValueError(f'External 3DGRUT checkout not found: {tool}')
    provenance = dataset / 'colmap/provenance.json'
    if not provenance.is_file() or json.loads(provenance.read_text()).get('status') != 'complete':
        raise ValueError('Complete the COLMAP stage successfully before training')
    if not (dataset / 'undistorted/images').is_dir():
        raise ValueError('Undistorted images are missing')
    if args.max_gaussians < 1 or args.workers < 0:
        raise ValueError('Invalid Gaussian limit or worker count')
    cuda = probe([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'])
    if cuda['returncode'] != 0:
        raise ValueError('Activate the installed 3dgrut environment first. ' + cuda['output'])
    run_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = HERE / 'runs' / f'{dataset.name}_{run_name}'
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, 'train.py', '--config-name', 'apps/colmap_3dgut_mcmc.yaml',
               f'path={dataset / "undistorted"}', f'out_dir={folder / "model"}',
               f'experiment_name={dataset.name}', f'num_workers={args.workers}',
               f'strategy.add.max_n_gaussians={args.max_gaussians}', 'export_ply.enabled=true']
    metadata = {'status': 'running', 'environment': environment(tool), 'dataset': str(dataset),
                'command': command, 'colmap_provenance': json.loads(provenance.read_text())}
    for name in ['dataset.json', 'splits.json', 'calibration.json']:
        if (dataset / name).is_file():
            metadata[name] = json.loads((dataset / name).read_text())
    if 'splits.json' in metadata:
        metadata['split_policy'] = 'Reference split recorded; external trainer uses its own split. Audit actual lists before comparison.'
        print('Reference split recorded in provenance; NVIDIA split selection is not overridden by this launcher.')
    save(folder / 'provenance.json', metadata)
    try:
        logged(command, folder / 'training.log', tool)
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=str(exc))
        save(folder / 'provenance.json', metadata)
        raise
    save(folder / 'provenance.json', metadata)
    print(f'Training finished; outputs and exact software revisions: {folder}')


def match_graph(image_ids, pairs, threshold):
    adjacent = {i: set() for i in image_ids}
    edges = 0
    for pair_id, matches in pairs:
        if matches < threshold:
            continue
        a, b = divmod(pair_id, 2147483647)
        if a in adjacent and b in adjacent:
            adjacent[a].add(b)
            adjacent[b].add(a)
            edges += 1
    unseen = set(adjacent)
    components = []
    while unseen:
        seed = unseen.pop()
        pending, component = [seed], [seed]
        while pending:
            for node in adjacent[pending.pop()]:
                if node in unseen:
                    unseen.remove(node)
                    component.append(node)
                    pending.append(node)
        components.append(component)
    return {'min_verified_matches': threshold, 'edges': edges,
            'component_sizes': sorted([len(c) for c in components], reverse=True),
            'isolated_images': sum(not adjacent[i] for i in adjacent)}


def diagnose(args):
    dataset = args.dataset.expanduser().resolve()
    database = dataset / 'colmap/database.db'
    if not database.is_file():
        raise ValueError(f'Database missing: {database}')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        images = list(db.execute('SELECT image_id,name FROM images'))
        pairs = list(db.execute('SELECT pair_id,rows FROM two_view_geometries WHERE rows>0'))
        cameras = []
        for camera_id, model, w, h, blob in db.execute('SELECT camera_id,model,width,height,params FROM cameras'):
            cameras.append({'camera_id': camera_id, 'model_id': model, 'width': w, 'height': h,
                            'params': list(struct.unpack('<' + 'd' * (len(blob) // 8), blob))})
        result = {'images': len(images), 'verified_pairs': len(pairs), 'cameras': cameras,
                  'graphs': [match_graph([i for i, _ in images], pairs, threshold) for threshold in [15, 30, 100]],
                  'two_view_config_counts': list(db.execute('SELECT config,COUNT(*) FROM two_view_geometries WHERE rows>0 GROUP BY config'))}
    for filename, key in [('calibration.json', 'recorded_calibration'), ('colmap/provenance.json', 'colmap_status')]:
        path = dataset / filename
        if path.exists():
            obj = json.loads(path.read_text())
            result[key] = obj if key == 'recorded_calibration' else {k: obj.get(k) for k in ['status', 'error', 'registered_images']}
    print(json.dumps(result, indent=2))
    log = dataset / 'colmap/mapper.log'
    if log.exists():
        relevant = [line for line in log.read_text().splitlines() if re.search(
            r'initial image pair|Initializing|Registering image|Filtered images|Discarding|Failed|ERROR|No good', line, re.IGNORECASE)]
        print('\nMapper events (last 40):\n' + '\n'.join(relevant[-40:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    p = commands.add_parser('prepare', help='Extract a fresh RGB dataset from a ROS 2 bag')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--bag', type=Path, help='Legacy convenience mode: extract directly from a bag')
    source.add_argument('--export', type=Path, help='Reuse existing native export; no bag decoding')
    p.add_argument('--name', required=True)
    p.add_argument('--start', type=float, default=33)
    p.add_argument('--end', type=float, default=70)
    p.add_argument('--rate', type=float, default=2)
    p.add_argument('--calibrated', action='store_true', help='Export matching camera_info and use fixed recorded intrinsics in COLMAP')
    p.add_argument('--rgb-topic', help='RGB image topic override, passed to the bag extractor')
    p.add_argument('--camera-info-topic', help='Explicit camera_info topic when its namespace differs from the RGB stream')
    p.add_argument('--extractor-python', type=Path, default=PROJECT / 'Workflow/Data/.venv/bin/python')
    p.set_defaults(func=prepare)
    p = commands.add_parser('colmap', help='Estimate poses, screen registration, and undistort images')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--gpu', action='store_true', help='Use GPU feature extraction/matching; default CPU works over SSH')
    p.add_argument('--min-registered-fraction', type=float, default=.7)
    p.set_defaults(func=run_colmap)
    p = commands.add_parser('diagnose', help='Inspect camera parameters, verified-match connectivity, and mapper failures')
    p.add_argument('--dataset', type=Path, required=True)
    p.set_defaults(func=diagnose)
    p = commands.add_parser('train', help='Launch the separately installed NVIDIA 3DGRUT')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--tool-dir', type=Path, default=Path.home() / 'tools/3dgrut')
    p.add_argument('--max-gaussians', type=int, default=500000)
    p.add_argument('--workers', type=int, default=2)
    p.set_defaults(func=train)
    args = parser.parse_args()
    try:
        args.func(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
