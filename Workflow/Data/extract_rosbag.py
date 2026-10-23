#!/usr/bin/env python3
"""Stream ROS bags into RGB, video, metric depth, and timestamped telemetry.

No ROS installation or GPU is required. See README.md for setup and formats.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from contextlib import ExitStack

try:
    import cv2
    import numpy as np
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
except ImportError as exc:
    raise SystemExit(f'Missing dependency: {exc}. Install requirements.txt in a venv; see README.md.')

HERE = Path(__file__).resolve().parent
IMAGE_TYPES = {'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bag', type=Path, help='Bag directory containing metadata.yaml (default: sole bag in rosbags/)')
    p.add_argument('-o', '--output', type=Path, help='New output directory; existing directories are refused')
    for flags, help_text in [
        (['--RGB', '--rgb'], 'RGB PNG images'),
        (['--MP4', '--mp4'], 'RGB-only H.264 MP4 at --fps, preserving recording timing'),
        (['--depth'], 'Metric depth .npy arrays plus native uint16 PNGs where available'),
        (['--vicon'], 'Vicon TransformStamped / pose data'),
        (['--position', '--positions', '--PX4'], 'PX4 local position, odometry, and external visual odometry'),
        (['--trajectory'], 'Commanded trajectory/setpoint messages'),
        (['--camera-info'], 'All recorded camera calibration messages'),
        (['--tf'], 'Recorded /tf and /tf_static messages'),
        (['--telemetry'], 'Every non-image topic, including status and ROS logs'),
        (['--all'], 'RGB, MP4, depth, and every non-image topic'),
    ]:
        p.add_argument(*flags, dest=flags[0].lstrip('-').lower().replace('-', '_'), action='store_true', help=help_text)
    p.add_argument('--list-topics', action='store_true', help='Show recorded types and counts without exporting')
    p.add_argument('--topic', action='append', default=[], help='Additional non-image topic to export; repeatable')
    p.add_argument('--rgb-topic', help='Override automatic RGB topic selection')
    p.add_argument('--depth-topic', help='Override automatic depth topic selection')
    p.add_argument('--start', type=float, default=0, help='Start seconds relative to first BAG record (default: 0)')
    p.add_argument('--end', type=float, help='Exclusive end seconds relative to first BAG record')
    p.add_argument('--image-rate', type=float, default=0, help='Maximum PNG/depth samples per second per stream; 0 exports all. MP4 is independent.')
    p.add_argument('--fps', type=float, default=20, help='MP4 output FPS, default 20')
    p.add_argument('--depth-scale', type=float, default=.001, help='Metres per native uint16 depth unit (RealSense default .001); verify for other sensors')
    p.add_argument('--depth-preview', action='store_true', help='Also save colorized depth PNGs (display only)')
    p.add_argument('--preview-max-depth', type=float, default=10, help='Depth-preview display range, metres (default 10)')
    p.add_argument('--msg-dir', action='append', type=Path, default=[], help='Extra custom definitions root with PACKAGE/msg/*.msg; overrides bundled types')
    args = p.parse_args(argv)
    for name in ['start', 'image_rate', 'fps', 'depth_scale', 'preview_max_depth']:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in ['fps', 'depth_scale', 'preview_max_depth'] and value == 0):
            p.error(f'--{name.replace("_", "-")} has an invalid value')
    if args.end is not None and (not math.isfinite(args.end) or args.end <= args.start):
        p.error('--end must be greater than --start')
    if args.depth_preview:
        args.depth = True
    if not (args.list_topics or args.topic or any(getattr(args, key) for key in
            ['rgb', 'mp4', 'depth', 'vicon', 'position', 'trajectory', 'camera_info', 'tf', 'telemetry', 'all'])):
        p.error('Choose exports, for example --RGB, --RGB --MP4 --vicon, or --all')
    return args


def resolve_bag(path):
    if path is not None:
        path = path.expanduser().resolve()
        if not path.exists():
            raise ValueError(f'Bag does not exist: {path}')
        return path
    bags = sorted(p.parent for p in (HERE / 'rosbags').glob('*/metadata.yaml'))
    if len(bags) != 1:
        raise ValueError(f'Found {len(bags)} bags in rosbags/. Specify --bag /path/to/bag.')
    return bags[0]


def make_typestore(extra_dirs=()):
    store = get_typestore(Stores.ROS2_HUMBLE)
    definitions, sources = {}, {}
    for root in [HERE / 'message_definitions', *extra_dirs]:
        if not root.is_dir():
            raise ValueError(f'Message-definition directory not found: {root}')
        for path in sorted(root.rglob('*.msg')):
            if path.parent.name != 'msg':
                continue
            typename = f'{path.parent.parent.name}/msg/{path.stem}'
            definitions.update(get_types_from_msg(path.read_text(), typename))
            sources[typename] = str(path.resolve())
    store.register(definitions)
    return store, sources


def topic_key(topic):
    # Encoding every non-alphanumeric byte prevents filename collisions.
    return ''.join(c if c.isascii() and c.isalnum() else f'_{ord(c):02x}' for c in topic)


def clean(value):
    """Portable JSON: preserve all message fields; non-finite numbers become null."""
    if dataclasses.is_dataclass(value):
        return {f.name: clean(getattr(value, f.name)) for f in dataclasses.fields(value) if not f.name.startswith('__')}
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def flatten(data, prefix=''):
    result = {}
    for key, value in data.items():
        key = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, key))
        else:
            result[key] = json.dumps(value, allow_nan=False) if isinstance(value, list) else value
    return result


def header_ns(msg):
    header = getattr(msg, 'header', None)
    if header is None:
        return None
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def time_fields(msg, stamp, origin):
    return {'bag_timestamp_ns': int(stamp), 'bag_elapsed_s': (stamp - origin) / 1e9,
            'header_timestamp_ns': header_ns(msg)}


def raw_array(msg, channels, dtype):
    dtype = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<')
    row_bytes = int(msg.width) * channels * dtype.itemsize
    if msg.step < row_bytes or len(msg.data) < int(msg.height) * int(msg.step):
        raise ValueError('Raw image dimensions/step exceed its data buffer')
    rows = np.frombuffer(msg.data, dtype=np.uint8, count=int(msg.height) * int(msg.step)).reshape(msg.height, msg.step)
    a = np.ascontiguousarray(rows[:, :row_bytes]).view(dtype).reshape(msg.height, msg.width, channels)
    return a.astype(dtype.newbyteorder('='), copy=False)


def decode_rgb(msg):
    if hasattr(msg, 'format'):
        image = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode compressed RGB image')
        return image
    encoding = msg.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(encoding)
    if channels is None:
        raise ValueError(f'Unsupported RGB encoding: {encoding}')
    image = raw_array(msg, channels, 'u1')
    code = {'rgb8': cv2.COLOR_RGB2BGR, 'rgba8': cv2.COLOR_RGBA2BGR,
            'bgra8': cv2.COLOR_BGRA2BGR, 'mono8': cv2.COLOR_GRAY2BGR}.get(encoding)
    return cv2.cvtColor(image, code) if code is not None else image


def decode_depth(msg, scale):
    """Return (float32 metres, native uint16 or None, original encoding)."""
    encoding = getattr(msg, 'encoding', '').upper()
    if hasattr(msg, 'format'):
        encoding = msg.format.split(';')[0].strip().upper()
        if 'rvl' in msg.format.lower():
            raise ValueError('RVL compressed depth is unsupported; record PNG compressedDepth or raw depth')
        payload = bytes(msg.data)
        offset = payload.find(b'\x89PNG\r\n\x1a\n', 0, 64)
        if offset < 0:
            raise ValueError('PNG signature missing from compressed depth message')
        image = cv2.imdecode(np.frombuffer(payload[offset:], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint16 or image.ndim != 2:
            raise ValueError('Expected a 16-bit single-channel depth PNG')
        if encoding == '32FC1':
            if offset != 12:
                raise ValueError('Expected 12-byte inverse-depth header for 32FC1 compressedDepth')
            _, a, b = struct.unpack('<iff', payload[:12])
            metres = np.full(image.shape, np.nan, dtype=np.float32)
            valid = (image > 0) & (image.astype(np.float32) > b)
            metres[valid] = a / (image[valid].astype(np.float32) - b)
            return metres, None, encoding
    else:
        if encoding not in ('16UC1', 'MONO16', '32FC1'):
            raise ValueError(f'Unsupported raw depth encoding: {encoding}')
        image = raw_array(msg, 1, 'f4' if encoding == '32FC1' else 'u2')[:, :, 0]
    if encoding in ('16UC1', 'MONO16'):
        metres = image.astype(np.float32) * scale
        metres[image == 0] = np.nan
        return metres, image, encoding
    if encoding == '32FC1':
        metres = image.astype(np.float32)
        metres[(metres <= 0) | ~np.isfinite(metres)] = np.nan
        return metres, None, encoding
    raise ValueError(f'Unsupported depth encoding: {encoding}')


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Image write failed: {path}')


class CsvSink:
    def __init__(self, path, stack):
        self.file = stack.enter_context(path.open('w', newline=''))
        self.writer = None
        self.count = 0

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count += 1


class VideoSink:
    """Constant-rate video with hold-last-frame resampling using BAG timestamps."""
    def __init__(self, folder, fps, stack):
        self.folder, self.fps, self.proc = folder, fps, None
        self.previous = None
        self.first_stamp = None
        self.count = 0
        self.timeline = CsvSink(folder / 'frames.csv', stack)

    def emit(self):
        frame, stamp, header_stamp = self.previous
        self.proc.stdin.write(frame.tobytes())
        self.timeline.write({'frame_index': self.count, 'video_time_s': self.count / self.fps,
                             'source_bag_timestamp_ns': stamp, 'source_header_timestamp_ns': header_stamp})
        self.count += 1

    def add(self, frame, stamp, header_stamp):
        if self.proc is None:
            h, w = frame.shape[:2]
            if h % 2 or w % 2:
                raise ValueError('H.264 yuv420p requires even image dimensions')
            self.shape = frame.shape
            self.proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(self.fps),
                '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.folder / 'rgb.mp4')], stdin=subprocess.PIPE)
            self.first_stamp = stamp
        if frame.shape != self.shape:
            raise ValueError('RGB resolution changed during recording; export images or split the bag')
        while self.previous is not None and self.first_stamp + round(self.count * 1e9 / self.fps) < stamp:
            self.emit()
        self.previous = (frame, stamp, header_stamp)

    def close(self, successful=True):
        if self.proc is not None:
            try:
                if successful:
                    self.emit()
            finally:
                self.proc.stdin.close()
                code = self.proc.wait()
            if successful and code:
                raise RuntimeError(f'ffmpeg failed with exit code {code}')


def select_image(connections, kind, override):
    candidates = {c.topic: c.msgtype for c in connections if c.msgtype in IMAGE_TYPES}
    if override:
        if override not in candidates:
            raise ValueError(f'{kind} topic is absent or not an Image/CompressedImage: {override}')
        return override
    candidates = [t for t in candidates if ('depth' in t.lower()) == (kind == 'depth')]
    if kind == 'rgb':
        preferred = [t for t in candidates if 'color' in t.lower() or 'rgb' in t.lower()]
        candidates = preferred or candidates
    if not candidates:
        raise ValueError(f'No {kind} stream found; inspect --list-topics')
    # Prefer aligned depth and recorded compressed streams.
    candidates.sort(key=lambda t: ('aligned' not in t.lower() if kind == 'depth' else False,
                                   'compressed' not in t.lower(), t))
    if len(candidates) > 1:
        print(f'Using {kind}: {candidates[0]}; alternatives: {candidates[1:]} (override with --{kind}-topic)')
    return candidates[0]


def category(conn):
    t, typename = conn.topic.lower(), conn.msgtype
    if 'vicon' in t:
        return 'vicon'
    if typename.endswith('/CameraInfo'):
        return 'camera_info'
    if typename.endswith('/TFMessage'):
        return 'tf'
    if 'trajectory' in t or 'setpoint' in t:
        return 'trajectory'
    if typename.endswith(('/VehicleLocalPosition', '/VehicleOdometry', '/Odometry')) or 'position' in t:
        return 'position'
    return 'telemetry'


def export(args, bag, store, definition_sources):
    with AnyReader([bag], default_typestore=store) as reader:
        inventory = [{'topic': c.topic, 'type': c.msgtype, 'message_count': c.msgcount} for c in reader.connections]
        if args.list_topics:
            for item in inventory:
                print(f'{item["message_count"]:8d}  {item["topic"]}  [{item["type"]}]')
            return
        want_rgb, want_video, want_depth = args.rgb or args.all, args.mp4 or args.all, args.depth or args.all
        if want_video and shutil.which('ffmpeg') is None:
            raise ValueError('MP4 export needs ffmpeg on PATH; install it or omit --MP4')
        rgb_topic = select_image(reader.connections, 'rgb', args.rgb_topic) if want_rgb or want_video else None
        depth_topic = select_image(reader.connections, 'depth', args.depth_topic) if want_depth else None
        available = {c.topic for c in reader.connections}
        missing = set(args.topic) - available
        if missing:
            raise ValueError(f'Requested topics absent: {sorted(missing)}')
        for conn in reader.connections:
            if conn.topic in args.topic and conn.msgtype in IMAGE_TYPES:
                raise ValueError('Use --RGB / --depth and the image-topic overrides for image topics')
        selected = []
        for conn in reader.connections:
            cat = category(conn)
            if conn.topic in (rgb_topic, depth_topic) or (conn.msgtype not in IMAGE_TYPES and
                    (args.all or args.telemetry or getattr(args, cat, False) or conn.topic in args.topic)):
                selected.append(conn)
        for key in ['vicon', 'position', 'trajectory', 'camera_info', 'tf']:
            if getattr(args, key) and not any(category(c) == key for c in selected):
                raise ValueError(f'No {key} topics found; inspect --list-topics')
        if not any(c.msgcount for c in selected):
            raise ValueError('No recorded messages match the selection')
        origin = int(reader.start_time)
        begin = origin + round(args.start * 1e9)
        stop = origin + round(args.end * 1e9) if args.end is not None else int(reader.end_time)
        if begin >= reader.end_time:
            raise ValueError('--start is beyond the end of the recording')
        out = args.output.expanduser().resolve() if args.output else HERE / 'exports' / bag.stem / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out.mkdir(parents=True, exist_ok=False)
        manifest = {'status': 'in_progress', 'source_bag': str(bag), 'created_utc': datetime.now(timezone.utc).isoformat(),
            'software': {'python': sys.version.split()[0], 'opencv': cv2.__version__, 'numpy': np.__version__},
            'bag_start_timestamp_ns': origin, 'bag_end_timestamp_ns_exclusive': int(reader.end_time),
            'arguments': {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if k == 'msg_dir' else v for k, v in vars(args).items()},
            'topic_inventory': inventory, 'selected_topics': [c.topic for c in selected],
            'rgb_topic': rgb_topic, 'depth_topic': depth_topic, 'message_definition_sources': definition_sources,
            'counts': {}, 'files': {},
            'conventions': {'image_order': 'bag record timestamp', 'time_window': 'bag elapsed seconds, start inclusive/end exclusive',
                'depth_npy': 'float32 metres, NaN invalid; uint16 units follow --depth-scale',
                'depth_png': 'native uint16 units, zero invalid', 'telemetry_nonfinite': 'JSON null / empty CSV cell',
                'px4_timestamp': 'original PX4 clock (typically microseconds since boot), NOT Unix time',
                'frames': 'Original recorded frames; no NED/ENU, Vicon-to-camera, or scale alignment is applied',
                'video': 'constant FPS; hold previous RGB frame across gaps; image-rate does not affect MP4',
                'synchronization': 'Streams exported independently; no RGB-depth-pose pairing or interpolation applied'}}
        manifest_path = out / 'manifest.json'
        def save_manifest():
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        save_manifest()
        if bag.is_dir() and (bag / 'metadata.yaml').exists():
            shutil.copy2(bag / 'metadata.yaml', out / 'source_metadata.yaml')
        print(f'Export directory: {out}', flush=True)
        seen, counts, last_image = {}, {}, {}
        sinks, json_sinks = {}, {}
        video = None
        completed = False
        try:
            with ExitStack() as stack:
                for name, enabled in [('rgb', want_rgb), ('depth', want_depth), ('video', want_video)]:
                    if enabled:
                        (out / name).mkdir()
                if want_rgb:
                    rgb_csv = CsvSink(out / 'rgb' / 'timestamps.csv', stack)
                if want_depth:
                    depth_csv = CsvSink(out / 'depth' / 'timestamps.csv', stack)
                    if args.depth_preview:
                        (out / 'depth' / 'preview').mkdir()
                if want_video:
                    video = VideoSink(out / 'video', args.fps, stack)
                try:
                    for conn, stamp, raw in reader.messages(connections=selected, start=begin, stop=stop):
                        msg = reader.deserialize(raw, conn.msgtype)
                        seen[conn.topic] = seen.get(conn.topic, 0) + 1
                        if conn.topic == rgb_topic:
                            due = stamp >= last_image.get(conn.topic, -1)
                            if want_video or (want_rgb and due):
                                frame = decode_rgb(msg)
                            if want_video:
                                video.add(frame, stamp, header_ns(msg))
                            if want_rgb and due:
                                idx = rgb_csv.count
                                name = f'{idx:06d}_{stamp}.png'
                                write_image(out / 'rgb' / name, frame)
                                rgb_csv.write({'index': idx, 'filename': name, **time_fields(msg, stamp, origin),
                                               'width': frame.shape[1], 'height': frame.shape[0]})
                                counts['rgb_images'] = rgb_csv.count
                                last_image[conn.topic] = stamp + round(1e9 / args.image_rate) if args.image_rate else stamp
                        elif conn.topic == depth_topic:
                            if stamp < last_image.get(conn.topic, -1):
                                continue
                            depth, native, encoding = decode_depth(msg, args.depth_scale)
                            idx = depth_csv.count
                            stem = f'{idx:06d}_{stamp}'
                            np.save(out / 'depth' / f'{stem}.npy', depth, allow_pickle=False)
                            if native is not None:
                                write_image(out / 'depth' / f'{stem}.png', native)
                            if args.depth_preview:
                                valid = np.isfinite(depth) & (depth > 0)
                                display = np.zeros(depth.shape, dtype=np.uint8)
                                display[valid] = (np.clip(depth[valid] / args.preview_max_depth, 0, 1) * 255).astype(np.uint8)
                                preview = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
                                preview[~valid] = 0
                                write_image(out / 'depth' / 'preview' / f'{stem}.png', preview)
                            depth_csv.write({'index': idx, 'metres_npy': f'{stem}.npy',
                                'native_png': f'{stem}.png' if native is not None else '',
                                **time_fields(msg, stamp, origin), 'source_encoding': encoding,
                                'metres_per_native_unit': args.depth_scale if native is not None else '',
                                'width': depth.shape[1], 'height': depth.shape[0],
                                'valid_fraction': float(np.isfinite(depth).mean())})
                            counts['depth_frames'] = depth_csv.count
                            last_image[conn.topic] = stamp + round(1e9 / args.image_rate) if args.image_rate else stamp
                        else:
                            key = conn.topic
                            if key not in sinks:
                                folder = out / category(conn)
                                folder.mkdir(exist_ok=True)
                                path = folder / topic_key(key)
                                sinks[key] = CsvSink(path.with_suffix('.csv'), stack)
                                json_sinks[key] = stack.enter_context(path.with_suffix('.jsonl').open('w'))
                                manifest['files'][key] = {'csv': str(path.with_suffix('.csv').relative_to(out)),
                                                         'jsonl': str(path.with_suffix('.jsonl').relative_to(out))}
                            data = clean(msg)
                            timing = time_fields(msg, stamp, origin)
                            sinks[key].write({**timing, **flatten(data)})
                            json_sinks[key].write(json.dumps({**timing, 'topic': key, 'type': conn.msgtype, 'message': data}, allow_nan=False) + '\n')
                            counts[key] = sinks[key].count
                    completed = True
                finally:
                    if video:
                        video.close(successful=completed)
                        counts['video_frames'] = video.count
        except BaseException as exc:
            manifest.update(status='failed', error=str(exc), counts=counts, messages_read=seen)
            save_manifest()
            raise
        manifest.update(status='complete', counts=counts, messages_read=seen,
                        empty_selected_topics=[c.topic for c in selected if not seen.get(c.topic)])
        save_manifest()
        print(json.dumps(counts, indent=2))
        print(f'Completed. Index, provenance, and topic inventory: {manifest_path}')


def main():
    args = parse_args()
    try:
        bag = resolve_bag(args.bag)
        store, sources = make_typestore(args.msg_dir)
        export(args, bag, store, sources)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Export failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
