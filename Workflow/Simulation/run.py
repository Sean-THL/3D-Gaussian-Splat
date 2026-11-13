#!/usr/bin/env python3
"""Local interactive drone simulator and reproducible headless rollouts."""
import argparse
import csv
import errno
import hashlib
import io
import json
import queue
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from Workflow.Simulation.figs.simulator import Simulator
from Workflow.Simulation.isaac_sim.bridge_client import encode_rgba


def jpeg(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format='JPEG', quality=88)
    return stream.getvalue()


def depth_rgb(depth):
    from Workflow.Evaluation.depth_check import colorize
    return np.asarray(colorize(depth, 8.))


def placeholder(width=640, height=360):
    image = Image.new('RGB', (width, height), '#172438')
    ImageDraw.Draw(image).text((35, height//2), 'Dynamics only - Gaussian rendering disabled', fill='white')
    return np.asarray(image)


def write_telemetry(path, rows):
    with path.open('w') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def draw_record_frame(rgb, state):
    width, height = rgb.shape[1], rgb.shape[0]
    image = Image.new('RGB', (width+320, height), '#101c2d')
    image.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(image)
    route, history = np.array(state['route']), np.array(state['history'])
    low, high = route[:, :2].min(0)-.12, route[:, :2].max(0)+.12
    def point(p):
        q = (np.array(p)[:2]-low)/(high-low)
        return (width+25+int(q[0]*270), 230-int(q[1]*190))
    draw.text((width+20, 12), state['trajectory_label'].upper()+' / VICON XY', fill='#9db1c8')
    draw.line([point(p) for p in route], fill='#3b526d', width=3)
    if len(history) > 1:
        draw.line([point(p) for p in history], fill='#58ddbf', width=2)
    x, y = point(state['position'])
    draw.ellipse((x-5, y-5, x+5, y+5), fill='#fbbf68')
    p = state['position']
    text = (f"Time {state['time_s']:.1f}s / {state['duration_s']:.1f}s\n"
            f"XYZ {p[0]:+.2f}  {p[1]:+.2f}  {p[2]:.2f} m\n"
            f"Speed {state['speed_m_s']:.2f} m/s\n"
            f"Tracking error {state['tracking_error_m']*100:.1f} cm\n"
            'Prototype dynamics / candidate alignment')
    draw.text((width+20, 250), text, fill='#dce8f5', spacing=6)
    return np.asarray(image)


class App:
    def __init__(self, config, renderer, output):
        self.sim = Simulator(config)
        self.renderer, self.output = renderer, output
        self.scene_preview = renderer.scene_preview() if renderer else dict(points_m=[], kind='no scene renderer')
        self.lock, self.commands, self.stop = threading.Lock(), queue.Queue(), threading.Event()
        self.frame, self.depth = jpeg(placeholder()), jpeg(placeholder())
        self.rgba, self.depth_rgba = encode_rgba(placeholder()), encode_rgba(placeholder())
        self.latest, self.latest_csv = self.sim.snapshot(), b''
        self.error, self.render_ms, self.frames = None, 0., 0
        self.control_hz = int(config['control_hz'])
        self.save_episode()
        self.render_every = round(self.control_hz/config['render_hz'])
        if self.render_every < 1 or not np.isclose(self.render_every*config['render_hz'], self.control_hz):
            raise ValueError('control_hz must be an integer multiple of render_hz')

    def save_episode(self, output=None):
        output = self.output if output is None else output
        write_telemetry(output/'telemetry.csv', self.sim.telemetry)
        (output/'episode_parameters.json').write_text(json.dumps(self.sim.episode_parameters, indent=2)+'\n')
        with (output/'demonstrations.jsonl').open('w') as f:
            for row in self.sim.demonstrations:
                f.write(json.dumps(row, allow_nan=False)+'\n')

    def publish(self, render=False):
        if render:
            before = time.monotonic()
            if self.renderer:
                rgb, depth = self.renderer.render(self.sim.state)
                depth_image = depth_rgb(depth)
            else:
                rgb, depth_image = placeholder(), placeholder()
            frames = jpeg(rgb), jpeg(depth_image)
            rgba_frames = encode_rgba(rgb), encode_rgba(depth_image)
            self.render_ms = (time.monotonic()-before)*1000
            self.frames += 1
        snapshot = self.sim.snapshot()
        snapshot.update(render_ms=self.render_ms, rendered_frames=self.frames,
                        renderer='Native 3DGUT' if self.renderer else 'Dynamics only',
                        gpu=self.renderer.device_name if self.renderer else None,
                        error=self.error, output=str(self.output))
        stream = io.StringIO()
        if self.sim.telemetry:
            writer = csv.DictWriter(stream, fieldnames=list(self.sim.telemetry[0]))
            writer.writeheader()
            writer.writerows(self.sim.telemetry)
        with self.lock:
            self.latest, self.latest_csv = snapshot, stream.getvalue().encode()
            if render:
                self.frame, self.depth = frames
                self.rgba, self.depth_rgba = rgba_frames

    def worker(self):
        tick, dirty = 0, True
        try:
            while not self.stop.is_set():
                start = time.monotonic()
                while not self.commands.empty():
                    try:
                        command = self.commands.get_nowait()
                        event_time = self.sim.time
                        if command.get('action') in ('reset', 'route', 'configure') and self.sim.telemetry:
                            previous = self.output/f'episode_{self.sim.episode:04d}'
                            previous.mkdir(exist_ok=True)
                            self.save_episode(previous)
                        self.sim.command(command)
                        if command.get('action') in ('reset', 'route', 'configure'):
                            self.save_episode()
                        with (self.output/'events.jsonl').open('a') as f:
                            f.write(json.dumps(dict(time_s=event_time, command=command,
                                                    effective_task=self.sim.route.config,
                                                    episode_parameters=self.sim.episode_parameters))+'\n')
                        self.error = None
                        dirty = True
                    except (ValueError, KeyError, TypeError) as error:
                        self.error = str(error)
                if not self.sim.paused:
                    self.sim.advance()
                    tick += 1
                    dirty = dirty or tick % self.render_every == 0
                self.publish(render=dirty)
                dirty = False
                self.stop.wait(max(0., self.sim.control_dt-(time.monotonic()-start)))
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            self.sim.paused = True
            self.publish()
            import traceback
            traceback.print_exc()
        finally:
            self.save_episode()
            (self.output/'final_config.json').write_text(json.dumps(self.sim.config, indent=2)+'\n')


def serve(app, port):
    page = (Path(__file__).parent/'viewer.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, kind='application/json', code=200, download=False):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="simulation_telemetry.csv"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            with app.lock:
                if self.path == '/':
                    return self.send(page, 'text/html; charset=utf-8')
                if self.path.startswith('/api/state'):
                    return self.send(json.dumps(app.latest, allow_nan=False).encode())
                if self.path == '/api/scene':
                    return self.send(json.dumps(app.scene_preview, allow_nan=False).encode())
                if self.path.startswith('/frame.jpg'):
                    return self.send(app.depth if 'mode=depth' in self.path else app.frame, 'image/jpeg')
                if self.path.startswith('/frame.rgba'):
                    return self.send(app.depth_rgba if 'mode=depth' in self.path else app.rgba, 'application/octet-stream')
                if self.path == '/api/export':
                    return self.send(app.latest_csv, 'text/csv', download=True)
            self.send(b'{"error":"Not found"}', code=404)

        def do_POST(self):
            if self.path != '/api/control':
                return self.send(b'{"error":"Not found"}', code=404)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8192:
                    raise ValueError('Invalid command size')
                command = json.loads(self.rfile.read(size))
                if not isinstance(command, dict) or not isinstance(command.get('action'), str):
                    raise ValueError('Expected a command object')
                app.commands.put(command)
                self.send(b'{"queued":true}')
            except (ValueError, TypeError) as error:
                self.send(json.dumps(dict(error=str(error))).encode(), code=400)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    worker = threading.Thread(target=app.worker, daemon=True)
    worker.start()
    print(f'SIMULATOR_URL=http://127.0.0.1:{port}', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        worker.join(timeout=15)
        server.server_close()


def headless(app, duration, record):
    sim, renderer = app.sim, app.renderer
    sim.paused = False
    duration = duration or sim.route.duration+(0. if sim.route.continuous else 2.)
    count = int(np.ceil(duration/sim.control_dt))
    video = None
    recorder = None
    if sim.config.get('record_rgb_actions', False):
        if renderer is None or sim.controller_type not in ('nmpc','mpc_geometric'):
            raise ValueError('--dataset requires Gaussian rendering and a thrust/body-rate expert')
        from Workflow.Simulation.dataset import DemonstrationRecorder
        recorder = DemonstrationRecorder(app.output, renderer)
    if record:
        width = int(sim.config['rendering']['width'])+320
        height = int(sim.config['rendering']['height'])
        video = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
            '-r', str(sim.config['render_hz']), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p', str(app.output/'simulation.mp4')],
            stdin=subprocess.PIPE)
    before = time.monotonic()
    try:
        for i in range(count):
            observation = recorder.capture(sim) if recorder else None
            sim.advance()
            if recorder:
                if sim.telemetry[-1]['error_m'] > sim.config.get('validation', {}).get('max_position_error_m', .2):
                    raise RuntimeError('Tracking exceeded the dataset development acceptance limit')
                recorder.write(sim.demonstrations[-1], observation)
            if i % app.render_every == 0:
                rgb = renderer.render(sim.state)[0] if renderer else placeholder(
                    sim.config['rendering']['width'], sim.config['rendering']['height'])
                if video:
                    video.stdin.write(draw_record_frame(rgb, sim.snapshot()).tobytes())
                Image.fromarray(rgb).save(app.output/'latest_frame.jpg', quality=92)
            if i % app.control_hz == 0:
                print(f"Simulated {sim.time:.1f}/{duration:.1f}s; error={sim.snapshot()['tracking_error_m']:.4f}m", flush=True)
    finally:
        if recorder:
            recorder.close()
        if video:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError('Video export failed')
        app.save_episode()
    errors = np.array([r['error_m'] for r in sim.telemetry])
    summary = dict(simulated_seconds=sim.time, wall_seconds=time.monotonic()-before,
                   position_error_rms_m=float(np.sqrt(np.mean(errors**2))),
                   position_error_max_m=float(errors.max()), final=sim.snapshot())
    if sim.is_expert:
        solve_times = [r['mpc_solve_ms'] for r in sim.telemetry]
        summary.update(mpc_mean_solve_ms=float(np.mean(solve_times)), mpc_max_solve_ms=float(np.max(solve_times)),
                       mpc_failed_solves=sum(r['mpc_status'] != 'solved' for r in sim.telemetry))
    (app.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'final'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'Workflow/Simulation/configs/quadrotor_nmpc.json')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float, default=0., help='0 runs the full task plus settling time')
    parser.add_argument('--record', action='store_true', help='Export an MP4 during headless mode')
    parser.add_argument('--dataset', action='store_true', help='Record synchronized pre-action RGB/thrust/body-rate labels')
    parser.add_argument('--pose-render-service', action='store_true', help='CUDA pose-to-RGB service for the native Isaac backend')
    parser.add_argument('--no-render', action='store_true', help='CPU dynamics only; no Gaussian image')
    parser.add_argument('--dynamics', choices=['single_integrator', 'double_integrator', 'quadrotor'],
                        help='Override the configured vehicle model')
    parser.add_argument('--quadrotor-plant', choices=['rate_response', 'rigid_body','upstream_motor'],
                        help='Custom quadrotor rotation: response approximation or force/torque rigid body')
    parser.add_argument('--yaw-mode', choices=['constant', 'velocity_aligned'], help='Reference heading mode')
    parser.add_argument('--controller', choices=['baseline', 'mpc', 'nmpc','mpc_geometric'], help='Reference tracking controller')
    parser.add_argument('--randomize', action='store_true', help='Sample task and plant settings on each reset')
    parser.add_argument('--seed', type=int, help='Randomization seed')
    parser.add_argument('--episodes', type=int, default=1, help='Number of headless episodes, with seeds seed+i')
    parser.add_argument('--trajectory', choices=['m', 'horizontal_circle', 'lissajous_figure8'],
                        help='Override the configured reference shape')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.duration < 0 or not np.isfinite(args.duration):
        parser.error('Duration must be finite and nonnegative')
    if args.dataset and (not args.headless or args.no_render):
        parser.error('--dataset requires --headless and CUDA rendering')
    if args.pose_render_service and (args.headless or args.no_render):
        parser.error('--pose-render-service requires CUDA and runs as a server, without --headless')
    if args.episodes < 1 or (args.episodes != 1 and not args.headless):
        parser.error('--episodes must be positive and multiple episodes require --headless')
    if not args.headless:
        if not 1 <= args.port <= 65535:
            parser.error('--port must be in [1, 65535]')
        # Check before creating outputs or initializing the expensive GPU renderer.
        # serve() still performs the definitive bind after initialization.
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(('127.0.0.1', args.port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                parser.error(f'Port {args.port} is already in use. Reuse the running simulator '
                             '(browser/Isaac DASC panel), or stop that simulator before relaunching. '
                             'No GPU model was loaded by this launch.')
            parser.error(f'Cannot bind viewer port {args.port}: {error}')
    config = json.loads(args.config.read_text())
    if args.dynamics:
        config['dynamics_model'] = args.dynamics
    if args.quadrotor_plant:
        config['vehicle']['rotation_model'] = args.quadrotor_plant
        if config['dynamics_model'] != 'quadrotor':
            parser.error('--quadrotor-plant requires quadrotor dynamics')
    if args.yaw_mode:
        config['task']['yaw_mode'] = args.yaw_mode
    if args.trajectory:
        config['task']['trajectory_type'] = args.trajectory
    if args.controller:
        config['controller']['type'] = args.controller
    if args.randomize:
        config.setdefault('randomization', {})['enabled'] = True
    if args.seed is not None:
        config['seed'] = args.seed
    config['record_rgb_actions'] = args.dataset
    if args.dataset and config['controller'].get('type') not in ('nmpc','mpc_geometric'):
        parser.error('--dataset requires nmpc or mpc_geometric with quadrotor dynamics')
    if config.get('controller', {}).get('type', 'baseline') == 'mpc' and config.get('dynamics_model') != 'double_integrator':
        parser.error('MPC requires double_integrator. Use --controller baseline for other dynamics.')
    if config.get('controller', {}).get('type') == 'nmpc' and config.get('dynamics_model') != 'quadrotor':
        parser.error('NMPC requires --dynamics quadrotor.')
    if config.get('controller',{}).get('type')=='mpc_geometric' and (
            config.get('dynamics_model')!='quadrotor' or config['vehicle'].get('rotation_model')!='upstream_motor'):
        parser.error('mpc_geometric requires quadrotor dynamics and upstream_motor plant; use configs/motor_mpc.json.')
    if config.get('randomization', {}).get('enabled') and config.get('dynamics_model') not in ('double_integrator', 'quadrotor'):
        parser.error('Randomization supports double_integrator or quadrotor.')
    output = args.output or ROOT/'data/scene_20261025_200034/simulation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    provenance = dict(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                       no_render=args.no_render, argv=sys.argv, controller=config['controller'].get('type', 'baseline'),
                       code_revision=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                       notes=config['limitations'])
    provenance['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT/'Workflow/Simulation').rglob('*'))
                                  if path.is_file() and path.suffix in ('.py', '.html', '.sh', '.json')}
    alignment = ROOT/config['rendering']['alignment']
    provenance['alignment_sha256'] = hashlib.sha256(alignment.read_bytes()).hexdigest() if alignment.exists() else None
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'OUTPUT={output}', flush=True)
    renderer = None
    if not args.no_render:
        try:
            from Workflow.Simulation.rendering.threedgut import GaussianRenderer
            renderer = GaussianRenderer(config['rendering'], ROOT)
            print(f'Renderer ready: {renderer.device_name}', flush=True)
        except Exception as error:
            (output/'failure.json').write_text(json.dumps(dict(phase='renderer_startup', error=str(error),
                                                              type=type(error).__name__), indent=2)+'\n')
            raise
    runs = []
    if args.pose_render_service:
        from Workflow.Simulation.dataset import serve_pose_renderer
        serve_pose_renderer(renderer, args.port, config)
        return
    for i in range(args.episodes):
        episode_config = json.loads(json.dumps(config))
        episode_config['seed'] = int(config.get('seed', 42))+i
        episode_output = output if args.episodes == 1 else output/f'episode_{i+1:04d}'
        episode_output.mkdir(exist_ok=True)
        (episode_output/'config.json').write_text(json.dumps(episode_config, indent=2)+'\n')
        try:
            app = App(episode_config, renderer, episode_output)
            if args.headless:
                headless(app, args.duration, args.record)
            else:
                serve(app, args.port)
        except Exception as error:
            (episode_output/'failure.json').write_text(json.dumps(dict(error=str(error), type=type(error).__name__), indent=2)+'\n')
            raise
        runs.append(dict(output=str(episode_output), seed=episode_config['seed']))
    (output/'episodes.json').write_text(json.dumps(runs, indent=2)+'\n')


if __name__ == '__main__':
    main()
