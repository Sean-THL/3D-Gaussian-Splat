"""Synchronized pre-action RGB/state/reference/expert-action recording.

The pose render service keeps native 3DGUT CUDA code outside Isaac Python.
"""
import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial.transform import Rotation

from Workflow.Simulation.figs.dynamics import State
from Workflow.Simulation.isaac_sim.bridge_client import SimulatorClient, decode_rgba, encode_rgba


class PoseRenderClient:
    def __init__(self, url):
        self.url = SimulatorClient(url).url  # Enforce localhost-only transport.
        with urlopen(self.url+'/health', timeout=10) as response:
            self.metadata = json.load(response)

    def render(self, state):
        payload = json.dumps(dict(position_m=state.position.tolist(), quaternion_xyzw=state.orientation.as_quat().tolist())).encode()
        request = Request(self.url+'/render', data=payload, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=30) as response:
            width, height, pixels = decode_rgba(response.read())
        return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy(), None


class DemonstrationRecorder:
    def __init__(self, output, renderer):
        self.output = Path(output)
        self.images = self.output/'rgb'
        self.images.mkdir(exist_ok=True)
        self.renderer = renderer
        self.count = 0
        self.visual_randomizer = None
        self.image_format = 'png'
        self.jpeg_quality = 95
        self.stream = (self.output/'rgb_actions.jsonl').open('w')
        self.metadata = dict(schema_version=1, image_timing='pre-action, same state as expert decision',
            state_frame='Vicon world Z-up; quaternion body-to-world XYZW',
            action_frame='collective thrust along body +Z, body angular rates XYZ',
            renderer=getattr(renderer, 'metadata', {'kind': 'native local 3DGUT'}),
            camera_mount_status='candidate calibration; not independently measured')
        (self.output/'dataset_metadata.json').write_text(json.dumps(self.metadata, indent=2)+'\n')

    def capture(self, sim):
        from PIL import Image
        rgb, _ = self.renderer.render(sim.state)
        if self.count == 0:
            storage = sim.config.get('dataset_storage', {})
            self.image_format = storage.get('image_format', 'png').lower()
            self.jpeg_quality = int(storage.get('jpeg_quality', 95))
            if self.image_format not in ('png', 'jpeg') or not 1 <= self.jpeg_quality <= 100:
                raise ValueError('Dataset image_format must be png/jpeg and JPEG quality 1–100')
            self.metadata['image_encoding'] = dict(format=self.image_format,
                jpeg_quality=self.jpeg_quality if self.image_format == 'jpeg' else None,
                lossy=self.image_format == 'jpeg')
            from Workflow.Simulation.visual_randomization import VisualRandomizer
            self.visual_randomizer = VisualRandomizer(sim.config.get('visual_randomization', {}),
                sim.config.get('seed', 42), sim.config.get('randomization', {}).get('enabled', False))
            self.metadata['visual_randomization'] = self.visual_randomizer.metadata
            self.metadata.update(control_dt_s=sim.control_dt, image_resolution_wh=[rgb.shape[1], rgb.shape[0]],
                action_limits=dict(max_thrust_n=sim.controller.max_thrust, max_body_rate_rad_s=sim.controller.max_rate),
                physics_backend=sim.snapshot().get('physics_backend', 'custom_quadrotor'))
            if sim.state.motor_speeds_rad_s is not None:
                self.metadata['motor_state']=dict(units='rad/s',count=4,
                    max_motor_speed_rad_s=sim.vehicle.parameters.max_motor_speed,
                    role='actual simulated rotor speeds; privileged diagnostic, not required student input')
            (self.output/'dataset_metadata.json').write_text(json.dumps(self.metadata, indent=2)+'\n')
        rgb = self.visual_randomizer.apply(rgb)
        extension = 'jpg' if self.image_format == 'jpeg' else 'png'
        relative = Path('rgb')/f'{self.count:06d}.{extension}'
        options = dict(quality=self.jpeg_quality, subsampling=0) if self.image_format == 'jpeg' else {}
        Image.fromarray(rgb).save(self.output/relative, **options)
        return dict(time_s=sim.time, position_m=sim.state.position.tolist(),
                    quaternion_xyzw=sim.state.orientation.as_quat().tolist(),
                    image=str(relative), image_sha256=hashlib.sha256((self.output/relative).read_bytes()).hexdigest())

    def write(self, demonstration, observation):
        if demonstration.get('action_type') != 'thrust_body_rates':
            raise ValueError('RGB training recorder requires thrust/body-rate expert labels')
        if observation['time_s'] != demonstration['time_s'] or not np.allclose(observation['position_m'], demonstration['position_m'], atol=1e-12, rtol=0):
            raise ValueError('RGB observation and expert decision are not synchronized')
        if not np.allclose(observation['quaternion_xyzw'], demonstration['quaternion_xyzw'], atol=1e-12, rtol=0):
            raise ValueError('Image orientation differs from expert observation')
        row = dict(demonstration)
        row.update(image=observation['image'], image_sha256=observation['image_sha256'], image_time_s=observation['time_s'])
        self.stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.stream.flush()
        self.count += 1

    def close(self):
        self.stream.close()
        (self.output/'dataset_count.json').write_text(json.dumps(dict(samples=self.count))+'\n')


def serve_pose_renderer(renderer, port, config):
    metadata = renderer.metadata
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def send(self, body, kind, code=200):
            self.send_response(code); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path != '/health':
                return self.send(b'{}', 'application/json', 404)
            self.send(json.dumps(metadata).encode(), 'application/json')
        def do_POST(self):
            try:
                if self.path != '/render':
                    return self.send(b'{}', 'application/json', 404)
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size < 2048:
                    raise ValueError('Invalid pose request size')
                data = json.loads(self.rfile.read(size))
                position = np.asarray(data['position_m'], dtype=float)
                quaternion = np.asarray(data['quaternion_xyzw'], dtype=float)
                if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(np.r_[position, quaternion]).all() or abs(np.linalg.norm(quaternion)-1.) > 1e-3:
                    raise ValueError('Expected finite XYZ and normalized XYZW quaternion')
                state = State(position, np.zeros(3), Rotation.from_quat(quaternion))
                rgb, _ = renderer.render(state)
                self.send(encode_rgba(rgb), 'application/octet-stream')
            except (ValueError, KeyError, TypeError) as error:
                self.send(json.dumps(dict(error=str(error))).encode(), 'application/json', 400)
            except Exception as error:
                self.send(json.dumps(dict(error=str(error))).encode(), 'application/json', 500)
    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f'POSE_RENDER_URL=http://127.0.0.1:{port}', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
