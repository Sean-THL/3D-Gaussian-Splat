"""Render moving virtual camera poses with the existing native 3DGUT checkpoint."""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

from Workflow.Simulation.rendering.assets import resolve_training_asset


class GaussianRenderer:
    def __init__(self, config, root):
        sys.path.insert(0, str(Path(config['tool_dir']).expanduser().resolve()))
        import torch
        try:
            torch.cuda.init()
        except (RuntimeError, AssertionError) as error:
            raise RuntimeError(
                'CUDA initialization failed before Gaussian rendering. '
                'Check nvidia-smi and CUDA initialization in this Python environment; '
                'nvidia-smi alone does not verify CUDA computation. '
                'Possible causes include driver/device state, device visibility, or process isolation. '
                'To run the MPC and browser demo on CPU, use '
                'python3 Workflow/Simulation/run.py --no-render --controller mpc '
                '--dynamics double_integrator --trajectory lissajous_figure8 '
                '--yaw-mode velocity_aligned. '
                f'Original CUDA error: {error}'
            ) from error
        from threedgrut.datasets.protocols import Batch
        from threedgrut.datasets.utils import read_colmap_intrinsics_binary
        from threedgrut.model.model import MixtureOfGaussians
        from Workflow.Calibration.align_vicon import vicon_body_to_gs

        self.torch, self.Batch, self.to_gs = torch, Batch, vicon_body_to_gs
        calibration_path = root/config['alignment']
        self.calibration_path = calibration_path
        self.calibration = json.loads(calibration_path.read_text())
        # Optional independently measured mount can replace the candidate estimate.
        if config.get('mount_override'):
            self.calibration['camera_to_tracked_body'] = config['mount_override']
        self.checkpoint = resolve_training_asset(root, config['checkpoint'])
        checkpoint = torch.load(self.checkpoint, map_location='cuda', weights_only=False)
        conf = checkpoint['config']
        if conf.render.method != '3dgut' or conf.dataset.normalize_world_space:
            raise ValueError('Expected a 3DGUT checkpoint in original reconstruction coordinates')
        self.model = MixtureOfGaussians(conf)
        self.model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        del checkpoint
        self.model.build_acc()
        cameras = read_colmap_intrinsics_binary(str(resolve_training_asset(root, config['camera_model'])))
        if len(cameras) != 1:
            raise ValueError('Prototype expects a single calibrated pinhole camera')
        camera = next(iter(cameras.values()))
        if camera.model != 'PINHOLE':
            raise ValueError('Use the undistorted PINHOLE camera model')
        self.width, self.height = int(config['width']), int(config['height'])
        fx, fy, cx, cy = np.array(camera.params)*[
            self.width/camera.width, self.height/camera.height,
            self.width/camera.width, self.height/camera.height]
        u, v = np.meshgrid(np.arange(self.width)+.5, np.arange(self.height)+.5)
        rays = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones_like(u)], -1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        self.ray_z = rays[..., 2].astype('float32')
        self.rays = torch.tensor(rays[None], dtype=torch.float32, device='cuda')
        self.origins = torch.zeros_like(self.rays)
        self.intrinsics = dict(resolution=np.array([self.width, self.height], dtype=np.uint64),
                               shutter_type='GLOBAL', principal_point=np.array([cx, cy], dtype=np.float32),
                               focal_length=np.array([fx, fy], dtype=np.float32),
                               radial_coeffs=np.zeros(6, dtype=np.float32),
                               tangential_coeffs=np.zeros(2, dtype=np.float32),
                               thin_prism_coeffs=np.zeros(4, dtype=np.float32))
        self.device_name = torch.cuda.get_device_name(0)

    @property
    def metadata(self):
        if not hasattr(self, '_metadata'):
            digest = hashlib.sha256()
            with Path(self.checkpoint).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            self._metadata = dict(renderer='native 3DGUT', gpu=self.device_name,
                checkpoint=str(self.checkpoint), checkpoint_sha256=digest.hexdigest(),
                alignment=str(self.calibration_path),
                alignment_sha256=hashlib.sha256(self.calibration_path.read_bytes()).hexdigest(),
                camera_to_body=self.calibration['camera_to_tracked_body'], gs_to_vicon=self.calibration['gs_to_vicon'],
                intrinsics={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.intrinsics.items()},
                width=self.width, height=self.height)
        return self._metadata

    def scene_preview(self, count=6000):
        """Decimated Gaussian centers in Vicon metres, not surfaces/colliders."""
        centers = self.model.get_positions()
        stride = max(1, int(np.ceil(len(centers)/count)))
        points = centers[::stride].detach().cpu().numpy()
        alignment = self.calibration['gs_to_vicon']
        points = alignment['scale_m_per_gs_unit']*(points@np.asarray(alignment['rotation_matrix']).T)
        points += np.asarray(alignment['translation_m'])
        points = points[np.isfinite(points).all(axis=1) & (np.abs(points) < 50.).all(axis=1)]
        return dict(points_m=points.tolist(), kind='decimated Gaussian centers; visual proxy, not collision geometry')

    def render(self, state):
        torch = self.torch
        position, orientation = self.to_gs(state.position, state.orientation, self.calibration)
        pose = np.eye(4, dtype='float32')
        pose[:3, :3], pose[:3, 3] = orientation.as_matrix(), position
        batch = self.Batch(rays_ori=self.origins, rays_dir=self.rays,
                           T_to_world=torch.tensor(pose[None], device='cuda'),
                           intrinsics_OpenCVPinholeCameraModelParameters=self.intrinsics)
        with torch.inference_mode():
            result = self.model(batch)
        rgb = result['pred_features'][0].detach().cpu().numpy().clip(0, 1)
        opacity = result['pred_opacity'][0].detach().cpu().numpy().reshape(self.height, self.width)
        distance = result['pred_dist'][0].detach().cpu().numpy().reshape(self.height, self.width)
        depth = distance/np.maximum(opacity, 1e-8)*self.ray_z*self.calibration['gs_to_vicon']['scale_m_per_gs_unit']
        depth[(opacity < .1) | ~np.isfinite(depth) | (depth <= 0)] = np.nan
        return (rgb*255).round().astype('uint8'), depth
