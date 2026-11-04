"""Render moving virtual camera poses with the existing native 3DGUT checkpoint."""
import json
import sys
from pathlib import Path

import numpy as np

from Workflow.Simulation.rendering.assets import resolve_training_asset


class GaussianRenderer:
    def __init__(self, config, root):
        sys.path.insert(0, str(Path(config['tool_dir']).expanduser().resolve()))
        import torch
        from threedgrut.datasets.protocols import Batch
        from threedgrut.datasets.utils import read_colmap_intrinsics_binary
        from threedgrut.model.model import MixtureOfGaussians
        from Workflow.Calibration.align_vicon import vicon_body_to_gs

        self.torch, self.Batch, self.to_gs = torch, Batch, vicon_body_to_gs
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Launch with run.sh outside the restricted sandbox.')
        calibration_path = root/config['alignment']
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
