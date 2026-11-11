"""Reproducible RGB appearance augmentation; does not alter poses or geometry."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from Workflow.Simulation.randomization import sample_uniform


class VisualRandomizer:
    def __init__(self, settings, seed, episode_randomization_enabled):
        self.enabled = bool(settings.get('enabled', False) and episode_randomization_enabled)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 73129]))
        self.metadata = dict(enabled=self.enabled, seed=int(seed),
            kind='RGB appearance augmentation, not physical relighting or geometry changes')
        if not self.enabled:
            return
        defaults = dict(brightness=[.8, 1.2], contrast=[.85, 1.15], saturation=[.8, 1.2],
                        gamma=[.9, 1.1], blur_sigma_px=[0., .8], noise_std_uint8=[0., 3.])
        for name, default in defaults.items():
            value = float(sample_uniform(settings, self.rng, name, default))
            if value < 0 or (name in ('brightness', 'contrast', 'saturation', 'gamma') and value <= 0):
                raise ValueError(f'Invalid visual parameter: {name}')
            self.metadata[name] = value
        gains = sample_uniform(settings, self.rng, 'channel_gain', [.95, 1.05], size=3)
        if min(gains) <= 0:
            raise ValueError('Visual channel gains must be positive')
        self.metadata['channel_gain'] = gains.tolist()

    def apply(self, rgb):
        if not self.enabled:
            return rgb
        image = Image.fromarray(rgb)
        for name, enhancer in [('brightness', ImageEnhance.Brightness),
                               ('contrast', ImageEnhance.Contrast), ('saturation', ImageEnhance.Color)]:
            image = enhancer(image).enhance(self.metadata[name])
        if self.metadata['blur_sigma_px'] > 0:
            image = image.filter(ImageFilter.GaussianBlur(self.metadata['blur_sigma_px']))
        pixels = np.asarray(image).astype(np.float32)/255.
        pixels = np.power(pixels, self.metadata['gamma'])*np.asarray(self.metadata['channel_gain'])
        pixels *= 255.
        if self.metadata['noise_std_uint8'] > 0:
            pixels += self.rng.normal(0., self.metadata['noise_std_uint8'], pixels.shape)
        return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
