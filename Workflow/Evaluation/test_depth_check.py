import tempfile
import unittest
from pathlib import Path

import numpy as np

from Workflow.Evaluation.depth_check import expected_z, fit_scale, summarize


class DepthCheckTests(unittest.TestCase):
    def test_opacity_normalization_and_range_to_z(self):
        # Same hit distance at different opacity, with an oblique second ray.
        result = expected_z(np.array([4., 2., 0.]), np.array([1., .5, 0.]),
                            np.array([1., .8, 1.]), .75)
        np.testing.assert_allclose(result[:2], [3., 2.4])
        self.assertTrue(np.isnan(result[2]))

    def test_scale_excludes_test_views_and_bad_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            observations = np.array([(i+.5, .5, i) for i in range(21)],
                                    dtype=[('x', '<f8'), ('y', '<f8'), ('id', '<i8')])
            points = {i: [0., 0., 1.] for i in range(21)}
            images, references = [], {}
            for i, (depth, dt) in enumerate([(999., 0.), (2., 0.), (100., 41.)]):
                name = f'{i:06d}_{i}.png'
                np.save(root/f'{i}.npy', np.full((1, 21), depth))
                images.append(dict(name=name, q=np.array([1., 0., 0., 0.]),
                                   t=np.zeros(3), points=observations))
                references[str(i)] = dict(depth_filename=f'{i}.npy', depth_dt_ms=dt)
            scale, estimates = fit_scale(images, points, references, root, 40.)
            self.assertEqual(scale, 2.)
            self.assertEqual([e['image'] for e in estimates], ['000001_1.png'])

    def test_error_summary_includes_signed_bias_and_tails(self):
        result = summarize(np.array([-.2, .1, .9]))
        self.assertAlmostEqual(result['mae_m'], .4)
        self.assertAlmostEqual(result['signed_mean_m'], .8/3)
        self.assertEqual(result['pixels'], 3)
        self.assertEqual(summarize(np.array([])), {'pixels': 0})


if __name__ == '__main__':
    unittest.main()
