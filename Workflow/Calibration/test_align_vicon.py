import unittest

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from Workflow.Calibration.align_vicon import Alignment, fit_similarity, transform_dict, vicon_body_to_gs


class AlignmentTests(unittest.TestCase):
    def test_similarity_direction_and_scale(self):
        rng = np.random.default_rng(42)
        source = rng.normal(size=(30, 3))
        r = Rotation.from_euler('xyz', [.2, -.3, .8])
        target = .75*r.apply(source)+[1., -2., .3]
        scale, fitted, translation = fit_similarity(source, target)
        self.assertAlmostEqual(scale, .75, places=10)
        np.testing.assert_allclose(fitted.as_matrix(), r.as_matrix(), atol=1e-10)
        np.testing.assert_allclose(translation, [1., -2., .3], atol=1e-10)

    def test_body_camera_conversion_and_orientation(self):
        x = np.r_[np.log(.75), [.2, -.3, .8], [1., -2., .3], [-1., .2, .5], [.04, -.01, .1], -.01]
        calibration = transform_dict(x)
        rw, mount = Rotation.from_rotvec(x[1:4]), Rotation.from_rotvec(x[7:10])
        original_p = np.array([2., 3., -1.])
        original_r = Rotation.from_euler('xyz', [.1, .3, -.2])
        body_r = rw*original_r*mount.inv()
        body_p = .75*rw.apply(original_p)+x[4:7]-body_r.apply(x[10:13])
        restored_p, restored_r = vicon_body_to_gs(body_p, body_r, calibration)
        np.testing.assert_allclose(restored_p, original_p, atol=1e-10)
        np.testing.assert_allclose(restored_r.as_matrix(), original_r.as_matrix(), atol=1e-10)
        np.testing.assert_allclose(np.array(calibration['vicon_to_gs']['position_matrix_4x4']) @
                                   calibration['gs_to_vicon']['position_matrix_4x4'], np.eye(4), atol=1e-10)

    def test_joint_fit_recovers_mount_and_time_offset(self):
        vt = np.linspace(-1, 21, 2201)
        vp = np.column_stack([np.sin(vt*.7), np.cos(vt*.4), .1*vt+np.sin(vt*.2)])
        vr = Rotation.from_euler('xyz', np.column_stack([.25*np.sin(vt*.5), .2*np.cos(vt*.8), .15*vt]))
        t = np.linspace(0, 20, 61)
        truth = np.r_[np.log(.75), [.2, -.3, .8], [1., -2., .3], [-1., .2, .5], [.04, -.01, .1], -.013]
        rb = Slerp(vt, vr)(t+truth[13])
        pb = np.column_stack([np.interp(t+truth[13], vt, vp[:, i]) for i in range(3)])
        rw = Rotation.from_rotvec(truth[1:4])
        camera = rw.inv().apply(pb+rb.apply(truth[10:13])-truth[4:7])/.75
        camera_r = rw.inv()*rb*Rotation.from_rotvec(truth[7:10])
        data = dict(time=t, centers=camera, camera_rotation=camera_r, vtime=vt,
                    body_position=vp, body_rotation=vr)
        problem = Alignment(data)
        fit = problem.fit(np.ones(len(t), dtype=bool))
        self.assertAlmostEqual(np.exp(fit.x[0]), .75, places=5)
        self.assertAlmostEqual(fit.x[13], -.013, places=5)
        np.testing.assert_allclose(fit.x[10:13], truth[10:13], atol=1e-5)
        error, angle = problem.errors(fit.x, np.ones(len(t), dtype=bool))
        np.testing.assert_allclose(error, 0, atol=1e-6)
        np.testing.assert_allclose(angle, 0, atol=1e-6)

    def test_rejects_collinear_trajectory(self):
        source = np.column_stack([np.arange(20), np.zeros(20), np.zeros(20)])
        with self.assertRaises(ValueError):
            fit_similarity(source, source)


if __name__ == '__main__':
    unittest.main()
