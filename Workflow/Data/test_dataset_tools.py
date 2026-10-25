"""Verify pixel geometry, metric depth, and frame identity across variants."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import dataset_tools as tools


class GeometryTests(unittest.TestCase):
    def test_crop_resize_preserves_projected_ray(self):
        calibration = {'model': 'FULL_OPENCV', 'width': 1280, 'height': 720,
                       'params': [643.6, 642.9, 644.4, 364.9, -.05, .06, -.001, .002, -.02, 0., 0., 0.]}
        t = tools.image_transform(1280, 720, [280, 0, 720, 720], 448, 448)
        changed = tools.transform_calibration(calibration, t)
        def project(c, x, y):
            fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = c['params']
            r2 = x*x + y*y
            radial = (1+k1*r2+k2*r2*r2+k3*r2**3)/(1+k4*r2+k5*r2*r2+k6*r2**3)
            return np.array([fx*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+cx,
                             fy*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+cy])
        for x, y in [(-.2, .1), (.3, -.15), (0., 0.)]:
            old = project(calibration, x, y)
            expected = (old - [280, 0]) * (448/720)
            np.testing.assert_allclose(project(changed, x, y), expected, atol=1e-10)
        self.assertEqual(changed['params'][4:], calibration['params'][4:])
        self.assertEqual(calibration['width'], 1280)

    def test_half_pixel_conversion_matches_opencv_linear_sampling(self):
        ramp = np.tile(np.arange(8, dtype=np.float32), (4, 1))
        resized = cv2.resize(ramp, (4, 2), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_allclose(resized[0], [.5, 2.5, 4.5, 6.5])
        # OpenCV centre x=3 maps to x'=s*(3+.5)-.5=1.25; COLMAP c=3.5 maps to 1.75.
        c = {'model': 'FULL_OPENCV', 'width': 8, 'height': 4, 'params': [4.,4.,3.5,2., *([0.]*8)]}
        result = tools.transform_calibration(c, tools.image_transform(8, 4, None, 4, 2))
        self.assertEqual(result['params'][2] - .5, 1.25)

    def test_invalid_crop_and_stretch_rejected(self):
        with self.assertRaises(ValueError):
            tools.image_transform(1280,720,[1000,0,720,720],448,448)
        with self.assertRaisesRegex(ValueError, 'stretch'):
            tools.image_transform(1280,720,None,448,448)
        self.assertEqual(tools.image_transform(1280,720,None,640,None)['height'],360)

    def test_timestamp_matching_does_not_exceed_tolerance(self):
        rows = [{'id':'a'}, {'id':'b'}]
        self.assertEqual(tools.nearest_depth(109, [100,120], rows, 9), rows[0])
        self.assertIsNone(tools.nearest_depth(109, [100,120], rows, 8))


class DatasetIntegrationTests(unittest.TestCase):
    def test_selection_variant_keeps_splits_metres_and_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export = root/'export'
            (export/'rgb').mkdir(parents=True)
            (export/'depth').mkdir()
            topic = '/camera/color/camera_info'
            camera = {'width':8,'height':6,'k':[4.,0.,3.5,0.,4.,2.5,0.,0.,1.],
                      'd':[0.]*5,'distortion_model':'plumb_bob'}
            (export/'camera.jsonl').write_text(json.dumps({'message':camera})+'\n')
            tools.save(export/'manifest.json',{'status':'complete','rgb_topic':'/camera/color/image_raw/compressed',
                'depth_topic':'/camera/aligned_depth_to_color/image_raw/compressedDepth',
                'files':{topic:{'jsonl':'camera.jsonl'}}})
            rgb_rows, depth_rows = [], []
            for i in range(4):
                stamp=2_000_000_000+i*100_000_000
                filename=f'{i:06d}_{stamp}.png'
                image=np.full((6,8,3),40+i,dtype=np.uint8)
                cv2.imwrite(str(export/'rgb'/filename),image)
                depth=np.full((6,8),3.25,dtype=np.float32)
                np.save(export/'depth'/f'{i}.npy',depth)
                rgb_rows.append({'filename':filename,'bag_timestamp_ns':stamp,'bag_elapsed_s':i*.1,
                                 'header_timestamp_ns':stamp,'width':8,'height':6})
                depth_rows.append({'metres_npy':f'{i}.npy','bag_timestamp_ns':stamp,'header_timestamp_ns':stamp})
            for directory,rows in [('rgb',rgb_rows),('depth',depth_rows)]:
                with (export/directory/'timestamps.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            digest=hashlib.sha256((export/'rgb'/rgb_rows[0]['filename']).read_bytes()).digest()
            selected=root/'selected'
            with patch.object(tools,'new_dataset_name',return_value=selected):
                tools.selection(argparse.Namespace(export=export,name='test',start=0,end=None,rate=0,
                    test_every=2,exclude_file=None,camera_info_topic=topic,with_depth=True,
                    require_depth=True,depth_clock='header',max_depth_dt=.04))
            derived=root/'derived'
            with patch.object(tools,'new_dataset_name',return_value=derived):
                tools.variant(argparse.Namespace(source=selected,name='small',crop=[2,1,4,4],
                    center_crop=None,width=2,height=2,allow_stretch=False))
            _, frames, calibration=tools.validate_dataset(derived)
            self.assertEqual((derived/'splits.json').read_text(),(selected/'splits.json').read_text())
            self.assertEqual(calibration['params'][:4],[2.,2.,1.,1.])
            self.assertEqual(len(frames),4)
            for frame in frames:
                np.testing.assert_array_equal(np.load(derived/'depth'/frame['depth_filename']),np.full((2,2),3.25,dtype=np.float32))
            self.assertEqual(hashlib.sha256((export/'rgb'/rgb_rows[0]['filename']).read_bytes()).digest(),digest)
            (derived/'images'/frames[0]['filename']).unlink()
            with self.assertRaisesRegex(ValueError,'images changed'):
                tools.validate_dataset(derived)


if __name__=='__main__':
    unittest.main()
