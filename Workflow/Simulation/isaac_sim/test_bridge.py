"""Frame transport validation, independent of Isaac/CUDA availability."""
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from Workflow.Simulation.isaac_sim.bridge_client import SimulatorClient, decode_rgba, encode_rgba


class BridgeTests(unittest.TestCase):
    def test_rgba_frame_preserves_pixels_and_rejects_corruption(self):
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        encoded = encode_rgba(rgb)
        width, height, pixels = decode_rgba(encoded)
        self.assertEqual((width, height), (3, 2))
        rgba = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
        np.testing.assert_array_equal(rgba[:, :, :3], rgb)
        np.testing.assert_array_equal(rgba[:, :, 3], 255)
        for invalid in (b'', encoded[:-1], encoded+b'extra'):
            with self.assertRaises(ValueError):
                decode_rgba(invalid)

    def test_client_state_control_and_image_mode(self):
        client = SimulatorClient()
        with patch('Workflow.Simulation.isaac_sim.bridge_client.urlopen', return_value=io.BytesIO(b'{"controller":"mpc"}')):
            self.assertEqual(client.get_state()['controller'], 'mpc')
        with patch('Workflow.Simulation.isaac_sim.bridge_client.urlopen', return_value=io.BytesIO(b'{"queued":true}')) as request:
            client.command('configure', task={'yaw_mode': 'constant'})
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload['task']['yaw_mode'], 'constant')
        pixels = encode_rgba(np.zeros((2, 3, 3), dtype=np.uint8))
        with patch('Workflow.Simulation.isaac_sim.bridge_client.urlopen', return_value=io.BytesIO(pixels)) as request:
            self.assertEqual(client.get_rgba(depth=True)[:2], (3, 2))
            self.assertTrue(request.call_args.args[0].endswith('?mode=depth'))
        with self.assertRaises(ValueError):
            SimulatorClient('http://example.com')


if __name__ == '__main__':
    unittest.main()
