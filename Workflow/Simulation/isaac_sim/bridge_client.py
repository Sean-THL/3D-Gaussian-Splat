"""HTTP client for the CUDA simulator; no Isaac or PyTorch dependency.

Keep simulation and rendering in the 3DGUT environment. Isaac consumes states
and RGBA pixels through this interface, never importing its CUDA extensions.
"""
import json
import struct
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class SimulatorClient:
    def __init__(self, url='http://127.0.0.1:8765', timeout=2.):
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1'):
            raise ValueError('Bridge requires a localhost HTTP simulator endpoint')
        self.url = url.rstrip('/')
        self.timeout = timeout

    def get_state(self):
        with urlopen(self.url+'/api/state', timeout=self.timeout) as response:
            return json.load(response)

    def command(self, action, **values):
        payload = json.dumps(dict(action=action, **values)).encode()
        request = Request(self.url+'/api/control', data=payload, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def get_scene(self):
        with urlopen(self.url+'/api/scene', timeout=self.timeout) as response:
            return json.load(response)

    def get_rgba(self, depth=False):
        suffix = '?mode=depth' if depth else ''
        with urlopen(self.url+'/frame.rgba'+suffix, timeout=self.timeout) as response:
            payload = response.read(32*1024*1024)
        return decode_rgba(payload)


def encode_rgba(rgb):
    """Wire format: little-endian uint32 width/height, then packed RGBA8."""
    import numpy as np
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB frame must have shape (height, width, 3)')
    height, width = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.full((height, width, 1), 255, dtype=np.uint8)], axis=2)
    return struct.pack('<II', width, height)+rgba.tobytes()


def decode_rgba(payload):
    if len(payload) < 8:
        raise ValueError('Incomplete RGBA header')
    width, height = struct.unpack('<II', payload[:8])
    if not 0 < width <= 4096 or not 0 < height <= 4096 or len(payload) != 8+width*height*4:
        raise ValueError('Invalid RGBA frame size')
    return width, height, payload[8:]
