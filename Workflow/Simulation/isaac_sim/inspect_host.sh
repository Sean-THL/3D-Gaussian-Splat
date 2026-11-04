#!/usr/bin/env bash
# Read-only installation/GPU inventory; does not install or launch Isaac Sim.
set -u
python3 - <<'PY'
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path


def command(args):
    if not shutil.which(args[0]):
        return dict(available=False)
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(available=True, returncode=result.returncode,
                    stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except subprocess.TimeoutExpired:
        return dict(available=True, error='Timed out after 20 seconds')


launchers, packages, versions, unreadable = [], [], [], []
skip = {'.git', '.cache', 'node_modules', 'datasets', 'data', 'exports', 'rosbags',
        'checkpoints', 'wandb', 'logs', '__pycache__', 'build', 'install'}
launcher_names = {'isaac-sim.sh', 'isaac-sim.streaming.sh', 'isaac-sim.selector.sh',
                  'isaac-sim.compatibility_check.sh', 'isaacsim'}
roots = [Path.home(), Path('/opt'), Path('/usr/local')]
for root in roots:
    if not root.exists():
        continue
    for directory, dirs, files in os.walk(root, onerror=lambda e: unreadable.append(str(e.filename))):
        path = Path(directory)
        depth = len(path.relative_to(root).parts)
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('PX4-Autopilot')]
        if depth >= 9:
            dirs[:] = []
        for name in launcher_names.intersection(files):
            launchers.append(str(path/name))
        if 'isaac' in str(path).lower() and 'VERSION' in files:
            try:
                versions.append(dict(path=str(path/'VERSION'), value=(path/'VERSION').read_text()[:200].strip()))
            except (OSError, UnicodeError):
                pass
        if path.name.startswith('isaacsim') and path.name.endswith('.dist-info') and 'METADATA' in files:
            try:
                metadata = (path/'METADATA').read_text()
                name = next((line[6:] for line in metadata.splitlines() if line.startswith('Name: ')), '')
                version = next((line[9:] for line in metadata.splitlines() if line.startswith('Version: ')), '')
                packages.append(dict(name=name, version=version, path=str(path)))
            except (OSError, UnicodeError):
                pass

docker_images = command(['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'])
if docker_images.get('returncode') == 0:
    docker_images['stdout'] = '\n'.join(line for line in docker_images['stdout'].splitlines()
                                         if any(k in line.lower() for k in ['isaac', 'omniverse']))
docker_running = command(['docker', 'ps', '--format', '{{.Image}} {{.Names}}'])
if docker_running.get('returncode') == 0:
    docker_running['stdout'] = '\n'.join(line for line in docker_running['stdout'].splitlines()
                                          if any(k in line.lower() for k in ['isaac', 'omniverse']))
report = dict(hostname=socket.gethostname(), platform=platform.platform(),
              os_release=Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else '',
              gpu=command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv']),
              ram=command(['free', '-h']), disk=command(['df', '-h', str(Path.home())]),
              display=os.environ.get('DISPLAY'), wayland_display=os.environ.get('WAYLAND_DISPLAY'),
              launchers=sorted(set(launchers)), python_packages=packages, version_files=versions[:30],
              docker_images=docker_images, docker_running=docker_running,
              search_roots=[str(p) for p in roots], max_search_depth=9,
              unreadable_directory_count=len(unreadable),
              note='No match is not proof of absence; custom install locations or stopped containers may need separate inspection.')
print(json.dumps(report, indent=2))
PY
