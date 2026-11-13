#!/usr/bin/env python3
"""Create a frozen dataset configuration and a separate pilot-review workspace."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Workflow.Simulation.rendering.assets import resolve_training_asset


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def create_version(config_path, destination, pilot_episodes=3, pilot_seed=90000000, parent=None):
    config = json.loads(config_path.read_text())
    if config.get('dynamics_model')!='quadrotor' or config.get('controller',{}).get('type') not in ('nmpc','mpc_geometric'):
        raise ValueError('Use a quadrotor thrust/body-rate expert configuration')
    backend=config.get('collection_backend','native')
    if backend not in ('native','custom_motor'):
        raise ValueError('collection_backend must be native or custom_motor')
    if backend=='native' and config.get('controller',{}).get('type')!='nmpc':
        raise ValueError('mpc_geometric motor dynamics currently use collection_backend=custom_motor')
    if pilot_episodes<1 or pilot_seed<0:
        raise ValueError('Pilot episode count must be positive and seed nonnegative')
    if destination.exists():
        raise FileExistsError(f'Version already exists; do not overwrite it: {destination}')
    if parent is not None and not (parent/'version.json').is_file():
        raise ValueError('Parent must contain version.json')
    # Resolve all required assets before creating the destination.
    assets = {}
    for name in ('alignment', 'checkpoint', 'camera_model'):
        path = resolve_training_asset(ROOT, config['rendering'][name])
        assets[name] = dict(configured_path=config['rendering'][name],sha256=file_hash(path))
    sources = {str(p.relative_to(ROOT)):p for p in sorted((ROOT/'Workflow/Simulation').rglob('*'))
               if p.is_file() and (p.suffix in ('.py','.sh','.json','.html','.md') or p.name in ('LICENSE','NOTICE'))}
    destination.mkdir(parents=True)
    frozen = destination/'config.json'
    frozen.write_text(json.dumps(config,indent=2)+'\n')
    for relative,source in sources.items():
        target = destination/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    shutil.copyfile(resolve_training_asset(ROOT,config['rendering']['alignment']),destination/'calibration_snapshot.json')
    git = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    status = subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True)
    parent_record = None if parent is None else dict(path=str(parent),
        version=json.loads((parent/'version.json').read_text())['version'])
    metadata = dict(schema_version=1,version=destination.name,created_utc=datetime.now(timezone.utc).isoformat(),
        status='prepared_not_collected',parent=parent_record,config_sha256=file_hash(frozen),assets=assets,
        collection_backend=backend,
        source_git_commit=git.stdout.strip() if git.returncode==0 else None,
        source_worktree_dirty=bool(status.stdout.strip()),
        source_sha256={relative:file_hash(path) for relative,path in sources.items()},
        pilot=dict(folder='pilot_test',episodes=pilot_episodes,seed_start=pilot_seed,
                   seed_end=pilot_seed+pilot_episodes-1,purpose='development/visual review; excluded from final held-out test'),
        collection_folder='collection')
    (destination/'version.json').write_text(json.dumps(metadata,indent=2)+'\n')
    pilot = destination/'pilot_test'
    (pilot/'runs').mkdir(parents=True)
    (pilot/'reviews').mkdir()
    (destination/'collection').mkdir()
    review = dict(status='pending',run=None,reviewed_by=None,
        checks={name:None for name in ['native_report_passed','dataset_audits_passed','rgb_coverage_acceptable',
            'camera_mount_and_tilt_plausible','trajectory_and_altitude_as_intended',
            'augmentation_acceptable','tracking_and_expert_actions_acceptable']},
        notes='Do not treat a structural audit as a geometry-fidelity or sim-to-real pass.')
    if backend=='custom_motor':
        review['checks'].pop('native_report_passed')
        review['checks']['full_motor_model_validated']=None
    (pilot/'reviews/review_template.json').write_text(json.dumps(review,indent=2)+'\n')
    command = '''#!/usr/bin/env bash
set -euo pipefail
version_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f Workflow/Simulation/isaac_sim/run_native.sh ]]; then
  echo 'Run this script from the project repository root.' >&2
  exit 1
fi
python3 - "$version_root" <<'PY'
import hashlib,json,sys,os
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse
from Workflow.Simulation.rendering.assets import resolve_training_asset
root=Path(sys.argv[1]);metadata=json.loads((root/'version.json').read_text())
if hashlib.sha256((root/'config.json').read_bytes()).hexdigest()!=metadata['config_sha256']:
    raise SystemExit('Frozen configuration changed. Create a new dataset version.')
for name,expected in metadata['source_sha256'].items():
    path=Path(name)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Collection source changed: '+name+'. Create a new dataset version.')
for name,asset in metadata['assets'].items():
    path=resolve_training_asset(Path.cwd(),asset['configured_path'])
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    if digest.hexdigest()!=asset['sha256']:
        raise SystemExit('Frozen model/calibration asset changed: '+name)
config=json.loads((root/'config.json').read_text())
if metadata.get('collection_backend','native')=='native':
    url=os.environ.get('RENDER_URL','http://127.0.0.1:8766')
    if urlparse(url).scheme!='http' or urlparse(url).hostname not in ('127.0.0.1','localhost','::1'):
        raise SystemExit('Use a localhost HTTP renderer URL.')
    with urlopen(url.rstrip('/')+'/health',timeout=10) as response:
        renderer=json.load(response)
    for name,key in [('checkpoint','checkpoint_sha256'),('alignment','alignment_sha256')]:
        if renderer[key]!=metadata['assets'][name]['sha256']:
            raise SystemExit('Running renderer does not match version asset: '+name)
    if [renderer['width'],renderer['height']]!=[config['rendering']['width'],config['rendering']['height']]:
        raise SystemExit('Running renderer resolution differs from the version configuration.')
PY
run_output="$version_root/pilot_test/runs/run_$(date -u +%Y%m%dT%H%M%SZ)_$RANDOM"
bash Workflow/Simulation/isaac_sim/run_native.sh \\
  --config "$version_root/config.json" --checks --randomize \\
  --seed PILOT_SEED --episodes PILOT_EPISODES \\
  --render-url "${RENDER_URL:-http://127.0.0.1:8766}" \\
  --output "$run_output"
echo "PILOT_OUTPUT=$run_output"
'''.replace('PILOT_SEED',str(pilot_seed)).replace('PILOT_EPISODES',str(pilot_episodes))
    if backend=='custom_motor':
        command=command.replace('bash Workflow/Simulation/isaac_sim/run_native.sh','bash Workflow/Simulation/run.sh')
        command=command.replace('--config "$version_root/config.json" --checks --randomize',
            '--config "$version_root/config.json" --headless --dataset --randomize')
        command=command.replace('  --render-url "${RENDER_URL:-http://127.0.0.1:8766}" \\\n','')
    launcher = pilot/'run_pilot.sh'
    launcher.write_text(command)
    launcher.chmod(0o755)
    (pilot/'README.md').write_text(f'''# First {pilot_episodes} episodes: review only

These episodes use seeds {pilot_seed}–{pilot_seed+pilot_episodes-1} and the version's
frozen `../config.json`. They are development checks, not the final test split.
Each attempt gets a new folder under `runs/`; existing data are not overwritten.

Start the Gaussian renderer first. Then, from the repository root with Conda
deactivated, run this folder's `run_pilot.sh`. The script validates frozen source
and configuration before starting native physics. It collects complete flights.

After collection, run `visualize_dataset.py` and `plot_collected_trajectories.py`
against the new `runs/run_*` directory. Copy `reviews/review_template.json` to a
run-specific review file and record what was accepted/rejected and why.

Inspect all episodes for visual artifacts, floor/wall deformation, camera/pose
consistency, correct route dimensions/altitudes, image augmentation and expert
tracking. Compare representative rendered views against recorded RGB/depth at
matched poses. Repeated views from the same reconstruction are not independent
real-world validation. Do not start bulk collection merely because audits pass.
''')
    if backend=='custom_motor':
        text=(pilot/'README.md').read_text().replace('Start the Gaussian renderer first. Then, from the repository root with Conda\ndeactivated, run',
            'The custom runner loads its own Gaussian renderer. From the repository root, run')
        text=text.replace('starting native physics','starting the full motor/RK4 physics')
        (pilot/'README.md').write_text(text)
    (destination/'README.md').write_text(f'''# {destination.name}

Status: prepared, no episodes collected by this setup command.

- `config.json`: frozen settings; create a new version instead of editing.
- `version.json`: configuration/model/calibration hashes and source provenance.
- `calibration_snapshot.json`: small copy of the candidate alignment.
- `source_snapshot/`: source/config archive; model weights are referenced, not copied.
- `pilot_test/`: first {pilot_episodes} development episodes and review records.
- `collection/`: reserved for the later full campaign and train/validation/test indexes.

The pilot is separate from the final held-out test set. Review it before bulk
collection. Original recordings and earlier pilots remain in their original
locations; nothing was moved or deleted. See `Workflow/Simulation/DATASET_VERSIONS.md`
in the repository for version creation, native launch and acceptance guidance.
''')
    frozen.chmod(0o444)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',required=True,help='New folder name, e.g. campaign_v1')
    parser.add_argument('--config',type=Path,default=ROOT/'Workflow/Simulation/configs/large_collection.json')
    parser.add_argument('--base',type=Path,default=ROOT/'data/scene_20261025_200034/isaac_sim/campaigns')
    parser.add_argument('--pilot-episodes',type=int,default=3)
    parser.add_argument('--pilot-seed',type=int,default=90000000)
    parser.add_argument('--parent',type=Path)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.version) or args.version in ('.','..'):
        parser.error('Use a simple version name without slashes')
    destination=args.base/args.version
    try:metadata=create_version(args.config,destination,args.pilot_episodes,args.pilot_seed,args.parent)
    except (ValueError,FileExistsError) as error:parser.error(str(error))
    print(json.dumps(dict(folder=str(destination),status=metadata['status'],pilot=metadata['pilot']),indent=2))


if __name__=='__main__':main()
