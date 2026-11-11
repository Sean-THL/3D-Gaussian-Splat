#!/usr/bin/env python3
"""Plan/resume sequential native Isaac collection with disjoint seed splits.

Default: plan only. --execute runs native batches, audits every pair and indexes
only passed batches. Interrupted/failed attempts are retained and never indexed.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def build_plan(config, counts, batch_size, seed_offset=0):
    if config.get('controller', {}).get('type') != 'nmpc' or config.get('dynamics_model') != 'quadrotor':
        raise ValueError('Campaign requires thrust/body-rate NMPC and quadrotor dynamics')
    if config['task'].get('trajectory_type') != 'lissajous_figure8':
        raise ValueError('This campaign planner currently supports figure-eight episodes')
    if not config.get('randomization', {}).get('enabled'):
        raise ValueError('Campaign config must enable randomization')
    duration_bounds = config['randomization'].get('duration_scale', [1., 1.])
    if len(duration_bounds) != 2 or not all(math.isfinite(v) and v>0 for v in duration_bounds) or duration_bounds[0]>duration_bounds[1]:
        raise ValueError('Invalid duration_scale bounds')
    nominal = config['task']['duration_s']
    if nominal*duration_bounds[0] < max(30., config['task'].get('minimum_duration_s', 0.)):
        raise ValueError('All campaign flights must last at least 30 seconds')
    starts = {name:value+seed_offset for name,value in dict(train=100000, validation=10000000, test=20000000).items()}
    intervals = [(starts[name], starts[name]+count) for name,count in counts.items() if count]
    if any(max(a,c)<min(b,d) for i,(a,b) in enumerate(intervals) for c,d in intervals[i+1:]):
        raise ValueError('Seed ranges overlap; reduce counts or adjust planner seed allocation')
    jobs = []
    for split, count in counts.items():
        for offset in range(0, count, batch_size):
            jobs.append(dict(id=f'{split}_{offset//batch_size+1:05d}', split=split,
                seed=starts[split]+offset, episodes=min(batch_size, count-offset)))
    rate = float(config['control_hz'])
    if not math.isfinite(rate) or rate<=0:
        raise ValueError('Invalid control rate')
    # Native runner records a full route plus two settling seconds.
    return dict(counts=counts, episodes_per_batch=batch_size, jobs=jobs,
        min_samples_per_episode=math.ceil((nominal*duration_bounds[0]+2)*rate),
        max_samples_per_episode=math.ceil((nominal*duration_bounds[1]+2)*rate),
        expected_samples_per_episode=(nominal*sum(duration_bounds)/2+2)*rate)


def audit_batch(folder, job):
    from Workflow.Simulation.audit_dataset import audit_episode
    report = json.loads((folder/'report.json').read_text())
    if report.get('status') != 'passed' or report.get('tests', {}).get('passed') is not True:
        raise ValueError('Batch failed native validation or tracking checks')
    reports = report.get('episodes', [])
    if len(reports) != job['episodes']:
        raise ValueError('Unexpected episode count')
    accepted = []
    for index, item in enumerate(reports):
        episode = folder/f'episode_{index+1:04d}'
        parameters = json.loads((episode/'episode_parameters.json').read_text())
        seed = job['seed']+index
        if item['seed'] != seed or parameters['seed'] != seed or not item.get('passed') or item.get('failed_solves') != 0:
            raise ValueError('Unexpected seed or failed expert episode')
        audit = audit_episode(episode)
        if audit['samples'] != item['steps']:
            raise ValueError('Report/image sample count mismatch')
        write_json(episode/'dataset_audit.json', audit)
        accepted.append(dict(seed=seed, samples=audit['samples'], folder=str(episode), split=job['split']))
    return accepted


def write_indexes(output, state, counts):
    for split in counts:
        accepted = [episode for item in state['jobs'].values() if item['status']=='passed'
                    for episode in item['episodes'] if episode['split']==split]
        write_json(output/f'{split}_episodes.json',accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'Workflow/Simulation/configs/large_collection.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-episodes', type=int, default=1000)
    parser.add_argument('--validation-episodes', type=int, default=100)
    parser.add_argument('--test-episodes', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--seed-offset', type=int, default=0, help='Choose a new offset for an additional non-overlapping campaign')
    parser.add_argument('--render-url', default='http://127.0.0.1:8766')
    parser.add_argument('--estimated-bytes-per-sample', type=int, default=260000,
                        help='Storage planning estimate; measure your first batch before scaling')
    parser.add_argument('--execute', action='store_true', help='Collect now; otherwise just write/inspect the resumable plan')
    args = parser.parse_args()
    counts = dict(train=args.train_episodes, validation=args.validation_episodes, test=args.test_episodes)
    if min(counts.values())<0 or not sum(counts.values()) or not 1<=args.batch_size<=100 or args.estimated_bytes_per_sample<=0 or args.seed_offset<0:
        parser.error('Use nonnegative episode counts, at least one episode, batch size 1–100 and positive storage estimate')
    config = json.loads(args.config.read_text())
    plan = build_plan(config, counts, args.batch_size, args.seed_offset)
    source = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT/'Workflow/Simulation').rglob('*.py'))}
    signature = digest(dict(config=config, plan=plan, source=source))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.campaign.lock').open('w') as lock:
        try:fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Another collector is already using this campaign')
        manifest = output/'campaign.json'
        if manifest.exists():
            state = json.loads(manifest.read_text())
            if state['signature'] != signature:
                parser.error('Config, counts or source code changed; create a new campaign folder')
        else:
            if any(p.name!='.campaign.lock' for p in output.iterdir()):
                parser.error('Output must be empty or an existing campaign')
            state = dict(schema_version=1, signature=signature, plan=plan, source_sha256=source,
                         status='planned', jobs={}, renderer_identity=None)
            write_json(output/'config.json', config)
            write_json(manifest, state)
        remaining = [j for j in plan['jobs'] if state['jobs'].get(j['id'],{}).get('status')!='passed']
        remaining_episodes = sum(j['episodes'] for j in remaining)
        worst_bytes = remaining_episodes*plan['max_samples_per_episode']*args.estimated_bytes_per_sample*1.15
        free = shutil.disk_usage(output).free
        print(json.dumps(dict(campaign=str(output), episodes=sum(counts.values()),
              estimated_samples=round(sum(counts.values())*plan['expected_samples_per_episode']),
              remaining_episodes=remaining_episodes, worst_remaining_storage_gib=round(worst_bytes/2**30,1),
              free_gib=round(free/2**30,1), execute=args.execute),indent=2),flush=True)
        if not args.execute:
            return
        write_indexes(output,state,counts)
        if worst_bytes>free:
            parser.error('Insufficient estimated storage for remaining campaign; use a larger disk or a smaller new campaign')
        if not remaining:
            print('Campaign already complete.',flush=True)
            return
        # Reuse the existing localhost-only render client validation.
        from Workflow.Simulation.dataset import PoseRenderClient
        renderer = PoseRenderClient(args.render_url)
        from Workflow.Simulation.rendering.assets import resolve_training_asset
        alignment_path = ROOT/config['rendering']['alignment']
        if hashlib.sha256(alignment_path.read_bytes()).hexdigest()!=renderer.metadata['alignment_sha256']:
            parser.error('Running renderer uses a different alignment; restart it with the campaign scene configuration')
        checkpoint = resolve_training_asset(ROOT, config['rendering']['checkpoint'])
        checkpoint_digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024),b''):
                checkpoint_digest.update(chunk)
        if checkpoint_digest.hexdigest()!=renderer.metadata['checkpoint_sha256']:
            parser.error('Running renderer uses a different Gaussian checkpoint')
        identity = {k:renderer.metadata[k] for k in ['checkpoint_sha256','alignment_sha256','camera_to_body','intrinsics']}
        if renderer.metadata['width']!=config['rendering']['width'] or renderer.metadata['height']!=config['rendering']['height']:
            parser.error('Renderer resolution differs from campaign configuration')
        if state['renderer_identity'] not in (None, identity):
            parser.error('Renderer checkpoint/calibration changed; use the original renderer or start a new campaign')
        state['renderer_identity'] = identity
        state['status'] = 'running'
        write_json(manifest, state)
        for job in remaining:
            if shutil.disk_usage(output).free < job['episodes']*plan['max_samples_per_episode']*args.estimated_bytes_per_sample*1.15:
                state['status']='stopped'
                write_json(manifest,state)
                raise SystemExit('Insufficient free disk for the next batch; free space and resume this campaign.')
            previous = state['jobs'].get(job['id'], {})
            attempt = previous.get('attempt', 0)+1
            folder = output/'batches'/job['id']/f'attempt_{attempt:03d}'
            folder.parent.mkdir(parents=True,exist_ok=True)
            log = folder.parent/f'attempt_{attempt:03d}.log'
            entry = dict(status='running', attempt=attempt, output=str(folder), log=str(log),
                         previous_attempt=previous.get('output'), episodes=[])
            state['jobs'][job['id']] = entry
            write_json(manifest,state)
            command = ['bash',str(ROOT/'Workflow/Simulation/isaac_sim/run_native.sh'),
                '--config',str(output/'config.json'),'--checks','--randomize',
                '--episodes',str(job['episodes']),'--seed',str(job['seed']),
                '--render-url',args.render_url,'--output',str(folder)]
            environment = os.environ.copy()
            # Native uses Isaac bundled Python; parent can use the project environment for audits.
            environment.pop('CONDA_PREFIX',None)
            print(f"START {job['id']} seeds {job['seed']}–{job['seed']+job['episodes']-1}; log={log}",flush=True)
            try:
                with log.open('w') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT,start_new_session=True)
                    try:code = child.wait()
                    except KeyboardInterrupt:
                        os.killpg(child.pid,signal.SIGINT)
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise
                if code:
                    raise RuntimeError(f'Native batch exited {code}; inspect {log}')
                entry['episodes'] = audit_batch(folder,job)
                for episode in entry['episodes']:
                    episode['folder'] = str(Path(episode['folder']).relative_to(output))
                entry['status'] = 'passed'
                print(f"PASSED {job['id']}: {sum(e['samples'] for e in entry['episodes'])} audited pairs",flush=True)
            except (Exception,KeyboardInterrupt) as error:
                entry.update(status='failed',error=f'{type(error).__name__}: {error}')
                state['status']='stopped'
                write_json(manifest,state)
                print(f'Campaign stopped: {error}. Rerun the same command to resume in a new attempt folder.',file=sys.stderr)
                raise SystemExit(1)
            write_json(manifest,state)
            # Index only audited completed batches; never failed partial attempts.
            write_indexes(output,state,counts)
        state['status']='complete'
        write_json(manifest,state)
        print('CAMPAIGN COMPLETE: '+str(output),flush=True)


if __name__=='__main__':main()
