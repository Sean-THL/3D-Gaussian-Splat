"""Large-campaign split, failed-batch/resume and augmentation contracts."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from PIL import Image

from Workflow.Simulation import collect_campaign
from Workflow.Simulation.randomization import sample_episode
from Workflow.Simulation.visual_randomization import VisualRandomizer
from Workflow.Simulation.figs.simulator import Simulator
from Workflow.Simulation.dataset import DemonstrationRecorder
from Workflow.Simulation.audit_dataset import audit_episode

CONFIG = json.loads((Path(__file__).parent/'configs/large_collection.json').read_text())


class CampaignTests(unittest.TestCase):
    def test_disjoint_split_seeds_and_route_bounds(self):
        plan = collect_campaign.build_plan(CONFIG, dict(train=11, validation=3, test=2), 4)
        seed_sets = {}
        for job in plan['jobs']:
            seeds = seed_sets.setdefault(job['split'], set())
            current = set(range(job['seed'], job['seed']+job['episodes']))
            self.assertFalse(seeds & current)
            seeds.update(current)
        self.assertEqual([len(seed_sets[k]) for k in ['train','validation','test']], [11,3,2])
        self.assertFalse(seed_sets['train'] & seed_sets['test'])
        self.assertFalse(seed_sets['validation'] & seed_sets['test'])
        for seed in range(100):
            task, _, sampled = sample_episode(CONFIG, np.random.default_rng(seed))
            self.assertTrue(.6 <= task['width_m'] <= 2.+1e-12)
            self.assertTrue(.6 <= task['height_m'] <= 2.+1e-12)
            self.assertAlmostEqual(task['width_m'], task['height_m'])
            self.assertTrue(30. <= task['duration_s'] <= 60.)
            self.assertGreaterEqual(sampled['route_size_scale'], 1.)
            self.assertTrue(1.0 <= task['altitude_m'] <= 1.6)
            self.assertTrue(-5. <= sampled['initial_roll_offset_deg'] <= 5.)
            self.assertTrue(-5. <= sampled['initial_pitch_offset_deg'] <= 5.)

    def test_visual_randomization_is_reproducible_and_optional(self):
        rgb = np.arange(8*9*3, dtype=np.uint8).reshape(8,9,3)
        a = VisualRandomizer(CONFIG['visual_randomization'], 10, True)
        b = VisualRandomizer(CONFIG['visual_randomization'], 10, True)
        c = VisualRandomizer(CONFIG['visual_randomization'], 11, True)
        np.testing.assert_array_equal(a.apply(rgb), b.apply(rgb))
        np.testing.assert_array_equal(a.apply(rgb), b.apply(rgb))
        self.assertNotEqual(a.metadata, c.metadata)
        self.assertIs(VisualRandomizer(CONFIG['visual_randomization'], 10, False).apply(rgb), rgb)
        self.assertEqual(a.apply(rgb).shape, rgb.shape)
        self.assertEqual(a.apply(rgb).dtype, np.uint8)

    def test_augmented_jpeg_recorder_keeps_pre_action_contract(self):
        import importlib.util
        if not importlib.util.find_spec('casadi'):
            self.skipTest('NMPC dependency required')
        class Renderer:
            def render(self, state):
                return np.full((6,8,3), 100, dtype=np.uint8), None
        config = copy.deepcopy(CONFIG)
        sim = Simulator(config)
        actual_initial_angles = sim.state.orientation.as_euler('xyz', degrees=True)
        sampled = sim.episode_parameters['randomization']
        self.assertAlmostEqual(actual_initial_angles[0], sampled['initial_roll_offset_deg'])
        self.assertAlmostEqual(actual_initial_angles[1], sampled['initial_pitch_offset_deg'])
        sim.paused = False
        with tempfile.TemporaryDirectory() as temp:
            recorder = DemonstrationRecorder(temp, Renderer())
            for _ in range(3):
                observation = recorder.capture(sim)
                sim.advance()
                recorder.write(sim.demonstrations[-1], observation)
            recorder.close()
            self.assertEqual(audit_episode(temp)['samples'], 3)
            metadata = json.loads((Path(temp)/'dataset_metadata.json').read_text())
            self.assertTrue(metadata['visual_randomization']['enabled'])
            self.assertTrue(metadata['image_encoding']['lossy'])
            self.assertTrue((Path(temp)/'rgb/000000.jpg').exists())

    def test_failed_batch_is_excluded_and_resume_retries_without_duplicates(self):
        metadata = dict(checkpoint_sha256='test-scene', alignment_sha256='test-calibration',
                        camera_to_body={},intrinsics={},width=8,height=6)
        class FakeNative:
            fail = True
            calls = 0
            def __init__(self, command, **_):
                FakeNative.calls += 1
                def option(name):return command[command.index(name)+1]
                folder = Path(option('--output'));folder.mkdir()
                if FakeNative.fail:
                    self.code=1
                    return
                self.code=0
                results=[]
                for i in range(int(option('--episodes'))):
                    seed=int(option('--seed'))+i
                    episode=folder/f'episode_{i+1:04d}';episode.mkdir();(episode/'rgb').mkdir()
                    image=episode/'rgb/000000.png';Image.new('RGB',(8,6),(100,100,100)).save(image)
                    collect_campaign.write_json(episode/'episode_parameters.json',dict(seed=seed))
                    collect_campaign.write_json(episode/'dataset_metadata.json',dict(control_dt_s=.05,
                        image_resolution_wh=[8,6],physics_backend='isaac_physx_forces_and_torques',
                        action_limits=dict(max_thrust_n=22.,max_body_rate_rad_s=4.)))
                    collect_campaign.write_json(episode/'dataset_count.json',dict(samples=1))
                    row=dict(time_s=0.,image_time_s=0.,position_m=[0,0,1.2],velocity_m_s=[0,0,0],
                        quaternion_xyzw=[0,0,0,1],action_type='thrust_body_rates',action_thrust_n=8.5,
                        action_body_rates_rad_s=[0,0,0],reference_horizon=[[0]*11]*21,horizon_dt_s=.05,
                        solver=dict(status='solved'),image='rgb/000000.png',image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
                    (episode/'rgb_actions.jsonl').write_text(json.dumps(row)+'\n')
                    results.append(dict(seed=seed,steps=1,passed=True,failed_solves=0))
                collect_campaign.write_json(folder/'report.json',dict(status='passed',tests=dict(passed=True),episodes=results))
            def wait(self, **_):return self.code
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);config=copy.deepcopy(CONFIG);config['rendering'].update(width=8,height=6)
            alignment=temp/'alignment.json';alignment.write_text('{}')
            checkpoint=temp/'checkpoint.pt';checkpoint.write_bytes(b'test model identity')
            config['rendering'].update(alignment=str(alignment),checkpoint=str(checkpoint))
            metadata['alignment_sha256']=hashlib.sha256(alignment.read_bytes()).hexdigest()
            metadata['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            cfg=temp/'input.json';cfg.write_text(json.dumps(config))
            output=temp/'campaign'
            argv=['collect_campaign','--config',str(cfg),'--output',str(output),'--train-episodes','2',
                  '--validation-episodes','1','--test-episodes','1','--batch-size','1','--execute']
            with patch.object(sys,'argv',argv), patch('Workflow.Simulation.dataset.PoseRenderClient',return_value=SimpleNamespace(metadata=metadata)), \
                 patch.object(collect_campaign.subprocess,'Popen',FakeNative), \
                 patch.object(collect_campaign.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):collect_campaign.main()
                state=json.loads((output/'campaign.json').read_text())
                self.assertEqual(state['status'],'stopped')
                self.assertEqual(json.loads((output/'train_episodes.json').read_text()),[])
                FakeNative.fail=False
                collect_campaign.main()
                calls=FakeNative.calls
                collect_campaign.main()
                self.assertEqual(calls,FakeNative.calls)
            state=json.loads((output/'campaign.json').read_text())
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['jobs']['train_00001']['attempt'],2)
            train=json.loads((output/'train_episodes.json').read_text())
            self.assertEqual(len(train),2)
            self.assertEqual(len({e['seed'] for e in train}),2)
            self.assertTrue(all(not Path(e['folder']).is_absolute() for e in train))
            self.assertTrue(all((output/e['folder']).exists() for e in train))


if __name__=='__main__':unittest.main()
