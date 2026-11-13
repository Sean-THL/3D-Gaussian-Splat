"""Version immutability, asset provenance and pilot/test separation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Workflow.Simulation import prepare_dataset_version as versions


class DatasetVersionTests(unittest.TestCase):
    def prepare(self, root):
        source=root/'Workflow/Simulation'
        source.mkdir(parents=True)
        (source/'minimal.py').write_text('value = 1\n')
        assets=root/'assets';assets.mkdir()
        for name in ['alignment.json','checkpoint.pt','cameras.bin']:
            (assets/name).write_bytes(b'test asset identity')
        config=dict(dynamics_model='quadrotor',controller=dict(type='nmpc'),vehicle=dict(mass_kg=.87),
            rendering=dict(alignment='assets/alignment.json',checkpoint='assets/checkpoint.pt',camera_model='assets/cameras.bin'))
        cfg=root/'input.json';cfg.write_text(json.dumps(config))
        return cfg,config

    def test_new_version_is_frozen_and_pilot_is_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg,config=self.prepare(root)
            v1=root/'campaigns/campaign_v1'
            with patch.object(versions,'ROOT',root):
                metadata=versions.create_version(cfg,v1,3,90000000)
                self.assertEqual(metadata['status'],'prepared_not_collected')
                self.assertTrue((v1/'pilot_test/runs').is_dir())
                self.assertTrue((v1/'collection').is_dir())
                self.assertEqual(list((v1/'pilot_test/runs').iterdir()),[])
                self.assertIn('excluded from final held-out test',metadata['pilot']['purpose'])
                self.assertEqual(metadata['config_sha256'],hashlib.sha256((v1/'config.json').read_bytes()).hexdigest())
                self.assertTrue((v1/'source_snapshot/Workflow/Simulation/minimal.py').is_file())
                self.assertFalse(list((v1/'source_snapshot').rglob('*.pt')))
                with self.assertRaises(FileExistsError):versions.create_version(cfg,v1)
                config2=copy.deepcopy(config);config2['vehicle']['mass_kg']=1.
                cfg.write_text(json.dumps(config2))
                v2=root/'campaigns/campaign_v2'
                second=versions.create_version(cfg,v2,parent=v1)
                self.assertEqual(second['parent']['version'],'campaign_v1')
                self.assertNotEqual(metadata['config_sha256'],second['config_sha256'])
                self.assertEqual(json.loads((v1/'config.json').read_text())['vehicle']['mass_kg'],.87)

    def test_missing_asset_does_not_create_partial_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg,_=self.prepare(root)
            (root/'assets/checkpoint.pt').unlink()
            destination=root/'campaign_v1'
            with patch.object(versions,'ROOT',root),self.assertRaises(FileNotFoundError):
                versions.create_version(cfg,destination)
            self.assertFalse(destination.exists())

    def test_custom_motor_launcher_and_license_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg,config=self.prepare(root)
            config.update(collection_backend='custom_motor',controller=dict(type='mpc_geometric'))
            cfg.write_text(json.dumps(config))
            license_path=root/'Workflow/Simulation/third_party/model/LICENSE'
            license_path.parent.mkdir(parents=True);license_path.write_text('MIT test license')
            destination=root/'campaigns/motor_v2'
            with patch.object(versions,'ROOT',root):
                metadata=versions.create_version(cfg,destination)
            command=(destination/'pilot_test/run_pilot.sh').read_text()
            self.assertEqual(metadata['collection_backend'],'custom_motor')
            self.assertIn('bash Workflow/Simulation/run.sh',command)
            self.assertIn('--headless --dataset --randomize',command)
            self.assertNotIn('--render-url',command)
            self.assertTrue((destination/'source_snapshot'/license_path.relative_to(root)).is_file())


if __name__=='__main__':unittest.main()
