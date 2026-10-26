"""Exercise orchestration/resume and calibration split isolation without GPUs."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from stage2.fit_calibration import fit_policy
from stage2.pipeline import main, run_step


class PipelineChecks(unittest.TestCase):
    def test_interrupted_target_export_is_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ('logs', 'completed', 'targets'):
                (out / name).mkdir()
            output = out / 'targets/stage2'
            job = dict(name='labels_stage2', gpu='2', command=['python', '-m', 'stage2.labels',
                '--kind', 'stage2', '--out', str(output)], outputs=[output / 'index.csv'])

            def interrupted(command, **kwargs):
                output.mkdir()
                raise subprocess.CalledProcessError(1, command)

            with patch('stage2.pipeline.subprocess.run', side_effect=interrupted), \
                    redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                run_step(job, out, resume=False)
            self.assertFalse((out / 'completed/labels_stage2.json').exists())

            def finish(command, **kwargs):
                self.assertIn('--resume', command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
                (output / 'index.csv').write_text('index')

            with patch('stage2.pipeline.subprocess.run', side_effect=finish), redirect_stdout(io.StringIO()):
                run_step(job, out, resume=True)
            self.assertTrue((out / 'completed/labels_stage2.json').exists())

    def test_full_plan_and_resume_do_not_relaunch_finished_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests, out = root / 'manifests', root / 'run'
            manifests.mkdir()
            pd.DataFrame([dict(orgl_img_path='src.jpg', chng_img_path='img.jpg',
                               gt_path='old.png', pair_id=42)]).to_csv(manifests / 'epoch01.csv', index=False)
            arguments = ['pipeline', '--root', str(root), '--weights', str(root),
                         '--manifests', str(manifests), '--out', str(out), '--gpus', '0', '1']
            calls = []

            def execute(command, **kwargs):
                calls.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                target = Path(command[command.index('--out') + 1])
                module = command[command.index('-m') + 1]
                if module == 'stage2.validate':
                    target.write_text('{}')
                    return
                target.mkdir(parents=True, exist_ok=True)
                if module == 'stage2.train':
                    epoch = 32 if 'student' in command else 12
                    (target / f'epoch{epoch}.pt').write_bytes(b'checkpoint')
                    (target / 'last.pt').write_bytes(b'resume')
                elif module == 'stage2.labels':
                    (target / ('index.csv' if 'stage2' in command else '42.png')).write_text('target')
                else:
                    for name in ('calibration.json', 'policy.pkl', 'encoder_transform.pkl'):
                        (target / name).write_text('{}')

            with patch('sys.argv', arguments), patch('stage2.pipeline.check_inputs', return_value={'fixed': True}), \
                    patch('stage2.pipeline.subprocess.run', side_effect=execute), redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(len(calls), 11)
            self.assertTrue((out / 'final/model.pt').is_file())
            for command, gpu in calls:
                if '--fold' in command:
                    self.assertEqual(gpu, str(int(command[command.index('--fold') + 1]) % 2))
            epoch = pd.read_csv(out / 'student_manifests/epoch01.csv')
            self.assertEqual(epoch.gt_path[0], str(out / 'targets/nb/42.png'))
            with patch('sys.argv', arguments + ['--resume']), \
                    patch('stage2.pipeline.check_inputs', return_value={'fixed': True}), \
                    patch('stage2.pipeline.subprocess.run') as execute_again, redirect_stdout(io.StringIO()):
                main()
                execute_again.assert_not_called()
            # Removing a committed output must fail instead of silently claiming success.
            (out / 'student/epoch32.pt').unlink()
            with patch('sys.argv', arguments + ['--resume']), \
                    patch('stage2.pipeline.check_inputs', return_value={'fixed': True}), \
                    redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                main()

    def test_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not_created'
            with patch('sys.argv', ['pipeline', '--dry-run', '--out', str(out)]), \
                    patch('stage2.pipeline.check_inputs') as check, redirect_stdout(io.StringIO()):
                main()
                check.assert_not_called()
            self.assertFalse(out.exists())

    def test_controls_cannot_change_fitted_calibration(self):
        rng = np.random.default_rng(42)
        count = 160
        entries = pd.DataFrame(dict(source=['NB'] * 80 + ['stage2'] * 80,
            positive=[True] * count, partition=['policy_fit'] * 64 + ['control'] * 16 +
                                              ['policy_fit'] * 64 + ['control'] * 16))
        # Each source needs positive and negative controls for FPR evaluation.
        entries.loc[[79, 159], 'positive'] = False
        data = dict(handcrafted=rng.random((count, 244)), encoder=rng.random((count, 40)).astype(np.float32),
                    dice=rng.random((count, 4)), area=np.full((count, 4), 100), pixels=np.full(count, 1000))
        policy, transform, _ = fit_policy(data, entries)
        controls = entries.partition.eq('control').to_numpy()
        changed = {key: value.copy() for key, value in data.items()}
        changed['encoder'][controls] += 100
        changed['dice'][controls] = 0
        second, second_transform, _ = fit_policy(changed, entries)
        np.testing.assert_array_equal(transform['scaler'].mean_, second_transform['scaler'].mean_)
        np.testing.assert_array_equal(transform['pca'].components_, second_transform['pca'].components_)
        features = rng.random((10, 276))
        policy['model'].n_jobs = second['model'].n_jobs = 1
        np.testing.assert_array_equal(policy['model'].predict(features), second['model'].predict(features))


if __name__ == '__main__':
    unittest.main()
