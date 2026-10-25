"""Small CPU checks for segmentation edge cases and the research entry point."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from common import Loss
from stage2.recipe import resolve
from stage2.runtime import Images
from stage2.search import best_epoch, candidates, main as search_main
from stage2.validate import check_disjoint, evaluate_nb


class MLChecks(unittest.TestCase):
    def test_loss_empty_soft_and_small_targets(self):
        for beta in (.5, .7, None):
            logits = torch.zeros(3, 1, 8, 8, requires_grad=True)
            target = torch.zeros_like(logits)
            target[1, :, 0, 0] = 1
            target[2] = .3
            loss = Loss(tversky_beta=beta)(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            # Empty images must push logits down; positive pixels must push up.
            self.assertGreater(float(logits.grad[0].mean()), 0)
            self.assertLess(float(logits.grad[1, 0, 0, 0]), 0)

    def test_soft_targets_and_seeded_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            image = np.random.default_rng(42).integers(0, 255, (48, 64, 3), dtype=np.uint8)
            Image.fromarray(image).save(p / 'image.png')
            Image.fromarray(np.full((48, 64), 128, np.uint8)).save(p / 'mask.png')
            frame = pd.DataFrame([dict(orgl_img_path=str(p / 'image.png'),
                chng_img_path=str(p / 'image.png'), gt_path=str(p / 'mask.png'))])
            for profile in ('minimal', 'forensic'):
                a = Images(frame, 64, True, profile, 42)[0]
                b = Images(frame, 64, True, profile, 42)[0]
                self.assertTrue(torch.equal(a[0], b[0]))
                self.assertTrue(torch.allclose(a[1], torch.full_like(a[1], 128 / 255)))

    def test_config_rejects_silent_mistakes(self):
        for overrides in ({'lr': float('nan')}, {'typo': 1}, {'encoder': 'unknown'},
                          {'img_size': 639}, {'schedule_epochs': 2}, {'seed': True}):
            with self.assertRaises(ValueError):
                resolve(overrides)
        with self.assertRaises(ValueError):
            candidates({'defaults': {}, 'variants': [{'name': '../escape', 'overrides': {}}]})

    def test_epoch_selection_tradeoff_and_tie(self):
        rows = [dict(epoch=e, nb=dict(aic=a, fpr_negative=f))
                for e, a, f in [(2, .8, .1), (3, .8, .05), (4, .8, .05), (1, .7, 0)]]
        self.assertEqual(best_epoch(rows)['epoch'], 3)

    def test_fpr_exact_boundary_and_restored_state(self):
        class Boundary(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
            def forward(self, x):
                y = torch.full((len(x), 1, 100, 100), -10.)
                y[:, :, 0] = 10.
                return y + self.anchor
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(p / 'image.png')
            mask = np.zeros((100, 100), np.uint8)
            mask[0] = 255
            Image.fromarray(mask).save(p / 'mask.png')
            frame = pd.DataFrame([dict(orgl_img_path=str(p / 'image.png'),
                chng_img_path=str(p / 'image.png'), gt_path=str(p / 'mask.png'))])
            net = Boundary().train()
            rng = torch.get_rng_state().clone()
            result = evaluate_nb(net, frame, 100)
            self.assertEqual((result['dice_positive'], result['fpr_negative'], result['aic']), (1., 1., 0.))
            self.assertTrue(net.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            (p / 'alias.png').symlink_to(p / 'image.png')
            alias = frame.copy()
            alias['orgl_img_path'] = str(p / 'alias.png')
            with self.assertRaises(ValueError):
                check_disjoint(frame, alias)

    def test_search_dry_run_and_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = []
            for i in range(16):
                for name in ('src', 'edit', 'mask'):
                    Image.fromarray(np.full((8, 8), 255, np.uint8)).save(p / f'{name}{i}.png')
                rows.append(dict(orgl_img_path=f'src{i}.png', chng_img_path=f'edit{i}.png',
                    gt_path=f'mask{i}.png', edit_type='add' if i < 8 else 'remove'))
            pd.DataFrame(rows).to_csv(p / 'nb.csv', index=False)
            unrelated = pd.DataFrame([dict(orgl_img_path='train-src.png', chng_img_path='train-edit.png', gt_path='')])
            unrelated.to_csv(p / 'stage2_train.csv', index=False)
            unrelated.to_csv(p / 'epoch01.csv', index=False)
            plan = dict(defaults=dict(initialization='imagenet', schedule_epochs=1, final_epoch=1),
                        variants=[dict(name='b2', overrides={}), dict(name='b1', overrides=dict(encoder='mit_b1'))])
            (p / 'plan.json').write_text(json.dumps(plan))
            args = ['search', '--plan', str(p / 'plan.json'), '--root', str(p), '--manifests', str(p),
                    '--nb-csv', str(p / 'nb.csv'), '--out', str(p / 'out'), '--dry-run']
            with patch.object(sys, 'argv', args):
                search_main()
            self.assertEqual(len(json.loads((p / 'out/commands.json').read_text())), 2)
            self.assertFalse((p / 'out/b2').exists())
            self.assertEqual(len(pd.read_csv(p / 'out/tune.csv')), 12)
            self.assertEqual(len(pd.read_csv(p / 'out/control.csv')), 4)
            with patch.object(sys, 'argv', args + ['--resume']):
                search_main()
            plan['variants'][0]['overrides'] = dict(lr=3e-5)
            (p / 'plan.json').write_text(json.dumps(plan))
            with patch.object(sys, 'argv', args + ['--resume']), self.assertRaises(ValueError):
                search_main()


if __name__ == '__main__':
    unittest.main()
