import unittest
from pathlib import Path
import tempfile
import torch
from models import HistoSMoEScore
from infer import forward_tiled, tile_starts
from runtime_io import load_checkpoint
from train import semantic, pearson, rgb_to_y


class IdentityModel(torch.nn.Module):
    def forward(self, image, de, ce):
        return image, image.new_zeros(())


class MainModelTests(unittest.TestCase):
    def test_parameter_contract(self):
        model = HistoSMoEScore()
        report = model.assert_parameter_contract()
        self.assertEqual(report['active_top1'], 17640692)
        self.assertEqual(report['total'], 19198364)
        self.assertEqual(len(model.state_dict()), 867)
        shared = sum(p.numel() for n, p in model.named_parameters() if not semantic(n))
        self.assertEqual(shared, 16615100)

    def test_tiling_and_padding(self):
        model = IdentityModel()
        for h, w in ((1, 7), (24, 32), (49, 101), (65, 67)):
            image = torch.rand(1, 3, h, w)
            context = torch.zeros(1, 512)
            restored = forward_tiled(model, image, context, context, size=32, overlap=8)
            torch.testing.assert_close(restored, image, rtol=1e-6, atol=1e-7)
        self.assertEqual(tile_starts(101, 32, 24), [0, 24, 48, 69])

    def test_strict_checkpoint(self):
        model = torch.nn.Linear(2, 3)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'checkpoint.pth'
            torch.save({'params_ema': {'weight': model.weight}}, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, path)

    def test_main_training_step(self):
        torch.manual_seed(3407)
        model = HistoSMoEScore().train()
        image = torch.rand(1, 3, 16, 16)
        de, ce = torch.randn(1, 512), torch.randn(1, 512)
        prediction, auxiliary = model(image, de, ce)
        loss = (prediction - image).abs().mean() + auxiliary
        loss = loss + 0.05 * torch.nn.functional.cross_entropy(model.last_router_logits, torch.tensor([0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None for p in model.semantic_router.parameters()))

    def test_training_objectives(self):
        x = torch.rand(2, 3, 16, 16)
        torch.testing.assert_close(pearson(x, x), torch.tensor(0.0), atol=1e-6, rtol=0)
        self.assertEqual(tuple(rgb_to_y(x).shape), (2, 1, 16, 16))


if __name__ == '__main__':
    unittest.main()
