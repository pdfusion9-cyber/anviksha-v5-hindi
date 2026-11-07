from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factorized_adafactor import FactorizedAdafactor, estimate_state_bytes


class FactorizedAdafactorTests(unittest.TestCase):
    def test_matrix_update_is_finite_and_factored_state_is_small(self):
        torch.manual_seed(7)
        parameter = torch.nn.Parameter(torch.randn(512, 256))
        optimizer = FactorizedAdafactor([parameter], lr=1e-3, chunk_rows=32)
        before = parameter.detach().clone()

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = parameter.square().mean()
            loss.backward()
            optimizer.step()

        self.assertTrue(torch.isfinite(parameter).all().item())
        self.assertFalse(torch.equal(before, parameter))
        state = optimizer.state[parameter]
        self.assertEqual(state["row_sq"].numel(), 512)
        self.assertEqual(state["col_sq"].numel(), 256)
        self.assertLess(
            estimate_state_bytes([parameter]),
            parameter.numel() * 4 // 100,
        )

    def test_vector_state_and_checkpoint_roundtrip(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
        optimizer = FactorizedAdafactor([parameter], lr=1e-2)
        parameter.grad = torch.tensor([0.5, -0.25, 0.75])
        optimizer.step()
        saved = optimizer.state_dict()

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = FactorizedAdafactor([restored_parameter], lr=1e-2)
        restored.load_state_dict(saved)
        self.assertEqual(restored.state[restored_parameter]["step"], 1)
        self.assertTrue(torch.equal(
            restored.state[restored_parameter]["sq"],
            optimizer.state[parameter]["sq"],
        ))

    def test_rejects_non_fp32_training(self):
        parameter = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        optimizer = FactorizedAdafactor([parameter])
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(TypeError, "FP32"):
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
