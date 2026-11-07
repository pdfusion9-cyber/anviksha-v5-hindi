from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from hcdm110_prototype import (
    AdaptiveLearningController,
    AssociativeMemory,
    CA1Integrator,
    ControllerSignals,
    DGPatternEncoder,
    DualProjectionAdapterBank,
    HCDM110Prototype,
    PFCWorkingMemory,
    ReplayPriority,
)


class TestHCDM110Prototype(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(110)

    def test_dg_expansion_shape_sparsity_and_frozen_projection(self):
        dg = DGPatternEncoder(input_dim=32, expansion_dim=256, active_k=12, seed=7)
        x = torch.randn(5, 32)
        code = dg(x)
        self.assertEqual(tuple(code.shape), (5, 256))
        self.assertTrue(torch.isfinite(code).all())
        self.assertTrue(torch.all((code != 0).sum(dim=-1) == 12))
        self.assertEqual(sum(p.numel() for p in dg.parameters()), 0)
        self.assertEqual(tuple(dg.projection.shape), (32, 256))

    def test_dg_rejects_bad_shape_and_nan(self):
        dg = DGPatternEncoder(16, 64, 8)
        with self.assertRaises(ValueError):
            dg(torch.randn(2, 15))
        x = torch.randn(2, 16)
        x[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            dg(x)

    def test_associative_memory_exact_key_retrieval(self):
        memory = AssociativeMemory(capacity=8, key_dim=16, value_dim=6)
        keys = F.normalize(torch.randn(4, 16), dim=-1)
        values = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        memory.add(keys, values)
        retrieved, scores, ids = memory.retrieve(keys[2:3], top_k=3)
        self.assertTrue(torch.equal(retrieved[0, 0].cpu(), values[2]))
        self.assertAlmostEqual(float(scores[0, 0]), 1.0, places=5)
        self.assertEqual(int(ids[0, 0]), 2)

    def test_associative_memory_capacity_and_empty_state(self):
        memory = AssociativeMemory(capacity=2, key_dim=4, value_dim=3)
        zero = torch.zeros(1, 4)
        values, scores, ids = memory.retrieve(zero)
        self.assertEqual(tuple(values.shape), (1, 0, 3))
        self.assertEqual(tuple(scores.shape), (1, 0))
        keys = torch.eye(4)[:3]
        vals = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
        memory.add(keys, vals)
        self.assertEqual(memory.size, 2)
        self.assertEqual(int(memory.valid.sum()), 2)
        result, _, _ = memory.retrieve(keys[2:3], top_k=1)
        self.assertTrue(torch.equal(result[0, 0], vals[2]))

    def test_ca1_zero_initialisation_is_exact_identity(self):
        ca1 = CA1Integrator(dim=24)
        x = torch.randn(3, 24)
        retrieved = torch.randn(3, 5, 24)
        out, weights, disagreement = ca1(x, retrieved)
        self.assertTrue(torch.equal(out, x))
        self.assertEqual(tuple(weights.shape), (3, 5))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3)))
        self.assertTrue(torch.isfinite(disagreement).all())

    def test_ca1_learns_residual_with_finite_gradients(self):
        ca1 = CA1Integrator(dim=16)
        x = torch.randn(2, 16, requires_grad=True)
        retrieved = torch.randn(2, 3, 16)
        out, _, _ = ca1(x, retrieved)
        loss = out.square().mean()
        loss.backward()
        self.assertIsNotNone(ca1.output.weight.grad)
        self.assertTrue(torch.isfinite(ca1.output.weight.grad).all())

    def test_pfc_working_memory_state_and_control_distribution(self):
        pfc = PFCWorkingMemory(dim=20)
        x = torch.randn(4, 20)
        state = pfc.initial_state(4)
        next_state, controls = pfc(x, state)
        self.assertEqual(tuple(next_state.hidden.shape), (4, 20))
        self.assertEqual(tuple(controls.shape), (4, 3))
        self.assertTrue(torch.allclose(controls.sum(-1), torch.ones(4), atol=1e-6))
        self.assertGreater(float(next_state.hidden.abs().sum()), 0.0)
        reset = pfc.initial_state(4)
        self.assertEqual(float(reset.hidden.abs().sum()), 0.0)

    def test_replay_priority_prefers_high_quality_novel_example(self):
        signals = {
            "novelty": torch.tensor([0.9, 0.2, 0.4]),
            "prediction_error": torch.tensor([0.8, 0.2, 0.5]),
            "quality": torch.tensor([1.0, 0.1, 0.7]),
            "learning_progress": torch.tensor([0.7, 0.2, 0.3]),
            "diversity": torch.tensor([0.9, 0.1, 0.4]),
        }
        scores = ReplayPriority.score(signals)
        selected = ReplayPriority.select(signals, 1)
        self.assertEqual(int(selected[0]), 0)
        self.assertTrue(torch.isfinite(scores).all())

    def test_replay_rejects_bad_quality_and_shape(self):
        bad = {
            "novelty": torch.tensor([0.1]),
            "prediction_error": torch.tensor([0.1]),
            "quality": torch.tensor([2.0]),
            "learning_progress": torch.tensor([0.1]),
            "diversity": torch.tensor([0.1]),
        }
        with self.assertRaises(ValueError):
            ReplayPriority.score(bad)

    def test_dual_adapters_are_identity_and_have_expected_budget(self):
        bank = DualProjectionAdapterBank(layers=8, dim=384, rank=8)
        self.assertEqual(sum(p.numel() for p in bank.parameters()), 98_304)
        q, v = torch.randn(2, 12, 384), torch.randn(2, 12, 384)
        q2, v2 = bank(0, q, v)
        self.assertTrue(torch.equal(q2, q))
        self.assertTrue(torch.equal(v2, v))

    def test_controller_is_bounded_and_validation_is_strict(self):
        s = ControllerSignals(0.8, 0.7, 0.9, 0.6, 0.5, 0.3, 1.2)
        decision = AdaptiveLearningController.decide(s)
        self.assertGreaterEqual(decision.learning_rate_multiplier, 0.5)
        self.assertLessEqual(decision.learning_rate_multiplier, 1.25)
        self.assertGreaterEqual(decision.replay_fraction, 0.05)
        self.assertLessEqual(decision.replay_fraction, 0.5)
        self.assertGreaterEqual(decision.load_balance_weight_multiplier, 0.5)
        self.assertLessEqual(decision.load_balance_weight_multiplier, 2.0)
        invalid = ControllerSignals(1.2, 0.7, 0.9, 0.6, 0.5, 0.3, 1.2)
        with self.assertRaises(ValueError):
            AdaptiveLearningController.decide(invalid)

    def test_end_to_end_synthetic_contract_and_parameter_report(self):
        model = HCDM110Prototype(dim=32, expanded_dim=128, dg_active_k=8,
                                 layers=2, rank=4)
        pooled = torch.randn(3, 32)
        keys = model.dg(pooled)
        self.assertEqual(tuple(keys.shape), (3, 128))
        memory = AssociativeMemory(capacity=12, key_dim=128, value_dim=32)
        memory.add(keys, pooled)
        retrieved, scores, _ = memory.retrieve(keys, top_k=2)
        integrated, weights, disagreement = model.integrate(pooled, retrieved)
        state, controls = model.pfc(pooled, model.pfc.initial_state(3))
        self.assertEqual(tuple(integrated.shape), (3, 32))
        self.assertEqual(tuple(state.hidden.shape), (3, 32))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(disagreement).all())
        self.assertTrue(torch.isfinite(controls).all())
        report = model.parameter_report()
        self.assertEqual(report["adapter_parameters"], 2 * 2 * 2 * 32 * 4)
        self.assertGreater(report["trainable_parameters"], 0)


# F is used only in this test module to construct normalized retrieval cues.
import torch.nn.functional as F


if __name__ == "__main__":
    unittest.main(verbosity=2)
