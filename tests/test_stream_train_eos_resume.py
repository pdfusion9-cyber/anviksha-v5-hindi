from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stream_train import WindowStream, restore_training_checkpoint


class TinyTokenizer:
    eos = 99

    def encode(self, text: str):
        return [ord(ch) % 50 for ch in text]


class ResumeTests(unittest.TestCase):
    def test_document_boundary_appends_eos(self):
        stream = WindowStream(
            iter(["abcdefghijklmnopqrstuvwx", "ZYXWVUTSRQPONMLKJIHGFEDC"]),
            TinyTokenizer(), seq_len=4, terms=[], buffer_windows=2,
        )
        batch = stream.batch(2, __import__("numpy").random.default_rng(7))
        self.assertIsNotNone(batch)
        self.assertTrue(any(99 in row for row in stream.buf))

    def test_stream_state_roundtrip_preserves_buffer_and_cursor(self):
        docs = ["abcdefghijklmnopqrstuvwx", "ZYXWVUTSRQPONMLKJIHGFEDC",
                "0123456789abcdefghijklmn"]
        tok = TinyTokenizer()
        first = WindowStream(iter(docs), tok, seq_len=4, terms=[], buffer_windows=2)
        rng = __import__("numpy").random.default_rng(17)
        first_batch = first.batch(1, rng)
        self.assertIsNotNone(first_batch)
        saved = first.state_dict()
        saved_rng_state = rng.bit_generator.state

        resumed = WindowStream(iter(docs), tok, seq_len=4, terms=[],
                               buffer_windows=2, state=saved)
        expected = first.batch(1, rng)
        resumed_rng = __import__("numpy").random.default_rng()
        resumed_rng.bit_generator.state = saved_rng_state
        actual = resumed.batch(1, resumed_rng)
        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual)
        self.assertEqual(expected.tolist(), actual.tolist())
        self.assertEqual(first.state_dict(), resumed.state_dict())

    def test_optimizer_state_is_restored_and_required(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.ones(2, 3)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()
        payload = {
            "model": model.state_dict(),
            "config": {},
            "step": 23,
            "optimiser": optimizer.state_dict(),
        }

        new_model = torch.nn.Linear(3, 2)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        step = restore_training_checkpoint(
            payload, new_model, new_optimizer, require_optimiser_state=True
        )
        self.assertEqual(step, 23)
        self.assertEqual(len(new_optimizer.state), len(optimizer.state))

        payload_without_optimizer = dict(payload)
        payload_without_optimizer["optimiser"] = None
        with self.assertRaisesRegex(ValueError, "no optimiser state"):
            restore_training_checkpoint(
                payload_without_optimizer, new_model, new_optimizer,
                require_optimiser_state=True
            )


if __name__ == "__main__":
    unittest.main()
