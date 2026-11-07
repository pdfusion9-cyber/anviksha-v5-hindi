from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from benchmark_ca3_dg import main as benchmark_main


class TestCA3DGBenchmark(unittest.TestCase):
    def test_one_thousand_example_benchmark_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "benchmark.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "benchmark_ca3_dg.py"),
                 "--out", str(out), "--examples", "1000", "--topics", "100",
                 "--dim", "64", "--expansion-dim", "512", "--active-k", "32",
                 "--batch-size", "50"],
                check=True, capture_output=True, text=True, timeout=120
            )
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["examples"], 1000)
            self.assertEqual(report["pattern_separation_collision"]["same_facet_distinct_pairs"], 2000)
            self.assertEqual(report["dimensions"]["retrieval_k"], 10)
            for section in ("raw_cosine_baseline", "dg_ca3_sparse_retrieval"):
                self.assertGreaterEqual(report[section]["precision_at_10"], 0.0)
                self.assertLessEqual(report[section]["precision_at_10"], 1.0)
                self.assertGreaterEqual(report[section]["recall_at_10"], 0.0)
                self.assertLessEqual(report[section]["recall_at_10"], 1.0)
            collision = report["pattern_separation_collision"]
            self.assertGreaterEqual(collision["exact_support_collision_rate"], 0.0)
            self.assertLessEqual(collision["near_support_collision_rate"], 1.0)
            self.assertIn("dg_ca3_retrieval", report["latency"])
            self.assertFalse(report["checkpoint_loaded"])
            self.assertFalse(report["production_model_modified"])
            self.assertIn('"examples": 1000', result.stdout)

    def test_rejects_non_contract_dataset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run(
                [sys.executable, str(ROOT / "benchmark_ca3_dg.py"),
                 "--out", str(Path(tmp) / "x.json"), "--examples", "900"],
                capture_output=True, text=True
            )
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("exactly 1,000", p.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
