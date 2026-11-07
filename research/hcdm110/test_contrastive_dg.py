from __future__ import annotations
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import torch
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from benchmark_contrastive_dg import rank_metrics

class TestContrastiveDG(unittest.TestCase):
    def test_facet_and_topic_error_metrics_are_distinct(self):
        # Query label 0; labels 0 and 1 are two facets of the same topic.
        labels=torch.arange(10)
        query_labels=torch.tensor([0])
        scores=torch.arange(10,0,-1,dtype=torch.float32).reshape(1,10)
        m=rank_metrics(scores,labels,query_labels)
        self.assertAlmostEqual(m["precision_at_10"],0.1,places=6)
        self.assertAlmostEqual(m["wrong_facet_at_10_rate"],0.1,places=6)
        self.assertAlmostEqual(m["unrelated_topic_at_10_rate"],0.8,places=6)

    def test_contrastive_benchmark_contract_and_validation_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"contrastive.json"
            subprocess.run([sys.executable,str(ROOT/"benchmark_contrastive_dg.py"),
                "--out",str(out),"--seeds","101","202","--top-k","32","64",
                "--noises","0.10","0.20","--epochs","2","--expansion","256",
                "--scale-capacities","1000","2000"],check=True,capture_output=True,
                text=True,timeout=180)
            d=json.loads(out.read_text())
            self.assertTrue(d["dataset"]["train_memory_validation_test_disjoint"])
            self.assertEqual(d["seeds"],[101,202])
            self.assertEqual(d["top_k_values"],[32,64])
            self.assertEqual(len(d["per_seed_per_k"]),4)
            self.assertEqual(len(d["memory_scaling"]),8)
            for run in d["per_seed_per_k"]:
                w=run["validation"]["fusion_learned_weight"]
                self.assertTrue(0.0<=w<=1.0)
                self.assertEqual(len(run["test"]),2)
                for result in run["test"]:
                    for method in ("raw_cosine","fixed_dg","learned_contrastive_dg","hybrid_validation_tuned"):
                        self.assertTrue(0.0<=result[method]["precision_at_10"]<=1.0)
                        self.assertTrue(0.0<=result[method]["wrong_facet_at_10_rate"]<=1.0)
            self.assertFalse(d["checkpoint_loaded"])
            self.assertFalse(d["production_model_modified"])

if __name__=="__main__": unittest.main(verbosity=2)
