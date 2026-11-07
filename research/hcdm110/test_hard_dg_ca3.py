from __future__ import annotations
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).parent
class TestHardDGCA3(unittest.TestCase):
    def test_multiseed_noise_and_scaling_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"hard.json"
            subprocess.run([sys.executable,str(ROOT/"benchmark_hard_dg_ca3.py"),
                "--out",str(out),"--seeds","11","22","--noises","0.05","0.20",
                "--scale-capacities","1000","10000"],check=True,capture_output=True,text=True,timeout=120)
            d=json.loads(out.read_text())
            self.assertEqual(d["dataset"]["stored_records"],1000)
            self.assertFalse(d["dataset"]["exact_stored_query_reuse"])
            self.assertEqual(len(d["results"]),2)
            for seed in d["results"]:
                self.assertEqual(len(seed["noise_levels"]),2)
                for level in seed["noise_levels"]:
                    for method in ("raw_cosine","dg_ca3"):
                        m=level[method]
                        self.assertTrue(0<=m["precision_at_10"]<=1)
                        self.assertTrue(0<=m["recall_at_10"]<=1)
                        self.assertTrue(0<=m["cross_facet_false_retrieval_rate"]<=1)
            self.assertEqual([x["capacity"] for x in d["memory_scaling_proxy"]],[1000,10000])
            self.assertFalse(d["checkpoint_loaded"])
            self.assertFalse(d["production_model_modified"])
if __name__=="__main__": unittest.main(verbosity=2)
