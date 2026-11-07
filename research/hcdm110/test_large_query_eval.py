from __future__ import annotations
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).parent
class TestLargeQueryEvaluation(unittest.TestCase):
 def test_paired_independent_query_contract(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/"large_eval.json"
   subprocess.run([sys.executable,str(ROOT/"benchmark_large_query_eval.py"),
    "--out",str(out),"--seeds","601","702","--noises","0.20",
    "--queries-per-facet","2","--epochs","2","--bootstrap-replicates","20",
    "--timing-repeats","1"],check=True,capture_output=True,text=True,timeout=180)
   d=json.loads(out.read_text())
   self.assertTrue(d["dataset"]["training_validation_memory_test_disjoint"])
   self.assertEqual(d["dataset"]["total_queries_per_noise_all_seeds"],800)
   self.assertEqual(len(d["per_seed"]),2)
   row=d["summary"][0]
   self.assertEqual(row["total_queries"],800)
   for method in ("learned","hybrid"):
    for metric in ("precision_at_10","recall_at_10","top1_relevant_accuracy","ndcg_at_10"):
     ci=row["paired_bootstrap_vs_raw"][method][metric]
     self.assertEqual(ci["queries"],800)
     self.assertTrue(ci["ci95_low"]<=ci["ci95_high"])
   for seed in d["per_seed"]:
    self.assertEqual(seed["validation"]["objective"],"validation NDCG@10 only")
    self.assertTrue(seed["test"][0]["end_to_end_latency"]["learned"]["includes_query_encoding"])
   self.assertFalse(d["checkpoint_loaded"])
   self.assertFalse(d["production_model_modified"])
if __name__=="__main__":unittest.main(verbosity=2)
