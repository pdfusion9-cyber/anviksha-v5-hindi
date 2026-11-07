from __future__ import annotations
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).parent
class TestLearnableDG(unittest.TestCase):
 def test_encoder_learns_and_metrics_are_well_formed(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/"learned.json"
   subprocess.run([sys.executable,str(ROOT/"benchmark_learnable_dg.py"),"--out",str(out),
    "--seeds","11","22","--noises","0.05","0.20","--epochs","5",
    "--expansion","256","--active-k","24"],check=True,capture_output=True,text=True,timeout=120)
   d=json.loads(out.read_text())
   self.assertTrue(d["dataset"]["train_memory_query_disjoint"])
   self.assertEqual(d["dataset"]["memory_records"],1000)
   self.assertEqual(len(d["summary_mean_over_seeds"]),2)
   for row in d["summary_mean_over_seeds"]:
    for method in ("raw_cosine","fixed_dg","learned_dg","hybrid"):
     for metric in ("precision_at_10","recall_at_10","wrong_facet_at_10_rate","unrelated_topic_at_10_rate"):
      self.assertTrue(0<=row[method+"_"+metric]<=1)
   self.assertFalse(d["checkpoint_loaded"])
   self.assertFalse(d["production_model_modified"])
if __name__=="__main__": unittest.main(verbosity=2)
