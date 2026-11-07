from __future__ import annotations
import csv, tempfile, unittest, sys
from pathlib import Path
import torch
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from benchmark_real_language_retrieval import (
 stable_split, hashed_text_vector, parse_phinc, rank_metrics,
 build_candidate_eval, symmetric_multi_positive_infonce
)

class TestRealLanguagePipeline(unittest.TestCase):
 def test_stable_split(self):
  self.assertEqual(stable_split("q1","salt"),stable_split("q1","salt"))
  self.assertNotEqual(stable_split("q1","salt"),stable_split("q1","other"))

 def test_unicode_hashing(self):
  a=hashed_text_vector("नमस्ते दुनिया")
  self.assertTrue(torch.equal(a,hashed_text_vector("नमस्ते दुनिया")))
  self.assertTrue(torch.isfinite(a).all())
  self.assertAlmostEqual(float(torch.linalg.vector_norm(a)),1.0,places=5)

 def test_phinc_parser_and_dedup(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"phinc.csv"
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["Sentence","English_Translation"]);w.writeheader()
    w.writerow({"Sentence":"yeh test hai","English_Translation":"this is a test"})
    w.writerow({"Sentence":"yeh test hai","English_Translation":"this is a test"})
   rows,fields=parse_phinc(p)
   self.assertEqual(len(rows),1)
   self.assertEqual(rows[0]["query"],"yeh test hai")
   self.assertIn("Sentence",fields)

 def test_qrels_rank_metrics(self):
  scores=torch.tensor([[.9,.8,.1],[.2,.9,.1]])
  qrels={"q1":{"d1":1},"q2":{"d2":1}}
  m=rank_metrics(scores,qrels,["q1","q2"],["d1","d2","d3"])
  self.assertEqual(m["recall_at_1"],1.0)
  self.assertEqual(m["mrr_at_10"],1.0)
  self.assertEqual(m["queries_scored"],2)

 def test_contrastive_loss_finite(self):
  q=torch.eye(3);d=torch.eye(3);g=torch.tensor([1,2,3])
  loss=symmetric_multi_positive_infonce(q,d,g)
  self.assertTrue(torch.isfinite(loss))

if __name__=="__main__":unittest.main(verbosity=2)
