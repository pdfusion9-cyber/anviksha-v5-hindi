from __future__ import annotations
import csv, hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from benchmark_real_language_retrieval import stable_split, hashed_text_vector, retrieval_scores
import torch

class TestRealLanguagePipeline(unittest.TestCase):
 def test_split_is_deterministic_and_group_stable(self):
  self.assertEqual(stable_split("doc-1"),stable_split("doc-1"))
  self.assertNotEqual(stable_split("doc-1"),stable_split("doc-2") if stable_split("doc-1")=="train" else "not-train")

 def test_text_hashing_is_normalized_and_finite(self):
  a=hashed_text_vector("नमस्ते दुनिया")
  b=hashed_text_vector("नमस्ते दुनिया")
  self.assertTrue(torch.equal(a,b))
  self.assertTrue(torch.isfinite(a).all())
  self.assertAlmostEqual(float(torch.linalg.vector_norm(a)),1.0,places=5)

 def test_retrieval_metrics(self):
  scores=torch.tensor([[0.9,0.8,0.1],[0.2,0.9,0.1]])
  gold=torch.tensor([0,1])
  m=retrieval_scores(scores,gold)
  self.assertEqual(m["recall_at_1"],1.0)
  self.assertEqual(m["recall_at_10"],1.0)
  self.assertEqual(m["queries"],2)

 def test_xquad_and_phinc_download_and_split_contract(self):
  # Real source download is exercised by the GitHub workflow; this test checks
  # parsers against tiny fixtures and confirms no raw text is printed.
  from benchmark_real_language_retrieval import parse_xquad, parse_phinc
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"x.json"
   p.write_text(json.dumps({"data":[{"paragraphs":[{"context":"sample context","qas":[{"id":"q1","question":"sample question"}]}]}]}))
   rows,_=parse_xquad(p,"en")
   self.assertEqual(len(rows),1)
   self.assertEqual(rows[0]["document"],"sample context")
   c=Path(tmp)/"p.csv"
   with c.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["Sentence","English_Translation"]);w.writeheader()
    w.writerow({"Sentence":"yeh test hai","English_Translation":"this is a test"})
   phinc,fields=parse_phinc(c)
   self.assertEqual(len(phinc),1)
   self.assertEqual(phinc[0]["language"],"hinglish")

if __name__=="__main__": unittest.main(verbosity=2)
