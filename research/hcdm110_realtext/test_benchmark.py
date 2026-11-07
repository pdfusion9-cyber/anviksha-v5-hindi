import sys, tempfile, unittest, csv
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from benchmark import feature_vector, normalize_miracl_row, candidate_pool, retrieval_metrics, load_phinc, split_phinc

class RealTextTests(unittest.TestCase):
 def test_unicode_feature_vector_is_deterministic(self):
  a=feature_vector("नमस्ते दुनिया")
  self.assertTrue(torch.equal(a,feature_vector("नमस्ते दुनिया")))
  self.assertTrue(torch.isfinite(a).all())
  self.assertAlmostEqual(float(torch.linalg.vector_norm(a)),1.0,places=5)
 def test_miracl_schema_and_qrels(self):
  row=normalize_miracl_row({"query_id":"q1","query":"question",
   "positive_passages":[{"docid":"p1","text":"positive"}],
   "negative_passages":[{"docid":"n1","text":"negative"}]})
  ev=candidate_pool([row])
  self.assertEqual(len(ev["doc_ids"]),2)
  scores=torch.zeros((1,len(ev["doc_ids"])))
  scores[0,ev["doc_ids"].index("p1")]=.9
  scores[0,ev["doc_ids"].index("n1")]=.1
  metrics=retrieval_metrics(scores,ev)
  self.assertEqual(metrics["mrr_at_10"],1.0)
 def test_phinc_dedup_and_document_disjoint_split(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"p.csv"
   with p.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["Sentence","English_Translation"]);w.writeheader()
    for i in range(80):
     w.writerow({"Sentence":f"yeh sentence {i} hai","English_Translation":f"this is translation number {i}"})
   rows,_=load_phinc(p); splits=split_phinc(rows)
   ids=[{x["docid"] for x in splits[k]} for k in splits]
   self.assertFalse(ids[0]&ids[1]);self.assertFalse(ids[0]&ids[2]);self.assertFalse(ids[1]&ids[2])
if __name__=="__main__":unittest.main(verbosity=2)
