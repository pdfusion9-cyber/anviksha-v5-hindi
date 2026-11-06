from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from corpus_quality_audit import audit

def test_language_domain_spam_and_duplicate_flags(tmp_path):
    p=tmp_path/"docs.txt"
    clean="Hindi सुरक्षा और firewall network security तथा encryption policy. "*5
    noisy="Best casino betting slots and adult dating website offer. "*5
    p.write_text("\n\n".join([clean,noisy,clean]),encoding="utf-8")
    r=audit(p,domain="network",min_domain_hits=1)
    assert r["documents"]==3
    assert r["exact_duplicate_documents"]==1
    assert r["documents_with_spam_adult_review_hits"]==1
    assert r["devanagari_letters"]>0
    assert r["documents_passing_domain_keyword_floor"]==2
    assert r["review_required"] is True

def test_jsonl_reader_uses_one_record_one_document(tmp_path):
    p=tmp_path/"docs.jsonl"
    p.write_text("\n".join([
      json.dumps({"text":"हिंदी भाषा में network firewall सुरक्षा की जानकारी."},ensure_ascii=False),
      json.dumps({"prompt":"ignored prompt","assistant":"English response with encryption details."}),
      "not-json"]),encoding="utf-8")
    r=audit(p)
    assert r["documents"]==2 and r["devanagari_letters"]>0 and r["latin_letters"]>0
