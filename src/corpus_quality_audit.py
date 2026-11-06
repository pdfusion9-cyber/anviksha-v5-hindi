"""Auditable corpus quality gate for Anviksha security and Hindi-English data.

This tool flags candidates for human review; it never silently deletes data.
It reports language mix, exact duplicates, domain-keyword coverage and obvious
spam/adult/promotional contamination signals at document granularity.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

DOMAIN_TERMS = {
    "crypto": {"encryption","cipher","aes","rsa","ecc","sha","hmac","hash","salt","bcrypt","argon2","pki","certificate","tls","ssl","signature","nonce","entropy","public key","private key","cryptographic","key exchange"},
    "network": {"firewall","router","switch","vpn","dns","tcp","udp","packet","subnet","vlan","proxy","gateway","port scanning","segmentation","ids","ips","waf","ddos","mitm","network security","routing"},
    "application": {"injection","xss","csrf","ssrf","deserialization","buffer overflow","rce","ssti","xxe","dependency","supply chain","sandbox","input validation","secure coding","authentication","authorization","web application","owasp"},
    "incident": {"incident response","security incident","data breach","forensics","containment","eradication","recovery","post-mortem","root cause","timeline","triage","escalation","playbook","indicators of compromise","lessons learned","security operations"},
    "governance": {"security policy","compliance","security audit","gdpr","iso 27001","nist","risk assessment","security control","framework","governance","privacy","consent","retention","data classification","standard","information security management"},
}
SPAM_RISK_TERMS = {"adult","casino","gambling","sportsbook","betting","slot machine","roulette","escort","hookup","porn","pornography","milf","dating site","free spins","online slots","jackpot"}
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

def iter_documents(path: Path):
    if path.suffix.lower()==".jsonl":
        with path.open(encoding="utf-8",errors="replace") as f:
            for line in f:
                if not line.strip(): continue
                try: rec=json.loads(line)
                except json.JSONDecodeError: continue
                if not isinstance(rec,dict): continue
                value=next((rec.get(k) for k in ("text","content","assistant","answer","document") if isinstance(rec.get(k),str) and rec[k].strip()),"")
                if value: yield value.strip()
    else:
        parts=[]
        with path.open(encoding="utf-8",errors="replace") as f:
            for line in f:
                if line.strip(): parts.append(line)
                elif parts: yield "".join(parts).strip(); parts.clear()
        if parts: yield "".join(parts).strip()

def language_counts(text):
    letters=dev=latin=0
    for ch in text:
        if ch.isalpha():
            letters+=1
            if "\u0900"<=ch<="\u097f": dev+=1
            elif "A"<=ch<="Z" or "a"<=ch<="z": latin+=1
    n=max(letters,1)
    return {"letters":letters,"devanagari_letters":dev,"latin_letters":latin,"devanagari_pct":round(100*dev/n,4),"latin_pct":round(100*latin/n,4)}

def _hits(text,terms):
    low=text.casefold()
    return sorted(t for t in terms if t in low)

def audit(path:Path,domain=None,min_domain_hits=2,sample_limit=30):
    if domain is not None and domain not in DOMAIN_TERMS: raise ValueError(f"unknown domain: {domain}")
    seen=set(); language=Counter(); flags=Counter(); total_chars=docs=dupes=shorts=domain_pass=spam_docs=0; samples=[]
    terms=DOMAIN_TERMS.get(domain,set()) if domain else set()
    for index,text in enumerate(iter_documents(path),1):
        docs+=1; total_chars+=len(text)
        digest=hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()
        duplicate=digest in seen; seen.add(digest); dupes+=int(duplicate)
        short=len(text.strip())<100; shorts+=int(short)
        language.update({k:int(v) for k,v in language_counts(text).items() if k in {"letters","devanagari_letters","latin_letters"}})
        domain_hits=_hits(text,terms) if terms else []
        domain_ok=(len(domain_hits)>=min_domain_hits) if domain else None
        domain_pass+=int(bool(domain_ok))
        spam_hits=_hits(text,SPAM_RISK_TERMS); suspicious=bool(spam_hits); spam_docs+=int(suspicious)
        reasons=[]
        if duplicate: reasons.append("exact_duplicate")
        if short: reasons.append("short_document")
        if domain and not domain_ok: reasons.append("low_domain_keyword_coverage")
        if suspicious: reasons.append("spam_or_adult_review")
        if reasons:
            flags.update(reasons)
            if len(samples)<sample_limit:
                preview=URL_RE.sub("[URL]",EMAIL_RE.sub("[EMAIL]",text[:260]))
                samples.append({"record":index,"reasons":reasons,"domain_hits":domain_hits[:12],"spam_risk_hits":spam_hits[:8],"preview":re.sub(r"\s+"," ",preview)})
    n=max(language["letters"],1)
    return {"input":str(path),"domain":domain,"documents":docs,"characters":total_chars,"unique_documents":docs-dupes,
      "exact_duplicate_documents":dupes,"short_documents_under_100_chars":shorts,"letters":language["letters"],
      "devanagari_letters":language["devanagari_letters"],"latin_letters":language["latin_letters"],
      "devanagari_pct_of_letters":round(100*language["devanagari_letters"]/n,4),
      "latin_pct_of_letters":round(100*language["latin_letters"]/n,4),
      "documents_passing_domain_keyword_floor":domain_pass if domain else None,
      "documents_with_spam_adult_review_hits":spam_docs,"flag_counts_nonexclusive":dict(flags),
      "review_required":bool(flags),"samples_for_human_review":samples}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("input",type=Path); p.add_argument("--domain",choices=sorted(DOMAIN_TERMS))
    p.add_argument("--min-domain-hits",type=int,default=2); p.add_argument("--sample-limit",type=int,default=30)
    p.add_argument("--report",type=Path,required=True); p.add_argument("--fail-on-flagged",action="store_true")
    a=p.parse_args(); result=audit(a.input,a.domain,a.min_domain_hits,a.sample_limit)
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="samples_for_human_review"},ensure_ascii=False,indent=2))
    if a.fail_on_flagged and result["review_required"]: raise SystemExit(2)
if __name__=="__main__": main()
