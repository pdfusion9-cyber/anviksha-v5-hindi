"""What Hindi data actually exists, under what licence, and what each is good for.

Growing the knowledge base and the training corpus is the right move, but it can
only be done safely if every source is recorded with its licence and its role.
A source that cannot be licensed cannot be trained on, however good the text is,
and a source that is not reachable from the machine doing the work is not a
source.

The catalogue below is checked against reality rather than remembered: each entry
declares a host, and `--probe` tests whether that host is reachable from wherever
this runs. The sandbox blocks most of the internet; a GitHub runner does not, so
the same file answers a different question depending on where it is run.

Stages, because the corpus is built in order rather than all at once:

    1  language        a broad, clean base of Hindi prose
    2  instruction     question/answer pairs with a verifiable answer
    3  reasoning       multi-step problems where the working is checkable
    4  science         domain text that is correct by source, not by guess

A source belongs to exactly one stage. Mixing them is how a corpus quietly
becomes mostly one kind of thing.
"""
from __future__ import annotations

import argparse
import json
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Source:
    name: str
    stage: int
    kind: str                    # monolingual | instruction | reasoning | science
    host: str                    # where it actually lives
    licence: str
    licence_ok: bool             # may this be trained on
    size_note: str
    docs: str = ""
    tokens: str = ""
    notes: str = ""
    verified: bool = False       # have I confirmed this figure myself
    reachable: bool | None = None


# Figures marked verified were read from the project's own page during this
# session. Figures marked unverified come from the dataset card or the plan and
# have not been confirmed here -- they are listed so they can be checked, not
# because they are trusted.
CATALOGUE: list[Source] = [
    Source("HPLT 3.0 Hindi (hin_Deva)", 1, "monolingual",
           "data.hplt-project.org",
           "CC0", True,
           "59.97 GB compressed, 17 shards", docs="36.33M", tokens="26.77B",
           notes="figures read from the project page. The Hugging Face mirror "
                 "HPLT/HPLT3.0 holds only a README, not the data, so the download "
                 "has to happen on a machine with open internet; the sandbox "
                 "blocks this host, a GitHub runner does not. WDS quality bins "
                 "let you take the good tail instead of all 59.97 GB",
           verified=True),
    Source("Wikipedia Hindi (20231101.hi)", 1, "monolingual",
           "huggingface.co/wikimedia/wikipedia", "CC BY-SA 4.0", True,
           "~700 MB", notes="encyclopaedic register, small next to HPLT but clean",
           verified=False),
    Source("ai4bharat indic-instruct-data-v0.1", 2, "instruction",
           "huggingface.co/ai4bharat/indic-instruct-data-v0.1",
           "mixed, per subset", True,
           "dolly / anudesh / flan_v2 / oasst1 / wikihow / lm_sys",
           notes="already used in the 59,430-pair base", verified=False),
    Source("smangrul/hindi_instruct_v1", 2, "instruction",
           "huggingface.co/smangrul/hindi_instruct_v1", "see card", True,
           "10,854 usable pairs", notes="contributed the second largest share "
           "of the current base", verified=False),
    Source("atharvanighot/Hindi-Instruct-500K", 2, "instruction",
           "huggingface.co/atharvanighot/Hindi-Instruct-500K", "see card", True,
           "39,429 usable pairs after the window filter",
           notes="largest single source in the current base", verified=False),
    Source("guneetsk99/hindi_instruction_set_187K", 2, "instruction",
           "huggingface.co/guneetsk99/hindi_instruction_set_187K", "see card", True,
           "~700 usable pairs", notes="mostly filtered out by the window; worth "
           "re-checking with a larger context", verified=False),
    Source("FreedomIntelligence/evol-instruct-hindi", 2, "instruction",
           "huggingface.co/FreedomIntelligence/evol-instruct-hindi", "see card",
           True, "831 usable pairs",
           notes="evolved instructions, closer to reasoning than plain QA",
           verified=False),
    Source("HPLT science slice (to be filtered)", 4, "science",
           "data.hplt-project.org", "CC0", True, "subset of stage 1",
           notes="not a separate download: science text has to be selected out "
           "of the monolingual base by topic, and that selection is a step to "
           "build, not a dataset to fetch", verified=False),
    Source("NCERT / Indian curriculum textbooks", 4, "science",
           "local upload only", "government publication, check per book", False,
           "unknown", notes="the right source for Hindi science at school level, "
           "but the licence has to be established per title before any use",
           verified=False),
]


def probe(host: str) -> bool:
    """Can this machine reach the host? Recorded, because it decides the plan.

    Tested with an HTTPS request rather than a raw socket: this sandbox reaches
    the internet only through an HTTP proxy, so a direct connection fails even
    for hosts that are allowed. A socket test would report everything as
    unreachable and send the work to a runner for no reason.
    """
    import urllib.error
    import urllib.request

    h = host.split("/")[0].strip()
    if "huggingface.co" in h:
        h = "huggingface.co"
    url = f"https://{h}/"
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=8)
        return True
    except urllib.error.HTTPError:
        return True          # answered, which is what reachability means
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="test host reachability")
    ap.add_argument("--out", type=Path, default=Path("runs/data_inventory.json"))
    args = ap.parse_args()

    if args.probe:
        for s in CATALOGUE:
            s.reachable = probe(s.host)

    print("=" * 92)
    print(f"{'source':<38} {'stage':>5} {'licence ok':>10} {'reach':>6}  licence")
    print("=" * 92)
    for s in CATALOGUE:
        reach = "-" if s.reachable is None else ("yes" if s.reachable else "NO")
        ok = "yes" if s.licence_ok else "CHECK"
        print(f"{s.name[:38]:<38} {s.stage:>5} {ok:>10} {reach:>6}  {s.licence}")

    print()
    by_stage: dict[int, list[Source]] = {}
    for s in CATALOGUE:
        by_stage.setdefault(s.stage, []).append(s)
    names = {1: "language base", 2: "instruction", 3: "reasoning", 4: "science"}
    for stage in sorted(by_stage):
        print(f"stage {stage} - {names.get(stage, '?')}")
        for s in by_stage[stage]:
            size = s.tokens or s.size_note
            print(f"   {s.name[:40]:<42} {size}")
            if s.notes:
                print(f"      {s.notes}")
        print()

    blocked = [s for s in CATALOGUE if s.reachable is False]
    if blocked:
        print(f"{len(blocked)} source(s) unreachable from here:")
        for s in blocked:
            print(f"   {s.name} -> {s.host.split('/')[0]}")
        print("   the sandbox blocks most hosts; a GitHub runner does not, so the")
        print("   heavy work belongs there and this list is about where to run it")

    unlicensed = [s for s in CATALOGUE if not s.licence_ok]
    if unlicensed:
        print(f"\n{len(unlicensed)} source(s) cannot be trained on until the licence "
              f"is established:")
        for s in unlicensed:
            print(f"   {s.name} -- {s.licence}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(s) for s in CATALOGUE], indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    print(f"\ninventory written to {args.out}")


if __name__ == "__main__":
    main()
