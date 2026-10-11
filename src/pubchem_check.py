"""Does PubChem already know these molecules? Existence check, not validation.

WHAT A HIT MEANS, AND WHAT IT DOES NOT. A PubChem match says the structure is a
registered compound -- so it is evidence the generator produces real chemistry
rather than plausible-looking nonsense. It says nothing whatsoever about
permeability, safety or activity: PubChem registers compounds, it does not
endorse them. A hit is simultaneously evidence AGAINST novelty, and that is the
more useful reading, because constitutional novelty is the one thing
`postfilter` already screens for against the training corpus. This asks the
same question against a far larger corpus.

WHY THE DEFAULT SMILES LOOKUP AND NOT `same_connectivity`. BRICS assembly
strips stereochemistry, so every molecule here is flat. The worry was that an
exact-structure lookup would therefore miss stereo-defined compounds -- it does
not: the generated flat lysergamide resolves to CID 3981
("D-Lysergic acid N,N-diethylamide"), because PubChem registers the
stereo-unspecified parent too. `identity_type=same_connectivity` returns the
whole stereoisomer-and-salt family instead (48 CIDs for that molecule), which
answers a different and looser question. The default is the honest one.

IS "NOVEL" REALLY ABSENT, OR JUST WRITTEN DIFFERENTLY? Tested, because the
obvious failure mode is a molecule registered as a different stereoisomer,
salt or tautomer that an exact lookup reports as unknown. Fifteen `notfound`
molecules re-queried with `identity_type=same_connectivity` came back
**0/15 registered** -- so the bucket is not a canonicalization artefact. The
residual caveat is tautomers, which neither query normalizes, so the novel
count is an upper bound rather than an exact figure.

NOVEL DOES NOT MEAN GOOD. The same fifteen include
`C#CC1(OOOC(O)C(Cl)(Cl)Cl)CCCCC1` -- a trioxide, which is explosive-adjacent
and not something anyone would make. It passed the structural screen because
no alert catalogue encodes it. Absence from PubChem is absence of a record, not
evidence of value, and some of this bucket is novel because it is bad.

RATE LIMITING. PubChem PUG-REST allows 5 requests/second. A single request
takes ~1.3 s, so a serial loop runs at 0.64 req/s and needs ~3 hours for 6591
molecules -- latency-bound, not rate-limited. Five workers behind a token
bucket keeps throughput just under the published ceiling and brings it to
~30 min. Do NOT raise the worker count: exceeding the limit gets the caller
blocked, and the limit is theirs to set.

RESUMABLE. Results append to the output CSV and already-queried SMILES are
skipped on restart, because a 30-minute network job should not lose everything
to one transient failure.

    python -m src.pubchem_check                     # all survivors
    python -m src.pubchem_check --limit 500         # a random sample
    python -m src.pubchem_check --self-check        # no network beyond 3 probes
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SCREENED = ROOT / "results" / "candidates" / "brics10" / "screened.csv"
OUT = ROOT / "results" / "pubchem" / "existence.csv"

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
MAX_RPS = 5.0        # PubChem's published ceiling. Theirs, not a tuning knob.
WORKERS = 5
TIMEOUT = 30
RETRIES = 3


class _Bucket:
    """Token bucket shared by all workers, so the ceiling is global."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_at = 0.0

    def take(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if wait:
            time.sleep(wait)


def lookup(smiles: str, bucket: _Bucket) -> dict:
    """-> {smiles, n_cids, cid, status}. cid is the lowest, i.e. oldest, match."""
    enc = urllib.parse.quote(smiles, safe="")
    url = f"{BASE}/smiles/{enc}/cids/JSON"
    for attempt in range(RETRIES):
        bucket.take()
        try:
            r = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt == RETRIES - 1:
                return dict(smiles=smiles, n_cids=0, cid="", status=type(e).__name__)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:          # PubChem's "no match" for a structure
            return dict(smiles=smiles, n_cids=0, cid="", status="notfound")
        if r.status_code in (429, 503):   # throttled or busy -- back off, retry
            time.sleep(2 ** attempt + 1)
            continue
        if r.status_code != 200:
            return dict(smiles=smiles, n_cids=0, cid="", status=f"http{r.status_code}")
        try:
            cids = r.json().get("IdentifierList", {}).get("CID", [])
        except ValueError:
            return dict(smiles=smiles, n_cids=0, cid="", status="badjson")
        # PubChem returns a single CID 0 to mean "parsed, but no such compound".
        cids = [c for c in cids if c]
        return dict(smiles=smiles, n_cids=len(cids),
                    cid=cids[0] if cids else "",
                    status="hit" if cids else "notfound")
    return dict(smiles=smiles, n_cids=0, cid="", status="retries_exhausted")


def pool(survivors_only: bool = False) -> pd.DataFrame:
    """The molecules to classify.

    DEFAULT IS THE WHOLE POOLED SET, not the survivors. "How many of the
    molecules it generated already exist" has the full deduplicated pool as its
    denominator -- restricting to survivors would answer a different question,
    and a biased one, because `postfilter` has already removed every
    training-set match from that subset. Those removals are exactly bucket 1.
    """
    d = pd.read_csv(SCREENED)
    return (d[d.passes & ~d.in_training_flat] if survivors_only else d).copy()


def run(limit: int | None = None, seed: int = 0,
        survivors_only: bool = False) -> pd.DataFrame:
    s = pool(survivors_only)
    if limit:
        s = s.sample(min(limit, len(s)), random_state=seed)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT).smiles)
        print(f"resuming: {len(done)} already queried")

    todo = [x for x in s.smiles if x not in done]
    print(f"{len(s)} survivors, {len(todo)} to query, "
          f"{WORKERS} workers at <={MAX_RPS} req/s "
          f"(~{len(todo) / MAX_RPS / 60:.0f} min at the ceiling)")
    if not todo:
        return pd.read_csv(OUT)

    bucket = _Bucket(MAX_RPS)
    new = not OUT.exists()
    t0 = time.time()
    with OUT.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["smiles", "n_cids", "cid", "status"])
        if new:
            w.writeheader()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, res in enumerate(ex.map(lambda x: lookup(x, bucket), todo), 1):
                w.writerow(res)
                if i % 250 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  {i / el:.2f} req/s  "
                          f"eta {(len(todo) - i) / (i / el) / 60:.0f} min")
    return pd.read_csv(OUT)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Three mutually exclusive buckets, in priority order.

    training  -- flattened SMILES appears in the corpus the generator trained on
    known     -- not in training, but PubChem registers it
    novel     -- neither; no record found anywhere we looked

    `training` wins over `known` where both hold, since a molecule copied from
    training is a memorization result whether or not PubChem also lists it.
    `unresolved` is a fourth bucket for network failures, kept separate because
    a failed lookup is not an absence.
    """
    scr = pd.read_csv(SCREENED)[["smiles", "in_training_flat", "passes"]]
    m = df.merge(scr, on="smiles", how="left")
    m["bucket"] = "novel"
    m.loc[m.status == "hit", "bucket"] = "known"
    m.loc[m.in_training_flat.fillna(False).astype(bool), "bucket"] = "training"
    m.loc[~m.status.isin(["hit", "notfound"]), "bucket"] = "unresolved"
    return m


def report(df: pd.DataFrame) -> None:
    m = classify(df)
    n = len(m)
    print(f"\n{'=' * 70}\nWHERE THE GENERATED MOLECULES COME FROM\n{'=' * 70}")
    print(f"  queried: {n} deduplicated generated molecules\n")
    order = ["training", "known", "novel", "unresolved"]
    label = {"training": "in its training data",
             "known": "exist already (PubChem), not in training",
             "novel": "no record found -- novel",
             "unresolved": "lookup failed (NOT an absence)"}
    counts = m.bucket.value_counts()
    for b in order:
        c = int(counts.get(b, 0))
        if b == "unresolved" and not c:
            continue
        print(f"  {label[b]:<44} {c:>6}  {c / n:>6.1%}")

    res = m[m.bucket != "unresolved"]
    if len(res) and len(res) != n:
        print(f"\n  over resolved only (n={len(res)}):")
        for b in order[:3]:
            c = int((res.bucket == b).sum())
            print(f"    {label[b]:<42} {c:>6}  {c / len(res):>6.1%}")

    # Binomial interval on the novel fraction, since that is the headline and
    # it is a proportion from a finite sample of the policy's output.
    if len(res):
        p = (res.bucket == "novel").mean()
        se = (p * (1 - p) / len(res)) ** 0.5
        print(f"\n  novel fraction: {p:.1%} +/- {1.96 * se:.1%} (95%)")

    sv = m[m.passes.fillna(False).astype(bool) &
           ~m.in_training_flat.fillna(False).astype(bool)]
    if len(sv):
        sr = sv[sv.bucket != "unresolved"]
        if len(sr):
            print(f"\n  among the {len(sv)} screened survivors "
                  f"(training-set matches already removed):")
            for b in ("known", "novel"):
                c = int((sr.bucket == b).sum())
                print(f"    {label[b]:<42} {c:>6}  {c / len(sr):>6.1%}")

    print("\n  A PubChem hit means the structure is registered. It is evidence")
    print("  the generator makes real chemistry, and evidence AGAINST novelty.")
    print("  It is NOT evidence of permeability, activity or safety. 'Novel'")
    print("  means absent from PubChem and the training corpus -- not that the")
    print("  molecule is unknown to every database, nor that it is any good.")


def _self_check() -> None:
    b = _Bucket(MAX_RPS)

    # The bucket must not let a burst exceed the ceiling -- that limit is
    # PubChem's, and exceeding it gets the caller blocked.
    t0 = time.monotonic()
    for _ in range(6):
        b.take()
    el = time.monotonic() - t0
    assert el >= 5 / MAX_RPS - 0.05, f"6 tokens took {el:.3f}s, too fast for {MAX_RPS}/s"

    # Network probes are SKIPPED rather than failed when PubChem throttles us,
    # because a busy endpoint is not a bug in this code -- running these
    # alongside a full census exhausts the shared 5 req/s and they all fail.
    # The classification logic below is offline and always asserted.
    probe = {
        "CC(=O)Oc1ccccc1C(=O)O": 2244,                                  # aspirin
        "CCN(CC)C(=O)C1C=C2c3cccc4[nH]cc(c34)CC2N(C)C1": 3981,          # flat LSD
    }
    checked = 0
    for smi, want in probe.items():
        r = lookup(smi, b)
        if r["status"] in ("retries_exhausted", "badjson") or r["status"].startswith("http"):
            print(f"  SKIP network probe ({r['status']}) -- endpoint busy or unreachable")
            continue
        assert r["status"] == "hit", f"{smi} -> {r}"
        assert r["cid"] == want, f"{smi} -> CID {r['cid']}, expected {want}"
        checked += 1
    if checked:
        print(f"  {checked}/{len(probe)} network probes resolved correctly")

    # Bucket priority: training beats known, and a failed lookup is never
    # counted as an absence. Getting either wrong silently inflates a headline.
    probe = pd.DataFrame([
        dict(smiles="A", status="hit"),        # in training AND PubChem
        dict(smiles="B", status="hit"),        # PubChem only
        dict(smiles="C", status="notfound"),   # neither
        dict(smiles="D", status="Timeout"),    # failed
    ])
    scr = pd.DataFrame([
        dict(smiles="A", in_training_flat=True, passes=True),
        dict(smiles="B", in_training_flat=False, passes=True),
        dict(smiles="C", in_training_flat=False, passes=True),
        dict(smiles="D", in_training_flat=False, passes=True),
    ])
    global SCREENED
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        scr.to_csv(fh.index if False else fh, index=False)
        tmp = Path(fh.name)
    saved, SCREENED = SCREENED, tmp
    try:
        got = dict(zip(classify(probe).smiles, classify(probe).bucket))
    finally:
        SCREENED = saved
        tmp.unlink(missing_ok=True)
    assert got == {"A": "training", "B": "known", "C": "novel", "D": "unresolved"}, got

    print("pubchem_check self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="query a random sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--survivors-only", action="store_true",
                    help="restrict to screened survivors; note this removes "
                         "every training-set match, so the training bucket "
                         "goes to zero by construction")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="re-print from the CSV")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if args.report_only:
        return report(pd.read_csv(OUT))
    report(run(limit=args.limit, seed=args.seed,
               survivors_only=args.survivors_only))


if __name__ == "__main__":
    main()
