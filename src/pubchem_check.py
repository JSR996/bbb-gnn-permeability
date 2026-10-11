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
OUT = ROOT / "results" / "pubchem" / "existence.csv"          # raw query log
CLASSIFIED = ROOT / "results" / "pubchem" / "classified_all.csv"
SURVIVORS_OUT = ROOT / "results" / "pubchem" / "classified_survivors.csv"
SUMMARY = ROOT / "results" / "pubchem" / "summary.csv"        # both populations
SURVIVOR_RAW = ROOT / "results" / "pubchem" / "existence_survivors.csv"
KEY_CACHE = ROOT / "results" / "pubchem" / "key_cache.csv"    # inchikey -> cid

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
MAX_RPS = 4.0        # under PubChem's published 5/s ceiling, for margin
WORKERS = 4
TIMEOUT = 30
RETRIES = 6          # throttling is transient; giving up early loses the row

# THE BUCKET IS PER-PROCESS, SO DO NOT RUN TWO OF THESE AT ONCE. Learned the
# hard way: a census plus a handful of ad-hoc probes from another process each
# honoured 5 req/s locally, exceeded it jointly at PubChem, and 86 of the first
# 450 rows came back `retries_exhausted` -- a 19% loss that looked like network
# flakiness and was self-inflicted. Finish or stop the census before running
# any other query against PubChem.


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


BATCH = 150          # InChIKeys per POST; ~4 KB of body, well within limits


def lookup_batch(keys: list[str], bucket: _Bucket) -> dict[str, int]:
    """{inchikey: cid} for the keys PubChem knows. Absent keys are absent.

    WHY BATCHED, AND WHY BY InChIKey. One GET per molecule is the obvious
    implementation and it is the wrong shape for this API: 19219 requests at
    ~1.3 s latency needs five workers just to reach 4 req/s, and running that
    alongside anything else got this client HTTP 429'd outright. PubChem
    accepts a comma-separated list, which turns the same census into ~130
    requests.

    It has to be the `property` endpoint, not `cids`. A batched `/cids/JSON`
    returns one flat CID list with NO correspondence to the inputs, so there is
    no way to tell which key matched -- the mapping is the whole point. Asking
    for the InChIKey property back gives CID and key on the same row.

    Sent as POST because 150 keys is ~4 KB, far past what a GET URL takes.
    """
    body = {"inchikey": ",".join(keys)}
    url = f"{BASE}/inchikey/property/InChIKey/JSON"
    for attempt in range(RETRIES):
        bucket.take()
        try:
            r = requests.post(url, data=body, timeout=TIMEOUT)
        except requests.RequestException:
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(2 ** attempt + 1, 30))
            continue
        if r.status_code == 404:
            return {}                      # none of these keys is registered
        if r.status_code in (429, 503):
            time.sleep(min(2 ** attempt + 1, 30))
            continue
        r.raise_for_status()
        props = r.json().get("PropertyTable", {}).get("Properties", [])
        out: dict[str, int] = {}
        for p in props:
            k, c = p.get("InChIKey"), p.get("CID")
            if k and c and k not in out:   # keep the lowest CID per key
                out[k] = c
        return out
    raise RuntimeError(f"batch of {len(keys)} exhausted retries")


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
            # PubChem's own guidance is to slow down rather than hammer. Capped
            # so one bad patch cannot stall a worker for minutes.
            time.sleep(min(2 ** attempt + 1, 30))
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


def inchikeys(smiles: list[str]) -> dict[str, str]:
    """Computed locally with RDKit -- no network. 19219/19219 succeed."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey
    RDLogger.DisableLog("rdApp.*")
    out = {}
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            k = MolToInchiKey(m)
        except Exception:
            continue
        if k:
            out[s] = k
    return out


def _load_cache() -> dict[str, int | None]:
    """{inchikey: cid or None}. None means PubChem was asked and said no."""
    if not KEY_CACHE.exists():
        return {}
    d = pd.read_csv(KEY_CACHE)
    return {r.inchikey: (int(r.cid) if pd.notna(r.cid) else None)
            for r in d.itertuples()}


def _save_cache(cache: dict[str, int | None]) -> None:
    KEY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(inchikey=k, cid=v) for k, v in sorted(cache.items())]
                 ).to_csv(KEY_CACHE, index=False)


def _resolve(keys: list[str], cache: dict[str, int | None],
             bucket: _Bucket, label: str) -> None:
    """Fill `cache` for every key in `keys`, querying only what is missing."""
    todo = [k for k in keys if k not in cache]
    if not todo:
        print(f"  {label}: all {len(keys)} keys already cached")
        return
    n_req = (len(todo) + BATCH - 1) // BATCH
    print(f"  {label}: {len(todo)} keys to query in {n_req} POSTs "
          f"({len(keys) - len(todo)} cached)")
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        found = lookup_batch(chunk, bucket)
        for k in chunk:
            cache[k] = found.get(k)       # None records a confirmed absence
        _save_cache(cache)                # after every batch: 150 keys is the
        done = min(i + BATCH, len(todo))  # most a crash can cost
        hits = sum(1 for k in todo[:done] if cache.get(k))
        print(f"    {done}/{len(todo)} keys, {hits} registered, "
              f"{time.time() - t0:.0f}s")


def _rows(smiles: list[str], keys: dict[str, str],
          cache: dict[str, int | None]) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        k = keys.get(smi)
        if k is None:
            rows.append(dict(smiles=smi, inchikey="", n_cids=0, cid="",
                             status="unkeyable"))
            continue
        cid = cache.get(k, "pending")
        if cid == "pending":
            continue                      # not queried yet; omit, never guess
        rows.append(dict(smiles=smi, inchikey=k, n_cids=1 if cid else 0,
                         cid=cid or "", status="hit" if cid else "notfound"))
    return pd.DataFrame(rows)


def run_batched(limit: int | None = None, seed: int = 0,
                survivors_only: bool = False) -> pd.DataFrame:
    """Survivors first, then the rest of the pool.

    The survivor set is what anyone actually reads, so it is resolved and
    reported before the remaining ~12600 molecules are touched. Both phases
    share one key cache, so a key needed by both is fetched once and an
    interrupted run resumes at the last 150-key batch.
    """
    full = pool(False)
    if limit:
        full = full.sample(min(limit, len(full)), random_state=seed)
    surv_mask = (full.passes.fillna(False).astype(bool)
                 & ~full.in_training_flat.fillna(False).astype(bool))
    surv = full[surv_mask]

    keys = inchikeys(list(full.smiles))
    unkeyable = len(full) - len(keys)
    print(f"{len(full)} molecules ({len(surv)} survivors), "
          f"{len(keys)} InChIKeys computed"
          + (f", {unkeyable} unkeyable" if unkeyable else ""))

    k_surv = sorted({keys[s] for s in surv.smiles if s in keys})
    k_rest = sorted({keys[s] for s in full.smiles if s in keys} - set(k_surv))
    print(f"{len(k_surv)} distinct survivor keys, {len(k_rest)} others\n")

    cache = _load_cache()
    if cache:
        print(f"key cache: {len(cache)} keys already resolved")
    bucket = _Bucket(MAX_RPS)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    _resolve(k_surv, cache, bucket, "survivors")
    sdf = _rows(list(surv.smiles), keys, cache)
    sdf.to_csv(SURVIVOR_RAW, index=False)
    print(f"\n  wrote {SURVIVOR_RAW.name} ({len(sdf)} rows)")
    report(sdf, write=False, populations=("survivors",))

    if survivors_only:
        return sdf

    print(f"\n{'=' * 70}\nphase 2: the remaining {len(k_rest)} keys\n{'=' * 70}")
    _resolve(k_rest, cache, bucket, "rest of pool")
    df = _rows(list(full.smiles), keys, cache)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.name} ({len(df)} rows)")
    return df


def run(limit: int | None = None, seed: int = 0,
        survivors_only: bool = False) -> pd.DataFrame:
    s = pool(survivors_only)
    if limit:
        s = s.sample(min(limit, len(s)), random_state=seed)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        # Only RESOLVED queries count as done. A throttled or timed-out lookup
        # must be retried, or transient network failures silently become
        # permanent "unresolved" rows and the denominators drift.
        resolved = prev[prev.status.isin(["hit", "notfound"])]
        done = set(resolved.smiles)
        stale = len(prev) - len(resolved)
        print(f"resuming: {len(done)} resolved"
              + (f", {stale} failed and will be retried" if stale else ""))
        if stale:
            resolved.to_csv(OUT, index=False)   # drop the failures, then re-add

    todo = [x for x in s.smiles if x not in done]
    print(f"{len(s)} survivors, {len(todo)} to query, "
          f"{WORKERS} workers at <={MAX_RPS} req/s "
          f"(~{len(todo) / MAX_RPS / 60:.0f} min at the ceiling)")
    if not todo:
        return pd.read_csv(OUT)

    bucket = _Bucket(MAX_RPS)
    new = not OUT.exists() or OUT.stat().st_size == 0
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


ORDER = ["training", "known", "novel", "unresolved"]
LABEL = {"training": "in its training data",
         "known": "exist already (PubChem), not in training",
         "novel": "no record found -- novel",
         "unresolved": "lookup failed (NOT an absence)"}


def _summarize(m: pd.DataFrame, title: str, note: str = "") -> pd.DataFrame:
    """Print one population's three-way split and return it as a frame."""
    n = len(m)
    res = m[m.bucket != "unresolved"]
    print(f"\n{'-' * 70}\n{title}  (n={n})\n{'-' * 70}")
    if note:
        print(f"  {note}\n")
    rows = []
    for b in ORDER:
        c = int((m.bucket == b).sum())
        if b == "unresolved" and not c:
            continue
        of_res = c / len(res) if (len(res) and b != "unresolved") else float("nan")
        rows.append(dict(population=title, bucket=b, n=c,
                         share_of_all=c / n if n else float("nan"),
                         share_of_resolved=of_res))
        extra = f"   ({of_res:.1%} of resolved)" if len(res) != n and b != "unresolved" else ""
        print(f"  {LABEL[b]:<44} {c:>6}  {c / n:>6.1%}{extra}")
    if len(res):
        p = (res.bucket == "novel").mean()
        se = (p * (1 - p) / len(res)) ** 0.5
        print(f"\n  novel fraction: {p:.1%} +/- {1.96 * se:.1%} (95%, n={len(res)})")
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, write: bool = True,
           populations: tuple[str, ...] = ("all", "survivors")) -> None:
    m = classify(df)
    print(f"\n{'=' * 70}\nWHERE THE GENERATED MOLECULES COME FROM\n{'=' * 70}")

    scr = pd.read_csv(SCREENED)
    # A PARTIAL CENSUS IS NOT A SAMPLE. The pool file is ordered such that
    # training-set matches run at 44.4% in its first eighth and 0.5% in the
    # last, so a run stopped part-way overstates the training bucket -- it read
    # 32.9% at 362/19219 against a random-sample estimate of 7.5%. Refuse to
    # present that as a result. A survivors-only phase is NOT partial: it is a
    # complete census of a defined population.
    pool_sum = surv_sum = pd.DataFrame()
    if "all" in populations:
        if len(m) < len(scr):
            print(f"\n  *** INCOMPLETE: {len(m)}/{len(scr)} resolved "
                  f"({len(m) / len(scr):.1%}). NOT a valid sample. ***")
            print("  The pool file is ordered with training-set matches")
            print("  concentrated at the top, so a partial census overstates")
            print("  that bucket. Finish the run, or use --limit N, which")
            print("  samples at random, before quoting anything.")
        pool_sum = _summarize(
            m, "ALL GENERATED",
            "every deduplicated molecule the ten policy seeds produced")

    if "survivors" in populations:
        # Survivors are a SUBSET of the pool, so this needs no extra queries.
        # The training bucket is necessarily zero: postfilter removes those,
        # which is the point of the screen. Quote this novel rate only against
        # this denominator.
        sv = m[m.passes.fillna(False).astype(bool) &
               ~m.in_training_flat.fillna(False).astype(bool)]
        n_expected = int((scr.passes.fillna(False).astype(bool)
                          & ~scr.in_training_flat.fillna(False).astype(bool)).sum())
        if len(sv) < n_expected:
            print(f"\n  *** survivors INCOMPLETE: {len(sv)}/{n_expected} ***")
        surv_sum = _summarize(
            sv, "SCREENED SURVIVORS",
            "after size, structural, alert and training-set screening. The "
            "training bucket is\n  zero by construction -- those were removed "
            "to form this set.")

    if write and len(m):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([pool_sum, surv_sum], ignore_index=True).to_csv(
            SUMMARY, index=False)
        m.to_csv(CLASSIFIED, index=False)
        sv.to_csv(SURVIVORS_OUT, index=False)
        print(f"\nwrote {SUMMARY.name}, {CLASSIFIED.name}, {SURVIVORS_OUT.name} "
              f"in {OUT.parent}")

    print("\n  A PubChem hit means the structure is registered. It is evidence")
    print("  the generator makes real chemistry, and evidence AGAINST novelty.")
    print("  It is NOT evidence of permeability, activity or safety. 'Novel'")
    print("  means absent from PubChem and the training corpus -- not that the")
    print("  molecule is unknown to every database, nor that it is any good:")
    print("  the novel bucket contains a trioxide no alert catalogue flags.")


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
    ap.add_argument("--serial", action="store_true",
                    help="one GET per molecule instead of batched POSTs. "
                         "~130x the requests; kept only for cross-checking "
                         "the batched path against a sample")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if args.report_only:
        return report(pd.read_csv(OUT))
    go = run if args.serial else run_batched
    report(go(limit=args.limit, seed=args.seed,
              survivors_only=args.survivors_only))


if __name__ == "__main__":
    main()
