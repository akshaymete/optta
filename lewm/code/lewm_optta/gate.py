"""Bit-identity check between two runs.

    python -m lewm_optta.gate results/<run_a> results/<run_b>

Per episode, compares frame hashes (env state, raw pixels, transformed frame), plan hashes, adaptation
records and success, and names the first layer that differs. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run: Path) -> list[dict]:
    with open(run / "episodes.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def adapt_signature(ep: dict) -> list[tuple]:
    return [(r["replan"], r["adapt_step"], r["pre_loss_pred"], r["loss_pred"], r.get("loss_opt"),
             r.get("alpha"), r.get("post_loss_pred")) for r in ep["adapt_records"]]


def first_frame_divergence(fa: list[dict], fb: list[dict]):
    """(frame index, layer, record a, record b) of the first differing hash, or None."""
    for j, (x, y) in enumerate(zip(fa, fb)):
        for layer in ("state", "pixels", "obs"):
            if x.get(layer) != y.get(layer):
                return j, layer, x, y
    if len(fa) != len(fb):
        return min(len(fa), len(fb)), "count", None, None
    return None


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    a, b = Path(argv[0]), Path(argv[1])
    ea, eb = load(a), load(b)
    ok = True
    n_bad = 0
    if len(ea) != len(eb):
        print(f"MISMATCH episode count {len(ea)} vs {len(eb)}")
        ok = False
    for x, y in zip(ea, eb):
        i = x["index"]
        problems = []
        fd = first_frame_divergence(x.get("frame_hashes", []), y.get("frame_hashes", []))
        if fd is not None:
            j, layer, ra, _ = fd
            t = ra["t"] if ra else "?"
            problems.append(f"frames diverge at captured frame {j} (env step {t}), first differing layer: {layer}"
                            + ("  [physics: env produced a different state]" if layer == "state" else
                               "  [rendering: same state, different raw pixels]" if layer == "pixels" else
                               "  [corruption/transform: same raw pixels, different frame]" if layer == "obs" else ""))
        if x["plan_hashes"] != y["plan_hashes"]:
            k = next((j for j, (p, q) in enumerate(zip(x["plan_hashes"], y["plan_hashes"])) if p != q),
                     min(len(x["plan_hashes"]), len(y["plan_hashes"])))
            problems.append(f"plans diverge at replan {k} ({len(x['plan_hashes'])} vs {len(y['plan_hashes'])} plans)")
        sa, sb = adapt_signature(x), adapt_signature(y)
        if sa != sb:
            k = next((j for j, (p, q) in enumerate(zip(sa, sb)) if p != q), min(len(sa), len(sb)))
            problems.append(f"adaptation records diverge at step {k}: A={sa[k] if k < len(sa) else None} B={sb[k] if k < len(sb) else None}")
        if x["success"] != y["success"]:
            problems.append(f"success {x['success']} vs {y['success']}")
        if problems:
            ok = False
            n_bad += 1
            print(f"MISMATCH episode {i}:")
            for p in problems:
                print(f"   {p}")
    n = min(len(ea), len(eb))
    sa = sum(1 for e in ea[:n] if e["success"])
    sb = sum(1 for e in eb[:n] if e["success"])
    print(f"episodes differing: {n_bad}/{n}; successes {sa} vs {sb}")
    print("GATE PASS: runs are bit-identical" if ok else "GATE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
