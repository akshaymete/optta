"""Tabulate finished runs: success per (group, variant) over seeds and paired per-seed deltas.

    python -m lewm_optta.summarize results/<glob> ...
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def load_runs(patterns):
    runs, missing = [], []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            p = Path(d) / "summary.json"
            if p.exists():
                with open(p) as f:
                    runs.append(json.load(f))
            else:
                missing.append(d)
    return runs, missing


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    runs, missing = load_runs(argv or ["results/*"])
    if missing:
        print("unfinished / missing summary.json:", *missing, sep="\n  ")
    if not runs:
        print("no finished runs")
        return

    def split_name(r) -> tuple[str, str]:
        """('<prefix>/<cell>', variant) from the run name '<prefix>_<cell>_<variant>_s<seed>'."""
        stem = str(r["run_name"]).rsplit(f"_s{int(r['seed'])}", 1)[0]
        cell = str(r.get("cell"))
        marker = f"_{cell}_"
        if r.get("cell") and marker in stem:
            prefix, var = stem.split(marker, 1)
            return f"{prefix}/{cell}", var
        return cell, str(r.get("arm"))

    by_cell_arm = defaultdict(dict)
    for r in runs:
        g, v = split_name(r)
        by_cell_arm[(g, v)][int(r["seed"])] = r

    print(f"\n{'group':<22}{'arm':<14}{'n':>3}{'success':>10}{'sd':>7}   solved within k replans")
    for (cell, arm), seeds in sorted(by_cell_arm.items()):
        vals = [s["success_rate"] for s in seeds.values()]
        curves = [s["solved_by_k"] for s in seeds.values()]
        kmax = min(len(c) for c in curves)
        ks = [k for k in (1, 2, 5, 10, 20) if k <= kmax]
        curve = "  ".join(f"k{k}={st.mean(c[k - 1] for c in curves):.0f}" for k in ks)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{cell:<22}{arm:<14}{len(vals):>3}{st.mean(vals):>10.1f}{sd:>7.1f}   {curve}")

    print("\npaired deltas (same cell, same seed): every variant vs frozen and vs adajepa")
    cells = sorted({c for c, _ in by_cell_arm})
    for cell in cells:
        variants = sorted(v for c, v in by_cell_arm if c == cell)
        pairs = [(v, "frozen") for v in variants if v != "frozen"] + [(v, "adajepa") for v in variants if v not in ("adajepa", "frozen")]
        for hi, lo in pairs:
            a, b = by_cell_arm.get((cell, hi), {}), by_cell_arm.get((cell, lo), {})
            common = sorted(set(a) & set(b))
            if not common:
                continue
            d = [a[s]["success_rate"] - b[s]["success_rate"] for s in common]
            pos = sum(1 for x in d if x > 0)
            print(f"  {cell:<22}{hi}-{lo:<10} n={len(d)} mean={st.mean(d):+.1f} "
                  f"min={min(d):+.1f} max={max(d):+.1f} positive={pos}/{len(d)}")


if __name__ == "__main__":
    main()
