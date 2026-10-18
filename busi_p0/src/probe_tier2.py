"""Probe 2: can 'normal (should abstain)' be separated from 'lesion missed'?"""
import sys
import numpy as np

SRC = "/mnt/workspace/busi_p0/src"
sys.path.insert(0, SRC)
from busi_common import build_index, load_split  # noqa
import diag_tier_prompt as P  # noqa

RES = "/mnt/workspace/busi_p0/results"
ck = sys.argv[1] if len(sys.argv) > 1 else f"{RES}/p1_unetpp_seed0.pth"
print("coarse ckpt:", ck)
sig_fn = P.signals_fn("unetpp", 256, False, ck)

index = build_index(); byname = {it["name"]: it for it in index}
split = load_split()
test = [byname[n] for n in split["test"]]
val = [byname[n] for n in split["val"]]
sigs = {r["name"]: sig_fn(r) for r in (val + test)}

def stats(recs, tag):
    pm = np.array([sigs[r["name"]]["pmax"] for r in recs])
    nobox = sum(1 for r in recs if sigs[r["name"]]["box"] is None)
    nolo = sum(1 for r in recs if sigs[r["name"]]["box_lo"] is None)
    print(f"[{tag}] n={len(recs)} pmax p10/25/50/75/90: "
          + " ".join(f"{np.percentile(pm, q):.3f}" for q in (10, 25, 50, 75, 90))
          + f"  no-box={nobox}  no-lo={nolo}")
    return pm

for tag, recs in (("val", val), ("test", test)):
    les = [r for r in recs if r["cls"] in ("benign", "malignant")]
    nor = [r for r in recs if r["cls"] == "normal"]
    pl = stats(les, tag + " lesion")
    pn = stats(nor, tag + " normal ")
    print("   normal pmax sorted:", " ".join(f"{v:.3f}" for v in np.sort(pn)))

# separation check at candidate thresholds (test only)
les = [r for r in test if r["cls"] in ("benign", "malignant")]
nor = [r for r in test if r["cls"] == "normal"]
print("\nblind-gate threshold sweep (test):")
print(f"  {'tau':>5} {'#lesion flagged':>16} {'#lesion w/ box':>15} {'#normal flagged':>16}")
for tau in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
    lf = sum(1 for r in les if sigs[r["name"]]["pmax"] < tau)
    lwb = sum(1 for r in les if sigs[r["name"]]["pmax"] < tau and sigs[r["name"]]["box"] is not None)
    nf = sum(1 for r in nor if sigs[r["name"]]["pmax"] < tau)
    print(f"  {tau:5.2f} {lf:16d} {lwb:15d} {nf:16d}")
print(f"\n  total lesion={len(les)}, normal={len(nor)}")
