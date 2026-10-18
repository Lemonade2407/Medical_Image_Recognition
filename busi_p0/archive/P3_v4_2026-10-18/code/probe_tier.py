"""Probe: how many test images have a usable prompt at each threshold level."""
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

for tag, recs in (("val", val), ("test", test)):
    les = [r for r in recs if r["cls"] in ("benign", "malignant")]
    pm = np.array([sigs[r["name"]]["pmax"] for r in les])
    has_hi = sum(1 for r in les if sigs[r["name"]]["box"] is not None)
    has_lo = sum(1 for r in les if sigs[r["name"]]["box_lo"] is not None)
    neither = sum(1 for r in les if sigs[r["name"]]["box"] is None and sigs[r["name"]]["box_lo"] is None)
    lo_only = sum(1 for r in les if sigs[r["name"]]["box"] is None and sigs[r["name"]]["box_lo"] is not None)
    print(f"\n[{tag}] lesion n={len(les)}")
    print("  pmax pct 5/10/20/50/80: " + " ".join(f"{np.percentile(pm, q):.3f}" for q in (5, 10, 20, 50, 80)))
    print(f"  has box@0.5={has_hi}  has box@0.3={has_lo}  neither={neither}  lo_only={lo_only}")
    none_img = [r for r in les if sigs[r["name"]]["box"] is None]
    print("  no-box images (name, cls, pmax, area_lo):")
    for r in sorted(none_img, key=lambda r: sigs[r["name"]]["pmax"]):
        s = sigs[r["name"]]
        print(f"    {r['name']:<28} {r['cls']:<10} pmax={s['pmax']:.4f} area_lo={s['area_lo']}")
