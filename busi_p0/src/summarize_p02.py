"""Summarize P0-2: aggregate 3 seeds per model, report mean+-std, run Wilcoxon.

Paired test: for each model we average each test image's Dice over the 3 seeds,
then run a paired Wilcoxon signed-rank test across the 116 test images between
the two models. Also reports a per-seed aggregate table.
"""
import os
import json
import glob
import numpy as np
from scipy.stats import wilcoxon

from busi_common import RESULTS

TAG = "s{size}{mode}_e{ep}"
MODELS = ["unet", "attunet"]
METRICS = ["dice_lesion", "iou_lesion", "miou_hist", "precision", "recall",
           "specificity", "accuracy", "hd95", "asd", "fg_ratio_normal",
           "dice_benign", "dice_malignant"]


def load(model, size=128, mode="g", ep=24, seeds=(0, 1, 2)):
    out = {}
    for s in seeds:
        p = os.path.join(RESULTS, f"{model}_{TAG.format(size=size, mode=mode, ep=ep)}_seed{s}.json")
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def per_image_dice(rec):
    return {m["name"]: m["dice"] for m in rec["per_image"]}


def main():
    size, mode, ep = 128, "g", 24
    data = {m: load(m, size, mode, ep) for m in MODELS}
    for m in MODELS:
        if len(data[m]) < 3:
            print(f"WARN: {m} only {len(data[m])} seed files found")

    summary = {"protocol": {"split": "splits/busi_seed0.json", "size": size,
                            "mode": "gray" if mode == "g" else "rgb",
                            "epochs": ep, "seeds": [0, 1, 2]}, "models": {}}

    for m in MODELS:
        recs = list(data[m].values())
        agg = {}
        for k in METRICS:
            vals = [r["agg"][k] for r in recs]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                      "per_seed": [round(float(v), 4) for v in vals]}
        summary["models"][m] = {
            "params": recs[0]["params"],
            "seeds": sorted(data[m].keys()),
            "aggregate": agg,
        }

    # paired Wilcoxon on seed-averaged per-image Dice
    if all(len(data[m]) >= 1 for m in MODELS):
        names = sorted(set(per_image_dice(next(iter(data["unet"].values()))).keys()))
        a = np.mean([[per_image_dice(r)[n] for n in names] for r in data["unet"].values()], axis=0)
        b = np.mean([[per_image_dice(r)[n] for n in names] for r in data["attunet"].values()], axis=0)
        stat, p = wilcoxon(a, b)
        summary["wilcoxon"] = {
            "comparison": "unet vs attunet (paired per-image Dice, seed-averaged)",
            "n_images": len(names), "mean_unet": float(a.mean()), "mean_attunet": float(b.mean()),
            "delta": float(b.mean() - a.mean()), "statistic": float(stat), "p_value": float(p),
        }

    out = os.path.join(RESULTS, "multiseed_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:2500])
    print("wrote", out)


if __name__ == "__main__":
    main()
