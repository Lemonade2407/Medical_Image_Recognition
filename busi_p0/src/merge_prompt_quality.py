"""Merge the per-method P2 outputs into a single JSON + combined curve figure."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/mnt/workspace/busi_p0/results"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    d1, d2 = load(f"{RES}/prompt_quality_medsam.json"), load(f"{RES}/prompt_quality_sam.json")
    if not d1 and not d2:
        print("no shard outputs found"); return
    base = d1 or d2
    out = {"scales": base["scales"], "shifts": base["shifts"], "methods": {}, "auto": {}}
    for d in (d1, d2):
        if d:
            out["methods"].update(d["methods"])
            out["auto"].update(d["auto"])
    with open(f"{RES}/prompt_quality.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, rows in out["methods"].items():
        xs, ys = [], []
        for s in out["scales"]:
            r = [x for x in rows if x["kind"] == "scale" and x["level"] == s]
            if r:
                xs.append(np.mean([x["box_iou"] for x in r])); ys.append(np.mean([x["dice"] for x in r]))
        ax.plot(xs, ys, "-o", label=method)
    for k, v in out["auto"].items():
        ax.scatter([v["box_iou"]], [v["dice"]], marker="*", s=150, label=f"{k} (auto)")
    ax.set_xlabel("box IoU vs GT box"); ax.set_ylabel("final Dice (lesion)")
    ax.set_title("Prompt quality vs final Dice"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{RES}/prompt_quality.png", dpi=140)
    print("wrote", f"{RES}/prompt_quality.json and .png")
    for method, rows in out["methods"].items():
        for s in out["scales"]:
            r = [x for x in rows if x["kind"] == "scale" and x["level"] == s]
            if r:
                print(f"  {method} s={s}: IoU {np.mean([x['box_iou'] for x in r]):.3f} -> dice "
                      f"{np.mean([x['dice'] for x in r]):.4f}")
    for k, v in out["auto"].items():
        print(f"  {k}: box IoU {v['box_iou']:.3f} -> dice {v['dice']:.4f} (n={v['n']})")


if __name__ == "__main__":
    main()
