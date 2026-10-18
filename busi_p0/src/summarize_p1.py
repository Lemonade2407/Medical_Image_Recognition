"""Summarise the 150-epoch loss comparison (3 seeds, unified probability path).

Reads:
  results/postproc_p1_<model>_s<size><mode>[_loss]_seed<k>.json  (baseline variant)
  results/p1_<model>[_loss]_seed<k>.json                         (best-epoch history)
Writes:
  results/p1_summary.md / .json
"""
import os
import json
import numpy as np

RES = "/mnt/workspace/busi_p0/results"
CFG = [("unet", "U-Net", 128, "g", ["ce_dice", "tversky", "boundary"]),
       ("attunet", "Attention U-Net", 128, "g", ["ce_dice", "tversky", "boundary"]),
       ("unetpp", "UNet++", 256, "rgb", ["ce_dice", "tversky"]),
       ("transunet", "TransUNet", 256, "rgb", ["ce_dice", "tversky"])]
LOSS_LABEL = {"ce_dice": "CE+Dice", "tversky": "Tversky(0.3/0.7)", "boundary": "CE+Dice+Boundary"}
SEEDS = (0, 1, 2)


def load(res, fn):
    p = os.path.join(res, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def stats(v):
    if not v:
        return None
    return {"mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "n": len(v)}


def ms(s):
    if not s:
        return "—"
    return f"{s['mean']:.4f}±{s['std']:.4f}" if s["std"] else f"{s['mean']:.4f}"


def collect(res=RES):
    """Per (model, loss) 150ep test Dice on the unified probability path."""
    rows = []
    for model, label, size, mode, losses in CFG:
        for loss in losses:
            suf = "" if loss == "ce_dice" else f"_{loss}"
            vals, perseed = [], []
            for s in SEEDS:
                rp = load(res, f"postproc_p1_{model}_s{size}{mode}{suf}_seed{s}.json")
                r1 = load(res, f"p1_{model}{suf}_seed{s}.json")
                v = rp["variants"]["baseline"]["agg"]["dice_lesion"] if rp else None
                if v is None and r1:
                    v = r1["agg"]["dice_lesion"]
                hist = (r1 or {}).get("history", [])
                best = int(np.argmax([h["val_dice"] for h in hist])) + 1 if hist else None
                perseed.append({"seed": s, "dice": v,
                                "epochs": len(hist) or None, "best_epoch": best})
                if v is not None:
                    vals.append(v)
            rows.append({"model": model, "label": label, "loss": loss,
                         "stats": stats(vals), "perseed": perseed})
    return rows


def _mean(rows, model, loss):
    for r in rows:
        if r["model"] == model and r["loss"] == loss:
            return r["stats"]["mean"] if r["stats"] else None
    return None


def main():
    rows = collect(RES)
    L = ["# 150ep 损失函数对照（3 种子，统一概率路径）\n",
         "| 模型 | 损失 | Dice(病灶) | 最优轮 |",
         "|---|---|---|---|"]
    for r in rows:
        eps = [p["best_epoch"] for p in r["perseed"] if p["best_epoch"]]
        L.append(f"| {r['label']} | {LOSS_LABEL[r['loss']]} | {ms(r['stats'])} | "
                 f"{int(np.mean(eps)) if eps else '—'} |")
    L.append("\n## 同模型内 Δ（相对 CE+Dice）\n")
    L.append("| 模型 | Tversky − CE | Boundary − CE |")
    L.append("|---|---|---|")
    for model, label, *_ in CFG:
        ce = _mean(rows, model, "ce_dice")
        tv = _mean(rows, model, "tversky")
        bd = _mean(rows, model, "boundary")
        L.append(f"| {label} | {'%+.4f' % (tv-ce) if (ce is not None and tv is not None) else '—'} | "
                 f"{'%+.4f' % (bd-ce) if (ce is not None and bd is not None) else '—'} |")
    L.append("")
    out = os.path.join(RES, "p1_summary.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(RES, "p1_summary.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
    print("\n".join(L))
    print("\nwrote", out, "+ p1_summary.json")


if __name__ == "__main__":
    main()
