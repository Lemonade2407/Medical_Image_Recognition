"""Compare 24-epoch (ab_*) vs long-training (p1_*) test Dice, to test whether the
loss-function conclusions ("Boundary harmful", "Tversky hurts UNet++") are just
underfitting. Writes results/p1_summary.md.
"""
import os
import json
import numpy as np

RES = "/mnt/workspace/busi_p0/results"
CFG = [("unet", 128, "g", ["ce_dice", "tversky", "boundary"]),
       ("attunet", 128, "g", ["ce_dice", "tversky", "boundary"]),
       ("unetpp", 256, "rgb", ["ce_dice", "tversky"])]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def ms(v):
    if not v:
        return "—"
    return f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}" if len(v) > 1 else f"{np.mean(v):.4f}"


def main():
    L = ["# P1 充分训练对照（24 → 150 epoch + 早停）\n",
         "| 模型 | 损失 | 24ep Dice | 150ep Dice | Δ | 150ep 最优轮到 | 结论 |",
         "|---|---|---|---|---|---|---|"]
    rows = []
    for model, size, mode, losses in CFG:
        for loss in losses:
            suf = "" if loss == "ce_dice" else f"_{loss}"
            b24, b150, eps = [], [], []
            for s in (0, 1, 2):
                r = load(f"{RES}/ab_{model}{suf}_seed{s}.json")
                if r:
                    b24.append(r["agg"]["dice_lesion"])
                r = load(f"{RES}/p1_{model}{suf}_seed{s}.json")
                if r:
                    b150.append(r["agg"]["dice_lesion"])
                    eps.append(len(r.get("history", [])))
            d = (np.mean(b150) - np.mean(b24)) if (b24 and b150) else None
            note = ""
            if d is not None:
                note = "延长后提升" if d > 0.005 else ("基本不变" if d > -0.005 else "延长后仍更低")
            L.append(f"| {model} | {loss} | {ms(b24)} | {ms(b150)} | "
                     f"{('%+.4f' % d) if d is not None else '—'} | "
                     f"{int(np.mean(eps)) if eps else '—'} | {note} |")
            rows.append({"model": model, "loss": loss,
                         "d24": (float(np.mean(b24)) if b24 else None),
                         "d150": (float(np.mean(b150)) if b150 else None),
                         "std24": (float(np.std(b24, ddof=1)) if len(b24) > 1 else 0.0),
                         "std150": (float(np.std(b150, ddof=1)) if len(b150) > 1 else 0.0),
                         "delta": (float(d) if d is not None else None),
                         "epochs": (int(np.mean(eps)) if eps else None), "note": note,
                         "n24": len(b24), "n150": len(b150)})
    L.append("\n> 判据：若延长训练后 Boundary 仍低于 ce_dice、Tversky 在 UNet++ 上仍为负 → "
             "结论（贡献 3）成立；否则需改写。")
    out = os.path.join(RES, "p1_summary.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(RES, "p1_summary.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
    print("\n".join(L))
    print("\nwrote", out, "+ p1_summary.json")


if __name__ == "__main__":
    main()
