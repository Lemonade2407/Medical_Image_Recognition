"""Summarize P1 + A group: A-group variants on top of the 150-epoch weights.

Reads results/postproc_p1_<model>_s<size><mode>[_loss]_seed<k>.json
(and the P1 baseline from results/p1_<model>[_loss]_seed<k>.json)
Writes results/p1a_summary.md / .json
"""
import os
import json
import numpy as np

RES = "/mnt/workspace/busi_p0/results"
CFG = [("unet", "U-Net", 128, "g", ["ce_dice", "tversky", "boundary"]),
       ("attunet", "Attention U-Net", 128, "g", ["ce_dice", "tversky", "boundary"]),
       ("unetpp", "UNet++", 256, "rgb", ["ce_dice", "tversky"]),
       ("transunet", "TransUNet", 256, "rgb", ["ce_dice", "tversky"])]
A_VARIANTS = ["baseline", "A1_postproc", "A2_tta", "A3_threshold", "A2+A1+A3"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def ms(v):
    if not v:
        return "—"
    return f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}" if len(v) > 1 else f"{np.mean(v):.4f}"


def main():
    L = ["# P1 权重 + A 组（后处理 / TTA / 阈值）\n",
         "> 在 150 epoch（+早停）的权重上叠加推理侧 A 组，检验 A 组在**充分训练**下是否仍然有效。\n",
         "| 模型 | 损失 | P1 baseline | P1+A1 | P1+A2 | P1+A3 | P1+A2+A1+A3 | 最优变体 | Δ(A) | HD95(base→best) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for model, label, size, mode, losses in CFG:
        for loss in losses:
            suf = "" if loss == "ce_dice" else f"_{loss}"
            base, hd_base = [], []
            av = {v: [] for v in A_VARIANTS}
            hd = {v: [] for v in A_VARIANTS}
            for s in (0, 1, 2):
                r = load(f"{RES}/p1_{model}{suf}_seed{s}.json")
                if r:
                    base.append(r["agg"]["dice_lesion"]); hd_base.append(r["agg"]["hd95"])
                rp = load(f"{RES}/postproc_p1_{model}_s{size}{mode}{suf}_seed{s}.json")
                if rp:
                    for v in A_VARIANTS:
                        if v in rp["variants"]:
                            av[v].append(rp["variants"][v]["agg"]["dice_lesion"])
                            hd[v].append(rp["variants"][v]["agg"]["hd95"])
            if not base and not any(av.values()):
                continue
            best, bv = None, -1.0
            for v in A_VARIANTS:
                if av[v] and np.mean(av[v]) > bv:
                    bv, best = float(np.mean(av[v])), v
            bm = float(np.mean(base)) if base else None
            bm_a = float(np.mean(av["baseline"])) if av["baseline"] else bm
            d = (bv - bm_a) if (best and bm_a is not None) else None
            hb = float(np.mean(hd["baseline"])) if hd["baseline"] else None
            hbest = float(np.mean(hd[best])) if (best and hd[best]) else None
            L.append(f"| {label} | {loss} | {ms(base)} | {ms(av['A1_postproc'])} | {ms(av['A2_tta'])} | "
                     f"{ms(av['A3_threshold'])} | {ms(av['A2+A1+A3'])} | {best or '—'} | "
                     f"{(('%+.4f' % d) if d is not None else '—')} | "
                     f"{(('%.1f→%.1f' % (hb, hbest)) if (hb is not None and hbest is not None) else '—')} |")
            rows.append({"model": model, "label": label, "loss": loss,
                         "base_mean": bm, "a_baseline_mean": bm_a,
                         "best": best, "best_mean": bv if best else None,
                         "delta": d, "hd95_base": hb, "hd95_best": hbest,
                         "variant_mean": {v: (float(np.mean(av[v])) if av[v] else None)
                                          for v in A_VARIANTS},
                         "variant_std": {v: (float(np.std(av[v], ddof=1)) if len(av[v]) > 1 else 0.0)
                                         for v in A_VARIANTS},
                         "n": len(base)})
    with open(f"{RES}/p1a_summary.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
    out = f"{RES}/p1a_summary.md"
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
