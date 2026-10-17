"""Generalized A/B summary across ALL methods.

Supervised: B = results/ab_<model>[_<loss>]_seed<k>.json
            A = results/postproc_<model>_s<size><mode>[_<loss>]_seed<k>.json
Prompt:     A = results/postproc_prompt_<method>.json   (no B)

Writes results/ab_summary.json (structured) and results/ab_summary.md (rendered).
"""
import os
import json
import numpy as np

RES = "/mnt/workspace/busi_p0/results"
A_VARIANTS = ["baseline", "A1_postproc", "A2_tta", "A3_threshold", "A2+A1+A3"]
LOSSES = ["ce_dice", "tversky", "boundary"]
SUP = [("unet", 128, "g", LOSSES), ("attunet", 128, "g", LOSSES),
       ("unetpp", 256, "rgb", ["ce_dice", "tversky"]),
       ("transunet", 256, "rgb", ["ce_dice", "tversky"])]
PROMPT = ["sam", "medsam", "bussam"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def stats(vals):
    if not vals:
        return None
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}


def main():
    out = {"supervised": [], "prompt": []}

    for model, size, mode, losses in SUP:
        for loss in losses:
            suf = "" if loss == "ce_dice" else f"_{loss}"
            bvals, a = [], {v: [] for v in A_VARIANTS}
            a_hd = {v: [] for v in A_VARIANTS}
            for s in (0, 1, 2):
                rb = load(f"{RES}/ab_{model}{suf}_seed{s}.json")
                if rb:
                    bvals.append(rb["agg"]["dice_lesion"])
                rp = load(f"{RES}/postproc_{model}_s{size}{mode}{suf}_seed{s}.json")
                if rp:
                    for v in A_VARIANTS:
                        if v in rp["variants"]:
                            a[v].append(rp["variants"][v]["agg"]["dice_lesion"])
                            a_hd[v].append(rp["variants"][v]["agg"]["hd95"])
            if not bvals and all(not a[v] for v in A_VARIANTS):
                continue
            best, bv = None, -1.0
            for v in A_VARIANTS:
                if a[v] and np.mean(a[v]) > bv:
                    bv, best = float(np.mean(a[v])), v
            out["supervised"].append({
                "model": model, "size": size, "mode": mode, "loss": loss,
                "b_dice": stats(bvals),
                "a": {v: {"dice": stats(a[v]), "hd95": (float(np.mean(a_hd[v])) if a_hd[v] else None)}
                      for v in A_VARIANTS},
                "best_variant": best, "best_dice": bv if best else None})

    for meth in PROMPT:
        rp = load(f"{RES}/postproc_prompt_{meth}.json")
        if not rp:
            continue
        a = {v: rp["variants"][v]["agg"]["dice_lesion"] for v in A_VARIANTS if v in rp["variants"]}
        hd = {v: rp["variants"][v]["agg"]["hd95"] for v in A_VARIANTS if v in rp["variants"]}
        best = max(a, key=a.get) if a else None
        out["prompt"].append({"method": meth,
                              "a": {v: {"dice": a[v], "hd95": hd[v]} for v in a},
                              "best_variant": best, "best_dice": a.get(best) if best else None})

    with open(f"{RES}/ab_summary.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # markdown
    L = ["# A/B 优化结果汇总\n",
         "## 监督式（B=损失函数重训 × 3 种子；A=推理侧）\n",
         "| 模型 | 损失(B) | B: Dice | A: baseline | A2 TTA | A3 阈值 | A2+A1+A3 | 最优变体 |",
         "|---|---|---|---|---|---|---|---|"]
    def fm(d):
        if not d:
            return "—"
        return f"{d['mean']:.4f}±{d['std']:.4f}" if d["std"] else f"{d['mean']:.4f}"
    for r in out["supervised"]:
        a = r["a"]
        L.append(f"| {r['model']} | {r['loss']} | {fm(r['b_dice'])} | {fm(a['baseline']['dice'])} | "
                 f"{fm(a['A2_tta']['dice'])} | {fm(a['A3_threshold']['dice'])} | "
                 f"{fm(a['A2+A1+A3']['dice'])} | {r['best_variant'] or '—'} |")
    L += ["\n## 提示式（仅 A，无 B）\n",
          "| 方法 | baseline | A1 后处理 | A2 TTA | A3 阈值 | A2+A1+A3 | 最优变体 |",
          "|---|---|---|---|---|---|---|"]
    for r in out["prompt"]:
        a = r["a"]
        def g(v):
            return f"{a[v]['dice']:.4f}" if v in a else "—"
        L.append(f"| {r['method']} | {g('baseline')} | {g('A1_postproc')} | {g('A2_tta')} | "
                 f"{g('A3_threshold')} | {g('A2+A1+A3')} | {r['best_variant'] or '—'} |")
    L.append("\n> A1=最大连通域+填洞+面积阈值；A2=多尺度/翻转 TTA；A3=阈值搜索（验证集上选）。")
    with open(f"{RES}/ab_summary.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote ab_summary.json / ab_summary.md")


if __name__ == "__main__":
    main()
