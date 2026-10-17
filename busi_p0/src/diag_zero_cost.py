"""Zero-training-cost diagnostics (reuse existing checkpoints + existing prompt JSONs).

(1) Supervised false-negative decomposition: small lesion / low contrast / complete miss.
(2) Coarse-model box quality -> final Dice transfer curve (auto box from a trained
    supervised model, plus a scaled-GT-box quality gradient, fed to MedSAM / SAM).

Outputs:
  results/diag_fn_decomp.md / .json
  results/diag_box_curve.json / .png
"""
import os
import sys
import json

import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/mnt/workspace/busi_p0/src"
RES = "/mnt/workspace/busi_p0/results"
sys.path.insert(0, SRC)
from busi_common import build_index, load_split, orig_size, load_mask, dice_binary  # noqa

import eval_prompt_quality as epq  # reuse prompt-model helpers  # noqa


# ============================ (1) FN decomposition ============================ #
def fn_decomposition(src_json, variant, tag):
    if not os.path.exists(src_json):
        print("missing", src_json); return None
    rec = json.load(open(src_json))
    if variant not in rec["variants"]:
        print("missing variant", variant); return None
    per = rec["variants"][variant]["per_image"]
    byname = {it["name"]: it for it in build_index()}

    rows = []
    for m in per:
        if m.get("gt_empty"):
            continue
        it = byname[m["name"]]
        gt = load_mask(it)
        if gt.sum() == 0:
            continue
        area = int(gt.sum()); w0, h0 = orig_size(it); rel = area / (w0 * h0)
        img = np.asarray(Image.open(it["image"]).convert("L"), dtype=np.float32)
        ring = ndimage.binary_dilation(gt, iterations=8) & ~gt
        if ring.sum() == 0:
            ring = ~gt
        contrast = abs(float(img[gt].mean()) - float(img[ring].mean())) / 255.0
        tp, fn = int(m["tp"]), int(m["fn"])
        rows.append({"name": m["name"], "cls": m["cls"], "area": area, "rel": rel, "contrast": contrast,
                     "tp": tp, "fn": fn, "dice": float(m["dice"]),
                     "recall": tp / max(tp + fn, 1), "miss": bool(tp == 0)})

    def summarize(sel, label):
        if not sel:
            return None
        fn_px = sum(r["fn"] for r in sel)
        return {"bucket": label, "n": len(sel),
                "recall": float(np.mean([r["recall"] for r in sel])),
                "dice": float(np.mean([r["dice"] for r in sel])),
                "miss_rate": float(np.mean([r["miss"] for r in sel])),
                "fn_px": fn_px}

    all_fn = sum(r["fn"] for r in rows) or 1
    rq = np.quantile([r["rel"] for r in rows], [1 / 3, 2 / 3])
    by_size = []
    for sel, lab in [([r for r in rows if r["rel"] <= rq[0]], f"小（相对面积≤{rq[0]*100:.1f}%）"),
                     ([r for r in rows if rq[0] < r["rel"] <= rq[1]], f"中（≤{rq[1]*100:.1f}%）"),
                     ([r for r in rows if r["rel"] > rq[1]], f"大（>{rq[1]*100:.1f}%）")]:
        s = summarize(sel, lab)
        if s:
            s["fn_share"] = s["fn_px"] / all_fn
            s["area_px"] = float(np.mean([r["area"] for r in sel]))
            by_size.append(s)
    ct = np.quantile([r["contrast"] for r in rows], [1 / 3, 2 / 3])
    by_con = []
    for sel, lab in [([r for r in rows if r["contrast"] <= ct[0]], "低对比"),
                     ([r for r in rows if ct[0] < r["contrast"] <= ct[1]], "中对比"),
                     ([r for r in rows if r["contrast"] > ct[1]], "高对比")]:
        s = summarize(sel, lab)
        if s:
            s["fn_share"] = s["fn_px"] / all_fn
            by_con.append(s)
    return {"tag": tag, "variant": variant, "n": len(rows),
            "overall": summarize(rows, "全部"),
            "by_size": by_size, "by_contrast": by_con,
            "contrast_q": [float(ct[0]), float(ct[1])], "rows": rows}


# ============================ (2) box-quality curve ============================ #
def box_curve(recs, sup_tag="postproc", viz_tag="viz"):
    predictor = epq.load_sam()
    mm, mp = epq.load_medsam()

    def run(method, img, box):
        return epq.medsam_box(mm, mp, img, box) if method == "medsam" else epq.sam_box(predictor, img, box)

    out = {"scales": epq.SCALES, "methods": {}, "auto": {}}
    for method in ("medsam", "sam"):
        rows = []
        for rec in recs:
            gt = load_mask(rec); gb = epq.gt_box(gt)
            img = Image.open(rec["image"]).convert("RGB")
            for s in epq.SCALES:
                b = epq.scale_box(gb, s)
                pred = run(method, np.array(img) if method == "sam" else img, b)
                rows.append({"kind": "scale", "level": s, "box_iou": epq.box_iou(b, gb),
                             "dice": dice_binary(pred, gt)})
        out["methods"][method] = rows
        print("done", method)

    for tag, mname, size, gray, ck in [
            ("unet", "unet", 128, True, f"{RES}/{viz_tag}_unet_seed0.pth"),
            ("unetpp", "unetpp", 256, False, f"{RES}/{viz_tag}_unetpp_seed0.pth")]:
        if not os.path.exists(ck):
            print("skip auto (no ckpt)", ck); continue
        fn = epq.sup_box_fn(mname, size, gray, ck)
        for method in ("medsam", "sam"):
            ds, ious, n = [], [], 0
            for rec in recs:
                b = fn(rec)
                if b is None:
                    continue
                gt = load_mask(rec)
                img = Image.open(rec["image"]).convert("RGB")
                pred = run(method, np.array(img) if method == "sam" else img, b)
                ds.append(dice_binary(pred, gt)); ious.append(epq.box_iou(b, epq.gt_box(gt))); n += 1
            out["auto"][f"{method}_{tag}"] = {"n": n, "box_iou": float(np.mean(ious)),
                                              "dice": float(np.mean(ds))}
            print(f"auto {method}<-{tag}: box IoU {np.mean(ious):.3f} -> dice {np.mean(ds):.4f}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1-only", action="store_true", help="only the FN decomposition (no inference)")
    ap.add_argument("--sup-tag", default="postproc",
                    help="postproc JSON prefix (postproc=24ep, postproc_p1=150ep)")
    ap.add_argument("--viz-tag", default="viz",
                    help="checkpoint prefix for auto boxes (viz=24ep, p1=150ep)")
    args = ap.parse_args()
    byname = {it["name"]: it for it in build_index()}
    split = load_split()
    recs = [byname[n] for n in split["test"] if byname[n]["cls"] in ("benign", "malignant")]

    # ---------- (1) ----------
    decomp = {}
    for tag, src, var in [("unet", f"{RES}/{args.sup_tag}_unet_s128g_seed0.json", "A2+A1+A3"),
                          ("transunet", f"{RES}/{args.sup_tag}_transunet_s256rgb_seed0.json", "A2+A1+A3")]:
        d = fn_decomposition(src, var, tag)
        if d:
            decomp[tag] = d
    with open(f"{RES}/diag_fn_decomp.json", "w") as f:
        json.dump(decomp, f, indent=2, ensure_ascii=False)

    L = ["# 零成本诊断（1）：监督式漏检来源分解\n"]
    for tag, d in decomp.items():
        L.append(f"## {tag}（变体 {d['variant']}，n={d['n']} 张病灶图）\n")
        A = L.append
        A(f"| 维度 | 分桶 | 张数 | 召回 | 完全漏检率 | Dice | FN 像素占比 |")
        A("|---|---|---|---|---|---|---|")
        for s in d["by_size"]:
            A(f"| 尺寸 | {s['bucket']} | {s['n']} | {s['recall']:.3f} | {s['miss_rate']:.3f} | "
              f"{s['dice']:.3f} | {s['fn_share']*100:.1f}% |")
        for s in d["by_contrast"]:
            A(f"| 对比度 | {s['bucket']} | {s['n']} | {s['recall']:.3f} | {s['miss_rate']:.3f} | "
              f"{s['dice']:.3f} | {s['fn_share']*100:.1f}% |")
        L.append("")
    open(f"{RES}/diag_fn_decomp.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # ---------- (2) ----------
    if args.part1_only:
        print("\n--part1-only: skipping box-quality curve")
        return
    curve = box_curve(recs, args.sup_tag, args.viz_tag)
    with open(f"{RES}/diag_box_curve.json", "w") as f:
        json.dump(curve, f, indent=2)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, rows in curve["methods"].items():
        xs = [np.mean([r["box_iou"] for r in rows if r["level"] == s]) for s in epq.SCALES]
        ys = [np.mean([r["dice"] for r in rows if r["level"] == s]) for s in epq.SCALES]
        ax.plot(xs, ys, "-o", label=method)
    for k, v in curve["auto"].items():
        ax.scatter([v["box_iou"]], [v["dice"]], marker="*", s=150, label=f"{k} (auto)")
    ax.set_xlabel("box IoU vs GT box"); ax.set_ylabel("final Dice (lesion)")
    ax.set_title("Box quality -> final Dice"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{RES}/diag_box_curve.png", dpi=140)
    print("\nwrote diag_fn_decomp.{md,json} and diag_box_curve.{json,png}")


if __name__ == "__main__":
    main()
