"""P3 v3: error-type-aware targeted prompt correction (zero retraining).

Findings that drove this version (measured, see probe_tier*.py):
  * of the 96 test lesion images, 85 have a coarse box at 0.5, 2 only at the low
    threshold (pmax 0.39 / 0.48) and 9 have NO region at any usable threshold
    (pmax 0.029-0.181). Those 9 are the "no box" hard failures.
  * the pmax distribution of those 9 is indistinguishable from the normal test
    images (normal median 0.078, p75 0.174), so a *percentile* gate on pmax can
    NOT raise the precision of "missing" detection: at any tau the flagged set is
    ~40% lesion / 60% normal. Precision is capped by the binary coarse model.
  -> hence the gate is structural and threshold-free ("argmax never calls any
     pixel a lesion"), and the recoverable bucket gets multi-candidate prompts
     whose aggregation is now the variable under test.

Aggregation modes: union / majority / intersect / select (by MedSAM's own
iou_score). `select` avoids the precision dilution of unioning 3 masks.

Outputs: results/tier_prompt.json / .md / .png
"""
import os
import sys
import json
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/mnt/workspace/busi_p0/src"
RES = "/mnt/workspace/busi_p0/results"
sys.path.insert(0, SRC)
from busi_common import build_index, load_split, orig_size, load_mask, dice_binary  # noqa
import eval_prompt_quality as epq  # noqa

_spec = importlib.util.spec_from_file_location("busi_models5", os.path.join(SRC, "models.py"))
_bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bm)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_AREA = 20
SCALES = (1.0, 1.15, 1.3)


# ------------------------------ coarse signals ------------------------------ #
def signals_fn(model_name, size, gray, ckpt):
    model = _bm.build_model(model_name, in_ch=1 if gray else 3, n_cls=2, size=size)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
    model.to(DEV).eval()

    def fn(rec):
        im = Image.open(rec["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
        if gray:
            a = a.mean(2, keepdims=True); a = (a - a.mean()) / (a.std() + 1e-8)
        else:
            a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-8)
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV).float()
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            prob = F.softmax(out, 1)[:, 1]
        w, h = orig_size(rec)
        p = F.interpolate(prob[:, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        def cc_stats(thr):
            lab, n = ndimage.label(p > thr)
            best = None; total = 0
            for i in range(1, n + 1):
                m = lab == i
                a_ = int(m.sum())
                if a_ < MIN_AREA:
                    continue
                total += a_
                ys, xs = np.where(m)
                b = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
                if best is None or a_ > best["area"]:
                    best = {"area": a_, "box": b, "centroid": np.array([xs.mean(), ys.mean()])}
            return best, total

        hi, _ = cc_stats(0.5)
        lo, area_lo = cc_stats(0.3)
        area_hc = hi["area"] if hi else 0
        shift = 0.0
        if hi is not None and lo is not None:
            d = np.linalg.norm(hi["centroid"] - lo["centroid"])
            diag = np.linalg.norm(hi["box"][2:] - hi["box"][:2]) + 1e-6
            shift = float(d / diag)
        return {"pmax": float(p.max()), "wh": (w, h),
                "box": hi["box"] if hi else None,
                "box_lo": lo["box"] if lo else None,
                "centroid_hc": hi["centroid"] if hi else None,
                "centroid_lo": lo["centroid"] if lo else None,
                "area_hc": area_hc, "area_lo": area_lo,
                "shift": shift, "ratio": float(area_lo / max(area_hc, 1))}
    return fn


# ------------------------------ MedSAM (batched + memoised) ------------------------------ #
_MS_CACHE = {}


def medsam_multi(model, proc, img, boxes, name=""):
    keys = [(name, tuple(np.round(np.asarray(b, dtype=np.float64), 2))) for b in boxes]
    todo = sorted({k[1] for k in keys if k not in _MS_CACHE})
    if todo:
        inp = proc(img, input_boxes=[[[float(v) for v in b] for b in todo]], return_tensors="pt")
        inp = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, multimask_output=False)
        ms = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                     inp["reshaped_input_sizes"].cpu())[0].numpy() > 0
        ms = ms.reshape(-1, *ms.shape[-2:])
        sc = out.iou_scores.cpu().numpy().reshape(-1)
        for i, b in enumerate(todo):
            _MS_CACHE[(name, b)] = (ms[i], float(sc[i]) if i < len(sc) else 0.0)
    return [_MS_CACHE[k] for k in keys]


# ------------------------------ candidates ------------------------------ #
def expand(box, s):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * s / 2, (box[3] - box[1]) * s / 2
    return np.array([cx - w, cy - h, cx + w, cy + h], dtype=np.float64)


def shift_box(box, dx, dy):
    b = box + np.array([dx, dy, dx, dy])
    return np.array([max(0, b[0]), max(0, b[1]), b[2], b[3]], dtype=np.float64)


def fallback_boxes(wh):
    """Region proposals that do not depend on the coarse map (for blind images)."""
    w, h = wh
    out = [np.array([0, 0, w - 1, h - 1], dtype=np.float64)]
    for frac in (0.70, 0.45):
        bw, bh = w * frac / 2, h * frac / 2
        out.append(np.array([w / 2 - bw, h / 2 - bh, w / 2 + bw, h / 2 + bh], dtype=np.float64))
    return out


def diagnose(sig, tau):
    """Errors are diagnosed from the coarse map only; no GT, no percentile gate.

    'box is None' means the mask argmax never calls any pixel a lesion, so the
    image is either hopeless ('全盲') or merely under-confident ('欠自信').
    """
    if sig["box"] is None:
        return "欠自信" if sig["box_lo"] is not None else "全盲"
    if sig["shift"] > tau["d"]:
        return "偏移"
    if sig["ratio"] > tau["s"]:
        return "尺度错"
    return "正确"


def candidates(sig, tau, policy):
    """policy: single | blind | blind+rescue | type.

    tau['p'] is an optional extra gate on the blind buckets: a blind image only
    gets a fallback prompt if its peak confidence is still below tau['p'].
    """
    if policy == "single":
        return [] if sig["box"] is None else [expand(sig["box"], 1.0)]

    t = diagnose(sig, tau)
    blind = t in ("全盲", "欠自信")
    if blind and sig["pmax"] >= tau["p"]:
        return []

    if policy in ("blind", "blind+rescue"):
        if t == "全盲":
            return fallback_boxes(sig["wh"])
        if t == "欠自信":
            if policy == "blind+rescue":
                return [expand(sig["box_lo"], s) for s in SCALES]
            return []
        return [expand(sig["box"], 1.0)]

    # 'type' — full type-aware (v1 B3 semantics, single-box elsewhere)
    if t == "全盲":
        return fallback_boxes(sig["wh"])
    if t == "欠自信":
        return [expand(sig["box_lo"], s) for s in SCALES]
    if t == "偏移":
        cs = [expand(sig["box"], 1.15)]
        if sig["centroid_lo"] is not None:
            d = sig["centroid_lo"] - sig["centroid_hc"]
            cs.append(shift_box(expand(sig["box"], 1.15), d[0], d[1]))
        return cs
    if t == "尺度错":
        cs = [expand(sig["box"], 1.15)]
        if sig["box_lo"] is not None:
            cs.append(expand(sig["box_lo"], 1.15))
        return cs
    return [expand(sig["box"], 1.15)]


def combine(masks, scores, how, shape):
    if not masks:
        return np.zeros(shape, dtype=bool)
    if len(masks) == 1:
        return masks[0]
    if how == "union":
        m = np.zeros(shape, dtype=bool)
        for x in masks:
            m |= x
        return m
    if how == "intersect":
        m = np.ones(shape, dtype=bool)
        for x in masks:
            m &= x
        return m
    if how == "majority":
        return np.mean([x.astype(np.float32) for x in masks], axis=0) > 0.5
    if how == "select":
        return masks[int(np.argmax(scores))]
    raise ValueError(how)


# ------------------------------ evaluation ------------------------------ #
def evaluate(records, signals, medsam, tau, policy, how):
    model, proc = medsam
    per = []
    for rec in records:
        sig = signals[rec["name"]]
        gt = load_mask(rec)
        cs = candidates(sig, tau, policy)
        if not cs:
            pred = np.zeros(gt.shape, dtype=bool); nobox = True
        else:
            img = Image.open(rec["image"]).convert("RGB")
            res = medsam_multi(model, proc, img, cs, rec["name"])
            pred = combine([r[0] for r in res], [r[1] for r in res], how, gt.shape)
            nobox = False
        per.append({"name": rec["name"], "cls": rec["cls"], "dice": dice_binary(pred, gt),
                    "fg_ratio": float(pred.mean()), "nobox": nobox,
                    "ncand": len(cs), "type": diagnose(sig, tau)})
    return per


def agg_metrics(per):
    les = [p for p in per if p["cls"] in ("benign", "malignant")]
    nor = [p for p in per if p["cls"] == "normal"]
    rec = [p for p in les if p["nobox"]]
    nfg = [p["fg_ratio"] for p in nor]
    return {"e2e_dice": float(np.mean([p["dice"] for p in les])),
            "nobox_rate": float(np.mean([p["nobox"] for p in les])),
            "nobox_n": len(rec),
            "normal_fg": float(np.mean(nfg)) if nfg else 0.0,
            "normal_clean": float(np.mean([1.0 if f < 0.001 else 0.0 for f in nfg])) if nfg else 0.0,
            "normal_spec": float(1.0 - np.mean(nfg)) if nfg else 0.0,
            "multi_n": int(sum(1 for p in les if p["ncand"] > 1))}


GATE = 0.999   # structural gate: "argmax never calls any pixel a lesion"
VARIANTS = [
    ("B0_single",     "single",       "union",     GATE, "单框 s=1.0（基线）"),
    ("F_union",       "blind",        "union",     GATE, "全盲兜底 + 并集"),
    ("F_majority",    "blind",        "majority",  GATE, "全盲兜底 + 多数投票"),
    ("F_intersect",   "blind",        "intersect", GATE, "全盲兜底 + 交集"),
    ("F_select",      "blind",        "select",    GATE, "全盲兜底 + 按 MedSAM 分数选"),
    ("FR_select",     "blind+rescue", "select",    GATE, "兜底+欠自信救回 + 选"),
    ("FR_union",      "blind+rescue", "union",     GATE, "兜底+欠自信救回 + 并集"),
    ("T_select",      "type",         "select",    GATE, "全类型定向 + 选"),
    ("FR_select_g50", "blind+rescue", "select",    0.50, "只兜底 pmax<0.5（= 全盲）"),
    ("FR_select_g20", "blind+rescue", "select",    0.20, "只兜底 pmax<0.2（更严）"),
    ("FR_select_g35", "blind+rescue", "select",    0.35, "只兜底 pmax<0.35"),
]


def main():
    index = build_index(); byname = {it["name"]: it for it in index}
    split = load_split()
    val = [byname[n] for n in split["val"]]
    test = [byname[n] for n in split["test"]]

    ck = f"{RES}/viz_unetpp_seed0.pth"
    if not os.path.exists(ck):
        ck = f"{RES}/viz_unet_seed0.pth"
    is_pp = "unetpp" in ck
    print("coarse:", ck)
    sig_fn = signals_fn("unetpp" if is_pp else "unet", 256 if is_pp else 128, not is_pp, ck)

    print("computing coarse signals once ...")
    signals = {r["name"]: sig_fn(r) for r in (val + test)}

    vbox = [signals[r["name"]] for r in val if signals[r["name"]]["box"] is not None]
    tau_d = float(np.percentile([s["shift"] for s in vbox], 80)) if vbox else 0.1
    tau_s = float(np.percentile([s["ratio"] for s in vbox], 80)) if vbox else 2.0

    medsam = epq.load_medsam()
    out = {"coarse": os.path.basename(ck), "tau_d": tau_d, "tau_s": tau_s, "variants": {},
           "type_dist": {}, "gate_sweep": {}}

    print("\n%-14s %-12s %5s %8s %7s %8s %6s" %
          ("variant", "policy", "gate", "dice", "nobox", "nor.fg", "multi"))
    lesion_dice = {}
    for name, policy, how, gate, desc in VARIANTS:
        tau = {"p": gate, "d": tau_d, "s": tau_s}
        per = evaluate(test, signals, medsam, tau, policy, how)
        a = agg_metrics(per)
        out["variants"][name] = {"desc": desc, "policy": policy, "agg": how,
                                 "gate": gate, **a}
        lesion_dice[name] = [p["dice"] for p in per if p["cls"] in ("benign", "malignant")]
        print("%-14s %-12s %5.2f %8.4f %7.3f %8.4f %6d" %
              (name, policy, gate, a["e2e_dice"], a["nobox_rate"], a["normal_fg"], a["multi_n"]))
        if name == "FR_select":
            types = [p["type"] for p in per]
            out["type_dist"] = {t: types.count(t) for t in ("正确", "偏移", "尺度错", "欠自信", "全盲")}
            out["per_image_FR_select"] = per

    from scipy.stats import wilcoxon
    base = np.array(lesion_dice["B0_single"])
    for name, d in lesion_dice.items():
        v = np.array(d)
        if name == "B0_single" or np.allclose(v, base):
            out["variants"][name]["p_vs_B0"] = 1.0
            out["variants"][name]["n_better"] = 0
            continue
        try:
            _, p = wilcoxon(v, base)
        except ValueError:
            p = 1.0
        out["variants"][name]["p_vs_B0"] = float(p)
        out["variants"][name]["n_better"] = int(np.sum(v > base + 1e-9))
        out["variants"][name]["n_worse"] = int(np.sum(v < base - 1e-9))

    # gate sensitivity: why tightening tau_p cannot raise the precision of "missing"
    les = [r for r in test if r["cls"] in ("benign", "malignant")]
    nor = [r for r in test if r["cls"] == "normal"]
    for tau in (0.10, 0.20, 0.35, 0.50):
        fl = [r["name"] for r in les if signals[r["name"]]["pmax"] < tau]
        fn_ = [r["name"] for r in nor if signals[r["name"]]["pmax"] < tau]
        out["gate_sweep"][str(tau)] = {
            "n_lesion_flagged": len(fl), "n_normal_flagged": len(fn_),
            "precision": len(fl) / max(len(fl) + len(fn_), 1),
            "recall_of_nobox": sum(1 for n in fl if signals[n]["box"] is None)
                                / max(sum(1 for r in les if signals[r["name"]]["box"] is None), 1)}
    with open(f"{RES}/tier_prompt.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    V = out["variants"]
    b0 = V["B0_single"]["e2e_dice"]
    best_name, best_v = max(V.items(), key=lambda kv: kv[1]["e2e_dice"])

    L = ["# P3　误差类型感知的定向提示纠正（v3）\n",
         f"> 粗模型 `{out['coarse']}`；τ_d=80分位={tau_d:.3f}、τ_s=80分位={tau_s:.2f}；"
         "“全盲”判据**无阈值**：掩膜 argmax 从未判为病灶（无 ≥0.5 连通域）。\n",
         "| 变体 | 说明 | 门控 | 聚合 | 端到端 Dice | Δ vs B0 | p | 无框率 | 正常图 fg | 正常图特异度 | 空掩膜率 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in V:
        v = V[k]
        g = "—" if v["gate"] > 0.9 else f"{v['gate']:.2f}"
        pv = v.get("p_vs_B0", 1.0)
        L.append(f"| {k} | {v['desc']} | {g} | {v['agg']} | **{v['e2e_dice']:.4f}** | "
                 f"{v['e2e_dice'] - b0:+.4f} | {pv:.3f} | {v['nobox_rate']:.3f} | "
                 f"{v['normal_fg']:.4f} | {v['normal_spec']:.4f} | {v['normal_clean']:.2f} |")
    L.append("")
    L.append(f"**Dice 最优**：`{best_name}` = **{best_v['e2e_dice']:.4f}** vs 基线 B0 = {b0:.4f} → "
             f"Δ = **{best_v['e2e_dice'] - b0:+.4f}**（p={best_v.get('p_vs_B0', 1.0):.3f}，"
             f"{best_v.get('n_better', 0)} 张变好 / {best_v.get('n_worse', 0)} 张变差）；"
             f"代价是正常图假阳 **{V['B0_single']['normal_fg']:.4f} → {best_v['normal_fg']:.4f}**。")
    L.append("")
    L.append("**聚合方式对比（同一批多候选，仅换聚合）**\n")
    L.append("| 聚合 | Dice | 正常图 fg | 正常图特异度 |")
    L.append("|---|---|---|---|")
    for k in ("F_union", "F_majority", "F_intersect", "F_select"):
        v = V[k]
        L.append(f"| {v['agg']} | {v['e2e_dice']:.4f} | {v['normal_fg']:.4f} | {v['normal_spec']:.4f} |")
    L.append("")
    L.append("**丢失判据的精度上限（为何收紧 τ_p 无效）**\n")
    L.append("| τ_p | 判为低置信的病灶 | 判为低置信的正常图 | “是病灶”精度 | 覆盖全盲图比例 |")
    L.append("|---|---|---|---|---|")
    for k, d in out["gate_sweep"].items():
        L.append(f"| {k} | {d['n_lesion_flagged']} | {d['n_normal_flagged']} | "
                 f"{d['precision']:.2f} | {d['recall_of_nobox']:.2f} |")
    L.append("")
    L.append("**误差类型分布（测试集 116 张）**：" +
             "、".join(f"{k} {v}" for k, v in out["type_dist"].items()))
    open(f"{RES}/tier_prompt.md", "w").write("\n".join(L) + "\n")

    fig, ax = plt.subplots(figsize=(9, 4))
    ks = list(V.keys())
    ax.bar(range(len(ks)), [V[k]["e2e_dice"] for k in ks],
           color=["#888" if k == "B0_single" else "#c33" for k in ks])
    ax.axhline(b0, ls="--", c="k", lw=1, label=f"B0 {b0:.4f}")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("end-to-end Dice (96 lesions)")
    ax.set_title("P3 v3: fallback prompts x aggregation")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{RES}/tier_prompt.png", dpi=140)

    print("\n".join(L))
    print("wrote tier_prompt.{json,md,png}")


if __name__ == "__main__":
    main()
