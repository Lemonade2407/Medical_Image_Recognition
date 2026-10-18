"""P3 v4: error-type-aware targeted prompt correction, on the 150-epoch weights.

Changes vs v3:
  * the coarse model is the P1 checkpoint (150 ep + early stop + best-val
    rollback) instead of the 24 ep viz_* one, i.e. the same weights the main
    comparison table uses;
  * all three seeds are run, so the effect is reported as mean±std and the
    paired test uses seed-averaged per-image Dice (project convention);
  * an extra control isolates the specificity killer: fallback restricted to
    centre boxes (no whole-image box).

Diagnosis (measured, probe_tier2.py on these weights): of the 96 test lesion
images 86 have a coarse box at 0.5, 0 have one only at the low threshold, and
10 have nothing. Those 10 are the recovery target; their pmax overlaps the 13
low-confidence normal images, so the gate precision is capped (~0.43 at
tau=0.5) no matter where a pmax percentile is set.

Outputs: results/<--out>.json / .md / .png (default prefix tier_prompt).
"""
import os
import sys
import json
import argparse
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/mnt/workspace/busi_p0/src"
RES = "/mnt/workspace/busi_p0/results"
sys.path.insert(0, SRC)
from busi_common import build_index, load_split, orig_size, load_mask, dice_binary  # noqa
import eval_prompt_quality as epq  # noqa

_spec = importlib.util.spec_from_file_location("busi_models6", os.path.join(SRC, "models.py"))
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


def fallback_boxes(wh, whole=True):
    """Region proposals that do not depend on the coarse map.

    The centre-only variant keeps three boxes on purpose: majority voting over an
    even number of candidates degenerates to intersection (a tie at 0.5 is not
    strictly greater), so a two-box set would make majority and intersect the
    same experiment.
    """
    w, h = wh
    out = [np.array([0, 0, w - 1, h - 1], dtype=np.float64)] if whole else []
    for frac in ((0.70, 0.45) if whole else (0.85, 0.70, 0.45)):
        bw, bh = w * frac / 2, h * frac / 2
        out.append(np.array([w / 2 - bw, h / 2 - bh, w / 2 + bw, h / 2 + bh], dtype=np.float64))
    return out


def diagnose(sig, tau):
    """Errors from the coarse map only; no GT and no percentile gate."""
    if sig["box"] is None:
        return "欠自信" if sig["box_lo"] is not None else "全盲"
    if sig["shift"] > tau["d"]:
        return "偏移"
    if sig["ratio"] > tau["s"]:
        return "尺度错"
    return "正确"


def candidates(sig, tau, policy):
    """policy: single | blind | blind_center | blind+rescue | type.

    The presence gate (PGBF) is applied in `evaluate`, so that it is switched on
    by `--presence` uniformly rather than per variant.
    """
    if policy == "single":
        return [] if sig["box"] is None else [expand(sig["box"], 1.0)]

    t = diagnose(sig, tau)
    blind = t in ("全盲", "欠自信")
    if blind and sig["pmax"] >= tau["p"]:
        return []

    if policy in ("blind", "blind_center", "blind+rescue"):
        if t == "全盲":
            return fallback_boxes(sig["wh"], whole=(policy != "blind_center"))
        if t == "欠自信":
            if policy == "blind+rescue":
                return [expand(sig["box_lo"], s) for s in SCALES]
            return []
        return [expand(sig["box"], 1.0)]

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
def evaluate(records, signals, medsam, tau, policy, how, presence=None, presence_thr=0.5):
    model, proc = medsam
    per = []
    for rec in records:
        sig = signals[rec["name"]]
        gt = load_mask(rec)
        t = diagnose(sig, tau)
        cs = candidates(sig, tau, policy)
        p_abn = presence.get(rec["name"]) if presence else None
        gated = bool(cs and t == "全盲" and presence is not None
                     and p_abn is not None and p_abn < presence_thr)
        if gated:
            cs = []
        if not cs:
            pred = np.zeros(gt.shape, dtype=bool); nobox = True
        else:
            img = Image.open(rec["image"]).convert("RGB")
            res = medsam_multi(model, proc, img, cs, rec["name"])
            pred = combine([r[0] for r in res], [r[1] for r in res], how, gt.shape)
            nobox = False
        per.append({"name": rec["name"], "cls": rec["cls"], "dice": dice_binary(pred, gt),
                    "fg_ratio": float(pred.mean()), "nobox": nobox,
                    "ncand": len(cs), "type": t, "p_abn": p_abn, "gated": gated})
    return per


def agg_metrics(per):
    les = [p for p in per if p["cls"] in ("benign", "malignant")]
    nor = [p for p in per if p["cls"] == "normal"]
    nfg = [p["fg_ratio"] for p in nor]
    return {"e2e_dice": float(np.mean([p["dice"] for p in les])),
            "nobox_rate": float(np.mean([p["nobox"] for p in les])),
            "normal_fg": float(np.mean(nfg)) if nfg else 0.0,
            "normal_clean": float(np.mean([1.0 if f < 0.001 else 0.0 for f in nfg])) if nfg else 0.0,
            "normal_spec": float(1.0 - np.mean(nfg)) if nfg else 0.0,
            "multi_n": int(sum(1 for p in les if p["ncand"] > 1))}


GATE = 0.999   # structural gate: "argmax never calls any pixel a lesion"
VARIANTS = [
    ("B0_single",     "single",       "union",     GATE, "单框 s=1.0（基线）"),
    ("F_union",       "blind",        "union",     GATE, "全盲兜底（含全图框）+ 并集"),
    ("F_majority",    "blind",        "majority",  GATE, "全盲兜底（含全图框）+ 多数投票"),
    ("F_intersect",   "blind",        "intersect", GATE, "全盲兜底（含全图框）+ 交集"),
    ("F_select",      "blind",        "select",    GATE, "全盲兜底（含全图框）+ 按分数选"),
    ("FC_majority",   "blind_center", "majority",  GATE, "全盲兜底（**仅中心框**）+ 多数投票"),
    ("FC_intersect",  "blind_center", "intersect", GATE, "全盲兜底（**仅中心框**）+ 交集"),
    ("FC_select",     "blind_center", "select",    GATE, "全盲兜底（**仅中心框**）+ 按分数选"),
    ("FR_select",     "blind+rescue", "select",    GATE, "兜底 + 欠自信救回 + 选"),
    ("T_select",      "type",         "select",    GATE, "全类型定向 + 选"),
    ("FR_select_g20", "blind+rescue", "select",    0.20, "只兜底 pmax<0.2"),
    ("FR_select_g35", "blind+rescue", "select",    0.35, "只兜底 pmax<0.35"),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unetpp")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--ckpt", default="", help="checkpoint path template with {seed}; "
                                              "empty -> results/p1_<model>_seed<k>.pth")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="tier_prompt")
    ap.add_argument("--presence", default="",
                    help="JSON from train_presence_clf (_scores.json, flat or {threshold,scores}); "
                         "gates the blind fallback (T2)")
    ap.add_argument("--presence-thr", type=float, default=None,
                    help="override the threshold stored in the presence JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    index = build_index(); byname = {it["name"]: it for it in index}
    split = load_split()
    val = [byname[n] for n in split["val"]]
    test = [byname[n] for n in split["test"]]

    medsam = epq.load_medsam()
    presence, pres_thr = None, 0.5
    if args.presence:
        raw = json.load(open(args.presence))
        if isinstance(raw, dict) and "scores" in raw:
            presence, file_thr = raw["scores"], raw.get("threshold")
        else:
            presence, file_thr = raw, None
        if args.presence_thr is not None:
            pres_thr = args.presence_thr
        elif file_thr is not None:
            pres_thr = float(file_thr)
    out = {"model": args.model, "size": args.size, "gray": bool(args.gray),
           "seeds": seeds, "ckpts": [], "variants": {}, "per_seed": {}, "gate_sweep": {},
           "presence": ({"path": os.path.basename(args.presence), "thr": pres_thr}
                        if presence else None)}

    for seed in seeds:
        ck = (args.ckpt.format(seed=seed) if args.ckpt
              else f"{RES}/p1_{args.model}_seed{seed}.pth")
        if not os.path.exists(ck):
            print("MISSING", ck); continue
        out["ckpts"].append(os.path.basename(ck))
        print(f"\n=== seed {seed}  ({os.path.basename(ck)}) ===", flush=True)

        sig_fn = signals_fn(args.model, args.size, args.gray, ck)
        signals = {r["name"]: sig_fn(r) for r in (val + test)}
        vbox = [signals[r["name"]] for r in val if signals[r["name"]]["box"] is not None]
        tau = {"p": GATE,
               "d": float(np.percentile([s["shift"] for s in vbox], 80)) if vbox else 0.1,
               "s": float(np.percentile([s["ratio"] for s in vbox], 80)) if vbox else 2.0}

        les = [r for r in test if r["cls"] in ("benign", "malignant")]
        nor = [r for r in test if r["cls"] == "normal"]
        sweep = {}
        for t in (0.10, 0.20, 0.35, 0.50):
            fl = [r["name"] for r in les if signals[r["name"]]["pmax"] < t]
            fn_ = [r["name"] for r in nor if signals[r["name"]]["pmax"] < t]
            sweep[str(t)] = {"n_lesion": len(fl), "n_normal": len(fn_),
                             "cov": sum(1 for n in fl if signals[n]["box"] is None)
                                    / max(sum(1 for r in les if signals[r["name"]]["box"] is None), 1)}

        ps = {"tau": tau, "variants": {}}
        print("%-15s %7s %7s %7s" % ("variant", "dice", "nobox", "nor.fg"), flush=True)
        for name, policy, how, gate, desc in VARIANTS:
            tg = dict(tau); tg["p"] = gate
            per = evaluate(test, signals, medsam, tg, policy, how,
                           presence, pres_thr)
            a = agg_metrics(per)
            ps["variants"][name] = dict(a, **{"desc": desc, "policy": policy, "agg": how,
                                              "gate": gate,
                                              "lesion_dice": [p["dice"] for p in per
                                                              if p["cls"] in ("benign", "malignant")]})
            print("%-15s %7.4f %7.3f %7.4f" % (name, a["e2e_dice"], a["nobox_rate"], a["normal_fg"]),
                  flush=True)
            if name == "FR_select":
                ty = [p["type"] for p in per]
                ps["type_dist"] = {t: ty.count(t) for t in ("正确", "偏移", "尺度错", "欠自信", "全盲")}
                ps["per_image"] = per
        ps["gate_sweep"] = sweep
        out["per_seed"][str(seed)] = ps

    if not out["per_seed"]:
        raise SystemExit("no seed ran")

    names = [n for n, *_ in VARIANTS]
    for name in names:
        rows = [out["per_seed"][s]["variants"][name] for s in out["per_seed"]]
        agg = {}
        for k in ("e2e_dice", "nobox_rate", "normal_fg", "normal_spec", "normal_clean", "multi_n"):
            v = np.array([r[k] for r in rows], dtype=float)
            agg[k] = float(v.mean())
            agg[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        agg.update({"desc": rows[0]["desc"], "policy": rows[0]["policy"],
                    "agg": rows[0]["agg"], "gate": rows[0]["gate"],
                    "_dice_vec": np.mean([r["lesion_dice"] for r in rows], axis=0).tolist()})
        out["variants"][name] = agg

    base = np.array(out["variants"]["B0_single"]["_dice_vec"])
    for name in names:
        v = np.array(out["variants"][name]["_dice_vec"])
        if name == "B0_single" or np.allclose(v, base):
            out["variants"][name].update({"p_vs_B0": 1.0, "n_better": 0, "n_worse": 0})
            continue
        try:
            _, p = wilcoxon(v, base)
        except ValueError:
            p = 1.0
        out["variants"][name].update({"p_vs_B0": float(p),
                                      "n_better": int(np.sum(v > base + 1e-9)),
                                      "n_worse": int(np.sum(v < base - 1e-9))})

    for t in ("0.1", "0.2", "0.35", "0.5"):
        rs = [out["per_seed"][s]["gate_sweep"][t] for s in out["per_seed"]]
        out["gate_sweep"][t] = {
            "n_lesion": float(np.mean([r["n_lesion"] for r in rs])),
            "n_normal": float(np.mean([r["n_normal"] for r in rs])),
            "precision": float(np.mean([r["n_lesion"] / max(r["n_lesion"] + r["n_normal"], 1)
                                        for r in rs])),
            "cov": float(np.mean([r["cov"] for r in rs]))}
    out["type_dist"] = {t: float(np.mean([out["per_seed"][s]["type_dist"][t]
                                          for s in out["per_seed"]]))
                        for t in out["per_seed"][list(out["per_seed"])[0]]["type_dist"]}

    for name in names:
        out["variants"][name].pop("_dice_vec", None)
        for s in out["per_seed"]:
            out["per_seed"][s]["variants"][name].pop("lesion_dice", None)

    with open(f"{RES}/{args.out}.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    V = out["variants"]
    b0 = V["B0_single"]["e2e_dice"]
    best = max(V.items(), key=lambda kv: kv[1]["e2e_dice"])
    ms = lambda v, k="e2e_dice": (f"{v[k]:.4f}±{v[k+'_std']:.4f}" if v[k + "_std"] else f"{v[k]:.4f}")

    L = ["# P3　误差类型感知的定向提示纠正（v4 · 150ep 权重）\n",
         f"> 粗模型：`{', '.join(out['ckpts'])}`（P1，150ep + 早停 + 最优轮回滚，"
         f"{len(seeds)} 种子均值）。τ_d、τ_s 取验证集 80 分位；"
         "“全盲”判据**无阈值**：掩膜 argmax 从未判为病灶。\n",
         "| 变体 | 说明 | 聚合 | Dice（3 种子） | Δ vs B0 | p | 无框率 | 正常图 fg | 特异度 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in names:
        v = V[k]
        g = "" if v["gate"] > 0.9 else f"（gate {v['gate']:.2f}）"
        L.append(f"| {k} | {v['desc']}{g} | {v['agg']} | **{ms(v)}** | "
                 f"{v['e2e_dice']-b0:+.4f} | {v['p_vs_B0']:.3f} | {v['nobox_rate']:.3f} | "
                 f"{v['normal_fg']:.4f} | {v['normal_spec']:.4f} |")
    L.append("")
    L.append(f"**Dice 最优**：`{best[0]}` = **{ms(best[1])}** vs B0 {ms(V['B0_single'])} → "
             f"Δ = **{best[1]['e2e_dice']-b0:+.4f}**（p={best[1]['p_vs_B0']:.3f}，"
             f"{best[1]['n_better']} 张好 / {best[1]['n_worse']} 张差）；正常图假阳 "
             f"{V['B0_single']['normal_fg']:.4f} → {best[1]['normal_fg']:.4f}。")
    L.append("")
    L.append("**兜底区域的选择（含全图框 vs 仅中心框）**\n")
    L.append("| 区域提议 | 聚合 | Dice | 正常图 fg | 特异度 |")
    L.append("|---|---|---|---|---|")
    for k in ("F_majority", "F_intersect", "FC_majority", "FC_intersect"):
        v = V[k]
        L.append(f"| {'含全图框' if k.startswith('F_') else '仅中心框'} | {v['agg']} | "
                 f"{ms(v)} | {v['normal_fg']:.4f} | {v['normal_spec']:.4f} |")
    L.append("")
    L.append("**丢失判据的精度上限**（3 种子平均）\n")
    L.append("| τ_p | 判为低置信的病灶 | 正常图 | “是病灶”精度 | 全盲覆盖 |")
    L.append("|---|---|---|---|---|")
    for k, d in out["gate_sweep"].items():
        L.append(f"| {k} | {d['n_lesion']:.1f} | {d['n_normal']:.1f} | "
                 f"{d['precision']:.2f} | {d['cov']:.2f} |")
    L.append("")
    L.append("**误差类型分布（116 张，3 种子平均）**：" +
             "、".join(f"{k} {v:.1f}" for k, v in out["type_dist"].items()))
    open(f"{RES}/{args.out}.md", "w").write("\n".join(L) + "\n")

    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.bar(range(len(names)), [V[k]["e2e_dice"] for k in names],
           yerr=[V[k]["e2e_dice_std"] for k in names], capsize=3,
           color=["#888" if k == "B0_single" else "#c33" for k in names])
    ax.axhline(b0, ls="--", c="k", lw=1, label=f"B0 {b0:.4f}")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("end-to-end Dice (96 lesions)")
    ax.set_title("P3 v4 (150ep weights, 3 seeds): fallback prompts x aggregation")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{RES}/{args.out}.png", dpi=140)

    print("\n".join(L))
    print(f"wrote {args.out}.{{json,md,png}}")


if __name__ == "__main__":
    main()
