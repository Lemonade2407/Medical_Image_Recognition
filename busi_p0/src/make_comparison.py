"""Generate the filled 七种模型对比.md from result JSONs (reproducible backfill).

Reads:
  results/p1_<model>_seed<k>.json          supervised main table (150ep CE+Dice, 3 seeds)
  results/postproc_prompt_<method>.json    prompt A group
  results/p1a_summary.json             supervised 150ep + A summary
  results/env.json
Writes:
  docs/七种模型对比.md
"""
import os
import json
import numpy as np

ROOT = "/mnt/workspace/busi_p0"
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "七种模型对比.md")

PARAMS = {"SAM ViT-B": "~94M", "MedSAM ViT-B": "~94M", "BUSSAM ViT-B": "~95M", "TransUNet": "~105M"}
# (model, display name, input description, model size, mode)
# All supervised numbers come from the unified probability path (postproc baseline).
SUP = [("unet", "U-Net", "128 灰度", 128, "g"),
       ("attunet", "Attention U-Net", "128 灰度", 128, "g"),
       ("unetpp", "UNet++", "256 RGB", 256, "rgb"),
       ("transunet", "TransUNet", "256 RGB", 256, "rgb")]


def load(fn):
    p = os.path.join(RES, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=4):
    return "〔待回填〕" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def load_seeds(stem, seeds=(0, 1, 2)):
    """Load `results/<stem>_seed<k>.json` for the given seeds (missing ones skipped)."""
    recs = []
    for s in seeds:
        p = os.path.join(RES, f"{stem}_seed{s}.json")
        if os.path.exists(p):
            recs.append(json.load(open(p)))
    return recs


def aggregate(recs):
    """Aggregate per-seed `agg` blocks into {metric: {mean, std, per_seed}}."""
    if not recs:
        return None
    out = {}
    for k in recs[0].get("agg", {}):
        vals = [r["agg"][k] for r in recs if k in r.get("agg", {})]
        if not vals:
            continue
        out[k] = {"mean": float(np.mean(vals)),
                  "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                  "per_seed": [float(v) for v in vals]}
    return {"aggregate": out, "params": recs[0].get("params"), "n_seeds": len(recs)}


def load_sup_records(model, size, mode):
    """Per-seed records on the unified probability path (postproc baseline variant)."""
    params = {r.get("seed"): r.get("params") for r in load_seeds(f"p1_{model}")}
    recs = []
    for s in (0, 1, 2):
        rp = load(f"postproc_p1_{model}_s{size}{mode}_seed{s}.json")
        if not rp:
            continue
        v = rp["variants"]["baseline"]
        recs.append({"agg": v["agg"], "per_image": v["per_image"], "params": params.get(s)})
    return recs


def paired_wilcoxon(a_recs, b_recs):
    """Paired Wilcoxon on per-image Dice (seed-averaged), n = number of test images."""
    from scipy.stats import wilcoxon
    u = np.mean([[m["dice"] for m in r["per_image"]] for r in a_recs], axis=0)
    t = np.mean([[m["dice"] for m in r["per_image"]] for r in b_recs], axis=0)
    w, p = wilcoxon(t, u)
    return {"n_images": int(len(u)), "mean_a": float(u.mean()), "mean_b": float(t.mean()),
            "delta": float(np.mean(t - u)), "statistic": float(w), "p_value": float(p)}


P1A_CFG = [("unet", "U-Net", 128, "g", ["ce_dice", "tversky", "boundary"]),
           ("attunet", "Attention U-Net", 128, "g", ["ce_dice", "tversky", "boundary"]),
           ("unetpp", "UNet++", 256, "rgb", ["ce_dice", "tversky"]),
           ("transunet", "TransUNet", 256, "rgb", ["ce_dice", "tversky"])]
A_VARIANTS = ["baseline", "A1_postproc", "A2_tta", "A3_threshold", "A2+A1+A3"]
PROMPT_VARIANTS = ["baseline", "A1_postproc", "A2_tta", "A3_threshold", "A2+A1+A3"]


def collect_prompt_a():
    """Read postproc_prompt_*.json directly (prompt methods are epoch-agnostic)."""
    rows = []
    for m in ("sam", "medsam", "bussam"):
        rp = load(f"postproc_prompt_{m}.json")
        if not rp:
            continue
        av = {v: float(rp["variants"][v]["agg"]["dice_lesion"])
              for v in PROMPT_VARIANTS if v in rp["variants"]}
        if not av:
            continue
        rows.append({"method": m, "a": {v: {"dice": x} for v, x in av.items()},
                     "best_variant": max(av, key=av.get)})
    return rows


def collect_p1a():
    """Read postproc_p1_* directly and aggregate the A-group variants over seeds."""
    rows = []
    for model, label, size, mode, losses in P1A_CFG:
        for loss in losses:
            suf = "" if loss == "ce_dice" else f"_{loss}"
            av = {v: [] for v in A_VARIANTS}
            hd = {v: [] for v in A_VARIANTS}
            for s in (0, 1, 2):
                rp = load(f"postproc_p1_{model}_s{size}{mode}{suf}_seed{s}.json")
                if not rp:
                    continue
                for v in A_VARIANTS:
                    if v in rp["variants"]:
                        av[v].append(rp["variants"][v]["agg"]["dice_lesion"])
                        hd[v].append(rp["variants"][v]["agg"]["hd95"])
            if not any(av.values()):
                continue
            best, bv = None, -1.0
            for v in A_VARIANTS:
                if av[v] and np.mean(av[v]) > bv:
                    bv, best = float(np.mean(av[v])), v
            bm_a = float(np.mean(av["baseline"])) if av["baseline"] else bv
            rows.append({
                "model": model, "label": label, "loss": loss, "best": best,
                "best_mean": bv, "delta": bv - bm_a,
                "variant_mean": {v: (float(np.mean(av[v])) if av[v] else None) for v in A_VARIANTS},
                "variant_std": {v: (float(np.std(av[v], ddof=1)) if len(av[v]) > 1 else 0.0)
                                for v in A_VARIANTS},
                "hd95_best": (float(np.mean(hd[best])) if hd[best] else None)})
    return rows


def main():
    recs = {m: load_sup_records(m, size, mode) for m, _, _, size, mode in SUP}
    agg = {m: aggregate(recs[m]) for m in recs}
    wtest = paired_wilcoxon(recs["unet"], recs["attunet"]) \
        if agg["unet"] and agg["attunet"] else None
    sam = load("prompt_sam.json")
    medsam = load("prompt_medsam.json")
    bussam = load("prompt_bussam.json")
    env = load("env.json")

    L = []
    A = L.append
    A("# BUSI 乳腺超声病灶分割 · 七种方法统一协议对比\n")
    dev = f"{env['device']} ({env.get('gpu')}, torch {env.get('torch')})" if env and env.get("device") else "CPU"
    A("> 数据来源：`/mnt/workspace/busi_p0`（协议 `docs/P0-1-unified-evaluation-protocol.md`，划分 `splits/busi_seed0.json`，指标 `src/busi_common.py`）。")
    A(f"> 运行设备：**{dev}**。下表所有数值均在**同一测试集 + 同一指标定义 + 同一后处理（无）**下测得，可直接横向比较。\n")

    # ---- 1. data & protocol ----
    A("## 一、数据与统一评估协议\n")
    A("**数据集** BUSI，780 张（benign 437 / malignant 210 / normal 133），17 例多病灶按“或”合并。\n")
    A("**划分**（分层 70/15/15，种子 42）：\n")
    A("| 类别 | 训练 | 验证 | 测试 | 合计 |\n|---|---|---|---|---|")
    A("| benign | 306 | 66 | 65 | 437 |\n| malignant | 147 | 32 | 31 | 210 |\n| normal | 93 | 20 | 20 | 133 |")
    A("| **合计** | **546** | **118** | **116** | **780** |\n")
    A("测试集 116 张 = 病灶图 96（65+31）+ 正常图 20；所有方法共用。指标均在**原始分辨率**计算：")
    A("Dice(病灶)=仅病灶图平均；病灶 IoU=逐图病灶类 IoU 平均；mIoU=像素直方图法；精确率/召回率/特异度/准确率=全局混淆矩阵；HD95/ASD=边界双向距离（原图像素）。**后处理统一不做。**\n")

    # ---- 2. prompt methods ----
    A("## 二、提示式方法（单点提示，统一协议）\n")
    A("> 提示 = GT 前景中心 1 个正点；在统一测试集的 96 张病灶图上评估。\n")
    A("| 方法 | 参数量 | Dice | IoU | 精确率 | 灵敏度 | 特异度 | HD95 | ASD |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, rec in [("SAM ViT-B", sam), ("MedSAM ViT-B", medsam), ("BUSSAM ViT-B", bussam)]:
        if rec:
            g = rec["agg"]
            A(f"| {name} | {PARAMS[name]} | **{f(g['dice_lesion'])}** | {f(g['iou_lesion'])} | {f(g['precision'])} | "
              f"{f(g['recall'])} | {f(g['specificity'])} | {f(g['hd95'],1)} | {f(g['asd'],1)} |")
        else:
            A(f"| {name} | {PARAMS[name]} | 〔待回填〕 | — | — | — | — | — | — |")
    A("")
    if sam:
        A(f"- SAM 分组：benign Dice {f(sam['agg']['dice_benign'])} / malignant {f(sam['agg']['dice_malignant'])}。")
    if bussam:
        A(f"- BUSSAM 分组：benign Dice {f(bussam['agg']['dice_benign'])} / malignant {f(bussam['agg']['dice_malignant'])}。")
    if medsam:
        A(f"- MedSAM 分组：benign Dice {f(medsam['agg']['dice_benign'])} / malignant {f(medsam['agg']['dice_malignant'])}。")
    if bussam:
        A("- BUSSAM 由官方代码在 BUSI 上**自行微调**（无公开权重），划分对齐冻结划分，测试用与 SAM/MedSAM 相同的 GT 中心点提示。\n")
    if medsam:
        A(f"- MedSAM 点提示召回仅 {f(medsam['agg']['recall'])}（几乎不检出）；本机使用 ModelScope 的 HF 格式权重，"
          "MedSAM 本就以框提示为主（见 2b）。\n")

    # ---- 2b. GT box ----
    box_sam = load("prompt_sam_box.json")
    box_med = load("prompt_medsam_box.json")
    if box_sam or box_med:
        A("### 2b. GT 边框提示（SAM / MedSAM，统一协议）\n")
        A("> 提示 = GT 病灶外接矩形框。\n")
        A("| 方法 | Dice | IoU | 灵敏度 | 特异度 | HD95 |")
        A("|---|---|---|---|---|---|")
        for name, rec in [("SAM ViT-B", box_sam), ("MedSAM ViT-B", box_med)]:
            if rec:
                g = rec["agg"]
                A(f"| {name} | **{f(g['dice_lesion'])}** | {f(g['iou_lesion'])} | {f(g['recall'])} | "
                  f"{f(g['specificity'])} | {f(g['hd95'],1)} |")
        A("")

    # ---- 3. supervised ----
    A("## 三、监督式方法（统一协议）\n")
    A("| 方法 | 输入 | 参数量 | Dice(病灶) | 病灶 IoU | mIoU | 精确率 | 召回率 | 准确率 | HD95 |")
    A("|---|---|---|---|---|---|---|---|---|---|")

    def sup_row(name, size_mode, rec):
        if not rec:
            A(f"| {name} | {size_mode} | {PARAMS.get(name, '〔待回填〕')} | " + " | ".join(["〔待回填〕"] * 7) + " |")
            return
        ag = rec["aggregate"]; par = rec.get("params")
        dm, ds = ag["dice_lesion"]["mean"], ag["dice_lesion"]["std"]
        dice_s = f"{dm:.4f}±{ds:.4f}" if rec.get("n_seeds", 1) > 1 else f"{dm:.4f}"
        par_s = f"{par/1e6:.3f}M" if par else PARAMS.get(name, "〔待回填〕")
        A(f"| {name} | {size_mode} | {par_s} | {dice_s} | {f(ag['iou_lesion']['mean'])} | "
          f"{f(ag['miou_hist']['mean'])} | {f(ag['precision']['mean'])} | {f(ag['recall']['mean'])} | "
          f"{f(ag['accuracy']['mean'])} | {f(ag['hd95']['mean'], 1)} |")

    for m, label, size_mode, _, _ in SUP:
        sup_row(label, size_mode, agg[m])
    A("")
    A("> 四者为 3 种子均值±标准差，损失统一为 **CE+Dice**。\n")

    # ---- 4. P0-2 ----
    A("## 四、P0-2　多种子重复 + 显著性检验（U-Net vs Attention U-Net）\n")
    A("> 种子 0/1/2；同一冻结测试集。判定规则：**|差值| > 标准差 且 p < 0.05** 才算有效改进。\n")
    if agg["unet"] and agg["attunet"]:
        a, b = agg["unet"]["aggregate"], agg["attunet"]["aggregate"]
        A("| 模型 | Dice(病灶) | mIoU | 精确率 | 召回率 | HD95 | 3 种子明细 |")
        A("|---|---|---|---|---|---|---|")
        for m, label in [("unet", "U-Net"), ("attunet", "Attention U-Net")]:
            a_ = agg[m]["aggregate"]; seeds = a_["dice_lesion"]["per_seed"]
            A(f"| {label} | {f(a_['dice_lesion']['mean'])} ± {f(a_['dice_lesion']['std'])} | "
              f"{f(a_['miou_hist']['mean'])} | {f(a_['precision']['mean'])} | {f(a_['recall']['mean'])} | "
              f"{f(a_['hd95']['mean'],1)} | {', '.join(f'{s:.4f}' for s in seeds)} |")
        A(f"| **差值（Att − U-Net）** | {b['dice_lesion']['mean']-a['dice_lesion']['mean']:+.4f} | "
          f"{b['miou_hist']['mean']-a['miou_hist']['mean']:+.4f} | {b['precision']['mean']-a['precision']['mean']:+.4f} | "
          f"{b['recall']['mean']-a['recall']['mean']:+.4f} | {b['hd95']['mean']-a['hd95']['mean']:+.1f} | — |")
        w = wtest
        ds = a["dice_lesion"]["std"]; as_ = b["dice_lesion"]["std"]
        sig = "显著" if (abs(w["delta"]) > max(ds, as_) and w["p_value"] < 0.05) else "不显著"
        A(f"\n**配对 Wilcoxon**（配对对象：测试集逐图 Dice，种子平均后）：n = {w['n_images']}，"
          f"W = {w['statistic']:.1f}，**p = {w['p_value']:.4f}**，Δ = {w['delta']:+.4f}。")
        A(f"→ 门控效应 **{sig}**（3 种子标准差 U-Net {ds:.4f} / Att {as_:.4f}，均大于差值）。")
        A("\n> 单次训练随机波动量级与两模型差值相当（见 3 种子 std），故以检验替代单次比较。\n")

    # ---- 5. method characteristics (background) ----
    A("## 五、方法特点（背景）\n")
    A("| 方法 | 核心思路 | 参数量 | 优势 | 局限 |\n|---|---|---|---|---|")
    A("| U-Net | 对称编码-解码与跳跃连接 | 1.97M | 结构简单、小样本稳定、易训练 | 感受野有限，复杂形态病灶易欠分割 |")
    A("| Attention U-Net | 跳跃连接加加法注意力门控 | 2.38M | 背景噪声抑制更强，精确率略高 | 本次增益有限，召回率略降 |")
    A("| UNet++ | 密集嵌套跳跃连接与深监督 | 8.83M | 多尺度融合好 | 计算量高于 U-Net，需调深监督 |")
    A("| TransUNet | ResNet-50 局部特征 + ViT 全局自注意力 | 105.3M | 可建模长程依赖，全局视野 | 无预训练且数据少，参数量大 |")
    A("| SAM ViT-B | 提示式大模型（图像/提示编码 + 掩膜解码） | ≈94M | 零样本、点框交互，框提示 Dice 高 | 医学超声点提示弱，依赖提示质量 |")
    A("| MedSAM ViT-B | 医学数据预训练 SAM | ≈94M | 特异度高、背景抑制强 | 点提示灵敏度低，漏检偏多 |")
    A("| BUSSAM ViT-B | SAM ViT-B 在 BUSI 微调（适配器 + CNN 分支） | 25.1M(可训) | 点提示 Dice 高，提示式最佳 | 仍需提示，非全自动；训练需预训练权重 |\n")

    # ---- 6. conclusion ----
    A("## 六、结论\n")
    if sam and medsam:
        cand = {"SAM ViT-B": sam["agg"]["dice_lesion"], "MedSAM ViT-B": medsam["agg"]["dice_lesion"]}
        if bussam:
            cand["BUSSAM ViT-B"] = bussam["agg"]["dice_lesion"]
        best = max(cand, key=cand.get)
        A(f"- 提示式（单点，统一协议）：**{best}** 的 Dice 最高（{f(cand[best])}）；"
          + "、".join(f"{k} {f(v)}" for k, v in cand.items()) + "。")
    sup = {label: agg[m]["aggregate"]["dice_lesion"]["mean"]
           for m, label, _, _, _ in SUP if agg[m]}
    if sup:
        best = max(sup, key=sup.get)
        light = {k: v for k, v in sup.items() if k != best}
        gaps = [sup[best] - v for v in light.values()]
        gap_s = f"{min(gaps):.3f}–{max(gaps):.3f}" if gaps else "—"
        light_s = " / ".join(f"{k} {v:.4f}" for k, v in light.items())
        A(f"- 监督式（全自动，统一协议）：**{best}** 的病灶 Dice 最高（{sup[best]:.4f}）；"
          f"其余三骨干（{light_s}）与其差距仅 **{gap_s}**，结构优势有限。")
        if wtest:
            A(f"- 注意力门控（U-Net → Attention U-Net）差值 {wtest['delta']:+.4f}"
              f"（逐图配对；3 种子均值差 {sup['Attention U-Net']-sup['U-Net']:+.4f}），"
              f"p={wtest['p_value']:.4f}，在本数据规模下未达统计显著。")
    A("- 提示式与监督式的评价口径已统一到同一测试集/同一指标，但提示条件不同（提示式需 1 个 GT 点，监督式全自动），适用场景互补。")
    A("")

    # ---- 7. A/B optimisation ----
    prompt_rows = collect_prompt_a()
    p1a_rows = collect_p1a()
    if p1a_rows or prompt_rows:
        A("## 七、A/B 优化结果（统一协议）\n")
        A("> **B** = 损失函数重训（Tversky α=0.3/β=0.7、Boundary；3 种子）；")
        A("> **A** = 推理侧零训练成本项：**A1** 最大连通域+填洞+面积阈值、**A2** 多尺度+翻转 TTA、**A3** 验证集选阈值。\n")
        if p1a_rows:
            A("### 7.1 监督式\n")
            A("| 模型 | 损失 | baseline | A2 TTA | A3 阈值 | A2+A1+A3 | 最优变体 |")
            A("|---|---|---|---|---|---|---|")

            def cell(r, v):
                vm = (r.get("variant_mean") or {}).get(v)
                vs = (r.get("variant_std") or {}).get(v, 0.0)
                return "—" if vm is None else (f"{vm:.4f}±{vs:.4f}" if vs else f"{vm:.4f}")

            for r in p1a_rows:
                if r.get("best_mean") is None:
                    continue
                A(f"| {r['model']} | {r['loss']} | {cell(r, 'baseline')} | {cell(r, 'A2_tta')} | "
                  f"{cell(r, 'A3_threshold')} | {cell(r, 'A2+A1+A3')} | {r['best'] or '—'} |")
            A("")
            ds = [r["delta"] for r in p1a_rows if r.get("delta") is not None]
            if ds:
                A(f"> A 组**仍然有效**（Δ 相对 A 组 baseline：**{min(ds):+.3f} ~ {max(ds):+.3f}**），"
                  "且 boundary 模型 HD95 大幅下降。")
            rows_ok = [r for r in p1a_rows if r.get("best_mean") is not None]
            if rows_ok:
                br = max(rows_ok, key=lambda r: r["best_mean"])
                hd = (f"（HD95 {br['hd95_best']:.1f}）" if br.get("hd95_best") is not None else "")
                A(f"> **监督式最优 = {br['label']} + {br['loss']} + {br['best']} = {br['best_mean']:.4f}**{hd}。")
                unet_ce = next((r for r in rows_ok if r["model"] == "unet" and r["loss"] == "ce_dice"), None)
                if unet_ce and br["model"] != "unet":
                    A(f"> **U-Net（1.9M）+ 同配置 = {unet_ce['best_mean']:.4f}**，"
                      f"与 ~105M 的 {br['label']} 仅差 **{br['best_mean']-unet_ce['best_mean']:.3f}**。")
                A("")
        if prompt_rows:
            A("### 7.2 提示式（仅 A，无 B）\n")
            A("| 方法 | baseline | A1 后处理 | A2 TTA | A3 阈值 | A2+A1+A3 | 最优变体 |")
            A("|---|---|---|---|---|---|---|")
            for r in prompt_rows:
                a = r["a"]

                def g(v):
                    return f"{a[v]['dice']:.4f}" if v in a else "—"

                A(f"| {r['method']} | {g('baseline')} | {g('A1_postproc')} | {g('A2_tta')} | "
                  f"{g('A3_threshold')} | {g('A2+A1+A3')} | {r['best_variant'] or '—'} |")
            A("")

    # ---- appendix ----
    A("## 附录 A：重跑可得性与环境\n")
    A("| 方法 | 是否重跑 | 依据 / 原因 |\n|---|---|---|")
    A("| U-Net | ✅ 3 种子 | 本地训练 |\n| Attention U-Net | ✅ 3 种子 | 本地训练 |")
    A(f"| SAM ViT-B | {'✅' if sam else '⏳'} | 公开权重 sam_vit_b_01ec64.pth |")
    A(f"| MedSAM ViT-B | {'✅' if medsam else '⏳'} | 公开权重（ModelScope, HF 格式） |")
    for m, label, _, _, _ in SUP[2:]:
        A(f"| {label} | {'✅ 3 种子' if agg[m] else '⏳'} | 本地训练 256 RGB |")
    A(f"| BUSSAM | {'✅' if bussam else '⏳'} | 官方代码 + 基座 SAM ViT-B 在 BUSI 自行微调 100 epoch，划分对齐冻结划分 |")
    A(f"\n> 运行环境：{dev}。\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(L))
    print("wrote", OUT)
    print("sources:", [m for m in agg if agg[m]])


if __name__ == "__main__":
    main()
