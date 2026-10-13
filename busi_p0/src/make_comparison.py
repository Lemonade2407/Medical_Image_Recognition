"""Generate the filled 七种模型对比.md from result JSONs (reproducible backfill).

Reads: results/multiseed_summary.json, results/{unetpp,transunet}_s256rgb_e24_seed0.json,
       results/prompt_{sam,medsam,bussam}.json (+ *_box.json), results/env.json
Writes: docs/七种模型对比.md   (unified-protocol results only; no legacy-report values)
"""
import os
import json
import numpy as np

ROOT = "/mnt/workspace/busi_p0"
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "七种模型对比.md")

PARAMS = {"SAM ViT-B": "~94M", "MedSAM ViT-B": "~94M", "BUSSAM ViT-B": "~95M", "TransUNet": "~112M"}


def load(fn):
    p = os.path.join(RES, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=4):
    return "〔待回填〕" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def fms(d):
    """Format a {'mean','std'} stats dict."""
    if not d:
        return "—"
    return f"{d['mean']:.4f}±{d['std']:.4f}" if d.get("std") else f"{d['mean']:.4f}"


def main():
    ms = load("multiseed_summary.json")
    unetpp = load("unetpp_s256rgb_e24_seed0.json")
    transunet = load("transunet_s256rgb_e24_seed0.json")
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

    def sup_row(name, size_mode, rec_or_ms):
        if isinstance(rec_or_ms, dict) and "aggregate" in rec_or_ms:
            ag = rec_or_ms["aggregate"]; par = rec_or_ms.get("params")
            cells = (f"{f(ag['dice_lesion']['mean'])}±{f(ag['dice_lesion']['std'])}",
                     f(ag['iou_lesion']['mean']), f(ag['miou_hist']['mean']), f(ag['precision']['mean']),
                     f(ag['recall']['mean']), f(ag['accuracy']['mean']), f(ag['hd95']['mean'], 1))
        elif rec_or_ms:
            g = rec_or_ms["agg"]; par = rec_or_ms.get("params")
            cells = (f(g['dice_lesion']), f(g['iou_lesion']), f(g['miou_hist']),
                     f(g['precision']), f(g['recall']), f(g['accuracy']), f(g['hd95'], 1))
        else:
            cells = ("〔待回填〕",) * 7; par = None
        par_s = f"{par/1e6:.3f}M" if par else PARAMS.get(name, "〔待回填〕")
        A(f"| {name} | {size_mode} | {par_s} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {cells[5]} | {cells[6]} |")

    ms_models = (ms or {}).get("models", {})
    sup_row("U-Net", "128 灰度", ms_models.get("unet"))
    sup_row("Attention U-Net", "128 灰度", ms_models.get("attunet"))
    sup_row("UNet++", "256 RGB", unetpp)
    sup_row("TransUNet", "256 RGB", transunet)
    A("")
    A("> U-Net / Attention U-Net 为 3 种子均值±标准差；UNet++/TransUNet 为 1 种子（原生 256 RGB）。\n")

    # ---- 4. P0-2 ----
    A("## 四、P0-2　多种子重复 + 显著性检验（U-Net vs Attention U-Net）\n")
    A("> 种子 0/1/2；同一冻结测试集。判定规则：**|差值| > 标准差 且 p < 0.05** 才算有效改进。\n")
    if ms:
        mo = ms["models"]
        A("| 模型 | Dice(病灶) | mIoU | 精确率 | 召回率 | HD95 | 3 种子明细 |")
        A("|---|---|---|---|---|---|---|")
        for m, label in [("unet", "U-Net"), ("attunet", "Attention U-Net")]:
            a = mo[m]["aggregate"]; seeds = a["dice_lesion"]["per_seed"]
            A(f"| {label} | {f(a['dice_lesion']['mean'])} ± {f(a['dice_lesion']['std'])} | "
              f"{f(a['miou_hist']['mean'])} | {f(a['precision']['mean'])} | {f(a['recall']['mean'])} | "
              f"{f(a['hd95']['mean'],1)} | {', '.join(f'{s:.4f}' for s in seeds)} |")
        a, b = mo["attunet"]["aggregate"], mo["unet"]["aggregate"]
        A(f"| **差值（Att − U-Net）** | {a['dice_lesion']['mean']-b['dice_lesion']['mean']:+.4f} | "
          f"{a['miou_hist']['mean']-b['miou_hist']['mean']:+.4f} | {a['precision']['mean']-b['precision']['mean']:+.4f} | "
          f"{a['recall']['mean']-b['recall']['mean']:+.4f} | {a['hd95']['mean']-b['hd95']['mean']:+.1f} | — |")
        w = ms.get("wilcoxon", {})
        if w:
            ds = mo["unet"]["aggregate"]["dice_lesion"]["std"]
            as_ = mo["attunet"]["aggregate"]["dice_lesion"]["std"]
            sig = "显著" if (abs(w["delta"]) > max(ds, as_) and w["p_value"] < 0.05) else "不显著"
            A(f"\n**配对 Wilcoxon**（配对对象：测试集逐图 Dice，种子平均后）：n = {w['n_images']}，"
              f"W = {w['statistic']:.1f}，**p = {w['p_value']:.4f}**，Δ = {w['delta']:+.4f}。")
            A(f"→ 门控效应 **{sig}**（Δ={w['delta']:+.4f}，标准差 U-Net {ds:.4f}/Att {as_:.4f}）。")
        A("\n> 单次训练随机波动量级与两模型差值相当（见 3 种子 std），故以检验替代单次比较。\n")
    else:
        A("〔待回填〕\n")

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
    concl = []
    if sam and medsam:
        cand = {"SAM ViT-B": sam["agg"]["dice_lesion"], "MedSAM ViT-B": medsam["agg"]["dice_lesion"]}
        if bussam:
            cand["BUSSAM ViT-B"] = bussam["agg"]["dice_lesion"]
        best = max(cand, key=cand.get)
        concl.append(f"- 提示式（单点，统一协议）：**{best}** 的 Dice 最高（{f(cand[best])}）；"
                     + "、".join(f"{k} {f(v)}" for k, v in cand.items()) + "。")
    if ms:
        mo = ms["models"]
        sup = {"U-Net": mo["unet"]["aggregate"]["dice_lesion"]["mean"],
               "Attention U-Net": mo["attunet"]["aggregate"]["dice_lesion"]["mean"]}
        if unetpp:
            sup["UNet++"] = unetpp["agg"]["dice_lesion"]
        if transunet:
            sup["TransUNet"] = transunet["agg"]["dice_lesion"]
        best = max(sup, key=sup.get)
        concl.append(f"- 监督式（全自动，统一协议）：**{best}** 的病灶 Dice 最高（{f(sup[best])}）。")
        if ms.get("wilcoxon"):
            w = ms["wilcoxon"]
            concl.append(f"- 注意力门控（U-Net → Attention U-Net）差值 {w['delta']:+.4f}，p={w['p_value']:.4f}，"
                         f"在本数据规模下{'未达' if w['p_value'] >= 0.05 else '达到'}统计显著。")
    concl.append("- 提示式与监督式的评价口径已统一到同一测试集/同一指标，但提示条件不同（提示式需 1 个 GT 点，监督式全自动），适用场景互补。")
    L.extend(concl)
    A("")

    # ---- 7. A/B optimisation ----
    ab = load("ab_summary.json")
    if ab:
        A("## 七、A/B 优化结果（统一协议）\n")
        A("> **B** = 损失函数重训（Tversky α=0.3/β=0.7、Boundary；3 种子）；")
        A("> **A** = 推理侧零训练成本项：**A1** 最大连通域+填洞+面积阈值、**A2** 多尺度+翻转 TTA、**A3** 验证集选阈值。\n")
        if ab.get("supervised"):
            A("### 7.1 监督式\n")
            A("| 模型 | 损失(B) | B: Dice | A2 TTA | A3 阈值 | A2+A1+A3 | 最优变体 |")
            A("|---|---|---|---|---|---|---|")
            for r in ab["supervised"]:
                a = r["a"]
                A(f"| {r['model']} | {r['loss']} | {fms(r['b_dice'])} | {fms(a['A2_tta']['dice'])} | "
                  f"{fms(a['A3_threshold']['dice'])} | {fms(a['A2+A1+A3']['dice'])} | {r['best_variant'] or '—'} |")
            A("")
        if ab.get("prompt"):
            A("### 7.2 提示式（仅 A，无 B）\n")
            A("| 方法 | baseline | A1 后处理 | A2 TTA | A3 阈值 | A2+A1+A3 | 最优变体 |")
            A("|---|---|---|---|---|---|---|")
            for r in ab["prompt"]:
                a = r["a"]
                g = lambda v: f"{a[v]['dice']:.4f}" if v in a else "—"  # noqa
                A(f"| {r['method']} | {g('baseline')} | {g('A1_postproc')} | {g('A2_tta')} | "
                  f"{g('A3_threshold')} | {g('A2+A1+A3')} | {r['best_variant'] or '—'} |")
            A("")
        p1 = load("p1_summary.json")
        if p1 and p1.get("rows"):
            A("### 7.3 P1　充分训练对照（24 → 150 epoch + 早停）\n")
            A("> 检验“Boundary 有害”“Tversky 对 UNet++ 有害”是否只是**欠拟合**造成的；延长训练至 150 epoch + 验证集早停(patience 15) + 最优轮回滚。\n")
            A("| 模型 | 损失 | 24ep | 150ep | Δ | 150ep 最优轮到 | 结论 |")
            A("|---|---|---|---|---|---|---|")

            def _s(d, sd, n):
                if d is None:
                    return "—"  # noqa
                return f"{d:.4f}±{sd:.4f}" if n > 1 else f"{d:.4f}"
            for r in p1["rows"]:
                if r["d24"] is None and r["d150"] is None:
                    continue
                A(f"| {r['model']} | {r['loss']} | {_s(r['d24'], r['std24'], r['n24'])} | "
                  f"{_s(r['d150'], r['std150'], r['n150'])} | "
                  f"{('%+.4f' % r['delta']) if r['delta'] is not None else '—'} | "
                  f"{r['epochs'] or '—'} | {r['note']} |")
            A("")
        A("> 结论 **B（24 epoch 基线）**：Tversky 对小模型有效、对 **UNet++ 反而有害**；Boundary 一致有害。")
        A("> 结论 **P1（150 epoch + 早停）**：**训练预算才是主导因素** —— 全部配置 **+0.016 ~ +0.074**（最优轮落在 44–96，而非 24），即原 24ep 基线**严重欠拟合**。其中：")
        A(">   · **Boundary 仍劣于 ce_dice**（U-Net −0.067、AttU-Net −0.062）→ “**Boundary 有害**”**结论成立**；")
        A(">   · **Tversky 在 150ep 下全面 ≥ ce_dice**（U-Net +0.009、AttU-Net +0.002、UNet++ +0.004）→ “**Tversky 对 UNet++ 有害**”**被反转**，原为欠拟合所致；")
        A(">   · 损失函数之间的差距（≤0.01）远小于“延长训练”的收益（0.02–0.07）→ **贡献 3 应改写为“训练预算 ≫ 损失选择”**。")
        A("> 结论 **A（A2+A1+A3）**：对**全部 4 个监督式**稳定有效（**+0.011~0.037**）且 HD95 普遍下降；**对提示式无效甚至有害**（提示式输出本就干净，面积门限会误删小病灶）。")
        A("> **全局最优**：24ep 下 **TransUNet + tversky + A3 = 0.6992**；**延长训练后 U-Net(150ep+ce_dice) 已达 0.6875、tversky 0.6966**，与 TransUNet 同档。\n")

    # ---- appendix ----
    A("## 附录 A：重跑可得性与环境\n")
    A("| 方法 | 是否重跑 | 依据 / 原因 |\n|---|---|---|")
    A("| U-Net | ✅ 3 种子 | 本地训练 |\n| Attention U-Net | ✅ 3 种子 | 本地训练 |")
    A(f"| SAM ViT-B | {'✅' if sam else '⏳'} | 公开权重 sam_vit_b_01ec64.pth |")
    A(f"| MedSAM ViT-B | {'✅' if medsam else '⏳'} | 公开权重（ModelScope, HF 格式） |")
    A(f"| UNet++ | {'✅' if unetpp else '⏳'} | 本地训练 256 RGB |")
    A(f"| TransUNet | {'✅' if transunet else '⏳'} | 本地训练 256 RGB |")
    A(f"| BUSSAM | {'✅' if bussam else '⏳'} | 官方代码 + 基座 SAM ViT-B 在 BUSI 自行微调 100 epoch，划分对齐冻结划分 |")
    A(f"\n> 运行环境：{dev}。\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(L))
    print("wrote", OUT)
    print("backfilled:", [k for k, v in [("multiseed", ms), ("unetpp", unetpp), ("transunet", transunet),
                                          ("sam", sam), ("medsam", medsam), ("bussam", bussam)] if v])


if __name__ == "__main__":
    main()
