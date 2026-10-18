"""Archive every P3 artifact into a self-contained, date-stamped folder.

Writes archive/P3_v3_<date>/ with:
  code/      the exact scripts that produced the numbers
  results/   tier_prompt.json / .png + run log + env + frozen split
  preds/     per-image boolean masks (npz) for B0, F_intersect, F_majority
  figs/      qualitative grids for the 25 "blind" images
  tables/    variants.csv, per_image.csv
  sha256sums.txt

Documents go to docs/ instead of the archive:
  docs/BUSI-P3-定向提示.md (manifest), docs/BUSI-P3-结果表.md (report table)

The masks are dumped here (not during the main run) so the archived results
stay byte-identical to what the report quotes.
"""
import os
import sys
import json
import shutil
import hashlib
import subprocess

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/mnt/workspace/busi_p0/src"
ROOT = "/mnt/workspace/busi_p0"
RES = f"{ROOT}/results"
sys.path.insert(0, SRC)
from busi_common import build_index, load_split, load_mask, dice_binary  # noqa
import diag_tier_prompt as P  # noqa

DATE = subprocess.check_output(["date", "-I"]).decode().strip()
ARCH = f"{ROOT}/archive/P3_v3_{DATE}"
VARIANTS = [("B0_single", "single", "union"), ("F_intersect", "blind", "intersect"),
            ("F_majority", "blind", "majority"), ("FR_select", "blind+rescue", "select")]


def eval_masks(recs, signals, medsam, tau, policy, how):
    model, proc = medsam
    out = []
    for rec in recs:
        sig = signals[rec["name"]]
        gt = load_mask(rec)
        cs = P.candidates(sig, tau, policy)
        if not cs:
            pred = np.zeros(gt.shape, dtype=bool)
        else:
            img = Image.open(rec["image"]).convert("RGB")
            r = P.medsam_multi(model, proc, img, cs, rec["name"])
            pred = P.combine([x[0] for x in r], [x[1] for x in r], how, gt.shape)
        out.append({"rec": rec, "gt": gt, "pred": pred, "ncand": len(cs),
                    "type": P.diagnose(sig, tau), "pmax": sig["pmax"]})
    return out


def grid(rows, path, title, ncol=4):
    """Qualitative grid. English labels only: no CJK font in this container."""
    n = len(rows)
    if not n:
        return
    fig, axes = plt.subplots(n, ncol, figsize=(ncol * 2.1, n * 1.75))
    if n == 1:
        axes = axes[None, :]
    for i, r in enumerate(rows):
        img = np.asarray(Image.open(r["rec"]["image"]).convert("L"))
        base = os.path.basename(r["rec"]["name"])
        panels = [(img, f"{base}\npmax {r['pmax']:.3f}", None),
                  (img, f"B0  D={r['b0']:.3f}", r["m0"]),
                  (img, f"intersect  D={r['inter']:.3f}", r["m1"]),
                  (img, f"majority  D={r['maj']:.3f}", r["m2"])]
        for j, (im, ttl, m) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(im, cmap="gray")
            if m is not None and m.any():
                ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.45)
            if m is not None:
                ax.contour(r["gt"].astype(float), levels=[0.5], colors="lime", linewidths=0.7)
            ax.set_title(ttl, fontsize=6)
            ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=125)
    plt.close(fig)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    for d in ("code", "results", "preds", "figs", "tables"):
        os.makedirs(f"{ARCH}/{d}", exist_ok=True)

    index = build_index(); byname = {it["name"]: it for it in index}
    split = load_split()
    test = [byname[n] for n in split["test"]]
    val = [byname[n] for n in split["val"]]

    ck = f"{RES}/viz_unetpp_seed0.pth"
    sig_fn = P.signals_fn("unetpp", 256, False, ck)
    print("signals ...")
    signals = {r["name"]: sig_fn(r) for r in (val + test)}
    vbox = [signals[r["name"]] for r in val if signals[r["name"]]["box"] is not None]
    tau = {"p": 0.999,
           "d": float(np.percentile([s["shift"] for s in vbox], 80)),
           "s": float(np.percentile([s["ratio"] for s in vbox], 80))}

    medsam = P.epq.load_medsam()
    per_variant = {}
    for name, policy, how in VARIANTS:
        print("eval", name)
        per_variant[name] = eval_masks(test, signals, medsam, tau, policy, how)
        np.savez_compressed(f"{ARCH}/preds/{name}.npz",
                            **{o["rec"]["name"].replace("/", "__"): o["pred"]
                               for o in per_variant[name]})

    # per-image table
    lines = ["name,cls,type,pmax,dice_B0,dice_F_intersect,dice_F_majority,dice_FR_select,"
             "fg_B0,fg_F_intersect,fg_F_majority,fg_FR_select,ncand_F_intersect"]
    for i, o in enumerate(per_variant["B0_single"]):
        nm = o["rec"]["name"]
        row = [nm, o["rec"]["cls"], o["type"], f"{o['pmax']:.4f}"]
        for v, _, _ in VARIANTS:
            oo = per_variant[v][i]
            row.append(f"{dice_binary(oo['pred'], oo['gt']):.6f}")
        for v, _, _ in VARIANTS:
            row.append(f"{per_variant[v][i]['pred'].mean():.6f}")
        row.append(str(o["ncand"]))
        lines.append(",".join(row))
    open(f"{ARCH}/tables/per_image.csv", "w").write("\n".join(lines) + "\n")

    # variants table
    j = json.load(open(f"{RES}/tier_prompt.json"))
    v = ["variant,desc,policy,agg,gate,e2e_dice,nobox_rate,normal_fg,normal_spec,"
         "normal_clean,multi_n,p_vs_B0,n_better,n_worse"]
    for k, d in j["variants"].items():
        v.append(",".join([k, f"\"{d['desc']}\"", d["policy"], d["agg"], f"{d['gate']:.3f}",
                           f"{d['e2e_dice']:.6f}", f"{d['nobox_rate']:.6f}", f"{d['normal_fg']:.6f}",
                           f"{d['normal_spec']:.6f}", f"{d['normal_clean']:.3f}", str(d["multi_n"]),
                           f"{d.get('p_vs_B0', 1.0):.6f}", str(d.get("n_better", 0)),
                           str(d.get("n_worse", 0))]))
    open(f"{ARCH}/tables/variants.csv", "w").write("\n".join(v) + "\n")

    # qualitative grids for the blind bucket
    blind = [o for o in per_variant["B0_single"] if o["type"] == "全盲"]
    les = [o["rec"]["name"] for o in blind if o["rec"]["cls"] in ("benign", "malignant")]
    nor = [o["rec"]["name"] for o in blind if o["rec"]["cls"] == "normal"]

    def rows_for(names):
        idx = {o["rec"]["name"]: i for i, o in enumerate(per_variant["B0_single"])}
        out = []
        for nm in names:
            i = idx[nm]
            o = per_variant["B0_single"][i]
            out.append({"rec": o["rec"], "gt": o["gt"], "pmax": o["pmax"],
                        "b0": dice_binary(o["pred"], o["gt"]), "m0": o["pred"],
                        "inter": dice_binary(per_variant["F_intersect"][i]["pred"], o["gt"]),
                        "m1": per_variant["F_intersect"][i]["pred"],
                        "maj": dice_binary(per_variant["F_majority"][i]["pred"], o["gt"]),
                        "m2": per_variant["F_majority"][i]["pred"]})
        return out

    grid(rows_for(les), f"{ARCH}/figs/blind_lesions.png",
         f"Blind lesions (n={len(les)}) -- green = GT, red = predicted mask")
    grid(rows_for(nor), f"{ARCH}/figs/blind_normals.png",
         f"Blind normals (n={len(nor)}) -- GT empty, all red pixels are false positives")

    # copies
    for f in ("tier_prompt.json", "tier_prompt.png",
              "tier_prompt_v3.log", "env.json"):
        if os.path.exists(f"{RES}/{f}"):
            shutil.copy2(f"{RES}/{f}", f"{ARCH}/results/{f}")
    if os.path.exists(f"{RES}/tier_prompt.md"):
        shutil.copy2(f"{RES}/tier_prompt.md", f"{ROOT}/docs/BUSI-P3-结果表.md")
    for f in ("diag_tier_prompt.py", "probe_tier.py", "probe_tier2.py", "archive_p3.py"):
        shutil.copy2(f"{SRC}/{f}", f"{ARCH}/code/{f}")
    shutil.copy2(f"{ROOT}/splits/busi_seed0.json", f"{ARCH}/results/busi_seed0.json")

    # manifest
    b0 = j["variants"]["B0_single"]; fm = j["variants"]["F_majority"]
    fi = j["variants"]["F_intersect"]; fr = j["variants"]["FR_select"]
    env = json.load(open(f"{RES}/env.json"))
    M = [f"# P3 归档（{DATE}）\n",
         "误差类型感知的定向提示纠正 —— 全部数据留档。",
         "归档由 `src/archive_p3.py` 生成，数值与 `docs/BUSI-算法优化方案.md` §4.1⑥ 完全一致。\n",
         "## 一、一句话结论\n",
         f"粗模型“无框”（全盲）是端到端最大的可回收损失：兜底提示把 Dice 从 **{b0['e2e_dice']:.4f}** 抬到 "
         f"**{fm['e2e_dice']:.4f}**（+{fm['e2e_dice']-b0['e2e_dice']:.4f}，p={fm['p_vs_B0']:.3f}，"
         f"{fm['n_better']} 好 / {fm['n_worse']} 差），无框率 {b0['nobox_rate']*100:.1f}%→{fm['nobox_rate']*100:.1f}%；"
         f"代价是正常图假阳 {b0['normal_fg']:.4f}→{fi['normal_fg']:.4f}（交集）~{fm['normal_fg']:.4f}（多数投票），"
         "因正常图与漏检病灶在粗模型输出上不可分（判据精度上限 0.39）。\n",
         "## 二、环境\n",
         f"- 设备：{env['device']} / {env['gpu']}；torch {env['torch']}（cuda {env['cuda']}）",
         f"- 解释器：`{env['python']}`",
         "- 粗模型：`results/viz_unetpp_seed0.pth`（UNet++，24ep，seed0，256 RGB）",
         "- 精修器：MedSAM ViT-B（`/mnt/workspace/models/medsam_vit_b`）\n",
         "## 三、复现\n",
         "```bash",
         "cd /mnt/workspace/busi_p0/src",
         "/mnt/workspace/busi_p0/.venv/bin/python -u diag_tier_prompt.py   # 消融 -> results/tier_prompt.*",
         "/mnt/workspace/busi_p0/.venv/bin/python -u archive_p3.py          # 本归档（含掩膜/图）",
         "/mnt/workspace/busi_p0/.venv/bin/python -u probe_tier.py          # 证据：各阈值下可用提示统计",
         "/mnt/workspace/busi_p0/.venv/bin/python -u probe_tier2.py         # 证据：正常图 vs 漏检可分性",
         "```\n",
         f"> 阈值：τ_d={tau['d']:.4f}、τ_s={tau['s']:.4f}（验证集 80 分位）；门控无阈值（argmax 从未判为病灶）。\n",
         "## 四、目录\n",
         "| 路径 | 内容 |",
         "|---|---|",
         "| `code/diag_tier_prompt.py` | v3 主脚本（信号缓存 + MedSAM 批处理/去重 + 4 种聚合） |",
         "| `code/probe_tier.py` | 诊断：0.5/0.3 阈值下有无连通域的统计 |",
         "| `code/probe_tier2.py` | 诊断：正常图与漏检病灶的 pmax 重叠（判据精度上限的证据） |",
         "| `code/archive_p3.py` | 本归档生成器 |",
         "| `results/tier_prompt.json` | 11 个变体的全部指标 + 逐图记录 + 门控扫描 |",
         "| `results/tier_prompt.png` | 柱状图（报告表见 `docs/BUSI-P3-结果表.md`） |",
         "| `results/tier_prompt_v3.log` | 完整运行日志 |",
         "| `results/env.json` | 运行环境 |",
         "| `results/busi_seed0.json` | 冻结划分（546/118/116） |",
         "| `preds/B0_single.npz` 等 4 个 | 116 张测试图的**布尔预测掩膜**（原始分辨率，键=图像名） |",
         "| `tables/variants.csv` | 变体指标表 |",
         "| `tables/per_image.csv` | 逐图 Dice / fg / 类型 / 候选数 |",
         "| `figs/blind_lesions.png` | 9 张全盲**病灶**：原图 / B0 / 交集 / 多数投票 |",
         "| `figs/blind_normals.png` | 16 张全盲**正常图**：同上（假阳代价） |",
         "| `sha256sums.txt` | 归档内全部文件的校验和 |",
         "",
         "## 五、关键数字\n",
         "| 变体 | Dice | Δ | p | 无框率 | 正常图 fg | 特异度 |",
         "|---|---|---|---|---|---|---|",
         f"| B0 单框 | {b0['e2e_dice']:.4f} | — | — | {b0['nobox_rate']:.3f} | {b0['normal_fg']:.4f} | {b0['normal_spec']:.4f} |",
         f"| 交集 F_intersect | {fi['e2e_dice']:.4f} | {fi['e2e_dice']-b0['e2e_dice']:+.4f} | {fi['p_vs_B0']:.3f} | {fi['nobox_rate']:.3f} | {fi['normal_fg']:.4f} | {fi['normal_spec']:.4f} |",
         f"| 多数投票 F_majority | {fm['e2e_dice']:.4f} | {fm['e2e_dice']-b0['e2e_dice']:+.4f} | {fm['p_vs_B0']:.3f} | {fm['nobox_rate']:.3f} | {fm['normal_fg']:.4f} | {fm['normal_spec']:.4f} |",
         f"| 兜底+救回 FR_select | {fr['e2e_dice']:.4f} | {fr['e2e_dice']-b0['e2e_dice']:+.4f} | {fr['p_vs_B0']:.3f} | {fr['nobox_rate']:.3f} | {fr['normal_fg']:.4f} | {fr['normal_spec']:.4f} |",
         "",
         f"误差类型分布（116 张）：" + "、".join(f"{k} {v}" for k, v in j["type_dist"].items()),
         "",
         "## 六、已知限制\n",
         "1. 粗模型仅 seed0、且为 24ep 欠拟合权重 → 增益的种子间稳健性**未验证**（+0.0233 ≈ 3 种子 std 0.02）。",
         "2. 正常图假阳代价**未解决**（特异度 0.9996→0.90），成功判据“正常图不劣化”未通过 → 只作归因证据。",
         "3. 掩膜为原始分辨率布尔图；npz 键把 `/` 替换为 `__`。",
         "4. 无随机性，重跑应逐位复现（MedSAM 推理确定性）。",
         ""]
    open(f"{ROOT}/docs/BUSI-P3-定向提示.md", "w").write("\n".join(M) + "\n")

    # checksums
    chk = []
    for root, _, files in sorted(os.walk(ARCH)):
        for f in sorted(files):
            if f == "sha256sums.txt":
                continue
            p = os.path.join(root, f)
            chk.append(f"{sha(p)}  {os.path.relpath(p, ARCH)}")
    open(f"{ARCH}/sha256sums.txt", "w").write("\n".join(chk) + "\n")

    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(ARCH) for f in fs)
    print(f"\narchived -> {ARCH}")
    print(f"files: {len(chk)}   size: {tot/1e6:.2f} MB")
    print(f"blind lesions {len(les)} / blind normals {len(nor)}")


if __name__ == "__main__":
    main()
