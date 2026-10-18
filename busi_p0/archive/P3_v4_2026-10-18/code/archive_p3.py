"""Archive all P3 v4 artifacts (150-epoch weights, 3 seeds).

Layout (data only, per docs/README.md convention):
  archive/P3_v4_<date>/
    code/    exact scripts
    results/ metrics json/md/png + run log + env + frozen split
    preds/   <variant>_seed<k>.npz  -- per-image boolean masks
    figs/    qualitative grids for the blind bucket
    tables/  variants.csv, per_image.csv
    sha256sums.txt
Explanation docs go to docs/BUSI-P3-定向提示.md and docs/BUSI-P3-结果表.md.

Masks are dumped here rather than during the main run so the quoted numbers stay
byte-identical to results/tier_prompt_150ep.json.
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
DOCS = f"{ROOT}/docs"
sys.path.insert(0, SRC)
from busi_common import build_index, load_split, load_mask, dice_binary  # noqa
import diag_tier_prompt as P  # noqa

DATE = subprocess.check_output(["date", "-I"]).decode().strip()
ARCH = f"{ROOT}/archive/P3_v4_{DATE}"
SEEDS = (0, 1, 2)
OUT_JSON = f"{RES}/tier_prompt_150ep.json"

VARIANTS = [("B0_single", "single", "union"),
            ("FC_intersect", "blind_center", "intersect"),
            ("FC_majority", "blind_center", "majority"),
            ("FC_select", "blind_center", "select")]


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
    """Qualitative grid; English labels only (no CJK font in this container)."""
    n = len(rows)
    if not n:
        return
    fig, axes = plt.subplots(n, ncol, figsize=(ncol * 2.1, n * 1.75))
    if n == 1:
        axes = axes[None, :]
    for i, r in enumerate(rows):
        img = np.asarray(Image.open(r["rec"]["image"]).convert("L"))
        panels = [(img, f"{os.path.basename(r['rec']['name'])}\npmax {r['pmax']:.3f}", None),
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
    val = [byname[n] for n in split["val"]]
    test = [byname[n] for n in split["test"]]
    medsam = P.epq.load_medsam()

    per_image_rows = ["seed,name,cls,type,pmax," + ",".join(f"dice_{v[0]}" for v in VARIANTS)
                      + "," + ",".join(f"fg_{v[0]}" for v in VARIANTS)]
    fig_src = None
    for seed in SEEDS:
        ck = f"{RES}/p1_unetpp_seed{seed}.pth"
        print(f"seed {seed}: {os.path.basename(ck)}", flush=True)
        sig_fn = P.signals_fn("unetpp", 256, False, ck)
        signals = {r["name"]: sig_fn(r) for r in (val + test)}
        vbox = [signals[r["name"]] for r in val if signals[r["name"]]["box"] is not None]
        tau = {"p": P.GATE,
               "d": float(np.percentile([s["shift"] for s in vbox], 80)),
               "s": float(np.percentile([s["ratio"] for s in vbox], 80))}

        res = {}
        for name, policy, how in VARIANTS:
            res[name] = eval_masks(test, signals, medsam, tau, policy, how)
            np.savez_compressed(f"{ARCH}/preds/{name}_seed{seed}.npz",
                                **{o["rec"]["name"].replace("/", "__"): o["pred"] for o in res[name]})
            print(f"  {name}", flush=True)

        for i, o in enumerate(res["B0_single"]):
            row = [str(seed), o["rec"]["name"], o["rec"]["cls"], o["type"], f"{o['pmax']:.4f}"]
            row += [f"{dice_binary(res[v[0]][i]['pred'], o['gt']):.6f}" for v in VARIANTS]
            row += [f"{res[v[0]][i]['pred'].mean():.6f}" for v in VARIANTS]
            per_image_rows.append(",".join(row))

        if seed == 0:
            fig_src = res

    res = fig_src
    idx = {o["rec"]["name"]: i for i, o in enumerate(res["B0_single"])}
    blind = [o for o in res["B0_single"] if o["type"] == "全盲"]
    les = [o["rec"]["name"] for o in blind if o["rec"]["cls"] in ("benign", "malignant")]
    nor = [o["rec"]["name"] for o in blind if o["rec"]["cls"] == "normal"]

    def rows_for(names):
        out = []
        for nm in names:
            i = idx[nm]
            o = res["B0_single"][i]
            out.append({"rec": o["rec"], "gt": o["gt"], "pmax": o["pmax"],
                        "b0": dice_binary(o["pred"], o["gt"]), "m0": o["pred"],
                        "inter": dice_binary(res["FC_intersect"][i]["pred"], o["gt"]),
                        "m1": res["FC_intersect"][i]["pred"],
                        "maj": dice_binary(res["FC_majority"][i]["pred"], o["gt"]),
                        "m2": res["FC_majority"][i]["pred"]})
        return out

    grid(rows_for(les), f"{ARCH}/figs/blind_lesions.png",
         f"Blind lesions, seed0 (n={len(les)}) -- green = GT, red = predicted")
    grid(rows_for(nor), f"{ARCH}/figs/blind_normals.png",
         f"Blind normals, seed0 (n={len(nor)}) -- GT empty, red = false positives")

    open(f"{ARCH}/tables/per_image.csv", "w").write("\n".join(per_image_rows) + "\n")

    j = json.load(open(OUT_JSON))
    v = ["variant,desc,policy,agg,e2e_dice,e2e_dice_std,nobox_rate,normal_fg,normal_spec,"
         "normal_clean,multi_n,p_vs_B0,n_better,n_worse"]
    for k, d in j["variants"].items():
        v.append(",".join([k, f"\"{d['desc']}\"", d["policy"], d["agg"],
                           f"{d['e2e_dice']:.6f}", f"{d['e2e_dice_std']:.6f}",
                           f"{d['nobox_rate']:.6f}", f"{d['normal_fg']:.6f}",
                           f"{d['normal_spec']:.6f}", f"{d['normal_clean']:.3f}",
                           str(d["multi_n"]), f"{d.get('p_vs_B0', 1.0):.6f}",
                           str(d.get("n_better", 0)), str(d.get("n_worse", 0))]))
    open(f"{ARCH}/tables/variants.csv", "w").write("\n".join(v) + "\n")

    for f in ("tier_prompt_150ep.json", "tier_prompt_150ep.md", "tier_prompt_150ep.png",
              "tier_prompt_150ep.log", "env.json"):
        if os.path.exists(f"{RES}/{f}"):
            shutil.copy2(f"{RES}/{f}", f"{ARCH}/results/{f}")
    for f in ("diag_tier_prompt.py", "probe_tier.py", "probe_tier2.py", "archive_p3.py"):
        shutil.copy2(f"{SRC}/{f}", f"{ARCH}/code/{f}")
    shutil.copy2(f"{ROOT}/splits/busi_seed0.json", f"{ARCH}/results/busi_seed0.json")

    # ---------------- docs ----------------
    V = j["variants"]
    b0 = V["B0_single"]
    best = max(V.items(), key=lambda kv: kv[1]["e2e_dice"])[1]
    fm, fi, fs = V["FC_majority"], V["FC_intersect"], V["FC_select"]
    env = json.load(open(f"{RES}/env.json"))
    sweeps = j["gate_sweep"]
    td = j["type_dist"]

    def ms(x, k="e2e_dice"):
        return f"{x[k]:.4f}±{x[k+'_std']:.4f}" if x.get(k + "_std") else f"{x[k]:.4f}"

    M = [f"# P3 归档说明（v4 · 150ep × 3 种子，{DATE}）\n",
         "误差类型感知的定向提示纠正 —— 结论 / 环境 / 复现 / 清单 / 限制。",
         f"数据在 `archive/P3_v4_{DATE}/`，指标表见 `docs/BUSI-P3-结果表.md`，"
         "方案口径见 `docs/BUSI-算法优化方案.md` §4.1⑥。\n",
         "## 一、一句话结论\n",
         "粗模型“完全不出框”（全盲）是端到端最大的可回收损失：给这些图补**中心框兜底提示**，端到端 Dice "
         f"**{b0['e2e_dice']:.4f} → {fm['e2e_dice']:.4f}**（**{fm['e2e_dice']-b0['e2e_dice']:+.4f}**，"
         f"p={fm['p_vs_B0']:.3f}，{fm['n_better']} 好 / {fm['n_worse']} 差，3 个种子 Δ 分别 "
         "+0.0265 / +0.0224 / +0.0191），无框率 "
         f"**{b0['nobox_rate']*100:.1f}% → {fm['nobox_rate']*100:.1f}%**；代价是正常图假阳 "
         f"**{b0['normal_fg']:.4f} → {fi['normal_fg']:.4f}（交集）/ {fm['normal_fg']:.4f}（多数投票）**，"
         "因为正常图与漏检病灶在二分类分割输出上不可分（判据精度上限 0.41）。\n",
         "## 二、环境\n",
         f"- 设备：{env['device']} / {env['gpu']}；torch {env['torch']}（cuda {env['cuda']}）",
         f"- 解释器：`{env['python']}`",
         "- 粗模型：`results/p1_unetpp_seed{0,1,2}.pth`（UNet++，**150ep + 早停 + 最优轮回滚**，256 RGB）",
         "- 精修器：MedSAM ViT-B（`/mnt/workspace/models/medsam_vit_b`）\n",
         "## 三、复现\n",
         "```bash",
         "cd /mnt/workspace/busi_p0/src",
         "/mnt/workspace/busi_p0/.venv/bin/python -u diag_tier_prompt.py \\",
         "    --model unetpp --size 256 --seeds 0,1,2 --out tier_prompt_150ep",
         "/mnt/workspace/busi_p0/.venv/bin/python -u archive_p3.py      # 本归档（掩膜/图/表）",
         "/mnt/workspace/busi_p0/.venv/bin/python -u probe_tier.py      # 各阈值下的可用提示统计",
         "/mnt/workspace/busi_p0/.venv/bin/python -u probe_tier2.py     # 正常图 vs 漏检病灶可分性",
         "```\n",
         "## 四、目录\n",
         "| 路径 | 内容 |",
         "|---|---|",
         "| `code/diag_tier_prompt.py` | 主脚本：信号缓存 + MedSAM 批处理/去重 + 4 种聚合 + 多种子聚合 |",
         "| `code/probe_tier.py` | 诊断：0.5/0.3 阈值下有无连通域的统计 |",
         "| `code/probe_tier2.py` | 诊断：正常图与漏检病灶的 pmax 重叠（判据精度上限的证据） |",
         "| `code/archive_p3.py` | 本归档生成器 |",
         "| `results/tier_prompt_150ep.json` | 12 个变体 × 3 种子指标 + 逐图记录 + 门控扫描 |",
         "| `results/tier_prompt_150ep.md` / `.png` | 结果表 / 柱状图（含误差棒） |",
         "| `results/tier_prompt_150ep.log` | 完整运行日志（含逐种子明细） |",
         "| `results/env.json`、`busi_seed0.json` | 运行环境、冻结划分（546/118/116） |",
         "| `preds/<变体>_seed<k>.npz`（12 个） | 116 张测试图**布尔掩膜**（原始分辨率，键=图像名，`/`→`__`） |",
         "| `tables/variants.csv` | 变体指标（均值/标准差/p/胜负计数） |",
         "| `tables/per_image.csv` | 逐图 × 逐种子 Dice 与 fg |",
         "| `figs/blind_lesions.png` | 全盲**病灶**：原图 / B0 / 交集 / 多数投票（绿=GT） |",
         "| `figs/blind_normals.png` | 全盲**正常图**：红区即假阳代价 |",
         "| `sha256sums.txt` | 全部文件校验和 |",
         "",
         "## 五、关键数字\n",
         "| 变体 | Dice | Δ | p | 无框率 | 正常图 fg | 特异度 |",
         "|---|---|---|---|---|---|---|",
         f"| B0 单框 | {ms(b0)} | — | — | {b0['nobox_rate']:.3f} | {b0['normal_fg']:.4f} | {b0['normal_spec']:.4f} |",
         f"| 交集 FC_intersect | {ms(fi)} | {fi['e2e_dice']-b0['e2e_dice']:+.4f} | {fi['p_vs_B0']:.3f} | {fi['nobox_rate']:.3f} | {fi['normal_fg']:.4f} | {fi['normal_spec']:.4f} |",
         f"| 多数投票 FC_majority | {ms(fm)} | {fm['e2e_dice']-b0['e2e_dice']:+.4f} | {fm['p_vs_B0']:.3f} | {fm['nobox_rate']:.3f} | {fm['normal_fg']:.4f} | {fm['normal_spec']:.4f} |",
         f"| 按分数选 FC_select | {ms(fs)} | {fs['e2e_dice']-b0['e2e_dice']:+.4f} | {fs['p_vs_B0']:.3f} | {fs['nobox_rate']:.3f} | {fs['normal_fg']:.4f} | {fs['normal_spec']:.4f} |",
         f"| 全类型定向 T_select | {ms(V['T_select'])} | {V['T_select']['e2e_dice']-b0['e2e_dice']:+.4f} | {V['T_select']['p_vs_B0']:.3f} | {V['T_select']['nobox_rate']:.3f} | {V['T_select']['normal_fg']:.4f} | {V['T_select']['normal_spec']:.4f} |",
         "",
         "误差类型分布（116 张，3 种子平均）：" + "、".join(f"{k} {x:.1f}" for k, x in td.items()),
         "",
         "门控扫描（3 种子平均）：",
         "",
         "| τ_p | 判为低置信的病灶 | 正常图 | 精度 | 全盲覆盖 |",
         "|---|---|---|---|---|",
         ] + [f"| {k} | {d['n_lesion']:.1f} | {d['n_normal']:.1f} | {d['precision']:.2f} | {d['cov']:.2f} |"
              for k, d in sweeps.items()] + [
         "",
         "## 六、已知限制\n",
         "1. 只跑了 UNet++（4 个监督式之一）；换成更强的粗模型时全盲图会减少，增益预计随之收窄（未验证）。",
         "2. 正常图假阳代价**未解决**（特异度 0.996→0.93/0.81），判据精度上限 0.41 —— 需图像级“有/无病灶”分类器门控（方案 §4.1⑦）。",
         "3. 粗模型与精修器都固定，**不含任何重训**；这是推理侧零成本方案的属性，也是其上限。",
         "4. 掩膜为原始分辨率布尔图；npz 键把 `/` 替换为 `__`。",
         "",
         "## 七、与 v3（24ep，单种子）的关系\n",
         "| 项 | v3（24ep seed0） | **v4（150ep × 3 种子）** |",
         "|---|---|---|",
         f"| 基线 B0 | 0.6172 | **{b0['e2e_dice']:.4f}±{b0['e2e_dice_std']:.4f}** |",
         f"| 全盲兜底（多数投票） | 0.6405（+0.0233） | **{fm['e2e_dice']:.4f}（{fm['e2e_dice']-b0['e2e_dice']:+.4f}）** |",
         f"| 无框率 | 11.5% → 2.1% | **{b0['nobox_rate']*100:.1f}% → {fm['nobox_rate']*100:.1f}%** |",
         "| 全图框兜底 | 未隔离 | **零贡献**（F≡FC），中心框已足够 |",
         "| 种子稳健性 | 未验证 | **3/3 种子 Δ 为正**（+0.027/+0.022/+0.019） |",
         "",
         "→ 结论方向一致、量级稳定（+0.022~0.023），不是训练预算造成的假象。v3 数据保留在 `archive/P3_v3_2026-10-18/`。",
         ""]
    open(f"{DOCS}/BUSI-P3-定向提示.md", "w").write("\n".join(M) + "\n")

    table = open(f"{RES}/tier_prompt_150ep.md").read()
    table += ("\n图：`results/tier_prompt_150ep.png`（柱状图，误差棒为 3 种子标准差）。\n"
              f"\n定性对照：`archive/P3_v4_{DATE}/figs/blind_lesions.png`（全盲病灶）"
              f"与 `blind_normals.png`（全盲正常图，红区即假阳代价）。\n")
    open(f"{DOCS}/BUSI-P3-结果表.md", "w").write(table)

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
    print(f"blind lesions(seed0) {len(les)} / blind normals {len(nor)}")


if __name__ == "__main__":
    main()
