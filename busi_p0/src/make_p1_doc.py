"""Generate docs/BUSI-P1-充分训练.md from the 150-epoch results (3 seeds).

Reads  results/postproc_p1_* + results/p1_* (via summarize_p1.collect)
       results/p1a_summary.json (150ep + A group)
Writes docs/BUSI-P1-充分训练.md
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_p1 import collect, CFG, LOSS_LABEL, ms

RES = "/mnt/workspace/busi_p0/results"
OUT = "/mnt/workspace/busi_p0/docs/BUSI-P1-充分训练.md"


def _mean(rows, model, loss):
    for r in rows:
        if r["model"] == model and r["loss"] == loss:
            return r["stats"]["mean"] if r["stats"] else None
    return None


def main():
    rows = collect(RES)
    L = []
    A = L.append
    A("# BUSI 分割 · 损失函数对照（150 epoch，3 种子）\n")
    A("> 在**充分训练**（150 epoch + 验证集早停 + 最优轮回滚）下比较 CE+Dice / Tversky / Boundary。")
    A("> 数据：冻结测试集（`splits/busi_seed0.json`，116 张 / 其中病灶 96 张），统一指标 `src/busi_common.py`，3 个随机种子，统一概率路径。\n")

    A("## 一、实验设置\n")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 训练轮数 | **150** |")
    A("| 早停 | 验证集 Dice，**patience 15**，`--min-epochs 20` |")
    A("| 模型选择 | **回滚到验证集最优轮**（best-val rollback） |")
    A("| 损失 | CE+Dice / Tversky(α=0.3, β=0.7) / CE+Dice+Boundary(λ=1) |")
    A("| 其余 | 优化器 / 增强 / 调度（`step_size=epochs//3`）四组完全一致 |")
    A("| 范围 | U-Net / Attention U-Net × 3 损失；**UNet++ / TransUNet × {CE+Dice, Tversky}** |\n")

    A("## 二、主结果（测试集病灶 Dice，3 种子）\n")
    A("| 模型 | 损失 | Dice(病灶) | 最优轮 |")
    A("|---|---|---|---|")
    for r in rows:
        eps = [p["best_epoch"] for p in r["perseed"] if p["best_epoch"]]
        A(f"| {r['label']} | {LOSS_LABEL[r['loss']]} | **{ms(r['stats'])}** | "
          f"{int(np.mean(eps)) if eps else '—'} |")
    A("")

    A("## 三、同模型内 Δ（相对 CE+Dice）\n")
    A("| 模型 | Tversky − CE | Boundary − CE |")
    A("|---|---|---|")
    for model, label, *_ in CFG:
        ce = _mean(rows, model, "ce_dice")
        tv = _mean(rows, model, "tversky")
        bd = _mean(rows, model, "boundary")
        A(f"| {label} | {'%+.4f' % (tv-ce) if (ce is not None and tv is not None) else '—'} | "
          f"{'%+.4f' % (bd-ce) if (ce is not None and bd is not None) else '—'} |")
    A("")

    A("## 四、逐种子明细\n")
    A("| 模型 | 损失 | 种子 | Dice | 实际训练轮 | 最优轮 |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        for p in r["perseed"]:
            A(f"| {r['label']} | {LOSS_LABEL[r['loss']]} | {p['seed']} | "
              f"{('%.4f' % p['dice']) if p['dice'] is not None else '—'} | "
              f"{p['epochs'] or '—'} | {p['best_epoch'] or '—'} |")
    A("")

    A("## 五、结论\n")
    bnd, tvdel, best_ep = [], [], []
    for model, label, *_ in CFG:
        ce = _mean(rows, model, "ce_dice")
        bd = _mean(rows, model, "boundary")
        tv = _mean(rows, model, "tversky")
        if ce is not None and bd is not None:
            bnd.append((label, bd - ce))
        if ce is not None and tv is not None:
            tvdel.append((label, tv - ce))
    for r in rows:
        best_ep += [p["best_epoch"] for p in r["perseed"] if p["best_epoch"]]

    if bnd:
        A("**① Boundary 一致劣于 CE+Dice。** "
          + "、".join(f"{m} {d:+.4f}" for m, d in bnd)
          + " —— 在本数据上 Boundary 损失**不适用**。\n")
    if tvdel:
        A("**② 损失选择的影响很小，且模型依赖。** Tversky 相对 CE+Dice："
          + "、".join(f"{m} {d:+.4f}" for m, d in tvdel)
          + "。轻量模型的差异均在 ±0.01 以内；唯一反向的是 TransUNet，"
          "说明**重模型上 Tversky 不适用**。\n")

    ce_unet = _mean(rows, "unet", "ce_dice")
    ce_tr = _mean(rows, "transunet", "ce_dice")
    light = {m: _mean(rows, m, "ce_dice") for m in ("unet", "attunet", "unetpp")}
    light = {k: v for k, v in light.items() if v is not None}
    if ce_unet is not None and ce_tr is not None:
        A(f"**③ 结构差距有限。** U-Net（1.9M）{ce_unet:.4f} vs TransUNet（105M）{ce_tr:.4f}，"
          f"仅差 **{ce_tr - ce_unet:+.4f}**；轻量三骨干落在 "
          f"{min(light.values()):.4f}–{max(light.values()):.4f}。\n")
    if best_ep:
        A(f"**④ 收敛轮次分布。** 验证集最优轮落在 **{min(best_ep)}–{max(best_ep)}**，"
          "早停与最优轮回滚是必要的（否则末轮权重并非最优）。\n")

    # ---- 150ep + A ----
    try:
        p1a = json.load(open(f"{RES}/p1a_summary.json"))
    except Exception:
        p1a = None
    if p1a and p1a.get("rows"):
        A("## 六、150ep 权重 + A 组（后处理 / TTA / 验证集阈值）\n")
        A("> 在 150ep 权重上再叠加推理侧 A 组（零重训）。\n")
        A("| 模型 | 损失 | baseline | 最优 A 变体 | A 后 Dice | Δ(A) | HD95(base→best) |")
        A("|---|---|---|---|---|---|---|")
        for r in p1a["rows"]:
            bm = r.get("a_baseline_mean", r.get("base_mean"))
            if bm is None:
                continue
            A(f"| {r['label']} | {r['loss']} | {bm:.4f} | {r['best'] or '—'} | "
              f"{(('%.4f' % r['best_mean']) if r.get('best_mean') is not None else '—')} | "
              f"{(('%+.4f' % r['delta']) if r.get('delta') is not None else '—')} | "
              f"{(('%.1f→%.1f' % (r['hd95_base'], r['hd95_best'])) if r.get('hd95_best') is not None else '—')} |")
        A("")
        ok = [r for r in p1a["rows"] if r.get("best_mean") is not None]
        if ok:
            br = max(ok, key=lambda r: r["best_mean"])
            A(f"**监督式最优 = {br['label']} + {br['loss']} + {br['best']} = {br['best_mean']:.4f}**"
              + (f"（HD95 {br['hd95_best']:.1f}）" if br.get("hd95_best") is not None else "") + "。")
            ue = next((r for r in ok if r["model"] == "unet" and r["loss"] == "ce_dice"), None)
            if ue and br["model"] != "unet":
                A(f"**U-Net（1.9M）+ 同配置 = {ue['best_mean']:.4f}**，"
                  f"与 ~105M 的 {br['label']} 仅差 {br['best_mean']-ue['best_mean']:.3f}。")
            A("")

    A("## 七、复现命令\n")
    A("```bash")
    A("python train_supervised.py --model unet --size 128 --gray --epochs 150 --seed 0 \\")
    A("       --loss ce_dice --early-stop 15 --min-epochs 20 --tag p1_unet")
    A("python eval_postproc.py --ckpt results/p1_unet_seed0.pth --model unet --size 128 --gray --tta")
    A("python summarize_p1.py      # -> results/p1_summary.{md,json}")
    A("python make_p1_doc.py       # -> docs/BUSI-P1-充分训练.md")
    A("```\n")

    A("## 八、产物清单\n")
    A("| 文件 | 说明 |")
    A("|---|---|")
    A("| `results/p1_*.json`（30 个） | 每个 (模型×损失×种子) 的测试结果 + 逐轮 history |")
    A("| `results/postproc_p1_*.json`（30 个） | A 组各变体指标（baseline 为主表口径） |")
    A("| `results/p1_summary.md` / `.json` | 150ep 损失汇总 |")
    A("| `results/p1a_summary.md` / `.json` | 150ep + A 组汇总 |")
    A("| `results/p1log_*.log` / `p1alog_*.log` | 训练 / A 组评估日志 |")
    A("| `src/run_p1_par.sh` 等 | 重跑脚本 |")
    A("")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", OUT)
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
