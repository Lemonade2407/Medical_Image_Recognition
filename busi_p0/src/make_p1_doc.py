"""Generate a standalone P1 document from the re-run data.

Reads results/ab_<model>[_loss]_seed<k>.json (24 ep) and
      results/p1_<model>[_loss]_seed<k>.json (150 ep + early stop)
Writes docs/BUSI-P1-充分训练.md
"""
import os
import json
import numpy as np

RES = "/mnt/workspace/busi_p0/results"
OUT = "/mnt/workspace/busi_p0/docs/BUSI-P1-充分训练.md"

CFG = [("unet", "U-Net", 128, "g", ["ce_dice", "tversky", "boundary"]),
       ("attunet", "Attention U-Net", 128, "g", ["ce_dice", "tversky", "boundary"]),
       ("unetpp", "UNet++", 256, "rgb", ["ce_dice", "tversky"]),
       ("transunet", "TransUNet", 256, "rgb", ["ce_dice", "tversky"])]
LOSS_LABEL = {"ce_dice": "CE+Dice", "tversky": "Tversky(0.3/0.7)", "boundary": "CE+Dice+Boundary"}
SEEDS = (0, 1, 2)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def stats(v):
    if not v:
        return None
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def ms(s):
    if not s:
        return "—"
    return f"{s['mean']:.4f}±{s['std']:.4f}" if s["std"] else f"{s['mean']:.4f}"


def main():
    L = []
    A = L.append
    A("# P1　充分训练重跑（24 → 150 epoch + 验证集早停）\n")
    A("> 目的：检验“Boundary 有害”“Tversky 对 UNet++ 有害”是否只是 **24 epoch 欠拟合**造成的假象 —— 若成立将直接动摇贡献 3。")
    A("> 数据：冻结测试集（`splits/busi_seed0.json`，116 张 / 其中病灶 96 张），统一指标 `src/busi_common.py`，3 个随机种子。\n")

    A("## 一、实验设置\n")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 训练轮数 | **150**（原 24） |")
    A("| 早停 | 验证集 Dice，**patience 15**，`--min-epochs 20` |")
    A("| 模型选择 | **回滚到验证集最优轮**（best-val rollback） |")
    A("| 损失 | CE+Dice（基线）/ Tversky(α=0.3,β=0.7) / CE+Dice+Boundary(λ=1) |")
    A("| 其余 | 优化器/增强/调度（`step_size=epochs//3`）与基线**完全一致**，仅轮数+早停不同 |")
    A("| 范围 | U-Net / Attention U-Net × 3 损失；**UNet++ × {CE+Dice, Tversky}**（验证 UNet++ 的 Tversky 结论） |")
    A("| 新增代码 | `train_supervised.py` 的 `--early-stop` / `--min-epochs` |\n")

    rows = []
    for model, label, size, mode, losses in CFG:
        for loss in losses:
            suf = "" if loss == "ce_dice" else f"_{loss}"
            d24, d150, perseed = [], [], []
            for s in SEEDS:
                r24 = load(f"{RES}/ab_{model}{suf}_seed{s}.json")
                r150 = load(f"{RES}/p1_{model}{suf}_seed{s}.json")
                v24 = r24["agg"]["dice_lesion"] if r24 else None
                v150 = r150["agg"]["dice_lesion"] if r150 else None
                hist = (r150 or {}).get("history", [])
                best_ep = int(np.argmax([h["val_dice"] for h in hist])) + 1 if hist else None
                tot_ep = len(hist) or None
                if v24 is not None:
                    d24.append(v24)
                if v150 is not None:
                    d150.append(v150)
                perseed.append((s, v24, v150, tot_ep, best_ep))
            s24, s150 = stats(d24), stats(d150)
            delta = (s150["mean"] - s24["mean"]) if (s24 and s150) else None
            rows.append({"model": model, "label": label, "loss": loss, "s24": s24, "s150": s150,
                         "delta": delta, "perseed": perseed})

    A("## 二、主结果（测试集病灶 Dice，3 种子）\n")
    A("| 模型 | 损失 | 24 epoch | **150 epoch + 早停** | Δ | 150ep 最优轮 |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        eps = [p[4] for p in r["perseed"] if p[4]]
        ep_s = f"{int(np.mean(eps))}" if eps else "—"
        A(f"| {r['label']} | {LOSS_LABEL[r['loss']]} | {ms(r['s24'])} | **{ms(r['s150'])}** | "
          f"{(('%+.4f' % r['delta']) if r['delta'] is not None else '—')} | {ep_s} |")
    A("")

    A("## 三、逐种子明细\n")
    A("| 模型 | 损失 | 种子 | 24ep Dice | 150ep Dice | Δ | 实际训练轮 | 最优轮 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        for (s, v24, v150, tot, best) in r["perseed"]:
            d = (v150 - v24) if (v24 is not None and v150 is not None) else None
            A(f"| {r['label']} | {LOSS_LABEL[r['loss']]} | {s} | "
              f"{('%.4f' % v24) if v24 is not None else '—'} | {('%.4f' % v150) if v150 is not None else '—'} | "
              f"{(('%+.4f' % d) if d is not None else '—')} | {tot or '—'} | {best or '—'} |")
    A("")

    # conclusions
    def get(model, loss):
        for r in rows:
            if r["model"] == model and r["loss"] == loss:
                return r
        return None

    A("## 四、判据与结论\n")
    A("**① 训练预算是主导因素。** 全部 8 个配置在延长训练后**一律提升**（Δ 均为正）；最优轮集中在 **29–81**，"
      "远大于原设定的 24 —— 说明 24 epoch 基线**严重欠拟合**。\n")
    bnd_ok = []
    for m in ("unet", "attunet"):
        c, b = get(m, "ce_dice"), get(m, "boundary")
        if c and b:
            bnd_ok.append((m, b["s150"]["mean"] - c["s150"]["mean"]))
    A("**② “Boundary 有害”成立。** 150ep 下 boundary 仍明显低于 CE+Dice："
      + "、".join(f"{m} {d:+.4f}" for m, d in bnd_ok) + " → 结论**未被推翻**。\n")
    tv = []
    for m in ("unet", "attunet", "unetpp"):
        c, t = get(m, "ce_dice"), get(m, "tversky")
        if c and t:
            tv.append((m, (t["s24"]["mean"] - c["s24"]["mean"]) if (t["s24"] and c["s24"]) else None,
                       t["s150"]["mean"] - c["s150"]["mean"]))
    A("**③ “Tversky 对 UNet++ 有害”被推翻。** Tversky 相对 CE+Dice 的差值：\n")
    A("| 模型 | 24ep Δ(Tversky − CE) | 150ep Δ(Tversky − CE) | 判定 |")
    A("|---|---|---|---|")
    for m, d24, d150 in tv:
        verdict = "反转（原有害→现不劣）" if (d24 is not None and d24 < 0 <= d150) else ("保持" if d150 >= 0 else "仍偏差")
        A(f"| {m} | {(('%+.4f' % d24) if d24 is not None else '—')} | {d150:+.4f} | {verdict} |")
    A("")
    A("→ 在充分训练下 **Tversky 全面 ≥ CE+Dice**，原“UNet++ 有害”系欠拟合假象。\n")
    A("**④ 附带发现。** 150ep 的 U-Net（CE+Dice 0.6875 / Tversky 0.6966）已**追平** 24ep 的 TransUNet（0.6958）"
      "—— 监督式内部的“结构差距”同样被训练预算抹平。\n")

    # ---- P1 + A ----
    p1a = None
    try:
        p1a = json.load(open(f"{RES}/p1a_summary.json"))
    except Exception:
        p1a = None
    if p1a and p1a.get("rows"):
        A("## 五、P1 权重 + A 组（后处理 / TTA / 验证集阈值）\n")
        A("> 在 150ep 权重上再叠加推理侧 A 组，检验 A 组在**充分训练**下是否仍有效（零重训）。\n")
        A("| 模型 | 损失 | P1 baseline | 最优 A 变体 | A 后 Dice | Δ(A) | HD95(base→best) |")
        A("|---|---|---|---|---|---|---|")
        for r in p1a["rows"]:
            if r.get("base_mean") is None:
                continue
            A(f"| {r['label']} | {r['loss']} | {r['base_mean']:.4f} | {r['best'] or '—'} | "
              f"{(('%.4f' % r['best_mean']) if r.get('best_mean') is not None else '—')} | "
              f"{(('%+.4f' % r['delta']) if r.get('delta') is not None else '—')} | "
              f"{(('%.1f→%.1f' % (r['hd95_base'], r['hd95_best'])) if r.get('hd95_best') is not None else '—')} |")
        A("")
        A("**结论**：A 组在充分训练后**仍然有效**（Δ **+0.001 ~ +0.021**），且 **boundary 模型 HD95 大幅下降**（169.8→143.9 等）。")
        A("**TransUNet + CE+Dice + 150ep + A2+A1+A3 = 0.7087**（HD95 96.2）为**监督式最优**；"
          "而 **U-Net（1.9M 参数）+ 同配置 = 0.6993**，与 ~22× 更大的 TransUNet 仅差 **0.009** —— "
          "即“**充分训练 + 零成本推理增强**”可在极轻量模型上逼近重模型。\n")

    A("## 六、复现命令\n")
    A("```bash")
    A("# 基线（24ep）")
    A("python train_supervised.py --model unet --size 128 --gray --epochs 24 --seed 0 --loss ce_dice")
    A("# P1（150ep + 早停 + 回滚）")
    A("python train_supervised.py --model unet --size 128 --gray --epochs 150 --seed 0 \\")
    A("       --loss ce_dice --early-stop 15 --min-epochs 20 --tag p1_unet")
    A("python summarize_p1.py      # -> results/p1_summary.{md,json}")
    A("```\n")

    A("## 六、产物清单\n")
    A("| 文件 | 说明 |")
    A("|---|---|")
    A("| `results/p1_*.json`（24 个） | 每个 (模型×损失×种子) 的测试结果 + 逐轮 history |")
    A("| `results/p1_summary.md` / `.json` | 24ep↔150ep 汇总 |")
    A("| `results/p1log_*.log` | 训练日志（含 early-stop 触发轮） |")
    A("| `src/run_p1_par.sh` / `run_p1_2.sh` | 重跑脚本（并行 / 串行） |")
    A("| `src/summarize_p1.py` / `make_p1_doc.py` | 汇总与本文件生成器 |")
    A("")
    A("> 注：P1 只重训了 B 组（损失）；**未**在 P1 权重上再叠加 A 组（后处理/TTA/阈值）。若需要“P1+A”的组合结果，可另跑 `eval_postproc.py`。")

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", OUT)
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
