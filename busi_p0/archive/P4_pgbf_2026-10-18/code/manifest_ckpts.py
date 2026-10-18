"""T5: checkpoint manifest instead of a 2.7 GB duplicate copy.

For every results/p1_*.pth record sha256 + size + the training metadata taken
from the sibling JSON (model, loss, seed, best epoch, best val dice, test dice),
so the checkpoints can be verified and restored without archiving them twice.

Writes results/CHECKPOINTS.md and results/checkpoints.sha256.
"""
import os
import re
import json
import hashlib

RES = "/mnt/workspace/busi_p0/results"
OUT_MD = f"{RES}/CHECKPOINTS.md"
OUT_SUM = f"{RES}/checkpoints.sha256"

LOSS_LABEL = {"": "CE+Dice", "_tversky": "Tversky(0.3/0.7)", "_boundary": "CE+Dice+Boundary"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    files = sorted(f for f in os.listdir(RES) if re.match(r"^p1_.*\.pth$", f))
    rows = []
    for f in files:
        p = os.path.join(RES, f)
        stem = f[:-4]
        m = re.match(r"^p1_(\w+?)(_tversky|_boundary)?_seed(\d+)$", stem)
        model, loss, seed = (m.group(1), m.group(2) or "", int(m.group(3))) if m else ("?", "", "?")
        js = os.path.join(RES, stem + ".json")
        meta = {}
        if os.path.exists(js):
            d = json.load(open(js))
            hist = d.get("history") or []
            meta = {"test": (d.get("agg") or {}).get("dice_lesion"),
                    "best_ep": (max(range(len(hist)), key=lambda i: hist[i]["val_dice"]) + 1)
                               if hist else None,
                    "best_val": max((h["val_dice"] for h in hist), default=None),
                    "epochs": len(hist) or None}
        rows.append({"file": f, "model": model, "loss": LOSS_LABEL.get(loss, loss),
                     "seed": seed, "size_mb": os.path.getsize(p) / 1e6,
                     "sha256": sha(p), **meta})

    with open(OUT_SUM, "w") as fh:
        for r in rows:
            fh.write(f"{r['sha256']}  results/{r['file']}\n")

    L = ["# P1 / P3 checkpoint 清单（T5）\n",
         "> 由 `src/manifest_ckpts.py` 生成。`sha256sum -c results/checkpoints.sha256` 可校验完整性。",
         "> 不复制文件本体（30 个、约 2.7 GB），只登记指纹与训练元数据。\n",
         "用途：**P3 的粗模型**（`p1_unetpp_seed{0,1,2}.pth`）与 **A 组**（`postproc_p1_*` 的输入权重）"
         "都依赖这些文件；缺一个就不可复现。\n",
         "| 文件 | 模型 | 损失 | 种子 | 大小(MB) | 最优轮 | 最优验证 Dice | 测试 Dice |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bv = f"{r['best_val']:.4f}" if r.get("best_val") is not None else "—"
        td = f"{r['test']:.4f}" if r.get("test") is not None else "—"
        L.append(f"| `{r['file']}` | {r['model']} | {r['loss']} | {r['seed']} | "
                 f"{r['size_mb']:.1f} | {r['best_ep'] or '—'} | {bv} | {td} |")
    L += ["",
          f"合计 **{len(rows)}** 个 checkpoint，**{sum(r['size_mb'] for r in rows)/1000:.2f} GB**。\n",
          "## 校验\n",
          "```bash",
          "cd /mnt/workspace/busi_p0 && sha256sum -c results/checkpoints.sha256",
          "```\n",
          "## 与 P3 的对应关系\n",
          "- P3 v4 粗模型 = `p1_unetpp_seed0/1/2.pth`（150ep + 早停 + 最优轮回滚）",
          "- A 组推理结果 `postproc_p1_<model>_<size><mode>[_loss]_seed<k>.json` 由同名 `.pth` 生成",
          "- 24ep 对照用的是 `ab_*.pth` 与 `viz_*.pth`，不在本清单内（历史产物，仅定性图使用）",
          ""]
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT_MD} and {OUT_SUM}  ({len(rows)} ckpts, "
          f"{sum(r['size_mb'] for r in rows)/1000:.2f} GB)")


if __name__ == "__main__":
    main()
