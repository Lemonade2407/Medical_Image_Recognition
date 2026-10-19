# P4 归档说明（presence-gated blind fallback，2026-10-18）

误差类型兜底的**代价修复**实验：用图像级"有/无病灶"分类器决定"全盲"图是否出兜底提示。
数据在 `archive/P4_pgbf_2026-10-18/`，方案口径见 `docs/BUSI-算法优化方案.md` §4.1⑦ / §5.4。

## 一、一句话结论

门控**机制有效且免费**：合法阈值（仅用验证集，τ=0.662）下端到端 Dice 与不做门控**逐位相同（0.6768）**，正常图假阳面积 **0.1888→0.0378（−80%）**；但**前置判据在 n=20 下不可判定**——4/20 的 95% CI 为 **8.1–41.6%**，跨过 10% 目标（精确二项 **p=0.133**），点估计超线但样本量不足以支持结论 → 不作新意主张，只作"机制有效、功效受限"的证据。

## 二、方法

1. **T1**：训练图像级二分类器（normal vs lesion），在测试集上分别看三组 P(abnormal)：20 张正常图 / 各自种子的全盲病灶 / 其余病灶。阈值只在验证集选。
2. **T2**：把 P3 的"全盲 → 兜底"分支加上门控：`P(abnormal) ≥ τ` 才出候选，否则按无框记 0；其余类型、聚合方式完全不变。
3. **T3**：与 gate off（`FC_majority`：Dice 0.6768 / fg 0.1888）对照。

阈值规则：正常图验证集 FP ≤ 预算，先最大化病灶召回，同召回取更高阈值；另跑一次 FP=0 预算得到"最严合法阈值" τ=0.662。

## 三、T1：分类器

| 分类器 | 阈值来源 | 全盲命中 | 正常图误报 | 判定 |
|---|---|---|---|---|
| ResNet18 128 | val（FP≤10%） | 0.83（5/10, 8/8, 6/6） | 0.38（7.7/20） | ⚠️ 不可判定 |
| **EfficientNet-B0 224** | val（FP≤10%，τ=0.50） | **1.00（10/10, 8/8, 6/6）** | 0.25（5/20） | ⚠️ 不可判定 |
| EfficientNet-B0 224 | val（FP=0，τ=0.662） | **24/24** | 4/20 | ⚠️ 不可判定（CI 跨 10%） |
| EfficientNet-B0 224 | **测试集 oracle τ=0.80** | 8/10 | **1/20** | ✅（非合法标定） |

**统计口径（为什么是"不可判定"而不是"失败"）**：
- n=20 时，4/20 的 **95% CI = 8.1–41.6%**，包含 10% 目标；精确二项检验 **p=0.133**（P(X≥4 | n=20, p=0.10)），不能拒绝"FP=10%"。
- 要区分 10% vs 20%（80% power, 双侧 5%）需 **~86 张正常图**；±10% 半宽需 ~62 张，±5% 需 ~246 张。
- 也就是说，**"≤2/20" 这条判据在 n=20 上本身没有功效**——观察到的 4/20 与 2/20 在统计上不可区分。

- **信号存在**：AUC（全盲 vs 正常）= **0.775**（p=0.008）；整体 AUC（正常 vs 全部病灶）= **0.957**。
- **失败在标定与功效，不在信号**：两类分数在 0.65–0.85 重叠；验证集只有 20 张正常图，其分数上限 **0.662** 低于测试集 7 张正常图 → val→test 漂移；同时 n=20 的 FP 置信区间宽达 8–42%，判据无功效。
- EffNet-B0 用 RGB（`gray=false`），与分割主线的 128 灰度预处理不同。

## 四、T2/T3：端到端

| 配置 | Dice | Δ vs B0 | 无框率 | 正常图 fg | 有前景的正常图 |
|---|---|---|---|---|---|
| B0 单框 | 0.6542±0.0359 | — | 9.0% | 0.0036 | 5.3/20 |
| gate off（`FC_majority`） | 0.6768±0.0325 | +0.0226 | 0.7% | 0.1888 | 16.7/20 |
| **gate on，τ=0.662（合法）** | **0.6768±0.0325** | **+0.0226** | 0.7% | **0.0378** | **7.3/20** |
| gate on，τ=0.80（oracle） | 0.6756±0.0347 | +0.0214 | 1.4% | **0.0092** | 5.7/20 |

> `FC_intersect` 在 τ=0.662 下 fg 更低（**0.0164**，已 ≤0.02），Dice 0.6715（p=0.060）。

## 五、结论

1. **门控本身有效且免费**：合法 τ 下 Dice 与 gate off 完全相同，正常图假阳 −80%，有前景正常图 16.7→7.3；oracle τ 可回到 5.7，接近 B0 的 5.3。即被挡掉的几乎都是"本来不该出兜底"的正常图。
2. **门控只能治"全盲"这一档**：B0 自身的 5.3 张正常图假阳来自**粗模型本来就出框**的正常图，不在 blind 分支，门控原理上管不到——这是残余 7.3 张的下限来源。要压低只能把 presence 前置到全部图像，那已是 BTEC-Net / RFS-UNet 的既有机制。
3. **最终定位**：P3 的"无框硬失败可回收"**成立**；"正常图代价可消除"**不成立**（这是二分类分割头的信息上限）。P4 是"机制有效、**判据功效不足**"的证据——T1 应写成不可判定，而非失败。

## 六、限制

1. 验证集仅 20 张正常图：合法阈值只能取到 0.662，且 4/20 的 95% CI 为 8.1–41.6%，与 10% 目标不可区分（p=0.133）；要判定需 ≥86 张正常图。oracle 0.80 不可用于论文结论。
2. 只测 UNet++（150ep × 3 种子）一个粗模型。
3. 门控不作用于非全盲分支（偏移 / 尺度错 / 正确），也不改变聚合方式。
4. EffNet-B0 使用 RGB 224，与分割主线的 128 灰度不同，属独立信号源。

## 七、复现

```bash
cd /mnt/workspace/busi_p0/src
PY=/mnt/workspace/busi_p0/.venv/bin/python

# T1：分类器（val FP<=10% 选阈值）
$PY -u train_presence_clf.py --arch efficientnet_b0 --size 224 --rgb \
       --seeds 0,1,2 --epochs 40 --patience 10 --out presence_clf_effb0
# 合法最严阈值（val FP=0 -> 0.662）
$PY -u train_presence_clf.py --arch efficientnet_b0 --size 224 --rgb \
       --seeds 0,1,2 --epochs 40 --patience 10 --fp-budget 0 --out presence_clf_effb0_fp0

# T2：门控（合法 / oracle）
$PY -u diag_tier_prompt.py --model unetpp --size 256 --seeds 0,1,2 \
       --presence results/presence_clf_effb0_scores.json --presence-thr 0.662 \
       --out tier_prompt_150ep_pgbf662
$PY -u diag_tier_prompt.py --model unetpp --size 256 --seeds 0,1,2 \
       --presence results/presence_clf_effb0_scores.json --presence-thr 0.80 \
       --out tier_prompt_150ep_pgbf080
```

## 八、产物

| 路径（`archive/P4_pgbf_2026-10-18/`） | 内容 |
|---|---|
| `code/train_presence_clf.py` | 分类器训练 / 三组统计 / 验证集选阈值 |
| `code/diag_tier_prompt.py` | P3 + `--presence` 门控 |
| `code/diag_zero_cost.py`、`code/make_qualitative.py`、`code/manifest_ckpts.py`、`code/archive_p4.py` | 150ep 诊断、定性图、checkpoint 指纹、归档生成器 |
| `results/presence_clf*.json` | 分类器逐种子指标、阈值、`pass` 标志 |
| `results/presence_clf_effb0_scores.json` | `{threshold, scores}`，门控输入 |
| `results/tier_prompt_150ep_pgbf{662,080}.{json,md,png}` | 门控前后的端到端指标 |
| `tables/t1_classifier.csv`、`tables/gate_ablation.csv` | 汇总表 |
| `results/diag_fn_decomp.*`、`diag_box_curve.*` | 150ep 版零成本诊断 |
| `figs/qualitative_*.png` | 150ep 定性图 |
| `results/CHECKPOINTS.md`、`checkpoints.sha256` | 30 个 checkpoint 的指纹（不复制本体） |
| `sha256sums.txt` | 归档校验和 |
