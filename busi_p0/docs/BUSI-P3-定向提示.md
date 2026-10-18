# P3 归档（2026-10-18）

误差类型感知的定向提示纠正 —— 全部数据留档。
归档由 `src/archive_p3.py` 生成，数值与 `docs/BUSI-算法优化方案.md` §4.1⑥ 完全一致。

## 一、一句话结论

粗模型“无框”（全盲）是端到端最大的可回收损失：兜底提示把 Dice 从 **0.6172** 抬到 **0.6405**（+0.0233，p=0.008，9 好 / 0 差），无框率 11.5%→2.1%；代价是正常图假阳 0.0004→0.1003（交集）~0.2601（多数投票），因正常图与漏检病灶在粗模型输出上不可分（判据精度上限 0.39）。

## 二、环境

- 设备：cuda / Tesla V100-SXM2-32GB；torch 2.5.1+cu124（cuda 12.4）
- 解释器：`/mnt/workspace/busi_p0/.venv/bin/python`
- 粗模型：`results/viz_unetpp_seed0.pth`（UNet++，24ep，seed0，256 RGB）
- 精修器：MedSAM ViT-B（`/mnt/workspace/models/medsam_vit_b`）

## 三、复现

```bash
cd /mnt/workspace/busi_p0/src
/mnt/workspace/busi_p0/.venv/bin/python -u diag_tier_prompt.py   # 消融 -> results/tier_prompt.*
/mnt/workspace/busi_p0/.venv/bin/python -u archive_p3.py          # 本归档（含掩膜/图）
/mnt/workspace/busi_p0/.venv/bin/python -u probe_tier.py          # 证据：各阈值下可用提示统计
/mnt/workspace/busi_p0/.venv/bin/python -u probe_tier2.py         # 证据：正常图 vs 漏检可分性
```

> 阈值：τ_d=0.0284、τ_s=1.4711（验证集 80 分位）；门控无阈值（argmax 从未判为病灶）。

## 四、目录

| 路径 | 内容 |
|---|---|
| `code/diag_tier_prompt.py` | v3 主脚本（信号缓存 + MedSAM 批处理/去重 + 4 种聚合） |
| `code/probe_tier.py` | 诊断：0.5/0.3 阈值下有无连通域的统计 |
| `code/probe_tier2.py` | 诊断：正常图与漏检病灶的 pmax 重叠（判据精度上限的证据） |
| `code/archive_p3.py` | 本归档生成器 |
| `results/tier_prompt.json` | 11 个变体的全部指标 + 逐图记录 + 门控扫描 |
| `results/tier_prompt.png` | 柱状图（报告表见 `docs/BUSI-P3-结果表.md`） |
| `results/tier_prompt_v3.log` | 完整运行日志 |
| `results/env.json` | 运行环境 |
| `results/busi_seed0.json` | 冻结划分（546/118/116） |
| `preds/B0_single.npz` 等 4 个 | 116 张测试图的**布尔预测掩膜**（原始分辨率，键=图像名） |
| `tables/variants.csv` | 变体指标表 |
| `tables/per_image.csv` | 逐图 Dice / fg / 类型 / 候选数 |
| `figs/blind_lesions.png` | 9 张全盲**病灶**：原图 / B0 / 交集 / 多数投票 |
| `figs/blind_normals.png` | 16 张全盲**正常图**：同上（假阳代价） |
| `sha256sums.txt` | 归档内全部文件的校验和 |

## 五、关键数字

| 变体 | Dice | Δ | p | 无框率 | 正常图 fg | 特异度 |
|---|---|---|---|---|---|---|
| B0 单框 | 0.6172 | — | — | 0.115 | 0.0004 | 0.9996 |
| 交集 F_intersect | 0.6358 | +0.0186 | 0.139 | 0.021 | 0.1003 | 0.8997 |
| 多数投票 F_majority | 0.6405 | +0.0233 | 0.008 | 0.021 | 0.2601 | 0.7399 |
| 兜底+救回 FR_select | 0.6341 | +0.0169 | 0.004 | 0.000 | 0.5356 | 0.4644 |

误差类型分布（116 张）：正确 67、偏移 13、尺度错 7、欠自信 4、全盲 25

## 六、已知限制

1. 粗模型仅 seed0、且为 24ep 欠拟合权重 → 增益的种子间稳健性**未验证**（+0.0233 ≈ 3 种子 std 0.02）。
2. 正常图假阳代价**未解决**（特异度 0.9996→0.90），成功判据“正常图不劣化”未通过 → 只作归因证据。
3. 掩膜为原始分辨率布尔图；npz 键把 `/` 替换为 `__`。
4. 无随机性，重跑应逐位复现（MedSAM 推理确定性）。

