# BUSI 分割 · 损失函数对照（150 epoch，3 种子）

> 在**充分训练**（150 epoch + 验证集早停 + 最优轮回滚）下比较 CE+Dice / Tversky / Boundary。
> 数据：冻结测试集（`splits/busi_seed0.json`，116 张 / 其中病灶 96 张），统一指标 `src/busi_common.py`，3 个随机种子，统一概率路径。

## 一、实验设置

| 项 | 值 |
|---|---|
| 训练轮数 | **150** |
| 早停 | 验证集 Dice，**patience 15**，`--min-epochs 20` |
| 模型选择 | **回滚到验证集最优轮**（best-val rollback） |
| 损失 | CE+Dice / Tversky(α=0.3, β=0.7) / CE+Dice+Boundary(λ=1) |
| 其余 | 优化器 / 增强 / 调度（`step_size=epochs//3`）四组完全一致 |
| 范围 | U-Net / Attention U-Net × 3 损失；**UNet++ / TransUNet × {CE+Dice, Tversky}** |

## 二、主结果（测试集病灶 Dice，3 种子）

| 模型 | 损失 | Dice(病灶) | 最优轮 |
|---|---|---|---|
| U-Net | CE+Dice | **0.6887±0.0421** | 72 |
| U-Net | Tversky(0.3/0.7) | **0.6976±0.0207** | 58 |
| U-Net | CE+Dice+Boundary | **0.6214±0.0513** | 81 |
| Attention U-Net | CE+Dice | **0.6699±0.0341** | 49 |
| Attention U-Net | Tversky(0.3/0.7) | **0.6720±0.0031** | 42 |
| Attention U-Net | CE+Dice+Boundary | **0.6081±0.0239** | 68 |
| UNet++ | CE+Dice | **0.6433±0.0328** | 29 |
| UNet++ | Tversky(0.3/0.7) | **0.6473±0.0106** | 42 |
| TransUNet | CE+Dice | **0.7002±0.0348** | 48 |
| TransUNet | Tversky(0.3/0.7) | **0.6731±0.0045** | 23 |

## 三、同模型内 Δ（相对 CE+Dice）

| 模型 | Tversky − CE | Boundary − CE |
|---|---|---|
| U-Net | +0.0089 | -0.0674 |
| Attention U-Net | +0.0021 | -0.0618 |
| UNet++ | +0.0040 | — |
| TransUNet | -0.0271 | — |

## 四、逐种子明细

| 模型 | 损失 | 种子 | Dice | 实际训练轮 | 最优轮 |
|---|---|---|---|---|---|
| U-Net | CE+Dice | 0 | 0.6530 | 48 | 33 |
| U-Net | CE+Dice | 1 | 0.7351 | 117 | 102 |
| U-Net | CE+Dice | 2 | 0.6781 | 96 | 81 |
| U-Net | Tversky(0.3/0.7) | 0 | 0.6847 | 68 | 53 |
| U-Net | Tversky(0.3/0.7) | 1 | 0.7216 | 70 | 55 |
| U-Net | Tversky(0.3/0.7) | 2 | 0.6866 | 81 | 66 |
| U-Net | CE+Dice+Boundary | 0 | 0.6492 | 103 | 88 |
| U-Net | CE+Dice+Boundary | 1 | 0.5622 | 47 | 32 |
| U-Net | CE+Dice+Boundary | 2 | 0.6527 | 139 | 124 |
| Attention U-Net | CE+Dice | 0 | 0.7038 | 91 | 76 |
| Attention U-Net | CE+Dice | 1 | 0.6355 | 50 | 35 |
| Attention U-Net | CE+Dice | 2 | 0.6703 | 53 | 38 |
| Attention U-Net | Tversky(0.3/0.7) | 0 | 0.6690 | 76 | 61 |
| Attention U-Net | Tversky(0.3/0.7) | 1 | 0.6752 | 50 | 35 |
| Attention U-Net | Tversky(0.3/0.7) | 2 | 0.6717 | 46 | 31 |
| Attention U-Net | CE+Dice+Boundary | 0 | 0.5933 | 86 | 71 |
| Attention U-Net | CE+Dice+Boundary | 1 | 0.6357 | 95 | 80 |
| Attention U-Net | CE+Dice+Boundary | 2 | 0.5953 | 69 | 54 |
| UNet++ | CE+Dice | 0 | 0.6085 | 32 | 17 |
| UNet++ | CE+Dice | 1 | 0.6736 | 52 | 37 |
| UNet++ | CE+Dice | 2 | 0.6476 | 49 | 34 |
| UNet++ | Tversky(0.3/0.7) | 0 | 0.6386 | 48 | 33 |
| UNet++ | Tversky(0.3/0.7) | 1 | 0.6442 | 58 | 43 |
| UNet++ | Tversky(0.3/0.7) | 2 | 0.6591 | 65 | 50 |
| TransUNet | CE+Dice | 0 | 0.7293 | 78 | 63 |
| TransUNet | CE+Dice | 1 | 0.7096 | 65 | 50 |
| TransUNet | CE+Dice | 2 | 0.6617 | 47 | 32 |
| TransUNet | Tversky(0.3/0.7) | 0 | 0.6719 | 34 | 19 |
| TransUNet | Tversky(0.3/0.7) | 1 | 0.6781 | 35 | 20 |
| TransUNet | Tversky(0.3/0.7) | 2 | 0.6693 | 45 | 30 |

## 五、结论

**① Boundary 一致劣于 CE+Dice。** U-Net -0.0674、Attention U-Net -0.0618 —— 在本数据上 Boundary 损失**不适用**。

**② 损失选择的影响很小，且模型依赖。** Tversky 相对 CE+Dice：U-Net +0.0089、Attention U-Net +0.0021、UNet++ +0.0040、TransUNet -0.0271。轻量模型的差异均在 ±0.01 以内；唯一反向的是 TransUNet，说明**重模型上 Tversky 不适用**。

**③ 结构差距有限。** U-Net（1.9M）0.6887 vs TransUNet（105M）0.7002，仅差 **+0.0115**；轻量三骨干落在 0.6433–0.6887。

**④ 收敛轮次分布。** 验证集最优轮落在 **17–124**，早停与最优轮回滚是必要的（否则末轮权重并非最优）。

## 六、150ep 权重 + A 组（后处理 / TTA / 验证集阈值）

> 在 150ep 权重上再叠加推理侧 A 组（零重训）。

| 模型 | 损失 | baseline | 最优 A 变体 | A 后 Dice | Δ(A) | HD95(base→best) |
|---|---|---|---|---|---|---|
| U-Net | ce_dice | 0.6887 | A2+A1+A3 | 0.6993 | +0.0106 | 114.4→108.1 |
| U-Net | tversky | 0.6976 | baseline | 0.6976 | +0.0000 | 120.5→120.5 |
| U-Net | boundary | 0.6214 | A2+A1+A3 | 0.6414 | +0.0201 | 169.6→132.0 |
| Attention U-Net | ce_dice | 0.6699 | A2+A1+A3 | 0.6784 | +0.0085 | 127.9→120.8 |
| Attention U-Net | tversky | 0.6720 | A2_tta | 0.6721 | +0.0002 | 124.9→132.4 |
| Attention U-Net | boundary | 0.6081 | A2+A1+A3 | 0.6274 | +0.0193 | 169.8→143.9 |
| UNet++ | ce_dice | 0.6433 | A2+A1+A3 | 0.6545 | +0.0113 | 131.3→136.4 |
| UNet++ | tversky | 0.6473 | A2_tta | 0.6677 | +0.0204 | 134.5→137.5 |
| TransUNet | ce_dice | 0.7002 | A2+A1+A3 | 0.7087 | +0.0085 | 106.8→96.2 |
| TransUNet | tversky | 0.6731 | A2+A1+A3 | 0.6930 | +0.0199 | 121.1→112.9 |

**监督式最优 = TransUNet + ce_dice + A2+A1+A3 = 0.7087**（HD95 96.2）。
**U-Net（1.9M）+ 同配置 = 0.6993**，与 ~105M 的 TransUNet 仅差 0.009。

## 七、复现命令

```bash
python train_supervised.py --model unet --size 128 --gray --epochs 150 --seed 0 \
       --loss ce_dice --early-stop 15 --min-epochs 20 --tag p1_unet
python eval_postproc.py --ckpt results/p1_unet_seed0.pth --model unet --size 128 --gray --tta
python summarize_p1.py      # -> results/p1_summary.{md,json}
python make_p1_doc.py       # -> docs/BUSI-P1-充分训练.md
```

## 八、产物清单

| 文件 | 说明 |
|---|---|
| `results/p1_*.json`（30 个） | 每个 (模型×损失×种子) 的测试结果 + 逐轮 history |
| `results/postproc_p1_*.json`（30 个） | A 组各变体指标（baseline 为主表口径） |
| `results/p1_summary.md` / `.json` | 150ep 损失汇总 |
| `results/p1a_summary.md` / `.json` | 150ep + A 组汇总 |
| `results/p1log_*.log` / `p1alog_*.log` | 训练 / A 组评估日志 |
| `src/run_p1_par.sh` 等 | 重跑脚本 |

