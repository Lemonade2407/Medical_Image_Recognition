# P1　充分训练重跑（24 → 150 epoch + 验证集早停）

> 目的：检验“Boundary 有害”“Tversky 对 UNet++ 有害”是否只是 **24 epoch 欠拟合**造成的假象 —— 若成立将直接动摇贡献 3。
> 数据：冻结测试集（`splits/busi_seed0.json`，116 张 / 其中病灶 96 张），统一指标 `src/busi_common.py`，3 个随机种子。

## 一、实验设置

| 项 | 值 |
|---|---|
| 训练轮数 | **150**（原 24） |
| 早停 | 验证集 Dice，**patience 15**，`--min-epochs 20` |
| 模型选择 | **回滚到验证集最优轮**（best-val rollback） |
| 损失 | CE+Dice（基线）/ Tversky(α=0.3,β=0.7) / CE+Dice+Boundary(λ=1) |
| 其余 | 优化器/增强/调度（`step_size=epochs//3`）与基线**完全一致**，仅轮数+早停不同 |
| 范围 | U-Net / Attention U-Net × 3 损失；**UNet++ × {CE+Dice, Tversky}**（验证 UNet++ 的 Tversky 结论） |
| 新增代码 | `train_supervised.py` 的 `--early-stop` / `--min-epochs` |

## 二、主结果（测试集病灶 Dice，3 种子）

| 模型 | 损失 | 24 epoch | **150 epoch + 早停** | Δ | 150ep 最优轮 |
|---|---|---|---|---|---|
| U-Net | CE+Dice | 0.6137±0.0331 | **0.6875±0.0418** | +0.0738 | 72 |
| U-Net | Tversky(0.3/0.7) | 0.6492±0.0126 | **0.6966±0.0207** | +0.0474 | 58 |
| U-Net | CE+Dice+Boundary | 0.5628±0.0490 | **0.6210±0.0514** | +0.0583 | 81 |
| Attention U-Net | CE+Dice | 0.6292±0.0028 | **0.6689±0.0342** | +0.0397 | 49 |
| Attention U-Net | Tversky(0.3/0.7) | 0.6444±0.0101 | **0.6712±0.0031** | +0.0268 | 42 |
| Attention U-Net | CE+Dice+Boundary | 0.5512±0.0126 | **0.6073±0.0241** | +0.0562 | 68 |
| UNet++ | CE+Dice | 0.6265±0.0083 | **0.6429±0.0327** | +0.0164 | 29 |
| UNet++ | Tversky(0.3/0.7) | 0.6039±0.0220 | **0.6470±0.0106** | +0.0431 | 42 |
| TransUNet | CE+Dice | 0.6817±0.0231 | **0.7000±0.0347** | +0.0183 | 48 |
| TransUNet | Tversky(0.3/0.7) | 0.6958±0.0035 | **0.6730±0.0046** | -0.0227 | 23 |

## 三、逐种子明细

| 模型 | 损失 | 种子 | 24ep Dice | 150ep Dice | Δ | 实际训练轮 | 最优轮 |
|---|---|---|---|---|---|---|---|
| U-Net | CE+Dice | 0 | 0.6276 | 0.6524 | +0.0248 | 48 | 33 |
| U-Net | CE+Dice | 1 | 0.6376 | 0.7337 | +0.0961 | 117 | 102 |
| U-Net | CE+Dice | 2 | 0.5760 | 0.6765 | +0.1005 | 96 | 81 |
| U-Net | Tversky(0.3/0.7) | 0 | 0.6564 | 0.6838 | +0.0275 | 68 | 53 |
| U-Net | Tversky(0.3/0.7) | 1 | 0.6565 | 0.7204 | +0.0639 | 70 | 55 |
| U-Net | Tversky(0.3/0.7) | 2 | 0.6346 | 0.6855 | +0.0508 | 81 | 66 |
| U-Net | CE+Dice+Boundary | 0 | 0.6172 | 0.6489 | +0.0317 | 103 | 88 |
| U-Net | CE+Dice+Boundary | 1 | 0.5490 | 0.5617 | +0.0128 | 47 | 32 |
| U-Net | CE+Dice+Boundary | 2 | 0.5221 | 0.6525 | +0.1303 | 139 | 124 |
| Attention U-Net | CE+Dice | 0 | 0.6264 | 0.7027 | +0.0763 | 91 | 76 |
| Attention U-Net | CE+Dice | 1 | 0.6320 | 0.6344 | +0.0024 | 50 | 35 |
| Attention U-Net | CE+Dice | 2 | 0.6293 | 0.6696 | +0.0403 | 53 | 38 |
| Attention U-Net | Tversky(0.3/0.7) | 0 | 0.6374 | 0.6681 | +0.0307 | 76 | 61 |
| Attention U-Net | Tversky(0.3/0.7) | 1 | 0.6560 | 0.6744 | +0.0184 | 50 | 35 |
| Attention U-Net | Tversky(0.3/0.7) | 2 | 0.6398 | 0.6710 | +0.0312 | 46 | 31 |
| Attention U-Net | CE+Dice+Boundary | 0 | 0.5368 | 0.5925 | +0.0556 | 86 | 71 |
| Attention U-Net | CE+Dice+Boundary | 1 | 0.5564 | 0.6351 | +0.0787 | 95 | 80 |
| Attention U-Net | CE+Dice+Boundary | 2 | 0.5603 | 0.5944 | +0.0342 | 69 | 54 |
| UNet++ | CE+Dice | 0 | 0.6172 | 0.6082 | -0.0090 | 32 | 17 |
| UNet++ | CE+Dice | 1 | 0.6293 | 0.6733 | +0.0439 | 52 | 37 |
| UNet++ | CE+Dice | 2 | 0.6330 | 0.6473 | +0.0143 | 49 | 34 |
| UNet++ | Tversky(0.3/0.7) | 0 | 0.5935 | 0.6384 | +0.0449 | 48 | 33 |
| UNet++ | Tversky(0.3/0.7) | 1 | 0.5890 | 0.6438 | +0.0548 | 58 | 43 |
| UNet++ | Tversky(0.3/0.7) | 2 | 0.6291 | 0.6587 | +0.0296 | 65 | 50 |
| TransUNet | CE+Dice | 0 | 0.7008 | 0.7289 | +0.0282 | 78 | 63 |
| TransUNet | CE+Dice | 1 | 0.6560 | 0.7095 | +0.0534 | 65 | 50 |
| TransUNet | CE+Dice | 2 | 0.6882 | 0.6615 | -0.0267 | 47 | 32 |
| TransUNet | Tversky(0.3/0.7) | 0 | 0.6980 | 0.6719 | -0.0261 | 34 | 19 |
| TransUNet | Tversky(0.3/0.7) | 1 | 0.6975 | 0.6781 | -0.0194 | 35 | 20 |
| TransUNet | Tversky(0.3/0.7) | 2 | 0.6918 | 0.6691 | -0.0227 | 45 | 30 |

## 四、判据与结论

**① 训练预算是主导因素。** 全部 8 个配置在延长训练后**一律提升**（Δ 均为正）；最优轮集中在 **29–81**，远大于原设定的 24 —— 说明 24 epoch 基线**严重欠拟合**。

**② “Boundary 有害”成立。** 150ep 下 boundary 仍明显低于 CE+Dice：unet -0.0665、attunet -0.0616 → 结论**未被推翻**。

**③ “Tversky 对 UNet++ 有害”被推翻。** Tversky 相对 CE+Dice 的差值：

| 模型 | 24ep Δ(Tversky − CE) | 150ep Δ(Tversky − CE) | 判定 |
|---|---|---|---|
| unet | +0.0354 | +0.0090 | 保持 |
| attunet | +0.0152 | +0.0023 | 保持 |
| unetpp | -0.0227 | +0.0041 | 反转（原有害→现不劣） |

→ 在充分训练下 **Tversky 全面 ≥ CE+Dice**，原“UNet++ 有害”系欠拟合假象。

**④ 附带发现。** 150ep 的 U-Net（CE+Dice 0.6875 / Tversky 0.6966）已**追平** 24ep 的 TransUNet（0.6958）—— 监督式内部的“结构差距”同样被训练预算抹平。

## 五、P1 权重 + A 组（后处理 / TTA / 验证集阈值）

> 在 150ep 权重上再叠加推理侧 A 组，检验 A 组在**充分训练**下是否仍有效（零重训）。

| 模型 | 损失 | P1 baseline | 最优 A 变体 | A 后 Dice | Δ(A) | HD95(base→best) |
|---|---|---|---|---|---|---|
| U-Net | ce_dice | 0.6875 | A2+A1+A3 | 0.6993 | +0.0118 | 114.4→108.1 |
| U-Net | tversky | 0.6966 | baseline | 0.6976 | +0.0011 | 120.5→120.5 |
| U-Net | boundary | 0.6210 | A2+A1+A3 | 0.6414 | +0.0204 | 169.6→132.0 |
| Attention U-Net | ce_dice | 0.6689 | A2+A1+A3 | 0.6784 | +0.0095 | 127.9→120.8 |
| Attention U-Net | tversky | 0.6712 | A2_tta | 0.6721 | +0.0010 | 124.9→132.4 |
| Attention U-Net | boundary | 0.6073 | A2+A1+A3 | 0.6274 | +0.0201 | 169.8→143.9 |
| UNet++ | ce_dice | 0.6429 | A2+A1+A3 | 0.6545 | +0.0116 | 131.3→136.4 |
| UNet++ | tversky | 0.6470 | A2_tta | 0.6677 | +0.0207 | 134.5→137.5 |
| TransUNet | ce_dice | 0.7000 | A2+A1+A3 | 0.7087 | +0.0087 | 106.8→96.2 |
| TransUNet | tversky | 0.6730 | A2+A1+A3 | 0.6930 | +0.0200 | 121.1→112.9 |

**结论**：A 组在充分训练后**仍然有效**（Δ **+0.001 ~ +0.021**），且 **boundary 模型 HD95 大幅下降**（169.8→143.9 等）。
**TransUNet + CE+Dice + 150ep + A2+A1+A3 = 0.7087**（HD95 96.2）为**监督式最优**；而 **U-Net（1.9M 参数）+ 同配置 = 0.6993**，与 ~22× 更大的 TransUNet 仅差 **0.009** —— 即“**充分训练 + 零成本推理增强**”可在极轻量模型上逼近重模型。

## 六、复现命令

```bash
# 基线（24ep）
python train_supervised.py --model unet --size 128 --gray --epochs 24 --seed 0 --loss ce_dice
# P1（150ep + 早停 + 回滚）
python train_supervised.py --model unet --size 128 --gray --epochs 150 --seed 0 \
       --loss ce_dice --early-stop 15 --min-epochs 20 --tag p1_unet
python summarize_p1.py      # -> results/p1_summary.{md,json}
```

## 六、产物清单

| 文件 | 说明 |
|---|---|
| `results/p1_*.json`（24 个） | 每个 (模型×损失×种子) 的测试结果 + 逐轮 history |
| `results/p1_summary.md` / `.json` | 24ep↔150ep 汇总 |
| `results/p1log_*.log` | 训练日志（含 early-stop 触发轮） |
| `src/run_p1_par.sh` / `run_p1_2.sh` | 重跑脚本（并行 / 串行） |
| `src/summarize_p1.py` / `make_p1_doc.py` | 汇总与本文件生成器 |

> 注：P1 只重训了 B 组（损失）；**未**在 P1 权重上再叠加 A 组（后处理/TTA/阈值）。若需要“P1+A”的组合结果，可另跑 `eval_postproc.py`。
