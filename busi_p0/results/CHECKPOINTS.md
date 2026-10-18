# P1 / P3 checkpoint 清单（T5）

> 由 `src/manifest_ckpts.py` 生成。`sha256sum -c results/checkpoints.sha256` 可校验完整性。
> 不复制文件本体（30 个、约 2.7 GB），只登记指纹与训练元数据。

用途：**P3 的粗模型**（`p1_unetpp_seed{0,1,2}.pth`）与 **A 组**（`postproc_p1_*` 的输入权重）都依赖这些文件；缺一个就不可复现。

| 文件 | 模型 | 损失 | 种子 | 大小(MB) | 最优轮 | 最优验证 Dice | 测试 Dice |
|---|---|---|---|---|---|---|---|
| `p1_attunet_boundary_seed0.pth` | attunet | CE+Dice+Boundary | 0 | 7.9 | 71 | 0.6826 | 0.5925 |
| `p1_attunet_boundary_seed1.pth` | attunet | CE+Dice+Boundary | 1 | 7.9 | 80 | 0.6850 | 0.6351 |
| `p1_attunet_boundary_seed2.pth` | attunet | CE+Dice+Boundary | 2 | 7.9 | 54 | 0.6475 | 0.5944 |
| `p1_attunet_seed0.pth` | attunet | CE+Dice | 0 | 7.9 | 76 | 0.7145 | 0.7027 |
| `p1_attunet_seed1.pth` | attunet | CE+Dice | 1 | 7.9 | 35 | 0.7008 | 0.6344 |
| `p1_attunet_seed2.pth` | attunet | CE+Dice | 2 | 7.9 | 38 | 0.7188 | 0.6696 |
| `p1_attunet_tversky_seed0.pth` | attunet | Tversky(0.3/0.7) | 0 | 7.9 | 61 | 0.7254 | 0.6681 |
| `p1_attunet_tversky_seed1.pth` | attunet | Tversky(0.3/0.7) | 1 | 7.9 | 35 | 0.6989 | 0.6744 |
| `p1_attunet_tversky_seed2.pth` | attunet | Tversky(0.3/0.7) | 2 | 7.9 | 31 | 0.6847 | 0.6710 |
| `p1_transunet_seed0.pth` | transunet | CE+Dice | 0 | 421.5 | 63 | 0.7366 | 0.7289 |
| `p1_transunet_seed1.pth` | transunet | CE+Dice | 1 | 421.5 | 50 | 0.7453 | 0.7095 |
| `p1_transunet_seed2.pth` | transunet | CE+Dice | 2 | 421.5 | 32 | 0.6511 | 0.6615 |
| `p1_transunet_tversky_seed0.pth` | transunet | Tversky(0.3/0.7) | 0 | 421.5 | 19 | 0.6581 | 0.6719 |
| `p1_transunet_tversky_seed1.pth` | transunet | Tversky(0.3/0.7) | 1 | 421.5 | 20 | 0.6775 | 0.6781 |
| `p1_transunet_tversky_seed2.pth` | transunet | Tversky(0.3/0.7) | 2 | 421.5 | 30 | 0.6965 | 0.6691 |
| `p1_unet_boundary_seed0.pth` | unet | CE+Dice+Boundary | 0 | 7.8 | 88 | 0.6975 | 0.6489 |
| `p1_unet_boundary_seed1.pth` | unet | CE+Dice+Boundary | 1 | 7.8 | 32 | 0.6213 | 0.5617 |
| `p1_unet_boundary_seed2.pth` | unet | CE+Dice+Boundary | 2 | 7.8 | 124 | 0.7184 | 0.6525 |
| `p1_unet_seed0.pth` | unet | CE+Dice | 0 | 7.8 | 33 | 0.7108 | 0.6524 |
| `p1_unet_seed1.pth` | unet | CE+Dice | 1 | 7.8 | 102 | 0.7245 | 0.7337 |
| `p1_unet_seed2.pth` | unet | CE+Dice | 2 | 7.8 | 81 | 0.7352 | 0.6765 |
| `p1_unet_tversky_seed0.pth` | unet | Tversky(0.3/0.7) | 0 | 7.8 | 53 | 0.7194 | 0.6838 |
| `p1_unet_tversky_seed1.pth` | unet | Tversky(0.3/0.7) | 1 | 7.8 | 55 | 0.7253 | 0.7204 |
| `p1_unet_tversky_seed2.pth` | unet | Tversky(0.3/0.7) | 2 | 7.8 | 66 | 0.7202 | 0.6855 |
| `p1_unetpp_seed0.pth` | unetpp | CE+Dice | 0 | 35.4 | 17 | 0.6578 | 0.6082 |
| `p1_unetpp_seed1.pth` | unetpp | CE+Dice | 1 | 35.4 | 37 | 0.6987 | 0.6733 |
| `p1_unetpp_seed2.pth` | unetpp | CE+Dice | 2 | 35.4 | 34 | 0.6862 | 0.6473 |
| `p1_unetpp_tversky_seed0.pth` | unetpp | Tversky(0.3/0.7) | 0 | 35.4 | 33 | 0.6940 | 0.6384 |
| `p1_unetpp_tversky_seed1.pth` | unetpp | Tversky(0.3/0.7) | 1 | 35.4 | 43 | 0.6759 | 0.6438 |
| `p1_unetpp_tversky_seed2.pth` | unetpp | Tversky(0.3/0.7) | 2 | 35.4 | 50 | 0.6772 | 0.6587 |

合计 **30** 个 checkpoint，**2.88 GB**。

## 校验

```bash
cd /mnt/workspace/busi_p0 && sha256sum -c results/checkpoints.sha256
```

## 与 P3 的对应关系

- P3 v4 粗模型 = `p1_unetpp_seed0/1/2.pth`（150ep + 早停 + 最优轮回滚）
- A 组推理结果 `postproc_p1_<model>_<size><mode>[_loss]_seed<k>.json` 由同名 `.pth` 生成
- 24ep 对照用的是 `ab_*.pth` 与 `viz_*.pth`，不在本清单内（历史产物，仅定性图使用）

