# 150ep 损失函数对照（3 种子，统一概率路径）

| 模型 | 损失 | Dice(病灶) | 最优轮 |
|---|---|---|---|
| U-Net | CE+Dice | 0.6887±0.0421 | 72 |
| U-Net | Tversky(0.3/0.7) | 0.6976±0.0207 | 58 |
| U-Net | CE+Dice+Boundary | 0.6214±0.0513 | 81 |
| Attention U-Net | CE+Dice | 0.6699±0.0341 | 49 |
| Attention U-Net | Tversky(0.3/0.7) | 0.6720±0.0031 | 42 |
| Attention U-Net | CE+Dice+Boundary | 0.6081±0.0239 | 68 |
| UNet++ | CE+Dice | 0.6433±0.0328 | 29 |
| UNet++ | Tversky(0.3/0.7) | 0.6473±0.0106 | 42 |
| TransUNet | CE+Dice | 0.7002±0.0348 | 48 |
| TransUNet | Tversky(0.3/0.7) | 0.6731±0.0045 | 23 |

## 同模型内 Δ（相对 CE+Dice）

| 模型 | Tversky − CE | Boundary − CE |
|---|---|---|
| U-Net | +0.0089 | -0.0674 |
| Attention U-Net | +0.0021 | -0.0618 |
| UNet++ | +0.0040 | — |
| TransUNet | -0.0271 | — |

