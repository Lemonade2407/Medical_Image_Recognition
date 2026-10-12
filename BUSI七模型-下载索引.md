# BUSI 七种方法 —— 模型/代码下载索引

> 依据 [TEMP-FILE-STATION/BUSI七种方法对比.md](TEMP-FILE-STATION/BUSI七种方法对比.md)
> 下载时间：2026-10-12 · 环境：PAI DSW（GPU 可见，torch 2.7.0+nv，CUDA available）
> 网络：本机可访问 **GitHub** 与 **ModelScope**；**HuggingFace**、`dl.fbaipublicfiles.com` 被墙。

---

## 1. 结果总览

| 方法 | 官方代码仓库（已克隆） | 公开权重（已下载） |
|---|---|---|
| SAM ViT-B | `code/segment-anything` | ✅ `models/sam_vit_b/sam_vit_b_01ec64.pth`（原始 Meta 格式） |
| MedSAM ViT-B | `code/MedSAM` | ✅ `models/medsam_vit_b/pytorch_model.bin`（HF transformers 格式） |
| BUSSAM ViT-B | `code/BUSSAM` | ❌ 无公开权重（官方仅提供基座 SAM 链接，微调权重需自行训练） |
| U-Net | `code/Pytorch-UNet` | ❌ 无公开 BUSI 权重（需自行训练） |
| Attention U-Net | `code/Attention-Gated-Networks` | ❌ 同上 |
| UNet++ | `code/UNetPlusPlus` | ❌ 同上 |
| TransUNet | `code/TransUNet` | ❌ 官方仅有 Synapse 预训练权重（Google Drive/Baidu，本机不可达） |

**能"下载"的公开权重只有 SAM ViT-B 与 MedSAM ViT-B 两个**（合计 ≈716 MB）。
其余 5 个方法在报告中均为**本地训练**得到，没有可直接下载的 BUSI 权重 —— 已按你的选择改为提供各自的**官方实现代码**。

---

## 2. 权重明细（`/mnt/workspace/models`）

| 文件 | 大小 | MD5 | 来源 |
|---|---|---|---|
| `sam_vit_b/sam_vit_b_01ec64.pth` | 375,042,383 B (357.7 MB) | `01ec64d29a2fca3f0661936605ae66f8` | ModelScope `muse/sam_vit_b_01ec64` |
| `medsam_vit_b/pytorch_model.bin` | 375,045,749 B (357.7 MB) | `8fe20d3d053a9094454c0e8f633144b1` | ModelScope `BeLuckyBePeace/MedSam` |
| `medsam_vit_b/config.json` | 6,502 B | — | 同上 |
| `medsam_vit_b/preprocessor_config.json` | 443 B | — | 同上 |
| `medsam_vit_b/README.md` | 3,344 B | — | 同上 |
| `sam_vit_b/README.md` | 103 B | — | 同上 |

- `sam_vit_b_01ec64.pth` 的 MD5 `01ec64…` 与 Meta 官方命名/校验一致，可直接被 `segment_anything` 加载：
  ```python
  from segment_anything import sam_model_registry
  sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b/sam_vit_b_01ec64.pth")
  ```
  该 checkpoint 同时是 **BUSSAM 训练所需的基座权重**（`code/BUSSAM/checkpoints/sam_vit_b_01ec64.pth`）。

- MedSAM 权重为 **HuggingFace `transformers` 格式**（`architectures: ["SamModel"]`），用 `transformers` 加载：
  ```python
  from transformers import SamModel
  model = SamModel.from_pretrained("models/medsam_vit_b")
  ```
  ⚠️ 注意格式差异：`code/MedSAM` 官方脚本默认加载**原生 `medsam_vit_b.pth`**（`sam_model_registry` 格式），
  与这里的 HF `pytorch_model.bin` 不是同一套 state_dict。官方 `medsam_vit_b.pth` 托管在 Google Drive/Baidu（本机不可达），
  如需原生格式请另行获取；或直接用 `transformers` 路径推理。

---

## 3. 代码仓库明细（`/mnt/workspace/code`，均 `git clone --depth 1`）

| 目录 | 仓库 | 对应报告中的方法 |
|---|---|---|
| `segment-anything` | github.com/facebookresearch/segment-anything | SAM ViT-B |
| `MedSAM` | github.com/bowang-lab/MedSAM | MedSAM ViT-B |
| `BUSSAM` | github.com/bscs12/BUSSAM | BUSSAM ViT-B（*Ultrasound SAM Adapter*，官方实现） |
| `Pytorch-UNet` | github.com/milesial/Pytorch-UNet | U-Net（PyTorch 参考实现；`Attention-Gated-Networks` 内亦含无门控 U-Net 基线） |
| `Attention-Gated-Networks` | github.com/ozan-oktay/Attention-Gated-Networks | Attention U-Net（报告所引用仓库） |
| `UNetPlusPlus` | github.com/MrGiovanni/UNetPlusPlus | UNet++（含多种 backbone/数据集脚本） |
| `TransUNet` | github.com/Beckschen/TransUNet | TransUNet |

> 报告中 `homework/Attention-Gated-Networks-master/` 是课程仓库，本工作区不存在；此处克隆的是其**上游官方仓库**（U-Net + Attention U-Net 实现）。

---

## 4. 复现时仍缺、需自行获取的部分

1. **数据集 BUSI**（780 张）——报告用的数据，未随本任务下载。
2. **5 个本地训练方法的权重**（U-Net / Attention U-Net / UNet++ / TransUNet / BUSSAM）：需按各自仓库自行训练，
   或由你提供已有 checkpoint 的地址。
3. **TransUNet 的 ImageNet/ViT 预训练权重**：官方放 Google Drive/Baidu，本机不可达；报告本身也标注其无预训练训练效果不佳（见优化方案文档 4.x）。

---

## 5. 目录结构

```
/mnt/workspace
├── BUSI七模型-下载索引.md        # 本文件
├── code/                         # 7 个官方代码仓库
│   ├── segment-anything/
│   ├── MedSAM/
│   ├── BUSSAM/
│   ├── Pytorch-UNet/
│   ├── Attention-Gated-Networks/
│   ├── UNetPlusPlus/
│   └── TransUNet/
├── models/                       # 已下载的公开权重
│   ├── sam_vit_b/sam_vit_b_01ec64.pth
│   └── medsam_vit_b/{pytorch_model.bin, config.json, preprocessor_config.json, README.md}
└── TEMP-FILE-STATION/            # 原始报告
```
