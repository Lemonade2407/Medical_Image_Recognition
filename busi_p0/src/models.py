"""2D segmentation models for the BUSI unified-protocol study.

Channel width matches the original report (encoder widths 16/32/64/128/256):
  - UNet:        plain encoder-decoder, no gating  (~1.97M params reported)
  - AttentionUNet: attention gates on skip connections (~2.38M reported)
  - UNetPP:      nested dense skips + deep supervision
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018). g: decoder signal, x: skip."""

    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.relu(self.W_g(g) + self.W_x(x))
        a = self.psi(a)
        return x * a


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_cls=2, chs=(16, 32, 64, 128, 256)):
        super().__init__()
        self.chs = chs
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(double_conv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, 2))
            self.dec.append(double_conv(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], n_cls, 1)

    def forward(self, x):
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x)
            feats.append(x)
            if i != len(self.enc) - 1:
                x = self.pool(x)
        for j, (u, d) in enumerate(zip(self.up, self.dec)):
            x = u(x)
            skip = feats[len(self.chs) - 2 - j]
            x = torch.cat([x, skip], dim=1)
            x = d(x)
        return self.head(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, n_cls=2, chs=(16, 32, 64, 128, 256)):
        super().__init__()
        self.chs = chs
        self.enc = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(double_conv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, 2))
            self.gate.append(AttentionGate(f_g=chs[i - 1], f_l=chs[i - 1], f_int=max(chs[i - 1] // 2, 1)))
            self.dec.append(double_conv(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], n_cls, 1)

    def forward(self, x):
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x)
            feats.append(x)
            if i != len(self.enc) - 1:
                x = self.pool(x)
        for j, (u, g, d) in enumerate(zip(self.up, self.gate, self.dec)):
            x = u(x)
            skip = feats[len(self.chs) - 2 - j]
            skip = g(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = d(x)
        return self.head(x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = double_conv(in_c, out_c)

    def forward(self, x):
        return self.block(x)


class UNetPP(nn.Module):
    """Nested U-Net (UNet++) with deep supervision. chs = per-level width."""

    def __init__(self, in_ch=1, n_cls=2, chs=(32, 64, 128, 256, 512), deep_supervision=True):
        super().__init__()
        self.chs = chs
        self.ds = deep_supervision
        self.nb = len(chs)
        self.pool = nn.MaxPool2d(2)
        self.conv0 = nn.ModuleDict()
        self.up = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        for i, c in enumerate(chs):
            self.conv0[str(i)] = ConvBlock(in_ch if i == 0 else chs[i - 1], c)
        for i in range(1, self.nb):
            self.up[str(i)] = nn.ConvTranspose2d(chs[i], chs[i - 1], 2, 2)
        for j in range(1, self.nb):
            for i in range(self.nb - j):
                self.conv[f"{i}_{j}"] = ConvBlock(chs[i] * (j + 1), chs[i])
        self.heads = nn.ModuleList([nn.Conv2d(chs[0], n_cls, 1) for _ in range(self.nb)])

    def forward(self, x):
        nodes = {}
        for i in range(self.nb):
            if i == 0:
                nodes[f"{i}_0"] = self.conv0["0"](x)
            else:
                nodes[f"{i}_0"] = self.conv0[str(i)](self.pool(nodes[f"{i-1}_0"]))
        for j in range(1, self.nb):
            for i in range(self.nb - j):
                up = self.up[str(i + 1)](nodes[f"{i+1}_{j-1}"])
                feats = [nodes[f"{i}_{k}"] for k in range(j)]
                nodes[f"{i}_{j}"] = self.conv[f"{i}_{j}"](torch.cat(feats + [up], dim=1))
        outs = [self.heads[i](F.interpolate(nodes[f"0_{i}"], size=x.shape[2:], mode="bilinear", align_corners=False))
                for i in range(self.nb)]
        if self.ds:
            return outs  # list; training averages losses
        return outs[-1]


def build_model(name, in_ch=1, n_cls=2, size=256):
    name = name.lower()
    if name == "unet":
        return UNet(in_ch, n_cls)
    if name in ("attunet", "attention_unet", "attentionunet"):
        return AttentionUNet(in_ch, n_cls)
    if name in ("unetpp", "unet++"):
        return UNetPP(in_ch, n_cls)
    if name in ("transunet",):
        import sys
        sys.path.insert(0, "/mnt/workspace/code/TransUNet")
        from networks.vit_seg_modeling import VisionTransformer, CONFIGS
        cfg = CONFIGS["R50-ViT-B_16"]
        model = VisionTransformer(cfg, img_size=size, num_classes=n_cls, zero_head=True)
        return model
    raise ValueError(f"unknown model: {name}")
