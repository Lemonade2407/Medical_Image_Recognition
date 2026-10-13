"""Train + evaluate one supervised model with one seed under the unified protocol.

Usage:
  python train_supervised.py --model unet   --size 128 --gray --epochs 24 --seed 0
  python train_supervised.py --model attunet --size 128 --gray --epochs 24 --seed 0
  python train_supervised.py --model unetpp  --size 256 --rgb  --epochs 24 --seed 0

Writes results/<model>_seed<k>.json with per-image and aggregate test metrics.
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from busi_common import (build_index, load_split, orig_size, RESULTS, LESION_CLASSES,
                         dice_binary, per_image_metrics)
from models import build_model


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class InMemoryBUSI(Dataset):
    def __init__(self, records, size, gray, train):
        self.recs = records
        self.train = train
        self.gray = gray
        self.size = size
        self.X = []
        self.Y = []          # model-resolution mask (for training loss)
        self.names = []
        for it in records:
            if gray:
                im = Image.open(it["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
                a = np.asarray(im, dtype=np.float32).mean(2, keepdims=True)
                a = (a - a.mean()) / (a.std() + 1e-8)
            else:
                im = Image.open(it["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
                a = np.asarray(im, dtype=np.float32)
                a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-8)
            self.X.append(torch.from_numpy(a).permute(2, 0, 1).contiguous())

            w, h = orig_size(it)
            m = np.zeros((h, w), dtype=bool)
            for mp in it["masks"]:
                arr = np.array(Image.open(mp).convert("L"))
                if arr.shape != (h, w):
                    arr = np.array(Image.open(mp).convert("L").resize((w, h), Image.NEAREST))
                m |= (arr > 127)
            mr = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127
            self.Y.append(torch.from_numpy(mr.astype(np.float32))[None])
            self.names.append(it["name"])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.train:
            if random.random() < 0.5:
                x, y = TF.hflip(x), TF.hflip(y)
            angle = random.uniform(-15, 15)
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            scale = random.uniform(0.85, 1.15)
            x = TF.affine(x, angle, translate, scale, 0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            y = TF.affine(y, angle, translate, scale, 0, interpolation=TF.InterpolationMode.NEAREST, fill=0)
        return x, (y > 0.5).float()


def ce_dice_loss(logits, target):
    if target.dim() == 4:
        target = target.squeeze(1)
    ce = F.cross_entropy(logits, target.long())
    prob = F.softmax(logits, 1)[:, 1]
    tgt = (target > 0.5).float()
    inter = (prob * tgt).sum()
    dice = 1 - (2 * inter + 1.0) / (prob.sum() + tgt.sum() + 1.0)
    return ce + dice


def tversky_loss(logits, target, alpha=0.3, beta=0.7):
    """B1: Tversky loss. alpha weights false positives, beta false negatives.

    With beta > alpha (0.3 / 0.7) missing lesion pixels cost more than over-
    predicting, which trades precision for recall -- the documented症结 of the
    supervised models (recall 0.715 vs precision 0.836).
    """
    if target.dim() == 4:
        target = target.squeeze(1)
    prob = F.softmax(logits, 1)[:, 1]
    tgt = (target > 0.5).float()
    tp = (prob * tgt).sum()
    fp = (prob * (1 - tgt)).sum()
    fn = ((1 - prob) * tgt).sum()
    tversky_index = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return 1.0 - tversky_index


def _signed_distance(gt):
    """Signed distance map of a binary mask: negative inside, positive outside."""
    from scipy import ndimage
    inside = ndimage.distance_transform_edt(gt).astype(np.float32)
    outside = ndimage.distance_transform_edt(~gt).astype(np.float32)
    return outside - inside  # inside -> negative, outside -> positive


def boundary_loss(logits, target):
    """B2: Boundary loss (Kervadec et al.).

    Aligns the foreground probability with a signed distance map of the GT, so
    the contour -- not the overlapping area -- is optimised directly. Dice is
    insensitive to uniform boundary shift; this term is not.
    """
    if target.dim() == 4:
        target = target.squeeze(1)
    prob = F.softmax(logits, 1)[:, 1]
    dms = []
    for i in range(target.shape[0]):
        gt = target[i].detach().cpu().numpy() > 0.5
        if gt.any() and not gt.all():
            dms.append(_signed_distance(gt))
        else:  # empty (normal images) or degenerate -> no boundary signal
            dms.append(np.zeros_like(gt, dtype=np.float32))
    dm = torch.from_numpy(np.stack(dms)).to(prob.device)
    return (prob * dm).mean()


def loss_term(logits, target, loss_name="ce_dice", lambda_bd=1.0):
    """Term for one output head. Baseline is CE+Dice; each option swaps/adds one
    component so the ablation changes a single variable at a time."""
    tgt = target.squeeze(1) if target.dim() == 4 else target
    if loss_name == "tversky":       # B1: Dice -> Tversky
        return F.cross_entropy(logits, tgt.long()) + tversky_loss(logits, tgt)
    if loss_name == "boundary":      # B2: CE+Dice + lambda * Boundary
        return ce_dice_loss(logits, tgt) + lambda_bd * boundary_loss(logits, tgt)
    return ce_dice_loss(logits, tgt)


def forward_loss(model, x, y, loss_name="ce_dice", lambda_bd=1.0):
    out = model(x)
    if isinstance(out, (list, tuple)):
        loss = 0.0
        for o in out:
            o = F.interpolate(o, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + loss_term(o, y, loss_name, lambda_bd)
        return loss / len(out), out[-1]
    return loss_term(out, y, loss_name, lambda_bd), out


@torch.no_grad()
def predict_logits(model, x):
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out


def dice_at_model_res(logits, y):
    pred = logits.argmax(1)
    pred = (pred == 1).float()
    inter = (pred * y).sum().item()
    denom = pred.sum().item() + y.sum().item()
    return (2 * inter + 1e-7) / (denom + 1e-7)


def train(args, records_train, records_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    in_ch = 1 if args.gray else 3
    model = build_model(args.model, in_ch=in_ch, n_cls=2, size=args.size).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(args.epochs // 3, 1), gamma=0.5)

    tr = InMemoryBUSI(records_train, args.size, args.gray, True)
    va = InMemoryBUSI(records_val, args.size, args.gray, False)
    nw = getattr(args, "workers", 4)
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=nw,
                       persistent_workers=nw > 0, pin_memory=True, drop_last=False)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=max(nw // 2, 0),
                       persistent_workers=nw > 0, pin_memory=True)

    best_dice, best_state, hist = -1.0, None, []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss, _ = forward_loss(model, x, y, args.loss, args.lambda_bd)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        sched.step()

        model.eval()
        dsum, dn = 0.0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = predict_logits(model, x)
                for b in range(x.size(0)):
                    dsum += dice_at_model_res(logits[b:b + 1], y[b:b + 1]); dn += 1
        vd = dsum / max(dn, 1)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_dice": vd, "sec": round(time.time() - t0, 1)})
        if vd > best_dice:
            best_dice = vd
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ep{ep+1:02d} loss={tot/len(tr):.4f} val_dice={vd:.4f} ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, n_par, best_dice, hist, device


@torch.no_grad()
def evaluate_original_res(model, records, size, gray, device):
    """Evaluate on the test set at ORIGINAL resolution (unified protocol)."""
    per = []
    for it in records:
        if gray:
            im = Image.open(it["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32).mean(2, keepdims=True)
            a = (a - a.mean()) / (a.std() + 1e-8)
        else:
            im = Image.open(it["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32)
            a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-8)
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(device)
        logits = predict_logits(model, x)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

        w, h = orig_size(it)
        gt = np.zeros((h, w), dtype=bool)
        for mp in it["masks"]:
            arr = np.array(Image.open(mp).convert("L"))
            if arr.shape != (h, w):
                arr = np.array(Image.open(mp).convert("L").resize((w, h), Image.NEAREST))
            gt |= (arr > 127)
        pred_o = np.array(Image.fromarray(pred * 255).resize((w, h), Image.NEAREST)) > 127

        m = per_image_metrics(pred_o, gt)
        m["name"] = it["name"]
        m["cls"] = it["cls"]
        per.append(m)
    return per


def aggregate(per):
    lesion = [m for m in per if m["cls"] in LESION_CLASSES]
    normal = [m for m in per if m["cls"] == "normal"]
    tp = sum(m["tp"] for m in per); fp = sum(m["fp"] for m in per)
    fn = sum(m["fn"] for m in per); tn = sum(m["tn"] for m in per)
    # pixel-histogram mIoU (includes background)
    iou_bg = tn / (tn + fp + 1e-7)
    iou_fg = tp / (tp + fn + fp + 1e-7)
    agg = {
        "n_test": len(per), "n_lesion": len(lesion), "n_normal": len(normal),
        "dice_lesion": float(np.mean([m["dice"] for m in lesion])),
        "dice_lesion_std": float(np.std([m["dice"] for m in lesion])),
        "dice_all": float(np.mean([m["dice"] for m in per])),
        "iou_lesion": float(np.mean([m["iou"] for m in lesion])),
        "miou_hist": float((iou_bg + iou_fg) / 2),
        "precision": tp / (tp + fp + 1e-7),
        "recall": tp / (tp + fn + 1e-7),
        "specificity": tn / (tn + fp + 1e-7),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "hd95": float(np.mean([m["hd95"] for m in lesion])),
        "asd": float(np.mean([m["asd"] for m in lesion])),
        "fg_ratio_normal": float(np.mean([m["fg_ratio"] for m in normal])) if normal else 0.0,
        "dice_benign": float(np.mean([m["dice"] for m in per if m["cls"] == "benign"])),
        "dice_malignant": float(np.mean([m["dice"] for m in per if m["cls"] == "malignant"])),
    }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--loss", default="ce_dice", choices=["ce_dice", "tversky", "boundary"],
                    help="ce_dice=baseline; tversky=B1; boundary=B2")
    ap.add_argument("--lambda-bd", type=float, default=1.0, help="B2 weight on the boundary term")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save the best-val weights so eval_postproc.py can reuse them")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.gray = args.gray or not args.rgb
    if args.rgb:
        args.gray = False

    set_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 8)
    index = build_index()
    byname = {it["name"]: it for it in index}
    split = load_split()
    rtr = [byname[n] for n in split["train"]]
    rva = [byname[n] for n in split["val"]]
    rte = [byname[n] for n in split["test"]]

    print(f"[{args.model} seed={args.seed} size={args.size} gray={args.gray} ep={args.epochs}]")
    model, n_par, best_val, hist, device = train(args, rtr, rva)
    per = evaluate_original_res(model, rte, args.size, args.gray, device)
    agg = aggregate(per)

    tag = args.tag or f"{args.model}_s{args.size}{'g' if args.gray else 'rgb'}_e{args.epochs}"
    if args.loss != "ce_dice":          # keep baseline filenames unchanged
        tag += f"_{args.loss}"
    out = {
        "model": args.model, "seed": args.seed, "size": args.size, "gray": args.gray,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "loss": args.loss, "lambda_bd": args.lambda_bd,
        "params": n_par, "best_val_dice": best_val,
        "device": str(device), "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "agg": agg, "per_image": per, "history": hist,
    }
    os.makedirs(RESULTS, exist_ok=True)
    ckpt_path = os.path.join(RESULTS, f"{tag}_seed{args.seed}.pth")
    torch.save(model.state_dict(), ckpt_path)
    path = os.path.join(RESULTS, f"{tag}_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    if args.save_ckpt:
        ckpt_dir = os.path.join(RESULTS, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        ck = os.path.join(ckpt_dir, f"{tag}_seed{args.seed}.pt")
        torch.save({"model": args.model, "size": args.size, "gray": args.gray,
                    "loss": args.loss, "state_dict": model.state_dict()}, ck)
        print("wrote", ck)

    print("params:", f"{n_par/1e6:.3f}M")
    print("TEST  dice_lesion=%.4f  miou=%.4f  prec=%.4f  recall=%.4f  hd95=%.1f  asd=%.1f" %
          (agg["dice_lesion"], agg["miou_hist"], agg["precision"], agg["recall"], agg["hd95"], agg["asd"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
