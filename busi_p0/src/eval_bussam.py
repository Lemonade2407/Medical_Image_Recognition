"""Evaluate a trained BUSSAM checkpoint on the frozen test set with the SAME
protocol/metrics as the other prompt-based methods.

Single GT-centre positive point (argmax of the GT distance transform, identical
rule to SAM/MedSAM), prediction thresholded at sigmoid>0.5, compared to GT at
ORIGINAL resolution. Writes results/prompt_bussam.json (per-image + aggregate).
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy import ndimage

BUSSAM = "/mnt/workspace/code/BUSSAM"
sys.path.insert(0, "/mnt/workspace/busi_p0/src")
from busi_common import build_index, load_split, orig_size, RESULTS, LESION_CLASSES, per_image_metrics  # noqa
sys.path.insert(0, BUSSAM)  # must precede src so `models`/`utils` resolve to BUSSAM's packages


def best_ckpt(outputs_dir):
    cands = glob.glob(os.path.join(outputs_dir, "BUSSAM_*", "checkpoints", "BUSSAM_*_*_*.pth"))
    if not cands:
        cands = glob.glob(os.path.join(outputs_dir, "BUSSAM_*", "checkpoints", "*.pth"))
    def score(p):
        try:
            return float(os.path.basename(p).split("_")[-1].replace(".pth", ""))
        except ValueError:
            return -1.0
    return max(cands, key=score)


def load_gt_orig(item):
    w, h = orig_size(item)
    m = np.zeros((h, w), dtype=bool)
    for mp in item["masks"]:
        arr = np.array(Image.open(mp).convert("L"))
        if arr.shape != (h, w):
            arr = np.array(Image.open(mp).convert("L").resize((w, h), Image.NEAREST))
        m |= (arr > 127)
    return m


def prep_image(item, size=256):
    im = Image.open(item["image"]).convert("L").resize((size, size), Image.BILINEAR)
    return to_tensor(im)  # (1,size,size) in [0,1]


def center_point_256(gt_orig, size=256):
    gt = np.array(Image.fromarray(gt_orig.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127
    if gt.sum() == 0:
        return np.array([size // 2, size // 2], dtype=np.float32)
    dt = ndimage.distance_transform_edt(gt)
    y, x = np.unravel_index(np.argmax(dt), dt.shape)
    return np.array([x, y], dtype=np.float32)  # (x, y)


def aggregate(per):
    def m(v):
        return float(np.mean(v)) if v else float("nan")
    tp = sum(x["tp"] for x in per); fp = sum(x["fp"] for x in per)
    fn = sum(x["fn"] for x in per); tn = sum(x["tn"] for x in per)
    iou_bg = tn / (tn + fp + fn + 1e-7); iou_fg = tp / (tp + fn + fp + 1e-7)
    return {
        "n_lesion": len(per),
        "dice_lesion": m([x["dice"] for x in per]),
        "dice_lesion_std": float(np.std([x["dice"] for x in per])) if per else float("nan"),
        "iou_lesion": m([x["iou"] for x in per]),
        "miou_hist": float((iou_bg + iou_fg) / 2),
        "precision": tp / (tp + fp + 1e-7), "recall": tp / (tp + fn + 1e-7),
        "specificity": tn / (tn + fp + 1e-7), "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-7),
        "hd95": m([x["hd95"] for x in per]), "asd": m([x["asd"] for x in per]),
        "dice_benign": m([x["dice"] for x in per if x["cls"] == "benign"]),
        "dice_malignant": m([x["dice"] for x in per if x["cls"] == "malignant"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--outputs", default=os.path.join(BUSSAM, "outputs"))
    args = ap.parse_args()

    from utils.config import get_config
    from models.model_dict import get_model
    opt = get_config("BUSI")
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    class A:
        encoder_input_size = 256
        low_image_size = 128
        vit_name = "vit_b"
        sam_ckpt = os.path.join(BUSSAM, "checkpoints", "sam_vit_b_01ec64.pth")
        n_gpu = 1
        modelname = "BUSSAM"
    a = A()

    ckpt = args.ckpt or best_ckpt(args.outputs)
    print("checkpoint:", ckpt)
    model = get_model("BUSSAM", args=a, opt=opt)
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(opt.device).eval()

    index = build_index()
    byname = {it["name"]: it for it in index}
    split = load_split()
    recs = [byname[n] for n in split["test"] if byname[n]["cls"] in LESION_CLASSES]

    per = []
    for i, it in enumerate(recs):
        gt = load_gt_orig(it)
        w, h = orig_size(it)
        x = prep_image(it, 256)[None].to(opt.device).float()
        pt = torch.as_tensor(center_point_256(gt)[None, None, :], dtype=torch.float32, device=opt.device)
        pl = torch.as_tensor(np.array([[1]]), dtype=torch.int, device=opt.device)
        with torch.no_grad():
            out = model(x, (pt, pl))
            prob = torch.sigmoid(out["masks"])[:, 0]  # (1,256,256)
        seg = (prob[0].cpu().numpy() > 0.5).astype(np.uint8)
        seg_o = np.array(Image.fromarray(seg * 255).resize((w, h), Image.NEAREST)) > 127

        mm = per_image_metrics(seg_o, gt)
        mm.update({"name": it["name"], "cls": it["cls"]})
        per.append(mm)
        if (i + 1) % 20 == 0:
            print(f"  bussam {i+1}/{len(recs)} running dice={np.mean([p['dice'] for p in per]):.4f}")

    agg = aggregate(per)
    out = {"method": "bussam", "prompt": "point", "seed": None, "size": 256, "gray": True,
           "device": opt.device, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "ckpt": ckpt, "agg": agg, "per_image": per}
    with open(os.path.join(RESULTS, "prompt_bussam.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("AGG:", json.dumps(agg, indent=2))
    print("wrote", os.path.join(RESULTS, "prompt_bussam.json"))


if __name__ == "__main__":
    main()
