"""P2: prompt-quality -> final Dice robustness curve.

For each test lesion image:
  * construct a box-quality gradient by scaling the GT box about its centre
    (s in SCALES) and by translating it, recording IoU(box, GT box);
  * also use an AUTO box derived from a trained supervised model's prediction;
  * feed each box to the prompt-based refiners (MedSAM / SAM) and record the
    final Dice against GT.
Aggregates (mean IoU, mean Dice) into results/prompt_quality.json and plots a
curve to results/prompt_quality.png.
"""
import os
import sys
import json
import importlib.util

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/mnt/workspace/busi_p0/src"
RES = "/mnt/workspace/busi_p0/results"
SAM_CKPT = "/mnt/workspace/models/sam_vit_b/sam_vit_b_01ec64.pth"
MEDSAM_DIR = "/mnt/workspace/models/medsam_vit_b"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, SRC)
from busi_common import build_index, load_split, orig_size, load_mask, dice_binary  # noqa

_spec = importlib.util.spec_from_file_location("busi_models2", os.path.join(SRC, "models.py"))
_bm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bm)

SCALES = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
SHIFTS = [0.0, 0.15]          # extra translation level (fraction of box size)


def gt_box(gt):
    ys, xs = np.where(gt)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def scale_box(box, s):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * s / 2, (box[3] - box[1]) * s / 2
    return np.array([cx - w, cy - h, cx + w, cy + h], dtype=np.float64)


def shift_box(box, frac, w0, h0):
    dx, dy = (box[2] - box[0]) * frac, (box[3] - box[1]) * frac
    b = box + np.array([dx, dy, dx, dy])
    b[0] = max(0, b[0]); b[1] = max(0, b[1])
    b[2] = min(w0 - 1, b[2]); b[3] = min(h0 - 1, b[3])
    return b


def box_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


# --------------------------- prompt models --------------------------- #
def load_sam():
    sys.path.insert(0, "/mnt/workspace/code/segment-anything")
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT); sam.to(DEV).eval()
    return SamPredictor(sam)


def sam_box(predictor, img, box):
    predictor.set_image(img)
    masks, _, _ = predictor.predict(box=box[None], multimask_output=False)
    return masks[0]


def load_medsam():
    from transformers import SamModel, SamProcessor
    return SamModel.from_pretrained(MEDSAM_DIR).to(DEV).eval(), SamProcessor.from_pretrained(MEDSAM_DIR)


def medsam_box(model, proc, img, box):
    inp = proc(img, input_boxes=[[[float(v) for v in box]]], return_tensors="pt")
    inp = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    m = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                inp["reshaped_input_sizes"].cpu())
    return m[0][0, 0].numpy() > 0


# --------------------------- auto box source --------------------------- #
def sup_box_fn(model_name, size, gray, ckpt):
    model = _bm.build_model(model_name, in_ch=1 if gray else 3, n_cls=2, size=size)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
    model.to(DEV).eval()

    def fn(rec):
        im = Image.open(rec["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
        if gray:
            a = a.mean(2, keepdims=True); a = (a - a.mean()) / (a.std() + 1e-8)
        else:
            a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-8)
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV).float()
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[-1]
        seg = out.argmax(1)[0].cpu().numpy().astype(np.uint8)
        w, h = orig_size(rec)
        seg = np.array(Image.fromarray(seg * 255).resize((w, h), Image.NEAREST)) > 127
        if seg.sum() == 0:
            return None
        lab, n = ndimage.label(seg)
        areas = np.bincount(lab.ravel()); areas[0] = 0
        cc = lab == int(areas.argmax())
        ys, xs = np.where(cc)
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
    return fn


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all", choices=["all", "medsam", "sam"])
    args = ap.parse_args()
    methods = ("medsam", "sam") if args.method == "all" else (args.method,)

    index = build_index(); byname = {it["name"]: it for it in index}
    split = load_split()
    recs = [byname[n] for n in split["test"] if byname[n]["cls"] in ("benign", "malignant")]
    print(f"test lesion: {len(recs)}")

    predictor = load_sam()
    mm, mp = load_medsam()

    def run(method, img, box):
        return medsam_box(mm, mp, img, box) if method == "medsam" else sam_box(predictor, img, box)

    out = {"scales": SCALES, "shifts": SHIFTS, "methods": {}}
    for method in methods:
        rows = []
        for rec in recs:
            gt = load_mask(rec)
            w0, h0 = orig_size(rec)
            gb = gt_box(gt)
            img = Image.open(rec["image"]).convert("RGB")
            img_np = np.array(img)
            for s in SCALES:
                b = scale_box(gb, s)
                pred = run(method, img_np if method == "sam" else img, b)
                rows.append({"kind": "scale", "level": s, "box_iou": box_iou(b, gb), "dice": dice_binary(pred, gt)})
            for sh in SHIFTS:
                if sh == 0.0:
                    continue
                b = shift_box(gb.copy(), sh, w0, h0)
                pred = run(method, img_np if method == "sam" else img, b)
                rows.append({"kind": "shift", "level": sh, "box_iou": box_iou(b, gb), "dice": dice_binary(pred, gt)})
        out["methods"][method] = rows
        print(f"done {method}: {len(rows)} runs")

    # auto boxes (U-Net / UNet++) -> medsam/sam
    auto = {}
    for tag, mname, size, gray, ck in [
            ("unet", "unet", 128, True, f"{RES}/p1_unet_seed0.pth"),
            ("unetpp", "unetpp", 256, False, f"{RES}/p1_unetpp_seed0.pth")]:
        if not os.path.exists(ck):
            print("skip auto box, no ckpt", ck); continue
        fn = sup_box_fn(mname, size, gray, ck)
        for method in methods:
            dices, ious, n = [], [], 0
            for rec in recs:
                b = fn(rec)
                if b is None:
                    continue
                gt = load_mask(rec)
                gb = gt_box(gt)
                img = Image.open(rec["image"]).convert("RGB")
                pred = run(method, np.array(img) if method == "sam" else img, b)
                dices.append(dice_binary(pred, gt)); ious.append(box_iou(b, gb)); n += 1
            auto[f"{method}_{tag}"] = {"n": n, "box_iou": float(np.mean(ious)), "dice": float(np.mean(dices))}
            print(f"auto {method} box from {tag}: IoU {np.mean(ious):.3f} -> dice {np.mean(dices):.4f}")
    out["auto"] = auto

    suffix = "" if len(methods) > 1 else f"_{methods[0]}"
    with open(f"{RES}/prompt_quality{suffix}.json", "w") as f:
        json.dump(out, f, indent=2)

    # plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, rows in out["methods"].items():
        xs, ys = [], []
        for s in SCALES:
            r = [x for x in rows if x["kind"] == "scale" and x["level"] == s]
            xs.append(np.mean([x["box_iou"] for x in r])); ys.append(np.mean([x["dice"] for x in r]))
        ax.plot(xs, ys, "-o", label=method)
    for k, v in auto.items():
        ax.scatter([v["box_iou"]], [v["dice"]], marker="*", s=140, label=f"{k} (auto)")
    ax.set_xlabel("box IoU vs GT box"); ax.set_ylabel("final Dice (lesion)")
    ax.set_title("Prompt quality vs final Dice"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{RES}/prompt_quality{suffix}.png", dpi=140)
    print("wrote", f"{RES}/prompt_quality{suffix}.json", "and .png")


if __name__ == "__main__":
    main()
