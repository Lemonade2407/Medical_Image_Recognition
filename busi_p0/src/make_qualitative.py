"""Qualitative comparison figure: original image / GT mask / each method's prediction
on a chosen set of test images (unified protocol: GT-centre point prompt for prompt
methods, same test split).

Outputs (to results/viz/):
  qualitative_lesion.png   6 lesion test images (3 benign + 3 malignant) x 9 rows
  qualitative_normal.png   3 normal test images x (原图/GT + 4 supervised)

Run AFTER the supervised checkpoints (p1_*.pth) exist.
"""
import os
import sys
import json
import glob
import importlib.util

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/mnt/workspace/busi_p0/src"
BUSSAM = "/mnt/workspace/code/BUSSAM"
RESULTS = "/mnt/workspace/busi_p0/results"
OUT = os.path.join(RESULTS, "viz")
SAM_CKPT = "/mnt/workspace/models/sam_vit_b/sam_vit_b_01ec64.pth"
MEDSAM_DIR = "/mnt/workspace/models/medsam_vit_b"

sys.path.insert(0, SRC)
from busi_common import build_index, load_split, orig_size, load_mask  # noqa

# load our own model factory under a distinct name (avoid clashing with BUSSAM's `models`)
_spec = importlib.util.spec_from_file_location("busi_models", os.path.join(SRC, "models.py"))
_bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bm)
build_model = _bm.build_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
def prep_sup(rec, size, gray):
    im = Image.open(rec["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    if gray:
        a = a.mean(2, keepdims=True)
        a = (a - a.mean()) / (a.std() + 1e-8)
    else:
        a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-8)
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV).float()


def to_orig(pred_small, rec):
    w, h = orig_size(rec)
    return np.array(Image.fromarray((pred_small * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 127


def center_point(gt):
    if gt.sum() == 0:
        h, w = gt.shape
        return np.array([w // 2, h // 2])
    dt = ndimage.distance_transform_edt(gt)
    y, x = np.unravel_index(np.argmax(dt), dt.shape)
    return np.array([x, y])


def sup_predict(rec, model_name, size, gray, ckpt):
    model = build_model(model_name, in_ch=1 if gray else 3, n_cls=2, size=size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(DEV).eval()
    with torch.no_grad():
        out = model(prep_sup(rec, size, gray))
        if isinstance(out, (list, tuple)):
            out = out[-1]
    pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return to_orig(pred, rec)


# --------------------------------------------------------------------------- #
def load_sam():
    sys.path.insert(0, "/mnt/workspace/code/segment-anything")
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
    sam.to(DEV).eval()
    return SamPredictor(sam)


def sam_predict(predictor, rec, gt):
    img = np.array(Image.open(rec["image"]).convert("RGB"))
    pt = center_point(gt)
    predictor.set_image(img)
    masks, _, _ = predictor.predict(point_coords=pt[None], point_labels=np.array([1]), multimask_output=False)
    return masks[0]


def load_medsam():
    from transformers import SamModel, SamProcessor
    return SamModel.from_pretrained(MEDSAM_DIR).to(DEV).eval(), SamProcessor.from_pretrained(MEDSAM_DIR)


def medsam_predict(model, proc, rec, gt):
    img = Image.open(rec["image"]).convert("RGB")
    pt = center_point(gt)
    inp = proc(img, input_points=[[[float(pt[0]), float(pt[1])]]], return_tensors="pt")
    inp = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())
    return masks[0][0, 0].numpy() > 0


def load_bussam():
    if BUSSAM not in sys.path:
        sys.path.insert(0, BUSSAM)
    from utils.config import get_config
    from models.model_dict import get_model

    class A:
        encoder_input_size = 256
        low_image_size = 128
        vit_name = "vit_b"
        sam_ckpt = os.path.join(BUSSAM, "checkpoints", "sam_vit_b_01ec64.pth")
        n_gpu = 1
        modelname = "BUSSAM"
    opt = get_config("BUSI")
    opt.device = DEV
    model = get_model("BUSSAM", args=A(), opt=opt)
    cands = glob.glob(os.path.join(BUSSAM, "outputs", "BUSSAM_*", "checkpoints", "BUSSAM_*_*_*.pth"))
    ckpt = max(cands, key=lambda p: float(os.path.basename(p).split("_")[-1].replace(".pth", "")))
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(DEV).eval()
    return model


def bussam_predict(model, rec, gt):
    im = Image.open(rec["image"]).convert("L").resize((256, 256), Image.BILINEAR)
    x = to_tensor(im)[None].to(DEV).float()
    g = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((256, 256), Image.NEAREST)) > 127
    if g.sum() == 0:
        pt = np.array([128, 128], dtype=np.float32)
    else:
        dt = ndimage.distance_transform_edt(g)
        y, xx = np.unravel_index(np.argmax(dt), dt.shape)
        pt = np.array([xx, y], dtype=np.float32)
    p = torch.as_tensor(pt[None, None, :], dtype=torch.float32, device=DEV)
    l = torch.as_tensor(np.array([[1]]), dtype=torch.int, device=DEV)
    with torch.no_grad():
        out = model(x, (p, l))
        prob = torch.sigmoid(out["masks"])[:, 0]
    seg = (prob[0].cpu().numpy() > 0.5).astype(np.uint8)
    return to_orig(seg, rec)


# --------------------------------------------------------------------------- #
def panel(ax, rec, gt, pred=None, overlay="pred", title=""):
    img = np.array(Image.open(rec["image"]).convert("L"))
    rgb = np.stack([img] * 3, -1).astype(np.float32) / 255.0
    if overlay == "none":
        pass
    elif overlay == "gt":
        rgb[gt] = rgb[gt] * 0.4 + np.array([0.1, 0.9, 0.1]) * 0.6
    else:
        if gt.any():
            rgb[gt ^ ndimage.binary_erosion(gt)] = np.array([0.1, 0.9, 0.1])
        if pred is not None and pred.any():
            rgb[pred ^ ndimage.binary_erosion(pred)] = np.array([1.0, 0.2, 0.2])
    ax.imshow(np.clip(rgb, 0, 1), cmap=None)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def main():
    os.makedirs(OUT, exist_ok=True)
    index = build_index()
    byname = {it["name"]: it for it in index}
    split = load_split()

    # rank lesion test images by BUSSAM dice -> pick easy/med/hard per class
    ref = json.load(open(os.path.join(RESULTS, "prompt_bussam.json")))
    byc = {"benign": [], "malignant": []}
    for m in ref["per_image"]:
        if m["cls"] in byc:
            byc[m["cls"]].append((m["dice"], m["name"]))
    samples = []
    for cls in ("benign", "malignant"):
        lst = sorted(byc[cls])
        idx = [int(len(lst) * p) for p in (0.2, 0.5, 0.8)]
        samples += [byname[lst[i][1]] for i in idx]

    # supervised checkpoints
    sup = [("U-Net", "unet", 128, True), ("AttU-Net", "attunet", 128, True),
           ("UNet++", "unetpp", 256, False), ("TransUNet", "transunet", 256, False)]
    preds = {name: {} for name, *_ in sup}
    for name, model, size, gray in sup:
        ck = os.path.join(RESULTS, f"p1_{model}_seed0.pth")
        if not os.path.exists(ck):
            print("MISSING ckpt (skip)", ck); continue
        for rec in samples:
            preds[name][rec["name"]] = sup_predict(rec, model, size, gray, ck)
        print("done", name)

    # prompt methods
    predictor = load_sam()
    for rec in samples:
        gt = load_mask(rec)
        preds.setdefault("SAM", {})[rec["name"]] = sam_predict(predictor, rec, gt)
    print("done SAM")
    mm, mp = load_medsam()
    for rec in samples:
        gt = load_mask(rec)
        preds.setdefault("MedSAM", {})[rec["name"]] = medsam_predict(mm, mp, rec, gt)
    print("done MedSAM")
    bm = load_bussam()
    for rec in samples:
        gt = load_mask(rec)
        preds.setdefault("BUSSAM", {})[rec["name"]] = bussam_predict(bm, rec, gt)
    print("done BUSSAM")

    # figure 1: lesion samples
    rows = ["Input", "GT"] + [n for n, *_ in sup] + ["SAM", "MedSAM", "BUSSAM"]
    ncol = len(samples)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.0 * ncol, 2.0 * len(rows)))
    for j, rec in enumerate(samples):
        gt = load_mask(rec)
        for i, r in enumerate(rows):
            ax = axes[i][j]
            if r == "Input":
                panel(ax, rec, gt, overlay="none", title=(f"{rec['cls']}" if i == 0 else ""))
            elif r == "GT":
                panel(ax, rec, gt, overlay="gt")
            else:
                panel(ax, rec, gt, preds.get(r, {}).get(rec["name"]), overlay="pred")
            if j == 0:
                ax.set_ylabel(r, fontsize=8)
    fig.suptitle("BUSI qualitative comparison (green = GT, red = prediction; single-centre-point prompt)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p1 = os.path.join(OUT, "qualitative_lesion.png")
    fig.savefig(p1, dpi=130); plt.close(fig)
    print("wrote", p1)

    # figure 2: normal samples (false positives, supervised only)
    normals = [byname[n] for n in split["test"] if byname[n]["cls"] == "normal"][:3]
    rows2 = ["Input", "GT"] + [n for n, *_ in sup]
    fig, axes = plt.subplots(len(rows2), len(normals), figsize=(2.0 * len(normals), 2.0 * len(rows2)))
    for j, rec in enumerate(normals):
        gt = load_mask(rec)
        for i, r in enumerate(rows2):
            ax = axes[i][j]
            if r == "Input":
                panel(ax, rec, gt, overlay="none", title=("normal" if i == 0 else ""))
            elif r == "GT":
                panel(ax, rec, gt, overlay="gt")
            else:
                panel(ax, rec, gt, preds.get(r, {}).get(rec["name"]), overlay="pred")
            if j == 0:
                ax.set_ylabel(r, fontsize=8)
    fig.suptitle("BUSI normal cases (no lesion) - false-positive check (red = prediction)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p2 = os.path.join(OUT, "qualitative_normal.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
