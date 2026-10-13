"""Shared utilities for the BUSI unified-protocol study (P0-1/P0-2).

Single source of truth for: dataset index, mask merging, image preprocessing,
metric definitions, and split IO. Every method (supervised and prompt-based)
must import its metrics and split from here so the evaluation protocol is
literally identical across methods.
"""
import os
import glob
import json
import numpy as np
import torch
from PIL import Image

ROOT = "/mnt/workspace/busi_p0"
DATA = os.path.join(ROOT, "data/Dataset_BUSI_with_GT")
SPLITS = os.path.join(ROOT, "splits")
RESULTS = os.path.join(ROOT, "results")
CLASSES = ["benign", "malignant", "normal"]
LESION_CLASSES = ["benign", "malignant"]


# --------------------------------------------------------------------------- #
# Dataset index
# --------------------------------------------------------------------------- #
def build_index():
    """Return one record per image: class, image path, its masks (merged later)."""
    items = []
    for cls in CLASSES:
        for p in sorted(glob.glob(os.path.join(DATA, cls, "*.png"))):
            base = os.path.basename(p)
            if "_mask" in base:
                continue
            stem = base[:-4]
            masks = sorted(glob.glob(os.path.join(DATA, cls, stem + "_mask*.png")))
            items.append({"cls": cls, "name": f"{cls}/{base}",
                          "image": p, "masks": masks})
    return items


def orig_size(item):
    with Image.open(item["image"]) as im:
        return im.size  # (w, h)


def load_mask(item):
    """Binary lesion mask at ORIGINAL resolution, all lesion sub-masks OR-ed."""
    w, h = orig_size(item)
    m = np.zeros((h, w), dtype=bool)
    for mp in item["masks"]:
        arr = np.array(Image.open(mp).convert("L"))
        if arr.shape != (h, w):
            arr = np.array(Image.open(mp).convert("L").resize((w, h), Image.NEAREST))
        m |= (arr > 127)
    return m


def load_image_gray(item, size):
    """Grayscale (or RGB-replicated) standardized image as float32 HxWxC."""
    im = Image.open(item["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    a = a.mean(axis=2, keepdims=True)  # grayscale
    a = (a - a.mean()) / (a.std() + 1e-8)  # per-image zero-mean unit-var
    return a


def load_image_rgb(item, size):
    im = Image.open(item["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    a = (a - a.mean(axis=(0, 1), keepdims=True)) / (a.std(axis=(0, 1), keepdims=True) + 1e-8)
    return a


# --------------------------------------------------------------------------- #
# Metrics (single definition reused by every method)
# --------------------------------------------------------------------------- #
def _safe_div(a, b, eps=1e-7):
    return float(a) / float(b + eps)


def dice_binary(pred, gt, eps=1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter + eps) / (pred.sum() + gt.sum() + eps))


def iou_binary(pred, gt, eps=1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def confusion(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def _surface_distances(pred, gt):
    """Symmetric surface (boundary) distances in pixels (2D, original resolution)."""
    from scipy import ndimage
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return None
    if pred.sum() == 0 or gt.sum() == 0:
        # one empty -> maximal penalty proportional to image diagonal
        return np.array([np.hypot(*pred.shape)])
    pred_b = pred ^ ndimage.binary_erosion(pred)
    gt_b = gt ^ ndimage.binary_erosion(gt)
    dt_gt = ndimage.distance_transform_edt(~gt_b)
    dt_pred = ndimage.distance_transform_edt(~pred_b)
    d_pred_to_gt = dt_gt[pred_b]
    d_gt_to_pred = dt_pred[gt_b]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def hd95_asd(pred, gt):
    d = _surface_distances(pred, gt)
    if d is None:
        return 0.0, 0.0
    return float(np.percentile(d, 95)), float(d.mean())


# --------------------------------------------------------------------------- #
# Inference-side add-on (A1): post-processing
# --------------------------------------------------------------------------- #
def postprocess_mask(pred, min_area_ratio=0.001):
    """A1: largest connected component + hole filling + area gate.

    Keeps the biggest connected component, fills its interior holes, and drops
    the result entirely if it covers less than `min_area_ratio` of the image.

    Rationale: stray speckle blobs far from the lesion dominate the 95th
    percentile of the surface distance (HD95/ASD), while tiny specks on normal
    images drive the false-positive rate. This is an INFERENCE-SIDE add-on and
    is reported separately from the main (no-post-processing) table.
    """
    from scipy import ndimage
    pred = np.asarray(pred).astype(bool)
    if not pred.any():
        return pred
    filled = ndimage.binary_fill_holes(pred)
    labels, n = ndimage.label(filled)
    if n == 0:
        return np.zeros_like(pred)
    areas = np.bincount(labels.ravel())
    areas[0] = 0  # background
    keep = labels == int(areas.argmax())
    if keep.sum() < max(1, int(min_area_ratio * pred.size)):
        return np.zeros_like(pred)
    return keep


def per_image_metrics(pred, gt):
    """All metrics for one image at original resolution. pred/gt: HxW boolean."""
    tp, fp, fn, tn = confusion(pred, gt)
    d = dice_binary(pred, gt)
    hd, asd = hd95_asd(pred, gt)
    return {
        "dice": d,
        "iou": iou_binary(pred, gt),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "hd95": hd,
        "asd": asd,
        "fg_ratio": float(pred.mean()),  # predicted foreground fraction
        "gt_empty": bool(gt.sum() == 0),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# --------------------------------------------------------------------------- #
# Split IO
# --------------------------------------------------------------------------- #
def load_split(path=None):
    path = path or os.path.join(SPLITS, "busi_seed0.json")
    with open(path) as f:
        return json.load(f)
