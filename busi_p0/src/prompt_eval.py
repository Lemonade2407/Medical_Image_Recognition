"""Zero-shot point-prompt inference for SAM ViT-B and MedSAM ViT-B on the unified test set.

Prompt = one positive point at the GT-foreground "center" (argmax of the distance
transform of the GT mask), matching the report's main prompt protocol.

  python prompt_eval.py --method sam    [--limit N]
  python prompt_eval.py --method medsam [--limit N]

Writes results/prompt_<method>.json (per-image + aggregate metrics).
"""
import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/workspace/code/segment-anything")
from busi_common import build_index, load_split, orig_size, RESULTS, LESION_CLASSES, per_image_metrics

SAM_CKPT = "/mnt/workspace/models/sam_vit_b/sam_vit_b_01ec64.pth"
MEDSAM_DIR = "/mnt/workspace/models/medsam_vit_b"


def center_point(gt):
    """Point inside the GT lesion farthest from the background (robust center)."""
    if gt.sum() == 0:
        h, w = gt.shape
        return np.array([w // 2, h // 2])
    dt = ndimage.distance_transform_edt(gt)
    y, x = np.unravel_index(np.argmax(dt), dt.shape)
    return np.array([x, y])  # SAM wants (x, y)


def test_records(limit=None):
    index = build_index()
    byname = {it["name"]: it for it in index}
    split = load_split()
    recs = [byname[n] for n in split["test"] if byname[n]["cls"] in LESION_CLASSES]
    return recs[:limit] if limit else recs


def load_gt(item):
    w, h = orig_size(item)
    m = np.zeros((h, w), dtype=bool)
    for mp in item["masks"]:
        arr = np.array(Image.open(mp).convert("L"))
        if arr.shape != (h, w):
            arr = np.array(Image.open(mp).convert("L").resize((w, h), Image.NEAREST))
        m |= (arr > 127)
    return m


def gt_box(gt):
    """[x0,y0,x1,y1] bounding box of the GT lesion."""
    ys, xs = np.where(gt)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


# --------------------------------------------------------------------------- #
def run_sam(records, prompt="point"):
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
    sam.to(device)
    sam.eval()
    pred = SamPredictor(sam)
    out = []
    for i, it in enumerate(records):
        img = np.array(Image.open(it["image"]).convert("RGB"))
        gt = load_gt(it)
        pred.set_image(img)
        if prompt == "box":
            masks, scores, _ = pred.predict(box=gt_box(gt)[None], multimask_output=False)
        else:
            pt = center_point(gt)
            masks, scores, _ = pred.predict(point_coords=pt[None], point_labels=np.array([1]),
                                            multimask_output=False)
        m = masks[0]
        mm = per_image_metrics(m, gt)
        mm.update({"name": it["name"], "cls": it["cls"]})
        out.append(mm)
        if (i + 1) % 10 == 0:
            print(f"  sam {i+1}/{len(records)} running dice={np.mean([o['dice'] for o in out]):.4f}")
    return out


def run_medsam(records, prompt="point"):
    import torch
    from transformers import SamModel, SamProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained(MEDSAM_DIR).to(device).eval()
    proc = SamProcessor.from_pretrained(MEDSAM_DIR)
    out = []
    for i, it in enumerate(records):
        img = Image.open(it["image"]).convert("RGB")
        gt = load_gt(it)
        if prompt == "box":
            bx = gt_box(gt)
            inputs = proc(img, input_boxes=[[[float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]]],
                          return_tensors="pt")
        else:
            pt = center_point(gt)
            inputs = proc(img, input_points=[[[float(pt[0]), float(pt[1])]]], return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        masks = proc.post_process_masks(outputs.pred_masks.cpu(),
                                        inputs["original_sizes"].cpu(),
                                        inputs["reshaped_input_sizes"].cpu())
        m = (masks[0][0, 0].numpy() > 0.0)
        mm = per_image_metrics(m, gt)
        mm.update({"name": it["name"], "cls": it["cls"]})
        out.append(mm)
        if (i + 1) % 10 == 0:
            print(f"  medsam {i+1}/{len(records)} running dice={np.mean([o['dice'] for o in out]):.4f}")
    return out


def aggregate(per):
    def m(vals):
        return float(np.mean(vals)) if vals else float("nan")
    tp = sum(x["tp"] for x in per); fp = sum(x["fp"] for x in per)
    fn = sum(x["fn"] for x in per); tn = sum(x["tn"] for x in per)
    iou_bg = tn / (tn + fp + fn + 1e-7); iou_fg = tp / (tp + fn + fp + 1e-7)
    return {
        "n_lesion": len(per),
        "dice_lesion": m([x["dice"] for x in per]),
        "dice_lesion_std": float(np.std([x["dice"] for x in per])) if per else float("nan"),
        "iou_lesion": m([x["iou"] for x in per]),
        "miou_hist": float((iou_bg + iou_fg) / 2),
        "precision": tp / (tp + fp + 1e-7),
        "recall": tp / (tp + fn + 1e-7),
        "specificity": tn / (tn + fp + 1e-7),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "hd95": m([x["hd95"] for x in per]),
        "asd": m([x["asd"] for x in per]),
        "dice_benign": m([x["dice"] for x in per if x["cls"] == "benign"]),
        "dice_malignant": m([x["dice"] for x in per if x["cls"] == "malignant"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["sam", "medsam"])
    ap.add_argument("--prompt", default="point", choices=["point", "box"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    recs = test_records(args.limit or None)
    print(f"[{args.method}/{args.prompt}] lesion test images: {len(recs)}")
    per = run_sam(recs, args.prompt) if args.method == "sam" else run_medsam(recs, args.prompt)
    agg = aggregate(per)
    import torch as _t
    out = {"method": args.method, "prompt": args.prompt, "seed": None, "size": 1024, "gray": False,
           "device": "cuda" if _t.cuda.is_available() else "cpu",
           "torch": _t.__version__, "gpu": _t.cuda.get_device_name(0) if _t.cuda.is_available() else None,
           "agg": agg, "per_image": per}
    os.makedirs(RESULTS, exist_ok=True)
    suffix = "" if args.prompt == "point" else f"_{args.prompt}"
    path = os.path.join(RESULTS, f"prompt_{args.method}{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("AGG:", json.dumps(agg, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
