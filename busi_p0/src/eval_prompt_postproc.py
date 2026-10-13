"""A-group (inference-side) optimisations for the PROMPT methods (SAM / MedSAM / BUSSAM).

Same machinery as eval_postproc.py (A1 largest-CC+fill+area, A2 hflip TTA,
A3 threshold chosen on the VAL lesion set), but for prompt-based models which
produce a foreground probability map instead of a supervised dense logit.

Usage:  python eval_prompt_postproc.py --method sam|medsam|bussam [--tta]
Writes: results/postproc_prompt_<method>.json
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy import ndimage

SRC = "/mnt/workspace/busi_p0/src"
BUSSAM = "/mnt/workspace/code/BUSSAM"
sys.path.insert(0, SRC)
from busi_common import (build_index, load_split, orig_size, load_mask, RESULTS,
                         LESION_CLASSES, dice_binary, per_image_metrics, postprocess_mask)  # noqa

SAM_CKPT = "/mnt/workspace/models/sam_vit_b/sam_vit_b_01ec64.pth"
MEDSAM_DIR = "/mnt/workspace/models/medsam_vit_b"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def center_point(gt):
    if gt.sum() == 0:
        h, w = gt.shape
        return np.array([w // 2, h // 2])
    dt = ndimage.distance_transform_edt(gt)
    y, x = np.unravel_index(np.argmax(dt), dt.shape)
    return np.array([x, y])


# --------------------------------------------------------------------------- #
def load_sam():
    sys.path.insert(0, "/mnt/workspace/code/segment-anything")
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
    sam.to(DEV).eval()
    return SamPredictor(sam)


def sam_probs(predictor, rec, gt, tta=False):
    img = np.array(Image.open(rec["image"]).convert("RGB"))
    w = img.shape[1]
    pt = center_point(gt)

    def one(im, p):
        predictor.set_image(im)
        _, _, low = predictor.predict(point_coords=p[None], point_labels=np.array([1]),
                                      multimask_output=False)
        lg = np.asarray(low)
        return lg[0] if lg.ndim == 3 else lg          # (256,256) logits, low-res frame
    lg = one(img, pt)
    if tta:
        lg = (lg + np.flip(one(np.flip(img, axis=1), np.array([w - 1 - pt[0], pt[1]])), axis=1)) / 2
    # IMPORTANT: undo SAM's square padding via the model's own postprocess_masks
    t = torch.as_tensor(lg, dtype=torch.float32)[None, None]      # (1,1,256,256)
    t = predictor.model.postprocess_masks(t, predictor.input_size, predictor.original_size)
    return torch.sigmoid(t)[0, 0].numpy()


def load_medsam():
    from transformers import SamModel, SamProcessor
    return SamModel.from_pretrained(MEDSAM_DIR).to(DEV).eval(), SamProcessor.from_pretrained(MEDSAM_DIR)


def medsam_probs(model, proc, rec, gt, tta=False):
    img = Image.open(rec["image"]).convert("RGB")
    w = img.size[0]
    pt = center_point(gt)

    def one(im, p):
        inp = proc(im, input_points=[[[float(p[0]), float(p[1])]]], return_tensors="pt")
        inp = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, multimask_output=False)
        m = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                    inp["reshaped_input_sizes"].cpu(), binarize=False)
        return m[0][0, 0].numpy()
    p = one(img, pt)
    if tta:
        p2 = one(img.transpose(Image.FLIP_LEFT_RIGHT), np.array([w - 1 - pt[0], pt[1]]))
        p = (p + np.flip(p2, axis=1)) / 2
    return p


def load_bussam():
    if BUSSAM not in sys.path:
        sys.path.insert(0, BUSSAM)
    import glob
    from utils.config import get_config
    from models.model_dict import get_model

    class A:
        encoder_input_size, low_image_size, vit_name = 256, 128, "vit_b"
        sam_ckpt = os.path.join(BUSSAM, "checkpoints", "sam_vit_b_01ec64.pth")
        n_gpu, modelname = 1, "BUSSAM"
    opt = get_config("BUSI"); opt.device = DEV
    model = get_model("BUSSAM", args=A(), opt=opt)
    cands = glob.glob(os.path.join(BUSSAM, "outputs", "BUSSAM_*", "checkpoints", "BUSSAM_*_*_*.pth"))
    ck = max(cands, key=lambda p: float(os.path.basename(p).split("_")[-1].replace(".pth", "")))
    sd = torch.load(ck, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd); model.to(DEV).eval()
    return model


def bussam_probs(model, rec, gt, tta=False):
    im = Image.open(rec["image"]).convert("L")
    g = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((256, 256), Image.NEAREST)) > 127
    if g.sum() == 0:
        pt = np.array([128., 128.], dtype=np.float32)
    else:
        dt = ndimage.distance_transform_edt(g)
        y, x = np.unravel_index(np.argmax(dt), dt.shape)
        pt = np.array([x, y], dtype=np.float32)

    def one(pil, p):
        x = to_tensor(pil.resize((256, 256), Image.BILINEAR))[None].to(DEV).float()
        pp = torch.as_tensor(p[None, None, :], dtype=torch.float32, device=DEV)
        ll = torch.as_tensor(np.array([[1]]), dtype=torch.int, device=DEV)
        with torch.no_grad():
            out = model(x, (pp, ll))
            return torch.sigmoid(out["masks"])[0, 0].cpu().numpy()
    pr = one(im, pt)
    if tta:
        pr2 = one(im.transpose(Image.FLIP_LEFT_RIGHT), np.array([255 - pt[0], pt[1]], dtype=np.float32))
        pr = (pr + np.flip(pr2, axis=1)) / 2
    w0, h0 = orig_size(rec)
    return F.interpolate(torch.from_numpy(pr)[None, None], size=(h0, w0),
                         mode="bilinear", align_corners=False)[0, 0].numpy()


# --------------------------------------------------------------------------- #
def search_threshold(probs, gts, use_pp, min_area, cands):
    best_t, best_d = 0.5, -1.0
    for t in cands:
        ds = []
        for pr, gt in zip(probs, gts):
            pred = pr > t
            if use_pp:
                pred = postprocess_mask(pred, min_area)
            ds.append(dice_binary(pred, gt))
        d = float(np.mean(ds))
        if d > best_d:
            best_t, best_d = float(t), d
    return best_t, best_d


def run_variant(probs, gts, cls, names, thr, use_pp, min_area):
    per = []
    for i in range(len(probs)):
        pred = probs[i] > thr
        if use_pp:
            pred = postprocess_mask(pred, min_area)
        m = per_image_metrics(pred, gts[i]); m["name"] = names[i]; m["cls"] = cls[i]
        per.append(m)
    tp = sum(x["tp"] for x in per); fp = sum(x["fp"] for x in per)
    fn = sum(x["fn"] for x in per); tn = sum(x["tn"] for x in per)
    iou_bg = tn / (tn + fp + 1e-7); iou_fg = tp / (tp + fn + fp + 1e-7)
    agg = {"n_lesion": len(per),
           "dice_lesion": float(np.mean([x["dice"] for x in per])),
           "iou_lesion": float(np.mean([x["iou"] for x in per])),
           "miou_hist": float((iou_bg + iou_fg) / 2),
           "precision": tp / (tp + fp + 1e-7), "recall": tp / (tp + fn + 1e-7),
           "specificity": tn / (tn + fp + 1e-7), "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-7),
           "hd95": float(np.mean([x["hd95"] for x in per])),
           "asd": float(np.mean([x["asd"] for x in per]))}
    return agg, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["sam", "medsam", "bussam"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--min-area", type=float, default=0.001)
    ap.add_argument("--steps", type=float, default=0.05)
    args = ap.parse_args()

    index = build_index(); byname = {it["name"]: it for it in index}
    split = load_split()
    rva = [byname[n] for n in split["val"] if byname[n]["cls"] in LESION_CLASSES]
    rte = [byname[n] for n in split["test"] if byname[n]["cls"] in LESION_CLASSES]
    print(f"[{args.method}] val lesion={len(rva)}  test lesion={len(rte)}")

    if args.method == "sam":
        obj = load_sam(); fn = lambda r, g, t: sam_probs(obj, r, g, t)      # noqa
    elif args.method == "medsam":
        m, p = load_medsam(); fn = lambda r, g, t: medsam_probs(m, p, r, g, t)  # noqa
    else:
        m = load_bussam(); fn = lambda r, g, t: bussam_probs(m, r, g, t)    # noqa

    def probs_gts(records, tta):
        P = [fn(r, load_mask(r), tta) for r in records]
        G = [load_mask(r) for r in records]
        C = [r["cls"] for r in records]; N = [r["name"] for r in records]
        return P, G, C, N

    print("val probs ...")
    vP, vG, vC, _ = probs_gts(rva, False)
    vPt = None
    if args.tta:
        print("val probs (TTA) ..."); vPt, _, _, _ = probs_gts(rva, True)
    cands = np.round(np.arange(args.steps, 1.0, args.steps), 4)
    thr_raw, dv_raw = search_threshold(vP, vG, False, args.min_area, cands)
    thr_pp, dv_pp = search_threshold(vP, vG, True, args.min_area, cands)
    thr_tta = thr_tta_pp = None
    if args.tta:
        thr_tta, _ = search_threshold(vPt, vG, False, args.min_area, cands)
        thr_tta_pp, _ = search_threshold(vPt, vG, True, args.min_area, cands)
    print(f"A3 thresholds (val): raw={thr_raw} ({dv_raw:.4f})  +pp={thr_pp} ({dv_pp:.4f})"
          + (f"  TTA raw={thr_tta}" if args.tta else ""))

    print("test probs ..."); P, G, C, N = probs_gts(rte, False)
    pt = None
    if args.tta:
        print("test probs (TTA) ..."); pt, _, _, _ = probs_gts(rte, True)

    variants = [("baseline", P, 0.5, False), ("A3_threshold", P, thr_raw, False),
                ("A1_postproc", P, 0.5, True), ("A1+A3", P, thr_pp, True)]
    if args.tta:
        variants += [("A2_tta", pt, 0.5, False), ("A2+A1", pt, 0.5, True),
                     ("A2+A3", pt, thr_tta, False), ("A2+A1+A3", pt, thr_tta_pp, True)]

    results = {"method": args.method, "tta": args.tta, "min_area": args.min_area,
               "n_val_lesion": len(rva), "n_test_lesion": len(rte),
               "val_thresholds": {"raw": thr_raw, "with_postproc": thr_pp,
                                  "tta_raw": thr_tta, "tta_with_postproc": thr_tta_pp},
               "variants": {}}
    print("\n%-14s %8s %8s %8s %8s %8s" % ("variant", "dice", "iou", "prec", "recall", "hd95"))
    for name, pr, thr, use_pp in variants:
        agg, per = run_variant(pr, G, C, N, thr, use_pp, args.min_area)
        results["variants"][name] = {"threshold": float(thr), "postproc": bool(use_pp),
                                     "agg": agg, "per_image": per}
        print("%-14s %8.4f %8.4f %8.4f %8.4f %8.1f" %
              (name, agg["dice_lesion"], agg["iou_lesion"], agg["precision"],
               agg["recall"], agg["hd95"]))

    out = os.path.join(RESULTS, f"postproc_prompt_{args.method}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
