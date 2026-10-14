"""Inference-side optimisations: A1 post-processing, A2 TTA, A3 threshold search.

These are reported as a UNIFIED ADD-ON on top of the frozen main table -- the
protocol keeps post-processing out of the main results, so this script writes
its own JSON and never touches the baseline numbers.

Usage:
  # 1) train once and keep the weights
  python train_supervised.py --model unet --size 128 --gray --epochs 24 --seed 0 --save-ckpt
  # 2) evaluate the A-group variants on top of those weights
  python eval_postproc.py --ckpt results/ckpt/unet_s128g_e24_seed0.pt \
                          --model unet --size 128 --gray --tta

Writes results/postproc_<tag>_seed<k>.json
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from busi_common import (build_index, load_split, orig_size, load_mask, RESULTS,
                         LESION_CLASSES, dice_binary, per_image_metrics,
                         postprocess_mask)
from models import build_model
from train_supervised import aggregate, predict_logits


@torch.no_grad()
def prob_map_original_res(model, item, size, gray, device, tta=False,
                          scales=(0.75, 1.0, 1.25)):
    """Foreground probability at ORIGINAL resolution (A2 is the `tta` branch)."""
    if gray:
        im = Image.open(item["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32).mean(2, keepdims=True)
        a = (a - a.mean()) / (a.std() + 1e-8)
    else:
        im = Image.open(item["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
        a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-8)

    x = torch.from_numpy(a).permute(2, 0, 1)[None].to(device)
    hi, wi = x.shape[-2:]

    if not tta:
        prob = F.softmax(predict_logits(model, x), 1)[:, 1]           # (1, hi, wi)
    else:
        acc, n = None, 0
        for s in scales:
            xs = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False) \
                if s != 1.0 else x
            for flip in (False, True):
                xin = torch.flip(xs, dims=[3]) if flip else xs        # hflip (width)
                p = F.softmax(predict_logits(model, xin), 1)[:, 1]
                if flip:
                    p = torch.flip(p, dims=[2])
                p = F.interpolate(p[:, None], size=(hi, wi),
                                  mode="bilinear", align_corners=False)[:, 0]
                acc = p if acc is None else acc + p
                n += 1
        prob = acc / n

    w, h = orig_size(item)
    prob_o = F.interpolate(prob[:, None], size=(h, w),
                           mode="bilinear", align_corners=False)[0, 0]
    return prob_o.cpu().numpy()


def search_threshold(probs, gts, cls, use_pp, min_area, candidates):
    """A3: pick the threshold that maximises lesion-only mean Dice on the VAL set."""
    lesion = [i for i, c in enumerate(cls) if c in LESION_CLASSES]
    best_t, best_d = 0.5, -1.0
    for t in candidates:
        ds = []
        for i in lesion:
            pred = probs[i] > t
            if use_pp:
                pred = postprocess_mask(pred, min_area)
            ds.append(dice_binary(pred, gts[i]))
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
        m = per_image_metrics(pred, gts[i])
        m["name"] = names[i]
        m["cls"] = cls[i]
        per.append(m)
    return aggregate(per), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--tta", action="store_true", help="also evaluate A2 (multi-scale + hflip)")
    ap.add_argument("--scales", default="0.75,1.0,1.25",
                    help="TTA scales; use 1.0 (flip-only) for fixed-pos-embed models like TransUNet")
    ap.add_argument("--min-area", type=float, default=0.001, help="A1 area gate (fraction)")
    ap.add_argument("--steps", type=float, default=0.05)
    ap.add_argument("--out-tag", default="", help="override the output tag (e.g. p1_unet_tversky)")
    args = ap.parse_args()
    args.gray = args.gray or not args.rgb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 1 if args.gray else 3
    model = build_model(args.model, in_ch=in_ch, n_cls=2, size=args.size).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    model.eval()
    loss_name = sd.get("loss", "ce_dice") if isinstance(sd, dict) else "ce_dice"
    print(f"loaded {args.ckpt}  (loss={loss_name})")

    index = build_index()
    byname = {it["name"]: it for it in index}
    split = load_split()
    rva = [byname[n] for n in split["val"]]
    rte = [byname[n] for n in split["test"]]

    scales = tuple(float(x) for x in str(args.scales).split(",") if x.strip())

    def probs_gts(records, tta):
        P = [prob_map_original_res(model, it, args.size, args.gray, device, tta=tta, scales=scales)
             for it in records]
        G = [load_mask(it) for it in records]
        C = [it["cls"] for it in records]
        N = [it["name"] for it in records]
        return P, G, C, N

    print(f"val  n={len(rva)}  computing probabilities ...")
    vP, vG, vC, _ = probs_gts(rva, tta=False)
    vPt = None
    if args.tta:
        vPt, _, _, _ = probs_gts(rva, tta=True)

    cands = np.round(np.arange(args.steps, 1.0, args.steps), 4)
    thr_raw, dv_raw = search_threshold(vP, vG, vC, False, args.min_area, cands)
    thr_pp, dv_pp = search_threshold(vP, vG, vC, True, args.min_area, cands)
    thr_tta, thr_tta_pp, dv_tta = None, None, None
    if args.tta:
        thr_tta, dv_tta = search_threshold(vPt, vG, vC, False, args.min_area, cands)
        thr_tta_pp, _ = search_threshold(vPt, vG, vC, True, args.min_area, cands)
    print(f"A3 thresholds (val): raw={thr_raw} (dice {dv_raw:.4f}), "
          f"with-postproc={thr_pp} (dice {dv_pp:.4f})")
    if args.tta:
        print(f"A3 thresholds (val, TTA): raw={thr_tta}, with-postproc={thr_tta_pp}")

    print(f"test n={len(rte)}  computing probabilities ...")
    P, G, C, N = probs_gts(rte, tta=False)
    Pt = None
    if args.tta:
        Pt, _, _, _ = probs_gts(rte, tta=True)

    variants = [
        ("baseline",        P,  0.5,     False),
        ("A3_threshold",    P,  thr_raw, False),
        ("A1_postproc",     P,  0.5,     True),
        ("A1+A3",           P,  thr_pp,  True),
    ]
    if args.tta:
        variants += [
            ("A2_tta",      Pt, 0.5,         False),
            ("A2+A1",       Pt, 0.5,         True),
            ("A2+A3",       Pt, thr_tta,     False),
            ("A2+A1+A3",    Pt, thr_tta_pp,  True),
        ]

    results = {"ckpt": os.path.basename(args.ckpt), "model": args.model, "seed": args.seed,
               "size": args.size, "gray": args.gray, "tta": args.tta,
               "min_area": args.min_area,
               "val_thresholds": {"raw": thr_raw, "with_postproc": thr_pp,
                                  "tta_raw": thr_tta, "tta_with_postproc": thr_tta_pp},
               "val_dice_at_threshold": {"raw": dv_raw, "with_postproc": dv_pp, "tta": dv_tta},
               "variants": {}}

    print("\n%-14s %8s %8s %8s %8s %7s %7s" %
          ("variant", "dice", "iou", "prec", "recall", "hd95", "asd"))
    for name, pr, thr, use_pp in variants:
        agg, per = run_variant(pr, G, C, N, thr, use_pp, args.min_area)
        results["variants"][name] = {"threshold": float(thr), "postproc": bool(use_pp),
                                     "agg": agg, "per_image": per}
        print("%-14s %8.4f %8.4f %8.4f %8.4f %7.1f %7.1f" %
              (name, agg["dice_lesion"], agg["iou_lesion"], agg["precision"],
               agg["recall"], agg["hd95"], agg["asd"]))

    tag = args.out_tag or f"postproc_{args.model}_s{args.size}{'g' if args.gray else 'rgb'}"
    if not args.out_tag and loss_name != "ce_dice":
        tag += f"_{loss_name}"
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"{tag}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
