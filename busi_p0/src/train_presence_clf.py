"""T1: image-level "lesion present / absent" classifier for the PGBF gate.

Trains a binary classifier on the frozen BUSI split and reports exactly the
three statistics the T2 decision needs:
  * normal test images  -> P(abnormal), false-positive rate
  * blind test lesions  -> P(abnormal), hit rate (per P3 seed)
  * other test lesions  -> P(abnormal), recall

"Blind" is read from a P3 result JSON (`per_seed.<k>.per_image[*].type == 全盲`),
so the classifier is evaluated on the same images the gate would have to save.

The operating threshold is selected on VAL (normal-FP <= 10%, maximise lesion
recall) and then applied unchanged to TEST, so no test-set tuning.

Outputs (in results/):
  <out>.json          per-seed metrics + aggregate + chosen threshold + pass
  <out>_scores.json   {image_name: mean P(abnormal)}  -> feed to
                      `diag_tier_prompt.py --presence`

Pass criterion (doc §5.1): blind hit >= 6/8 (>=75%) and normal FP <= 2/20.
"""
import os
import sys
import json
import random
import argparse

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

SRC = "/mnt/workspace/busi_p0/src"
RES = "/mnt/workspace/busi_p0/results"
sys.path.insert(0, SRC)
from busi_common import build_index, load_split, RESULTS  # noqa

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------- data ------------------------------- #
def load_input(item, size, gray):
    im = Image.open(item["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    if gray:
        a = a.mean(2, keepdims=True)
        a = (a - a.mean()) / (a.std() + 1e-8)
        a = np.repeat(a, 3, axis=2)
    else:
        a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-8)
    return a.transpose(2, 0, 1)


class BUSICls(Dataset):
    def __init__(self, items, size, gray):
        self.items = items
        self.size = size
        self.gray = gray

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = torch.from_numpy(load_input(it, self.size, self.gray)).float()
        y = 0 if it["cls"] == "normal" else 1
        return x, y


# ------------------------------- model ------------------------------- #
def build_model(arch, pretrained):
    if arch == "resnet18":
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = torchvision.models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(arch)
    return m


# ------------------------------- metrics ------------------------------- #
def auc_score(scores, labels):
    """Rank-based AUC (Mann-Whitney U); labels 0/1."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rate(scores, labels, thr):
    """(normal FP rate, lesion recall) at a fixed threshold."""
    s = np.asarray(scores); y = np.asarray(labels)
    fp = float(np.mean(s[y == 0] > thr)) if np.any(y == 0) else float("nan")
    rec = float(np.mean(s[y == 1] > thr)) if np.any(y == 1) else float("nan")
    return fp, rec


def select_threshold(val_scores, val_labels, fp_budget=0.10, grid=None):
    """Pick the threshold on VAL: FP <= budget, then max recall (ties -> higher thr)."""
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.05)
    cands = []
    for t in grid:
        fp, rec = rate(val_scores, val_labels, float(t))
        if fp == fp and fp <= fp_budget:
            cands.append((rec, float(t), fp))
    if not cands:  # no threshold meets the budget -> fall back to 0.5
        fp, rec = rate(val_scores, val_labels, 0.5)
        return 0.5, rec, fp
    rec, t, fp = max(cands, key=lambda x: (x[0], x[1]))
    return t, rec, fp


def blind_sets(tier_json):
    """{seed: set(image names of blind lesions)} from a P3 result file."""
    if not tier_json or not os.path.exists(tier_json):
        return {}
    d = json.load(open(tier_json))
    out = {}
    for seed, ps in d.get("per_seed", {}).items():
        out[int(seed)] = {p["name"] for p in ps.get("per_image", [])
                          if p.get("type") == "全盲" and p.get("cls") in ("benign", "malignant")}
    return out


# ------------------------------- train / eval ------------------------------- #
def train_one(seed, tr, va, te, args):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    dl_tr = DataLoader(BUSICls(tr, args.size, args.gray), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers, pin_memory=True)
    dl_va = DataLoader(BUSICls(va, args.size, args.gray), batch_size=args.batch,
                       shuffle=False, num_workers=args.workers, pin_memory=True)
    dl_te = DataLoader(BUSICls(te, args.size, args.gray), batch_size=args.batch,
                       shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, args.pretrained).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_acc, best_sd, best_ep, ran = -1.0, None, None, 0

    for ep in range(1, args.epochs + 1):
        ran = ep
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                pred = model(xb.to(DEV)).argmax(1).cpu()
                correct += int((pred == yb).sum()); total += len(yb)
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc, best_ep = acc, ep
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.patience and ep - best_ep >= args.patience:
            break

    model.load_state_dict(best_sd)
    model.eval()

    def probs(dl):
        out = []
        with torch.no_grad():
            for xb, _ in dl:
                out.append(F.softmax(model(xb.to(DEV)), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    return {"seed": seed, "best_val_acc": float(best_acc), "best_epoch": best_ep,
            "epochs_run": ran, "val_scores": probs(dl_va), "test_scores": probs(dl_te)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--gray", dest="gray", action="store_true", default=True)
    ap.add_argument("--rgb", dest="gray", action="store_false")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fp-budget", type=float, default=0.10,
                    help="val normal-FP budget used to pick the threshold")
    ap.add_argument("--thr", type=float, default=None,
                    help="fixed threshold; default = select on val")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--tier-json", default=f"{RESULTS}/tier_prompt_150ep.json")
    ap.add_argument("--out", default="presence_clf")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    index = {it["name"]: it for it in build_index()}
    split = load_split()
    tr = [index[n] for n in split["train"]]
    va = [index[n] for n in split["val"]]
    te = [index[n] for n in split["test"]]
    blind = blind_sets(args.tier_json)
    if not blind:
        print("WARN: no blind sets found in", args.tier_json)

    runs = []
    for s in seeds:
        r = train_one(s, tr, va, te, args)
        print(f"seed {s}: val_acc {r['best_val_acc']:.4f}  "
              f"(ep {r['best_epoch']}/{r['epochs_run']})", flush=True)
        runs.append(r)

    val_labels = np.array([0 if it["cls"] == "normal" else 1 for it in va])
    if args.thr is None:
        pooled_val = np.mean([r["val_scores"] for r in runs], axis=0)
        thr, v_rec, v_fp = select_threshold(pooled_val, val_labels, args.fp_budget)
        thr_src = f"val(fp<={args.fp_budget:.0%}, max recall)"
    else:
        thr, thr_src = args.thr, "fixed"
        v_fp, v_rec = rate(np.mean([r["val_scores"] for r in runs], axis=0), val_labels, thr)

    names = [it["name"] for it in te]
    test_labels = np.array([0 if it["cls"] == "normal" else 1 for it in te])

    per_seed = {}
    for r in runs:
        s = r["seed"]
        fp, rec = rate(r["test_scores"], test_labels, thr)
        p = dict(zip(names, r["test_scores"].tolist()))
        bs = blind.get(s, set())
        hits = sum(1 for n in bs if p.get(n, 0.0) > thr)
        per_seed[str(s)] = {
            "best_val_acc": r["best_val_acc"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"],
            "val_scores": r["val_scores"].tolist(),
            "auc": auc_score(r["test_scores"], test_labels),
            "normal_fp": fp, "lesion_recall": rec,
            "blind": {"n": len(bs), "hits": int(hits),
                      "rate": float(hits / len(bs)) if bs else float("nan"),
                      "missed": sorted(n for n in bs if p.get(n, 0.0) <= thr)},
        }

    def agg(key, sub=None):
        v = [d[key] if sub is None else d[key][sub] for d in per_seed.values()]
        v = [x for x in v if x == x]
        return {"mean": float(np.mean(v)) if v else None,
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}

    hits = [d["blind"]["hits"] for d in per_seed.values()]
    ns = [d["blind"]["n"] for d in per_seed.values()]
    mean_rate = float(np.mean([h / n for h, n in zip(hits, ns) if n])) if any(ns) else float("nan")
    fp_mean = agg("normal_fp")["mean"]

    summary = {
        "arch": args.arch, "size": args.size, "gray": bool(args.gray),
        "seeds": seeds, "threshold": thr, "threshold_source": thr_src,
        "val_names": [it["name"] for it in va],
        "val": {"normal_fp": float(v_fp), "lesion_recall": float(v_rec),
                "fp_budget": args.fp_budget},
        "per_seed": per_seed,
        "aggregate": {"auc": agg("auc"), "normal_fp": agg("normal_fp"),
                      "lesion_recall": agg("lesion_recall"),
                      "blind_rate": agg("blind", "rate")},
        "blind_totals": {"hits": hits, "n": ns},
    }
    summary["pass"] = bool(mean_rate >= 0.75 and fp_mean is not None and fp_mean <= 0.10)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{args.out}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    mean_scores = {n: float(np.mean([dict(zip(names, r["test_scores"].tolist()))[n]
                                     for r in runs])) for n in names}
    with open(os.path.join(RESULTS, f"{args.out}_scores.json"), "w") as f:
        json.dump({"threshold": thr, "scores": mean_scores}, f, indent=2, ensure_ascii=False)

    print(f"\nthreshold {thr:.2f} [{thr_src}]  val FP {v_fp:.3f} / rec {v_rec:.3f}")
    print("blind hits per seed:", list(zip(hits, ns)), f"mean rate {mean_rate:.3f}")
    print("test normal FP:", summary["aggregate"]["normal_fp"])
    print("PASS" if summary["pass"] else "FAIL", "->", f"results/{args.out}.json")
    print("gate input ->", f"results/{args.out}_scores.json")


if __name__ == "__main__":
    main()
