"""P0-1: build the single fixed stratified split shared by every method.

Produces splits/busi_seed0.json with train/val/test name lists, stratified by
class (benign/malignant/normal) at 70/15/15. Allocation uses the largest-
remainder method with a val-before-test tie-break, which reproduces the
546/118/116 (test 65/31/20) composition of the original report exactly.
"""
import os
import sys
import json
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from busi_common import build_index, SPLITS, CLASSES

SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def allocate(n, ratios=RATIOS):
    raw = [n * r for r in ratios]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    order = sorted(range(3), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    return base  # train, val, test


def main():
    index = build_index()
    by = defaultdict(list)
    for it in index:
        by[it["cls"]].append(it["name"])

    rng = random.Random(SEED)
    train, val, test = [], [], []
    per_class = {}
    for cls in CLASSES:
        names = sorted(by[cls])
        rng.shuffle(names)
        n_tr, n_va, n_te = allocate(len(names))
        test += names[:n_te]
        val += names[n_te:n_te + n_va]
        train += names[n_te + n_va:]
        per_class[cls] = {"train": n_tr, "val": n_va, "test": n_te, "total": len(names)}

    split = {
        "seed": SEED, "ratios": RATIOS, "classes": CLASSES,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "per_class": per_class,
        "train": sorted(train), "val": sorted(val), "test": sorted(test),
    }
    os.makedirs(SPLITS, exist_ok=True)
    out = os.path.join(SPLITS, "busi_seed0.json")
    with open(out, "w") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)

    print("wrote", out)
    print("counts:", split["counts"])
    for c in CLASSES:
        print(f"  {c:10s} {per_class[c]}")
    # sanity: no overlap, complete cover
    s = set(train) | set(val) | set(test)
    assert len(train) + len(val) + len(test) == len(index) == len(s), "split not a partition"
    print("partition OK:", len(s), "unique names")


if __name__ == "__main__":
    main()
