"""Archive the P4 run: presence-gate (PGBF) + 150ep §2 diagnostics + T5 manifest.

Data-only layout (explanation docs live in docs/):
  archive/P4_pgbf_<date>/
    code/ results/ tables/ figs/ sha256sums.txt
Explanation: docs/BUSI-算法优化方案.md §5.4, docs/BUSI-查新与新意.md §四·补2
"""
import os
import sys
import json
import csv
import random
import shutil
import hashlib
import subprocess

import numpy as np

ROOT = "/mnt/workspace/busi_p0"
SRC, RES, DOCS = f"{ROOT}/src", f"{ROOT}/results", f"{ROOT}/docs"
DATE = subprocess.check_output(["date", "-I"]).decode().strip()
ARCH = f"{ROOT}/archive/P4_pgbf_{DATE}"

CODE = ["train_presence_clf.py", "diag_tier_prompt.py", "diag_zero_cost.py",
        "manifest_ckpts.py", "make_qualitative.py", "archive_p4.py"]
RES_FILES = ["presence_clf.json", "presence_clf_scores.json",
             "presence_clf_effb0.json", "presence_clf_effb0_scores.json",
             "tier_prompt_150ep_pgbf662.json", "tier_prompt_150ep_pgbf662.md",
             "tier_prompt_150ep_pgbf662.png",
             "tier_prompt_150ep_pgbf080.json", "tier_prompt_150ep_pgbf080.md",
             "tier_prompt_150ep_pgbf080.png",
             "diag_fn_decomp.md", "diag_fn_decomp.json",
             "diag_box_curve.json", "diag_box_curve.png",
             "CHECKPOINTS.md", "checkpoints.sha256", "env.json"]
LOGS = ["presence_clf.log", "presence_clf_effb0.log", "pgbf662.log", "pgbf080.log",
        "diag_zero_cost_150ep.log", "viz_150ep.log"]
FIGS = ["viz/qualitative_lesion.png", "viz/qualitative_normal.png"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    for d in ("code", "results", "logs", "tables", "figs"):
        os.makedirs(f"{ARCH}/{d}", exist_ok=True)

    for f in CODE:
        if os.path.exists(f"{SRC}/{f}"):
            shutil.copy2(f"{SRC}/{f}", f"{ARCH}/code/{f}")
    for f in RES_FILES:
        if os.path.exists(f"{RES}/{f}"):
            shutil.copy2(f"{RES}/{f}", f"{ARCH}/results/{f}")
    for f in LOGS:
        if os.path.exists(f"{RES}/{f}"):
            shutil.copy2(f"{RES}/{f}", f"{ARCH}/logs/{f}")
    for f in FIGS:
        if os.path.exists(f"{RES}/{f}"):
            shutil.copy2(f"{RES}/{f}", f"{ARCH}/figs/{os.path.basename(f)}")
    shutil.copy2(f"{ROOT}/splits/busi_seed0.json", f"{ARCH}/results/busi_seed0.json")

    # ---- gate ablation table ----
    rows = [["setting", "tau", "variant", "dice", "dice_std", "delta_vs_B0", "p_vs_B0",
             "nobox_rate", "normal_fg", "normal_spec", "normals_with_fg", "n_better", "n_worse"]]
    runs = [("gate off", "-", "tier_prompt_150ep"),
            ("gate on", "0.662", "tier_prompt_150ep_pgbf662"),
            ("gate on (oracle)", "0.800", "tier_prompt_150ep_pgbf080")]
    for tag, tau, stem in runs:
        p = f"{RES}/{stem}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        b0 = d["variants"]["B0_single"]["e2e_dice"]
        for v in ("B0_single", "FC_majority", "FC_intersect", "FC_select"):
            V = d["variants"][v]
            # `per_image` is only stored for FR_select, so derive the count from
            # normal_clean = fraction of normals with fg < 0.001.
            n_fg = (1.0 - V["normal_clean"]) * 20.0
            rows.append([tag, tau, v, f"{V['e2e_dice']:.4f}", f"{V['e2e_dice_std']:.4f}",
                         f"{V['e2e_dice']-b0:+.4f}", f"{V.get('p_vs_B0', 1.0):.4f}",
                         f"{V['nobox_rate']:.3f}", f"{V['normal_fg']:.4f}",
                         f"{V['normal_spec']:.4f}", f"{n_fg:.1f}",
                         str(V.get("n_better", 0)), str(V.get("n_worse", 0))])
    with open(f"{ARCH}/tables/gate_ablation.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # ---- T1 classifier table ----
    t1 = [["classifier", "arch", "size", "threshold", "thr_source", "blind_rate",
           "blind_hits", "blind_n", "normal_fp", "auc"]]
    for tag, stem in [("resnet18", "presence_clf"), ("effb0", "presence_clf_effb0")]:
        p = f"{RES}/{stem}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        t1.append([tag, d["arch"], str(d["size"]), f"{d['threshold']:.4f}",
                   d["threshold_source"], f"{d['aggregate']['blind_rate']['mean']:.3f}",
                   "/".join(str(x) for x in d["blind_totals"]["hits"]),
                   "/".join(str(x) for x in d["blind_totals"]["n"]),
                   f"{d['aggregate']['normal_fp']['mean']:.3f}",
                   f"{d['aggregate']['auc']['mean']:.3f}"])
    with open(f"{ARCH}/tables/t1_classifier.csv", "w", newline="") as f:
        csv.writer(f).writerows(t1)

    chk = []
    for root, _, files in sorted(os.walk(ARCH)):
        for f in sorted(files):
            if f == "sha256sums.txt":
                continue
            p = os.path.join(root, f)
            chk.append(f"{sha(p)}  {os.path.relpath(p, ARCH)}")
    open(f"{ARCH}/sha256sums.txt", "w").write("\n".join(chk) + "\n")

    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(ARCH) for f in fs)
    print(f"archived -> {ARCH}\nfiles: {len(chk)}  size: {tot/1e6:.2f} MB")


if __name__ == "__main__":
    main()
