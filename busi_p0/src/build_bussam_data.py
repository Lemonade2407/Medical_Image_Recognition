"""Convert our BUSI + frozen split into BUSSAM's expected dataset layout.

Produces (under code/BUSSAM/datasets):
  BUSI/img/<flat>.png         grayscale image
  BUSI/label/<flat>.png       binary mask (0/255)
  MainPatient/BUSI_train.txt  lines: "1/BUSI/<flat>"
  MainPatient/BUSI_val.txt    lines: "1/BUSI/<flat>"
  MainPatient/BUSI_test.txt   lines: "BUSI/<flat>"
  MainPatient/class.json      {"BUSI": 2}

The split comes from the frozen splits/busi_seed0.json, so BUSSAM is evaluated
on exactly the same test images as the other six methods.
"""
import os
import re
import json
import shutil
import numpy as np
from PIL import Image

ROOT = "/mnt/workspace/busi_p0"
BUSSAM = "/mnt/workspace/code/BUSSAM"
DS = os.path.join(BUSSAM, "datasets")
sys_path = os.path.join(ROOT, "src")
import sys
sys.path.insert(0, sys_path)
from busi_common import build_index, load_split, load_mask, orig_size  # noqa


def flat_id(name):
    cls = name.split("/")[0]
    base = os.path.basename(name)[:-4]
    m = re.search(r"\((\d+)\)", base)
    return f"{cls}_{m.group(1)}" if m else f"{cls}_{base}"


def main():
    index = build_index()
    split = load_split()
    byname = {it["name"]: it for it in index}

    img_dir = os.path.join(DS, "BUSI", "img")
    lab_dir = os.path.join(DS, "BUSI", "label")
    mp = os.path.join(DS, "MainPatient")
    for d in (img_dir, lab_dir, mp):
        os.makedirs(d, exist_ok=True)

    # copy images/masks
    for it in index:
        fid = flat_id(it["name"])
        Image.open(it["image"]).convert("L").save(os.path.join(img_dir, fid + ".png"))
        mask = (load_mask(it).astype(np.uint8)) * 255
        Image.fromarray(mask).save(os.path.join(lab_dir, fid + ".png"))

    def lines(names, prefix):
        return "\n".join(f"{prefix}/{flat_id(n)}" for n in names) + "\n"

    with open(os.path.join(mp, "BUSI_train.txt"), "w") as f:
        f.write(lines(split["train"], "1/BUSI"))
    with open(os.path.join(mp, "BUSI_val.txt"), "w") as f:
        f.write(lines(split["val"], "1/BUSI"))
    with open(os.path.join(mp, "BUSI_test.txt"), "w") as f:
        f.write(lines(split["test"], "BUSI"))

    with open(os.path.join(mp, "class.json"), "w") as f:
        json.dump({"BUSI": 2}, f)

    # base SAM checkpoint where BUSSAM expects it
    ckpt_dir = os.path.join(BUSSAM, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    dst = os.path.join(ckpt_dir, "sam_vit_b_01ec64.pth")
    src = "/mnt/workspace/models/sam_vit_b/sam_vit_b_01ec64.pth"
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy(src, dst)

    print("images:", len(os.listdir(img_dir)), "labels:", len(os.listdir(lab_dir)))
    for s in ("train", "val", "test"):
        print(s, len(split[s]))
    print("class.json:", open(os.path.join(mp, "class.json")).read())
    print("ckpt:", os.path.exists(dst), dst)


if __name__ == "__main__":
    main()
