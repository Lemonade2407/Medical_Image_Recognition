"""P0-1 verification: every method's per-image results cover exactly the frozen test set."""
import os
import json
import glob

from busi_common import load_split, RESULTS, LESION_CLASSES


def main():
    split = load_split()
    test_all = set(split["test"])
    test_lesion = {n for n in split["test"] if n.split("/")[0] in LESION_CLASSES}

    files = sorted(glob.glob(os.path.join(RESULTS, "*.json")))
    files = [f for f in files if "summary" not in os.path.basename(f)]
    ok = True
    print(f"frozen test: all={len(test_all)}  lesion={len(test_lesion)}\n")
    for f in files:
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        per = rec.get("per_image")
        if not per:
            continue
        names = {m["name"] for m in per}
        is_prompt = os.path.basename(f).startswith("prompt_")
        expected = test_lesion if is_prompt else test_all
        same = names == expected
        ok &= same
        tag = "OK " if same else "FAIL"
        print(f"{tag} {os.path.basename(f):34s} n={len(names):3d} expected={len(expected):3d} "
              f"extra={len(names - expected)} missing={len(expected - names)}")
    print("\nALL CONSISTENT" if ok else "\nINCONSISTENT — fix before comparing")


if __name__ == "__main__":
    main()
