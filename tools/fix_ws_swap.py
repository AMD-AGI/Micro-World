# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Fix reversed W/S in the capsaicin metadata-detection jsons.

Observation: the stored `ws` values are opposite to the actual forward/backward
motion (keyboard_to_onehot maps ws==1 -> forward/W, ws==2 -> backward/S, but the
data has them reversed). This swaps ws 1<->2 in every action frame so W/S match
the real motion in both training and visualization.

Idempotent: each processed json gets action_encoding["_sprite_ws_refixed"] = True;
files already carrying that flag are skipped, so re-running is safe (won't double-swap).

Usage (inside the container or any python with no special deps):
  python tools/fix_ws_swap.py                       # defaults to 200f
  python tools/fix_ws_swap.py --dir datasets/capsaicin/200f/metadata-detection \
                              --dir datasets/capsaicin/400f/metadata-detection
  python tools/fix_ws_swap.py --dry_run             # report only, no writes
"""
import argparse
import glob
import json
import os

SWAP = {1: 2, 2: 1}  # 0 (none) stays 0


def process_file(path, dry_run=False):
    with open(path, "r") as f:
        data = json.load(f)

    ae = data.setdefault("action_encoding", {})
    if ae.get("_sprite_ws_refixed"):
        return "skipped"

    actions = data.get("actions", {})
    swapped = 0
    for _, a in actions.items():
        if "ws" in a:
            ws = a["ws"]
            if ws in SWAP:
                a["ws"] = SWAP[ws]
                swapped += 1

    if dry_run:
        return f"would swap {swapped}"

    ae["_sprite_ws_refixed"] = True
    note = ae.get("_postprocessed", "")
    ae["_postprocessed"] = (note + " | " if note else "") + "sprite fix: ws 1<->2 swapped (W/S matched to real motion)"

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return f"swapped {swapped}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        action="append",
        default=None,
        help="metadata-detection dir(s); repeatable. Defaults to 200f.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dirs = args.dir or ["datasets/capsaicin/200f/metadata-detection"]

    total = {"swapped": 0, "skipped": 0}
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        print(f"[{d}] {len(files)} json files")
        for p in files:
            r = process_file(p, dry_run=args.dry_run)
            if r == "skipped":
                total["skipped"] += 1
            else:
                total["swapped"] += 1
        print(f"  -> processed={len(files)}")
    print(f"done. files_modified={total['swapped']} skipped(already fixed)={total['skipped']} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
