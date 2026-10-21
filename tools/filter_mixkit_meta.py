# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env python3
"""Filter the Mixkit caption CSV down to categories suitable for action-conditioned
first-person-camera DMD training.

The DMD training here drives motion purely from a first-person CAMERA trajectory
(sampled from the Capsaicin action pool), so it only makes sense on relatively static,
camera-drivable scenes. Subject-dominated clips (people/sport/dance/animals) and full-
frame effects (smoke/fire/space/macro) fight the camera-motion prior, so they are dropped.

Categories are taken from the first path component of the "original video name" column,
e.g. "Street/mixkit-....mp4" -> "Street".

Usage:
    python tools/filter_mixkit_meta.py \
        --src datasets/mixkit/video_captions_mixkit.csv \
        --dst datasets/mixkit/video_captions_mixkit_filtered.csv
Edit WHITELIST below to change which categories are kept.
"""
import argparse
import csv

# Static, camera-drivable scene categories to KEEP.
WHITELIST = {
    "Street", "City", "beach", "sea", "sky", "forest", "mountain",
    "night", "water", "snow", "clouds", "sunset", "sun", "earth", "rain",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/mixkit/video_captions_mixkit.csv")
    ap.add_argument("--dst", default="datasets/mixkit/video_captions_mixkit_filtered.csv")
    args = ap.parse_args()

    kept = 0
    total = 0
    per = {}
    with open(args.src, newline="") as f, open(args.dst, "w", newline="") as g:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(g, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            category = row["original video name"].split("/")[0]
            if category in WHITELIST:
                writer.writerow(row)
                kept += 1
                per[category] = per.get(category, 0) + 1

    print(f"total={total} kept={kept} dropped={total - kept} -> {args.dst}")
    for cat, count in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5d}  {cat}")


if __name__ == "__main__":
    main()
