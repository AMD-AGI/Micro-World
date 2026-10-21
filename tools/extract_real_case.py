# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Extract a real training sample (first frame + caption + action) for long-video eval.

Reuses VideoGameDataset so the first frame, prompt and action tensors are produced
with exactly the same logic (frame indexing, mouse deg->mc->deg conversion, keyboard
one-hot) as training. Outputs:
  <out_dir>/first_frame.png   - first frame, resized to sample_size (matches training)
  <out_dir>/actions.npz       - mouse [T,2]=(pitch,yaw), keyboard [T,7]
  <out_dir>/prompt.txt        - caption used for this sample

Then feed them to examples/wan2.2/predict_causal_action_evalcase.py via:
  --validation_image_start <out_dir>/first_frame.png
  --prompt "$(cat <out_dir>/prompt.txt)"
  --real_action_npz <out_dir>/actions.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from microworld.data.dataset_game_video import VideoGameDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann_path", default="datasets/DL3DV-480P/video_captions_v0.csv")
    p.add_argument("--data_root", default="datasets")
    p.add_argument("--index", type=int, default=0, help="CSV row index of the sample to extract.")
    p.add_argument("--n_frames", type=int, default=401)
    p.add_argument("--sample_size", type=int, nargs=2, default=[352, 640])
    p.add_argument("--drop_camera_sentences", action="store_true",
                   help="Match training prompt cleaning if it was enabled.")
    p.add_argument("--pitch_sign", type=float, default=-1.0,
                   help="Sign applied to the extracted pitch (mouse_y). The dataset flips "
                        "DL3DV pitch to MC convention (pitch_factor=-1/15); the synthetic "
                        "presets feed mouse_y unflipped. Use -1.0 to flip it back so up/down "
                        "matches the preset convention (fixes inverted pitch); 1.0 keeps the "
                        "raw dataset (training) sign.")
    p.add_argument("--out_dir", default="samples/real_case")
    args = p.parse_args()

    ds = VideoGameDataset(
        ann_path=args.ann_path,
        data_root=args.data_root,
        video_sample_size=args.sample_size,
        video_sample_stride=1,
        video_sample_n_frames=args.n_frames,
        text_drop_ratio=0.0,
        drop_camera_sentences=args.drop_camera_sentences,
        use_action=True,
        process_collisions=True,
        enable_bucket=False,
    )

    # get_batch is deterministic for a given CSV row (start frame index from the CSV).
    sample = ds.get_batch(args.index)
    pixel_values = sample["pixel_values"]          # [T,3,H,W] in [-1,1]
    text = sample["text"]
    mouse = sample["mouse_actions"].cpu().numpy().copy()  # [T,2] = (pitch, yaw)
    keyboard = sample["keyboard_actions"].cpu().numpy()  # [T,7]
    name = sample["name"]

    # Align pitch sign with the synthetic-preset / inference convention (yaw untouched).
    mouse[:, 0] = mouse[:, 0] * args.pitch_sign

    os.makedirs(args.out_dir, exist_ok=True)

    # First frame: invert Normalize(0.5,0.5) -> [0,255] RGB. Already resized to sample_size.
    first = pixel_values[0]
    img = ((first * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(os.path.join(args.out_dir, "first_frame.png"))

    np.savez(os.path.join(args.out_dir, "actions.npz"), mouse=mouse, keyboard=keyboard)
    with open(os.path.join(args.out_dir, "prompt.txt"), "w") as f:
        f.write(text)

    # Report action stats so we know what motion this real case contains.
    yaw = mouse[:, 1]
    pitch = mouse[:, 0]
    print(f"[real_case] name      : {name}")
    print(f"[real_case] frames    : {pixel_values.shape[0]} (requested {args.n_frames})")
    print(f"[real_case] prompt    : {text}")
    print(f"[real_case] mouse shape: {mouse.shape}, keyboard shape: {keyboard.shape}")
    print(f"[real_case] yaw   abs mean/p95/max: {np.abs(yaw).mean():.4f} / "
          f"{np.percentile(np.abs(yaw),95):.4f} / {np.abs(yaw).max():.4f}")
    print(f"[real_case] pitch abs mean/p95/max: {np.abs(pitch).mean():.4f} / "
          f"{np.percentile(np.abs(pitch),95):.4f} / {np.abs(pitch).max():.4f}")
    # Keyboard column order from keyboard_to_onehot: [w, s, a, d, ...].
    print(f"[real_case] keyboard active-frame counts (per dim): {keyboard.astype(bool).sum(0).tolist()}")
    print(f"[real_case] saved to  : {args.out_dir}")


if __name__ == "__main__":
    main()
