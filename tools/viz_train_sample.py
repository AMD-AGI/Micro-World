# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Visualize one (random) training sample with the SAME UI-overlay principle used in
validation (see pipeline_wan_causal.py's with_ui path): keyboard highlight +
cumulative mouse cursor, cursor_gain=20, cursor_y_sign=1.0 (no pitch flip),
mouse_scale=0.2, mouse_rotation=-20.

The video frames and actions are pulled through the exact training dataset
(VideoGameDataset, use_action=True), so the actions shown are precisely what the
model is trained on (capsaicin: mouse net x1, no scaling/negation).

Run inside the training container (needs torch/decord/cv2/imageio):
  python tools/viz_train_sample.py [--idx N] [--seed S]
"""
import argparse
import os
import random
import sys

import cv2
import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from microworld.data.dataset_game_video import VideoGameDataset
from microworld.pipeline.pipeline_wan_action_t2w import (
    draw_keys_on_frame,
    overlay_icon,
    parse_config,
)

ANN = "datasets/capsaicin/200f/video_captions_v0.csv"
ROOT = "datasets/capsaicin/200f"
SAMPLE_SIZE = [352, 640]  # H, W (matches training --video_sample_size 352 640)
N_FRAMES = 81
FPS = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="dataset index; random if unset")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="output_viz")
    parser.add_argument(
        "--cursor_gain",
        type=float,
        default=3.0,
        help="Visualization-only cursor amplification. capsaicin/blender deltas are "
             "~0.6/frame, so gain 3 keeps the cumulative cursor legible without saturating "
             "the frame edges (matches the validation UI's cursor_vis_gain).",
    )
    parser.add_argument("--num", type=int, default=1, help="number of random samples to render")
    parser.add_argument("--ann", type=str, default=ANN, help="captions csv path")
    parser.add_argument("--root", type=str, default=ROOT, help="dataset root dir")
    parser.add_argument("--n_frames", type=int, default=N_FRAMES, help="frames to sample")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ds = VideoGameDataset(
        ann_path=args.ann,
        data_root=args.root,
        video_sample_size=SAMPLE_SIZE,
        video_sample_stride=1,
        video_sample_n_frames=args.n_frames,
        text_drop_ratio=0.0,
        drop_camera_sentences=False,
        use_action=True,
    )

    if args.idx is not None:
        indices = [args.idx]
    else:
        indices = random.sample(range(len(ds)), min(args.num, len(ds)))

    os.makedirs(args.out_dir, exist_ok=True)
    for idx in indices:
        render_one(ds, idx, args)


def render_one(ds, idx, args):
    sample = ds[idx]

    pixel_values = sample["pixel_values"]          # [T, C, H, W], normalized to ~[-1, 1]
    mouse_actions = sample["mouse_actions"]         # [T, 2] = (pitch, yaw) == (mouse_y, mouse_x)
    keyboard_actions = sample["keyboard_actions"]   # [T, 7] = w, s, a, d, jump, sneak, sprint
    name = sample["name"]

    # De-normalize back to RGB uint8 (mean=std=0.5 -> x*0.5+0.5).
    vid = (pixel_values * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    vid = (vid.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # [T, H, W, C] RGB
    T, H, W, _ = vid.shape

    # Same overlay principle as validation (gain tunable; see --cursor_gain help).
    cursor_vis_gain = args.cursor_gain
    # parse_config expects a leading batch dim [1, T, *] (mirrors the validation call).
    key_data, mouse_data = parse_config(
        mouse_actions.float().cpu()[None],
        keyboard_actions.cpu()[None],
        sample_size=(W, H),
        cursor_gain=cursor_vis_gain,
        cursor_y_sign=1.0,   # no pitch flip (matches the updated validation UI)
        cursor_vis_scale=1.0,
    )
    mouse_scale = 0.2
    mouse_rotation = -20
    mouse_icon = cv2.imread("./asset/mouse.png", cv2.IMREAD_UNCHANGED)

    frames = np.ascontiguousarray(vid)
    out_frames = []
    for i in range(T):
        keys = key_data.get(
            i,
            {"W": False, "A": False, "S": False, "D": False,
             "Space": False, "Shift": False, "Ctrl": False},
        )
        mouse_position = mouse_data.get(i, (W // 2, H // 2))
        frame = frames[i].copy()
        draw_keys_on_frame(frame, keys, key_size=(50, 50), spacing=10, bottom_margin=20)
        overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
        out_frames.append(frame)

    out_path = os.path.join(args.out_dir, f"train_viz_idx{idx}_{name}.mp4")
    imageio.mimsave(out_path, out_frames, fps=FPS)

    # Quick stats so we can sanity-check the action magnitudes.
    pitch = mouse_actions[:, 0]
    yaw = mouse_actions[:, 1]
    print(f"saved: {out_path}")
    print(f"idx={idx} name={name} frames={T} size={W}x{H}")
    print(f"pitch: min={pitch.min():.3f} max={pitch.max():.3f} mean|abs|={pitch.abs().mean():.3f}")
    print(f"yaw  : min={yaw.min():.3f} max={yaw.max():.3f} mean|abs|={yaw.abs().mean():.3f}")
    print(f"keys pressed (any frame): "
          f"W={int(keyboard_actions[:,0].sum())} S={int(keyboard_actions[:,1].sum())} "
          f"A={int(keyboard_actions[:,2].sum())} D={int(keyboard_actions[:,3].sum())} "
          f"jump={int(keyboard_actions[:,4].sum())} sneak={int(keyboard_actions[:,5].sum())} "
          f"sprint={int(keyboard_actions[:,6].sum())}")


if __name__ == "__main__":
    main()
