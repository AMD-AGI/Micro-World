# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Visualize the Mixkit DMD data-reading mechanism
(scripts/wan2.2/train_causal_dmd_ti2v_action_teacher_mixkit_161f.sh).

What this shows, per sample, is EXACTLY what the dataset feeds the DMD student:
  - The mp4 FRAMES + PROMPT come from a Mixkit clip. Because DMD only uses the clip's
    first latent frame as the i2v/sink condition (the rest is student-generated), we
    render the WHOLE clip as that single FIRST FRAME held static for N_FRAMES.
  - The PROMPT is drawn as a banner ABOVE the video.
  - The ACTION (mouse cursor + keyboard) is DECOUPLED: it is drawn at random from a
    Capsaicin 200f clip via action_pool_root (random json + random 81-frame window),
    then overlaid with the same UI principle as validation.

So each output video = "Mixkit first frame + prompt" with a "virtual Capsaicin action"
UI playing on top -- a faithful picture of the mixed-source training sample.

Run inside the training container (needs torch/decord/cv2/imageio), from repo root:
  python tools/viz_mixkit_dmd_sample.py --num 4 --seed 0
"""
import argparse
import os
import random
import sys
import textwrap

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from microworld.data.dataset_game_video import VideoGameDataset
from microworld.pipeline.pipeline_wan_action_t2w import (
    draw_keys_on_frame,
    overlay_icon,
    parse_config,
)

# Mixkit = frames + prompt (no action labels). Capsaicin = decoupled action pool.
ANN = "datasets/mixkit/video_captions_mixkit.csv"
ROOT = "datasets/mixkit"
ACTION_POOL_ROOT = "datasets/capsaicin/200f/metadata-detection"
SAMPLE_SIZE = [352, 640]  # H, W (matches training --video_sample_size 352 640)
N_FRAMES = 81
FPS = 16


def make_prompt_banner(text, width, font_scale=0.5, thickness=1, pad=10, line_gap=6):
    """Render a black banner containing the (word-wrapped) prompt, matching `width`."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Estimate chars-per-line from the average glyph width, then wrap.
    (char_w, char_h), _ = cv2.getTextSize("A", font, font_scale, thickness)
    max_chars = max(8, (width - 2 * pad) // max(1, char_w))
    lines = []
    for para in (text or "(empty prompt)").splitlines() or ["(empty prompt)"]:
        lines.extend(textwrap.wrap(para, width=max_chars) or [""])
    lines = lines[:4]  # cap height; prompts are short captions anyway

    line_h = char_h + line_gap
    banner_h = pad * 2 + line_h * len(lines)
    banner = np.zeros((banner_h, width, 3), dtype=np.uint8)
    y = pad + char_h
    for ln in lines:
        cv2.putText(banner, ln, (pad, y), font, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        y += line_h
    return banner


def render_one(ds, idx, args):
    sample = ds[idx]

    pixel_values = sample["pixel_values"]          # [T, C, H, W], normalized ~[-1, 1]
    mouse_actions = sample["mouse_actions"]         # [T, 2] = (pitch, yaw)
    keyboard_actions = sample["keyboard_actions"]   # [T, 7] = w,s,a,d,jump,sneak,sprint
    text = sample["text"]
    name = sample["name"]

    # De-normalize to RGB uint8.
    vid = (pixel_values * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    vid = (vid.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # [T, H, W, C] RGB
    T, H, W, _ = vid.shape

    # Hold the FIRST frame static for all T frames (use first frame for entire mp4).
    first_frame = vid[0]

    # Same overlay principle as validation / viz_train_sample.
    key_data, mouse_data = parse_config(
        mouse_actions.float().cpu()[None],
        keyboard_actions.cpu()[None],
        sample_size=(W, H),
        cursor_gain=args.cursor_gain,
        cursor_y_sign=1.0,
        cursor_vis_scale=1.0,
    )
    mouse_scale = 0.2
    mouse_rotation = -20
    mouse_icon = cv2.imread("./asset/mouse.png", cv2.IMREAD_UNCHANGED)

    banner = make_prompt_banner(text, W)

    out_frames = []
    for i in range(T):
        keys = key_data.get(
            i,
            {"W": False, "A": False, "S": False, "D": False,
             "Space": False, "Shift": False, "Ctrl": False},
        )
        mouse_position = mouse_data.get(i, (W // 2, H // 2))
        frame = first_frame.copy()
        draw_keys_on_frame(frame, keys, key_size=(50, 50), spacing=10, bottom_margin=20)
        overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
        out_frames.append(np.vstack([banner, frame]))

    os.makedirs(args.out_dir, exist_ok=True)
    safe = name.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"mixkit_dmd_viz_idx{idx}_{safe}.mp4")
    imageio.mimsave(out_path, out_frames, fps=FPS)

    pitch, yaw = mouse_actions[:, 0], mouse_actions[:, 1]
    print(f"saved: {out_path}")
    print(f"  mixkit frame : idx={idx} name={name} size={W}x{H} (first frame held x{T})")
    print(f"  prompt       : {text[:120]}")
    print(f"  action(pool) : pitch|abs|={pitch.abs().mean():.3f} yaw|abs|={yaw.abs().mean():.3f} "
          f"W={int(keyboard_actions[:,0].sum())} S={int(keyboard_actions[:,1].sum())} "
          f"A={int(keyboard_actions[:,2].sum())} D={int(keyboard_actions[:,3].sum())} "
          f"jump={int(keyboard_actions[:,4].sum())} sprint={int(keyboard_actions[:,6].sum())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None, help="dataset index; random if unset")
    parser.add_argument("--num", type=int, default=1, help="number of random samples to render")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="output_viz_mixkit_dmd")
    parser.add_argument("--cursor_gain", type=float, default=3.0,
                        help="Visualization-only cursor amplification (capsaicin deltas ~0.6/frame).")
    parser.add_argument("--ann", type=str, default=ANN)
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--action_pool_root", type=str, default=ACTION_POOL_ROOT)
    parser.add_argument("--n_frames", type=int, default=N_FRAMES)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ds = VideoGameDataset(
        ann_path=args.ann,
        data_root=args.root,
        video_sample_size=SAMPLE_SIZE,
        video_sample_stride=1,
        video_sample_n_frames=args.n_frames,
        text_drop_ratio=0.0,           # always show the prompt
        drop_camera_sentences=False,
        use_action=True,
        action_pool_root=args.action_pool_root,  # decoupled capsaicin actions
    )

    if args.idx is not None:
        indices = [args.idx]
    else:
        indices = random.sample(range(len(ds)), min(args.num, len(ds)))

    for idx in indices:
        render_one(ds, idx, args)


if __name__ == "__main__":
    main()
