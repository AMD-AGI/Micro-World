# Modifications Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

# This file is a modified version of code from VideoXFun
# (https://github.com/aigc-apps/VideoX-Fun),
# which is licensed under the Apache License, Version 2.0.

import csv
import gc
import json
import os
import random
import re
from contextlib import contextmanager

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_timeout
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

VIDEO_READER_TIMEOUT = 20


def keyboard_to_onehot(ws, ad, scs):
    arr = [0] * 7
    if ws == 1:
        arr[0] = 1  # forward
    elif ws == 2:
        arr[1] = 1  # backward
    if ad == 1:
        arr[2] = 1  # left
    elif ad == 2:
        arr[3] = 1  # right
    if scs == 1:
        arr[4] = 1  # jump
    elif scs == 2:
        arr[5] = 1  # sneak
    elif scs == 3:
        arr[6] = 1  # sprint
    return arr

def process_collisions(action):
    if action["collision"] == 1:
        action["ws"] = 0
        action["ad"] = 0
    if action["jump_invalid"] == 1:
        action["scs"] = 0
    return action

class GameVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = []

    def __iter__(self):
        for idx in self.sampler:
            self.bucket.append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket) == self.batch_size:
                yield self.bucket[:]
                del self.bucket[:]

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames 

class VideoGameDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=[352, 640], video_sample_stride=1, video_sample_n_frames=81,
        text_drop_ratio=0.1,
        drop_camera_sentences=False,
        use_action=False,
        process_collisions=True,
        enable_bucket=False,
        action_pool_root=None,
        enable_random_temporal_sample=False,
        temporal_reverse_ratio=0.0,
        frame_scaling="crop",
        enforce_sample_n_frames=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                self.dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            self.dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.drop_camera_sentences = drop_camera_sentences

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        # "crop" keeps the 1.0 framing: center-crop to the target size, discarding whatever
        # falls outside. "resize" instead rescales the whole frame, preserving the full field
        # of view at the cost of a small aspect-ratio distortion, which is what the
        # open-domain 2.0 data (mixkit / DL3DV) is trained with.
        if frame_scaling not in ("crop", "resize"):
            raise ValueError(f"frame_scaling must be 'crop' or 'resize', got {frame_scaling!r}")
        self.frame_scaling = frame_scaling
        if frame_scaling == "resize":
            spatial_transform = transforms.Resize(self.video_sample_size, antialias=True)
        else:
            spatial_transform = transforms.CenterCrop(self.video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                spatial_transform,
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.use_action = use_action
        self.process_collisions = process_collisions
        # When True, each __getitem__ draws a RANDOM temporal window of length
        # video_sample_n_frames from within the clip (instead of always anchoring at
        # the CSV `start frame index`). This lets a 200-frame/12.5s clip contribute
        # many different 81-frame/~5s windows across epochs. Video frames and action
        # trajectories share the same start/end, so they stay aligned automatically.
        self.enable_random_temporal_sample = enable_random_temporal_sample

        # When True every sample is forced to exactly video_sample_n_frames frames instead of
        # using the annotated [start, end] window as-is. Random temporal sampling needs a
        # fixed window length to slide, so it implies this.
        self.enforce_sample_n_frames = enforce_sample_n_frames or enable_random_temporal_sample

        # Temporal-reversal augmentation. With probability temporal_reverse_ratio, play the
        # sampled clip backwards and reverse its actions accordingly. This is only valid for
        # STATIC scenes where the sole motion is the first-person camera (e.g. capsaicin /
        # DL3DV walkthroughs): a forward walkthrough reversed is a legal backward walkthrough.
        # Action reversal is NOT a plain flip (see _reverse_temporal): mouse deltas are
        # negated + shifted one frame, and keyboard forward<->backward / left<->right are
        # swapped. Assumes no jump actions in the data.
        self.temporal_reverse_ratio = temporal_reverse_ratio

        # Optional decoupled-action pool: when set, action trajectories are drawn at
        # random from the json files under this directory (e.g. MW-capsaicin's
        # metadata-detection) instead of from each video's own json. Video frames and
        # prompts still come from the primary dataset (e.g. DL3DV).
        self.action_pool_paths = []
        if action_pool_root:
            self.action_pool_paths = sorted(
                os.path.join(action_pool_root, f)
                for f in os.listdir(action_pool_root)
                if f.endswith(".json")
            )
            print(f"action pool: {len(self.action_pool_paths)} json files from {action_pool_root}")

    @staticmethod
    def strip_camera_motion_sentences(text):
        if not text:
            return text

        raw_text = text.strip()

        # Strong rule: if caption contains "The camera", drop everything after it.
        m = re.search(r"\bthe\s+camera\b", raw_text, flags=re.IGNORECASE)
        if m is not None:
            prefix = raw_text[: m.start()].strip()
            return prefix if prefix else text

        # Fallback: sentence-level filtering for generic camera/motion descriptions.
        sentences = re.split(r'(?<=[。！？!?\.])\s+', raw_text)
        if len(sentences) <= 1:
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

        camera_keywords = [
            # English
            "camera", "motion", "moving", "moves", "move", "handheld", "body-mounted",
            "pan", "tilt", "zoom", "tracking", "dolly", "pov",
            # Chinese
            "镜头", "相机", "摄影机", "机位", "视角", "运动", "移动",
        ]
        camera_keywords = [kw.lower() for kw in camera_keywords]

        kept = []
        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(kw in lower_sentence for kw in camera_keywords):
                continue
            kept.append(sentence.strip())

        cleaned = " ".join([s for s in kept if s])
        # Fallback to original text to avoid empty prompt.
        return cleaned if cleaned else text
    
    def get_batch(self, idx):
        data_info = self.dataset[idx]
        start_frame_index = int(data_info["start frame index"])
        end_frame_index = int(data_info["end frame index"])
        # By default the annotated [start, end] window is used verbatim, as in 1.0.
        # With enforce_sample_n_frames the window is instead forced to exactly
        # video_sample_n_frames, anchored at the annotated start and clamped, extended or
        # padded to the video bounds below. That gives every batch a uniform temporal
        # length even when the annotations describe clips of varying length.
        target_num_frames = self.video_sample_n_frames if self.enforce_sample_n_frames else None
        if target_num_frames is not None:
            end_frame_index = start_frame_index + target_num_frames - 1
        
        video_id, text = data_info['original video name'], data_info['prompt']
        # Datasets converted from other formats (e.g. DL3DV) may not carry a `dir`
        # column; fall back to `video_dir` then to "" (so data_root points directly
        # at the dataset folder).
        data_dir = data_info.get("dir") or data_info.get("video_dir") or ""
        json_id = os.path.splitext(video_id)[0] + ".json"
        video_dir = os.path.join(self.data_root, data_dir, "video", video_id)

        with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
            video_total_frames = len(video_reader)
            if video_total_frames <= 0:
                raise ValueError(f"Video has no frames: {video_dir}")

            # Optionally draw a random temporal window so each clip contributes many
            # different sub-windows across epochs. Both the video frames and the action
            # tensors below use start_frame_index/end_frame_index, so they stay aligned.
            if self.enable_random_temporal_sample:
                max_start = max(0, video_total_frames - target_num_frames)
                start_frame_index = random.randint(0, max_start)
                end_frame_index = start_frame_index + target_num_frames - 1

            if target_num_frames is not None:
                # Clamp and shift sampling window to stay within bounds while preserving
                # the target temporal length as much as possible.
                start_frame_index = max(0, start_frame_index)
                end_frame_index = min(end_frame_index, video_total_frames - 1)

                current_num_frames = end_frame_index - start_frame_index + 1
                if current_num_frames < target_num_frames:
                    need = target_num_frames - current_num_frames
                    right_room = video_total_frames - 1 - end_frame_index
                    right_extend = min(need, right_room)
                    end_frame_index += right_extend
                    need -= right_extend
                    if need > 0:
                        left_room = start_frame_index
                        left_extend = min(need, left_room)
                        start_frame_index -= left_extend

            batch_index = range(start_frame_index, end_frame_index+1)
            try:
                sample_args = (video_reader, batch_index)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                # resized_frames = []
                # for i in range(len(pixel_values)):
                #     frame = pixel_values[i]
                #     resized_frame = resize_frame(frame, self.shorter_size)
                #     resized_frames.append(resized_frame)
                # pixel_values = np.array(resized_frames)
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader

            # For short videos, keep training stable by padding with the last frame.
            if target_num_frames is not None:
                if pixel_values.shape[0] < target_num_frames:
                    pad_count = target_num_frames - pixel_values.shape[0]
                    pad_frames = pixel_values[-1:].repeat(pad_count, 1, 1, 1)
                    pixel_values = torch.cat([pixel_values, pad_frames], dim=0)
                elif pixel_values.shape[0] > target_num_frames:
                    pixel_values = pixel_values[:target_num_frames]

            pixel_values = self.video_transforms(pixel_values)

        if self.drop_camera_sentences:
            text = self.strip_camera_motion_sentences(text)

        # Random use no text generation
        if random.random() < self.text_drop_ratio:
            text = ''

        sample = {"pixel_values": pixel_values,
                "text": text,
                "data_type": "video",
                "name": video_id.split(".")[0],
                "idx" : idx}
        if self.use_action:
            # Without a forced clip length the action trajectory simply spans the frames that
            # were actually read, which makes the padding in _actions_to_tensors a no-op.
            action_num_frames = (
                target_num_frames if target_num_frames is not None
                else end_frame_index - start_frame_index + 1
            )
            if self.action_pool_paths:
                # Decoupled-action mode: take the action trajectory from a RANDOM json in
                # the action pool (e.g. MW-capsaicin), not from this video's own json. The
                # video frames + prompt still come from the primary dataset (e.g. DL3DV).
                # A random window of the pooled clip is used so we see varied trajectories.
                pool_path = random.choice(self.action_pool_paths)
                with open(pool_path, 'r') as f:
                    action_data = json.load(f)
                pool_frames = [int(k) for k in action_data.get("actions", {}).keys() if k != "0"]
                pool_max = max(pool_frames) if pool_frames else action_num_frames
                a_start = random.randint(0, max(0, pool_max - action_num_frames))
                a_end = a_start + action_num_frames - 1
                mouse_actions, keyboard_actions = self._actions_to_tensors(
                    action_data, a_start, a_end, action_num_frames
                )
            else:
                action_dir = os.path.join(self.data_root, data_dir, "metadata-detection", json_id)
                with open(action_dir, 'r') as f:
                    action_data = json.load(f)
                mouse_actions, keyboard_actions = self._actions_to_tensors(
                    action_data, start_frame_index, end_frame_index, action_num_frames
                )

            sample.update({
                "mouse_actions": mouse_actions,  # Include mouse actions in the sample
                "keyboard_actions": keyboard_actions,  # Include keyboard actions in the sample
            })

        # Temporal-reversal augmentation: flip frames + actions together so they stay aligned.
        if self.temporal_reverse_ratio > 0 and random.random() < self.temporal_reverse_ratio:
            pv, ma, ka = self._reverse_temporal(
                sample["pixel_values"],
                sample.get("mouse_actions"),
                sample.get("keyboard_actions"),
            )
            sample["pixel_values"] = pv
            if ma is not None:
                sample["mouse_actions"] = ma
            if ka is not None:
                sample["keyboard_actions"] = ka

        return sample

    def _reverse_temporal(self, pixel_values, mouse_actions, keyboard_actions):
        """Play the clip backwards and reverse its actions to stay physically consistent.

        Only valid for static scenes where the sole motion is the first-person camera.
          - pixel_values [T, C, H, W]: reversed along time.
          - mouse_actions [T, 2] = [pitch_delta, yaw_delta]: these are frame-to-frame
            deltas, so reversing time negates each traversed delta and shifts it by one
            frame. If m[i] is the delta INTO frame i, the reversed clip needs
            new[i] = -m[T - i] for i >= 1, and new[0] = 0 (no preceding frame).
          - keyboard_actions [T, 7] = [fwd, back, left, right, jump, sneak, sprint]:
            walking a static scene backwards swaps forward<->backward and left<->right;
            jump/sneak/sprint columns keep their (time-reversed) order.
        """
        pixel_values = torch.flip(pixel_values, dims=[0])

        if mouse_actions is not None:
            rev = torch.flip(mouse_actions, dims=[0])  # rev[i] = old[T-1-i]
            new_mouse = torch.zeros_like(mouse_actions)
            new_mouse[1:] = -rev[:-1]                   # new[i] = -old[T-i]
            mouse_actions = new_mouse

        if keyboard_actions is not None:
            kb = torch.flip(keyboard_actions, dims=[0])
            # swap fwd<->back (0<->1) and left<->right (2<->3)
            kb = kb[:, [1, 0, 3, 2, 4, 5, 6]].contiguous()
            keyboard_actions = kb

        return pixel_values, mouse_actions, keyboard_actions

    def _actions_to_tensors(self, action_data, start_frame_index, end_frame_index, target_num_frames):
        """Parse mouse/keyboard action tensors from a loaded action json over the frame
        window (start_frame_index, end_frame_index], padded/truncated to target_num_frames.
        Handles per-source mouse unit conversion (see notes below)."""
        actions = action_data.get("actions", {})

        # Mouse unit handling. The shared `* 15` below maps MC mouse units -> degrees,
        # which is what the (Minecraft-trained) action model expects.
        #   - Minecraft json: mouse already in MC units -> factor 1.0 (then * 15).
        #   - DL3DV json: mouse stored in real-scale DEGREES (see action_encoding) and
        #     carries `mouse_normalized_to_mc` = {yaw_factor: 1/15, pitch_factor: -1/15}.
        #     Applying these factors first converts deg -> MC units (and flips pitch to
        #     the MC convention); the subsequent * 15 then yields the correct degree scale.
        #     Without this, DL3DV degrees go straight through * 15 and end up ~15x too large.
        #   - Capsaicin / Blender json: also store DEGREES (pitch/yaw already sign-flipped to
        #     the MC convention, see action_encoding._postprocessed) but ship no
        #     `mouse_normalized_to_mc`. They are DL3DV-like, so they must NOT be * 15'd: apply
        #     1/15 to both axes (no extra pitch flip, since already MC-signed) so the later
        #     * 15 cancels and the model sees the real degree scale. Blender's action_encoding
        #     explicitly declares pitch/yaw as "degrees, Capsaicin-compatible" and mouse_delta
        #     in "Capsaicin mouse units", so it is handled identically to Capsaicin.
        norm_to_mc = action_data.get("mouse_normalized_to_mc")
        if isinstance(norm_to_mc, dict):
            yaw_to_mc = norm_to_mc.get("yaw_factor", 1.0 / 15)
            pitch_to_mc = norm_to_mc.get("pitch_factor", -1.0 / 15)
        elif action_data.get("source") in ("Capsaicin", "Blender"):
            yaw_to_mc = 1.0 / 15
            pitch_to_mc = 1.0 / 15
        else:
            yaw_to_mc = 1.0
            pitch_to_mc = 1.0

        filtered_actions = {
            int(frame): {
                "ws": action["ws"],
                "ad": action["ad"],
                "scs": action["scs"],
                "pitch_delta": action["pitch_delta"] * pitch_to_mc * 15,
                "yaw_delta": action["yaw_delta"] * yaw_to_mc * 15,
                # Non-Minecraft sources (e.g. DL3DV) have no collision/jump_invalid.
                "collision": action.get("collision", 0),
                "jump_invalid": action.get("jump_invalid", 0)
            }
            for frame, action in actions.items()
            if frame != "0" and int(frame) > start_frame_index and int(frame) <= end_frame_index
        }

        mouse_actions = []
        keyboard_actions = []
        for frame in range(start_frame_index, end_frame_index + 1):
            action = filtered_actions.get(frame, None)
            if action:
                mouse_actions.append([action["pitch_delta"], action["yaw_delta"]])
                if self.process_collisions:
                    action = process_collisions(action)
                keyboard_action_now = keyboard_to_onehot(action["ws"], action["ad"], action["scs"])
                keyboard_actions.append(keyboard_action_now)
            else:
                mouse_actions.append([0.0, 0.0])
                keyboard_actions.append([0, 0, 0, 0, 0, 0, 0])

        if len(mouse_actions) < target_num_frames:
            pad_count = target_num_frames - len(mouse_actions)
            mouse_actions.extend([mouse_actions[-1] if len(mouse_actions) > 0 else [0.0, 0.0]] * pad_count)
            keyboard_actions.extend([keyboard_actions[-1] if len(keyboard_actions) > 0 else [0, 0, 0, 0, 0, 0, 0]] * pad_count)
        elif len(mouse_actions) > target_num_frames:
            mouse_actions = mouse_actions[:target_num_frames]
            keyboard_actions = keyboard_actions[:target_num_frames]

        mouse_actions = torch.tensor(mouse_actions, dtype=torch.float32)  # [T, mouse_dim]
        keyboard_actions = torch.tensor(keyboard_actions, dtype=torch.float32)  # [T, keyboard_dim]
        return mouse_actions, keyboard_actions

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # if len(self.check_idx) != 0:
        #     idx = self.check_idx.pop()
        while True:
            sample = {}
            try:
                sample = self.get_batch(idx)
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        return sample
    

if __name__ == "__main__":
    dataset = VideoGameDataset(
        ann_path="datasets/dataset/video_captions_v0.csv",
        data_root="datasets/",
        video_sample_size=[360, 640],
        video_sample_n_frames=201,
        use_action=True)
    data = dataset[104]
    print(1)