# Modifications Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

# This file is a modified version of code from VideoXFun
# (https://github.com/aigc-apps/VideoX-Fun),
# which is licensed under the Apache License, Version 2.0.

import csv
import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from func_timeout import FunctionTimedOut, func_timeout
from torch.utils.data.dataset import Dataset
from .dataset_game_video import VideoReader_contextmanager, get_video_reader_batch

VIDEO_READER_TIMEOUT = 20

def calculate_action_diversity_score(actions, start_idx, end_idx):
    """
    Calculate diversity score for a video segment.
    Higher score = more diverse actions (prioritized for sampling)
    
    Args:
        actions: dict of frame actions
        start_idx: start frame index
        end_idx: end frame index
    
    Returns:
        float: diversity score (higher is better)
    """
    ws_counts = {0: 0, 1: 0, 2: 0}  # none, foward, backward
    ad_counts = {0: 0, 1: 0, 2: 0}  # none, left, right
    total = 0
    
    for frame in range(start_idx, end_idx):
        action = actions.get(str(frame), None)
        if action:
            ws = action.get("ws", 0)
            ad = action.get("ad", 0)
            ws_counts[ws] += 1
            ad_counts[ad] += 1
            total += 1
    
    if total == 0:
        return 0.0
    
    # Calculate diversity score
    score = 0.0
    
    # Penalize excessive forward movement (ws=1)
    forward_ratio = ws_counts[1] / total
    if forward_ratio > 0.8:  # If >80% forward
        score -= (forward_ratio - 0.8) * 10  # Heavy penalty
    
    # Reward backward movement (ws=2) and still status (ws=0) - rare action
    backward_ratio = ws_counts[2] / total
    still_ratio = ws_counts[0] / total
    score += (backward_ratio + still_ratio) * 5.0
    
    # Reward left/right movement (ad=1 or ad=2) - diverse actions
    left_ratio = ad_counts[1] / total
    right_ratio = ad_counts[2] / total
    score += (left_ratio + right_ratio) * 3.0
    
    # Reward diversity (having multiple different actions)
    # num_ws_types = sum(1 for v in ws_counts.values() if v > 0)
    # num_ad_types = sum(1 for v in ad_counts.values() if v > 0)
    # score += (num_ws_types + num_ad_types) * 0.5
    
    return score

def keyboard_to_onehot(ws, ad):
    arr = [0] * 7    # last three are placeholder
    if ws == 1:
        arr[0] = 1  # forward
    elif ws == 2:
        arr[1] = 1  # backward
    if ad == 1:
        arr[2] = 1  # left
    elif ad == 2:
        arr[3] = 1  # right
    return arr

class VideoDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=[352, 640], video_sample_stride=1, video_sample_n_frames=81,
        text_drop_ratio=0.1,
        use_action=False,
        process_collisions=True,
        enable_bucket=False,
        balanced_sampling=False,  # NEW: Enable balanced action sampling
        balance_sample_candidates=10,  # NEW: Number of candidate segments to evaluate
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

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                # transforms.Resize(self.video_sample_size),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.use_action = use_action
        self.process_collisions = process_collisions

        # NEW: Balanced sampling parameters
        self.balanced_sampling = balanced_sampling
        self.balance_sample_candidates = balance_sample_candidates

    def _select_balanced_start_idx(self, actions, total_frames):
        """
        Select start index that leads to more diverse actions
        
        Args:
            actions: dict of all actions in the video
            total_frames: total number of frames
            
        Returns:
            int: selected start index
        """
        max_start = total_frames - self.video_sample_n_frames
        if max_start <= 0:
            return 0
        
        # Generate candidate start indices
        if max_start < self.balance_sample_candidates:
            # If video is short, try all possible starts
            candidates = list(range(max_start + 1))
        else:
            # Randomly sample candidates
            candidates = random.sample(range(max_start + 1), self.balance_sample_candidates)
        
        # Calculate diversity score for each candidate
        candidate_scores = []
        for start_idx in candidates:
            end_idx = start_idx + self.video_sample_n_frames
            score = calculate_action_diversity_score(actions, start_idx, end_idx)
            candidate_scores.append((start_idx, score))
        
        # Sort by score (higher is better)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select with weighted probability (favor higher scores)
        # Use softmax-like selection to still maintain some randomness
        scores = [score for _, score in candidate_scores]

        # If all scores are very low (all forward-only), fall back to random
        if max(scores) < -2:
            return random.choice(candidates)
        
        # # Weight by exponential of scores
        # scores_array = np.array(scores)
        # # Shift scores to avoid negative exponents
        # scores_array = scores_array - scores_array.min() + 1
        # weights = np.exp(scores_array / 2.0)
        # weights = weights / weights.sum()
        
        # # Select based on weights
        # selected_idx = np.random.choice(len(candidates), p=weights)
        # return candidate_scores[selected_idx][0]

        return candidate_scores[0][0]
    
    def get_batch(self, idx):
        data_info = self.dataset[idx]
        
        video_id, text = data_info['video_path'], data_info['prompt']
        json_id = os.path.splitext(video_id)[0] + ".json"
        video_dir = os.path.join(self.data_root, "videos", video_id)

        if self.use_action:
            action_dir = os.path.join(self.data_root, "metadata-detection", json_id)

            # Load actions from JSON file
            with open(action_dir, 'r') as f:
                action_data = json.load(f)
            actions = action_data.get("actions", {})
            
        with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
            total_frames = len(video_reader)
            max_start = total_frames - self.video_sample_n_frames
            assert max_start >= 0, f"Video {video_id} has only {total_frames} frames, but video_sample_n_frames is {self.video_sample_n_frames}"
            
            # NEW: Select start index based on sampling strategy
            if self.balanced_sampling:
                start_idx = self._select_balanced_start_idx(actions, total_frames)
            else:
                start_idx = random.randint(0, max_start)            
            
            batch_index = range(start_idx, start_idx + self.video_sample_n_frames)
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
            pixel_values = self.video_transforms(pixel_values)

        # Random use no text generation
        if random.random() < self.text_drop_ratio:
            text = ''

        sample = {"pixel_values": pixel_values,
                "text": text,
                "data_type": "video",
                "name": video_id.split(".")[0],
                "idx" : idx}

        if self.use_action:

            # Filter actions for frames 1 to 1,999
            filtered_actions = {
                int(frame): {
                    "ws": action["ws"],
                    "ad": action["ad"],
                    "pitch_delta": action["pitch_delta"],
                    "yaw_delta": action["yaw_delta"],
                }
                for frame, action in actions.items()
                if frame != "0" and int(frame) > start_idx and int(frame) <= start_idx + self.video_sample_n_frames
            }


            # Prepare mouse_actions and keyboard_actions tensors
            mouse_actions = []
            keyboard_actions = []

            for frame in range(start_idx, start_idx + self.video_sample_n_frames):
                action = filtered_actions.get(frame, None)
                if action:
                    # Extract mouse actions (pitch_delta, yaw_delta)
                    mouse_actions.append([action["pitch_delta"], action["yaw_delta"]])

                    keyboard_action_now = keyboard_to_onehot(action["ws"], action["ad"])
                    keyboard_actions.append(keyboard_action_now)
                else:
                    # Append zeros if no action is available for the frame
                    mouse_actions.append([0.0, 0.0])
                    keyboard_actions.append([0, 0, 0, 0, 0, 0, 0])

            # Convert to tensors and ensure correct shape
            mouse_actions = torch.tensor(mouse_actions, dtype=torch.float32)  # Shape [rn, mouse_dim]
            keyboard_actions = torch.tensor(keyboard_actions, dtype=torch.float32)  # Shape [rn, keyboard_dim]
            sample.update({
                "mouse_actions": mouse_actions,  # Include mouse actions in the sample
                "keyboard_actions": keyboard_actions,  # Include keyboard actions in the sample
            })

        return sample

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # if len(self.check_idx) != 0:
        #     idx = self.check_idx.pop()
        sample = self.get_batch(idx)
        return sample
    

if __name__ == "__main__":
    dataset = VideoDataset(
        ann_path="/wekafs/yugeng12/datasets/sekai-real-walking-hq/train/sekai-real-walking-hq.csv",
        data_root="/wekafs/yugeng12/datasets/sekai-real-walking-hq",
        video_sample_size=[720, 1280],
        video_sample_n_frames=121,
        use_action=True,
        balanced_sampling=True,           # enabled
        balance_sample_candidates=10,     # candidate clips evaluated per balance draw
        )
    data = dataset[1997]