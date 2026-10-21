# Tools

Standalone helpers for preparing training data and inspecting what the model
actually sees. All of them are run from the repository root, e.g.
`python tools/viz_train_sample.py`, because they add the repo root to `sys.path`
relative to their own location.

## Dataset metadata

| Script | Purpose |
| --- | --- |
| `filter_mixkit_meta.py` | Filter the Mixkit caption csv down to the static, camera-drivable categories used for DMD distillation. Writes `video_captions_mixkit_filtered.csv`. |
| `fix_ws_swap.py` | Repair reversed W/S labels in the capsaicin action jsons. Idempotent, safe to re-run. |
| `extract_real_case.py` | Pull one training sample out through `VideoGameDataset` into a self-contained eval case (`first_frame.png`, `actions.npz`, `prompt.txt`) for `--real_action_npz` inference. |

## Visualisation

| Script | Purpose |
| --- | --- |
| `viz_train_sample.py` | Render a capsaicin training sample with the validation UI overlay, so the actions shown are exactly the ones the action module is trained on. |
| `viz_mixkit_dmd_sample.py` | Inspect the DMD data path, where the frames and prompt come from Mixkit while the action trajectory is drawn at random from the capsaicin pool. |

Both need a GPU-less but full runtime (`torch`, `decord`, `cv2`, `imageio`), i.e.
run them inside the training container.
