# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env bash
# Launch an interactive ROCm container with this repository mounted.
#
# Overridable:
#   REPO_DIR   host path to mount (defaults to this repository)
#   MOUNT_DIR  path inside the container
#   IMAGE      container image
#   NAME       container name
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MOUNT_DIR="${MOUNT_DIR:-/workspace/Sprite}"
IMAGE="${IMAGE:-yugeng12/rocm7.0.2_pytorch2.8.0_fa2.8.3_yg_251029:latest}"
NAME="${NAME:-sprite_train}"

docker run -it \
  --name "$NAME" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  --shm-size=64g \
  --network=host \
  -v "$REPO_DIR:$MOUNT_DIR" \
  -w "$MOUNT_DIR" \
  "$IMAGE" \
  /bin/bash
