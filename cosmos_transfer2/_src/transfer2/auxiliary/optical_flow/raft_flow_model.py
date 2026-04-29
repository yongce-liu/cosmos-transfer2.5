# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RAFT optical-flow estimation and HSV visualization for control input.

Uses torchvision's RAFT (large) checkpoint. Output is a 3-channel uint8
RGB visualization in CTHW layout, shaped to match the input frames.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_MODEL_CACHE: dict[str, tuple[torch.nn.Module, str]] = {}


def _load_raft(device: str) -> torch.nn.Module:
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device][0]
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    _MODEL_CACHE[device] = (model, weights)
    return model


def _flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """HSV-style flow visualization. flow: (N, 2, H, W) float -> (N, 3, H, W) uint8."""
    fx, fy = flow[:, 0], flow[:, 1]
    mag = torch.sqrt(fx * fx + fy * fy)
    ang = torch.atan2(fy, fx)  # [-pi, pi]
    hue = (ang / (2 * math.pi) + 0.5).clamp(0.0, 1.0)  # [0,1]
    # Per-frame normalize magnitude with the 99th percentile to suppress outliers.
    N = mag.shape[0]
    mag_flat = mag.reshape(N, -1)
    q = torch.quantile(mag_flat, 0.99, dim=1).clamp_min(1e-6)
    sat = (mag / q[:, None, None]).clamp(0.0, 1.0)
    val = torch.ones_like(sat)
    # HSV -> RGB
    h6 = hue * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)
    p = val * (1.0 - sat)
    q_ = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    r = torch.zeros_like(val)
    g = torch.zeros_like(val)
    b = torch.zeros_like(val)
    masks = [(i == k) for k in range(6)]
    rgb_table = [(val, t, p), (q_, val, p), (p, val, t), (p, q_, val), (t, p, val), (val, p, q_)]
    for m, (cr, cg, cb) in zip(masks, rgb_table):
        r = torch.where(m, cr, r)
        g = torch.where(m, cg, g)
        b = torch.where(m, cb, b)
    rgb = torch.stack([r, g, b], dim=1)  # (N, 3, H, W)
    return (rgb * 255.0).round().clamp(0, 255).to(torch.uint8)


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    pad = (0, pad_w, 0, pad_h)
    return F.pad(x, pad, mode="replicate"), pad


@torch.no_grad()
def compute_flow_visualization(
    frames_cthw: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = 4,
) -> np.ndarray:
    """Compute per-frame forward optical flow and return RGB visualization.

    Args:
        frames_cthw: uint8 video, shape (3, T, H, W).
        device: torch device. Defaults to cuda if available.
        batch_size: number of frame-pairs per RAFT forward pass.

    Returns:
        uint8 array (3, T, H, W). Flow for the last frame is replicated from T-1.
    """
    assert frames_cthw.ndim == 4 and frames_cthw.shape[0] == 3, (
        f"Expected (3, T, H, W) frames, got {frames_cthw.shape}"
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T = frames_cthw.shape[1]
    if T < 2:
        return np.zeros_like(frames_cthw)

    model = _load_raft(device)
    # (T, 3, H, W) float in [-1, 1] (RAFT preprocessing in torchvision normalizes inputs).
    frames = torch.from_numpy(frames_cthw).permute(1, 0, 2, 3).contiguous().to(device)
    frames = frames.float() / 255.0
    frames = frames * 2.0 - 1.0
    frames, pad = _pad_to_multiple(frames, multiple=8)
    H_pad, W_pad = frames.shape[-2:]

    flow_rgb_chunks = []
    for start in range(0, T - 1, batch_size):
        end = min(start + batch_size, T - 1)
        img1 = frames[start:end]
        img2 = frames[start + 1 : end + 1]
        flow_list = model(img1, img2)  # list of refinement steps
        flow = flow_list[-1]  # (N, 2, H_pad, W_pad)
        flow_rgb_chunks.append(_flow_to_rgb(flow))
    flow_rgb = torch.cat(flow_rgb_chunks, dim=0)  # (T-1, 3, H_pad, W_pad)
    # Replicate flow for the last frame.
    flow_rgb = torch.cat([flow_rgb, flow_rgb[-1:].clone()], dim=0)  # (T, 3, H_pad, W_pad)
    # Crop padding back to original size.
    H_orig, W_orig = frames_cthw.shape[-2:]
    flow_rgb = flow_rgb[:, :, :H_orig, :W_orig]
    # (T, 3, H, W) -> (3, T, H, W)
    return flow_rgb.permute(1, 0, 2, 3).contiguous().cpu().numpy()
