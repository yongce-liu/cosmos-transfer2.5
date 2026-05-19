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

"""Shared HSV flow visualization and padding utilities.

Used by all backends (RAFT, WAFT, MemFlow) so the produced control
videos are visually comparable regardless of the estimator.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """HSV-style flow visualization. flow: (N, 2, H, W) float -> (N, 3, H, W) uint8."""
    fx, fy = flow[:, 0], flow[:, 1]
    mag = torch.sqrt(fx * fx + fy * fy)
    ang = torch.atan2(fy, fx)
    hue = (ang / (2 * math.pi) + 0.5).clamp(0.0, 1.0)
    N = mag.shape[0]
    mag_flat = mag.reshape(N, -1)
    q = torch.quantile(mag_flat, 0.99, dim=1).clamp_min(1e-6)
    sat = (mag / q[:, None, None]).clamp(0.0, 1.0)
    val = torch.ones_like(sat)
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
    rgb = torch.stack([r, g, b], dim=1)
    return (rgb * 255.0).round().clamp(0, 255).to(torch.uint8)


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    pad = (0, pad_w, 0, pad_h)
    return F.pad(x, pad, mode="replicate"), pad
