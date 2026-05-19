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

"""PTLFlow-backed optical-flow estimation and RGB visualization.

PTLFlow provides one common API for RAFT, MemFlow, WAFT, SEA-RAFT, GMFlow, and
many other optical-flow models. This module keeps Cosmos' control-video output
format stable while delegating model loading and inference to PTLFlow.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .flow_visualization import flow_to_rgb

LEGACY_FLOW_MODEL_ALIASES = {
    "raft": "raft",
    "waft": "waft_dav2_a2",
    "memflow": "memflow",
}

DEFAULT_PTLFLOW_CHECKPOINTS = {
    "raft": "sintel",
    "raft_small": "things",
    "memflow": "sintel",
    "memflow_t": "sintel",
    "waft_dav2_a1": "tar-c-t-sintel",
    "waft_dav2_a2": "zero_shot",
    "waft_dinov3_a2": "zero_shot",
    "waft_twins_a2": "zero_shot",
}

_MODEL_CACHE: dict[tuple[str, str | None, str], torch.nn.Module] = {}


def resolve_ptlflow_model(model_name: str, ckpt_path: str | None = None) -> tuple[str, str | None]:
    """Resolve legacy aliases and default checkpoint names for PTLFlow."""
    model_name = LEGACY_FLOW_MODEL_ALIASES.get(model_name, model_name)
    if ckpt_path == "":
        ckpt_path = None
    if ckpt_path is None:
        ckpt_path = DEFAULT_PTLFLOW_CHECKPOINTS.get(model_name, "sintel")
    return model_name, ckpt_path


def _load_ptlflow_model(model_name: str, ckpt_path: str | None, device: str) -> torch.nn.Module:
    cache_key = (model_name, ckpt_path, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        import ptlflow
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PTLFlow is required for on-the-fly optical-flow control. "
            "Install it with `pip install ptlflow` or run the project dependency sync."
        ) from exc

    model = ptlflow.get_model(model_name, ckpt_path=ckpt_path)
    model = model.to(device).eval()
    _MODEL_CACHE[cache_key] = model
    return model


def prefetch_ptlflow_model(model_name: str, ckpt_path: str | None = None) -> None:
    """Download the PTLFlow checkpoint into the local cache without moving it to GPU.

    Used to pre-download flow model weights at setup time (similar to how the
    main checkpoints are resolved/downloaded before inference), so the first
    flow sample does not stall on a network fetch.
    """
    resolved_name, resolved_ckpt = resolve_ptlflow_model(model_name, ckpt_path)
    cpu_key = (resolved_name, resolved_ckpt, "cpu")
    cuda_key = (resolved_name, resolved_ckpt, "cuda")
    if cpu_key in _MODEL_CACHE or cuda_key in _MODEL_CACHE:
        return

    try:
        import ptlflow
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PTLFlow is required for on-the-fly optical-flow control. "
            "Install it with `pip install ptlflow` or run the project dependency sync."
        ) from exc

    model = ptlflow.get_model(resolved_name, ckpt_path=resolved_ckpt).eval()
    _MODEL_CACHE[cpu_key] = model


@torch.no_grad()
def compute_flow_visualization(
    frames_cthw: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = 1,
    flow_model: str = "raft",
    flow_ckpt_path: str | None = None,
) -> np.ndarray:
    """Compute per-frame forward optical flow and return an RGB visualization.

    Args:
        frames_cthw: uint8 RGB video, shape (3, T, H, W).
        device: torch device. Defaults to cuda if available.
        batch_size: number of frame-pairs per PTLFlow forward pass. MemFlow is
            always run with batch size 1 so its recurrent memory follows the
            sequence order.
        flow_model: PTLFlow model name, or legacy alias ``raft``, ``waft``, or
            ``memflow``.
        flow_ckpt_path: PTLFlow checkpoint name or local checkpoint path. If
            None, a model-specific pretrained default is used, falling back to
            ``"sintel"`` for other PTLFlow models.

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

    model_name, ckpt_path = resolve_ptlflow_model(flow_model, flow_ckpt_path)
    model = _load_ptlflow_model(model_name, ckpt_path, device)

    from ptlflow.utils.io_adapter import IOAdapter

    _, _, H, W = frames_cthw.shape
    io_adapter = IOAdapter(
        output_stride=model.output_stride,
        input_size=(H, W),
        cuda=False,
    )

    # PTLFlow models follow the OpenCV convention at the API boundary (BGR,
    # float in [0, 1]); Cosmos video tensors are RGB uint8.
    frames = torch.from_numpy(frames_cthw).permute(1, 0, 2, 3).contiguous()
    frames = frames[:, [2, 1, 0]].float() / 255.0

    if hasattr(model, "clear_memory"):
        model.clear_memory()

    effective_batch_size = 1 if model_name.startswith("memflow") else batch_size
    flow_rgb_chunks = []
    for start in range(0, T - 1, effective_batch_size):
        end = min(start + effective_batch_size, T - 1)
        images = torch.stack((frames[start:end], frames[start + 1 : end + 1]), dim=1)
        inputs = io_adapter.prepare_inputs(inputs={"images": images})
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs["meta"] = {"is_seq_start": start == 0, "is_seq_end": end == T - 1}
        preds = model(inputs)
        preds = io_adapter.unscale(preds)
        flow = preds["flows"]
        if flow.dim() == 5:
            flow = flow[:, -1]
        if flow.dim() != 4:
            raise RuntimeError(f"PTLFlow model {model_name!r} returned unexpected flows shape {tuple(flow.shape)}")
        flow_rgb_chunks.append(flow_to_rgb(flow.float()))

    if hasattr(model, "clear_memory"):
        model.clear_memory()

    flow_rgb = torch.cat(flow_rgb_chunks, dim=0)
    flow_rgb = torch.cat([flow_rgb, flow_rgb[-1:].clone()], dim=0)
    return flow_rgb.permute(1, 0, 2, 3).contiguous().cpu().numpy()
