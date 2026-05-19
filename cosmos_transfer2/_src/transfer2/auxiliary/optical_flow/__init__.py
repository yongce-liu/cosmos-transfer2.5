# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optical flow estimation through PTLFlow.

``flow_model`` accepts any PTLFlow model name. The legacy names ``"raft"``,
``"waft"``, and ``"memflow"`` are still accepted and resolved to PTLFlow models.
"""

from __future__ import annotations

from .ptlflow_flow_model import (
    DEFAULT_PTLFLOW_CHECKPOINTS,
    LEGACY_FLOW_MODEL_ALIASES,
    compute_flow_visualization,
    prefetch_ptlflow_model,
    resolve_ptlflow_model,
)

SUPPORTED_FLOW_MODELS = tuple(sorted(set(DEFAULT_PTLFLOW_CHECKPOINTS) | set(LEGACY_FLOW_MODEL_ALIASES)))

__all__ = [
    "DEFAULT_PTLFLOW_CHECKPOINTS",
    "LEGACY_FLOW_MODEL_ALIASES",
    "SUPPORTED_FLOW_MODELS",
    "compute_flow_visualization",
    "prefetch_ptlflow_model",
    "resolve_ptlflow_model",
]
