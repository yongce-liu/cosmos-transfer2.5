"""Download Cosmos-Transfer2.5 checkpoints into the local HF cache.

Examples
--------
List all registered checkpoints (grouped):

    uv run --env-file .env python ./scripts/download_checkpoints.py -l

Download the minimum, ready-to-run bundle (prerequisites + base predict + edge/
depth transfer + sam2/grounding-dino/depth-anything/siglip/siglip2/qwen-guard
auxiliary models + guardrail). One-click setup for inference:

    uv run --env-file .env python ./scripts/download_checkpoints.py --minimum

Download all "prerequisite" assets (tokenizer / VAE / Reason1 / Guardrail):

    uv run --env-file .env python ./scripts/download_checkpoints.py -g prerequisites

Download every transfer-control checkpoint plus prerequisites:

    uv run --env-file .env python ./scripts/download_checkpoints.py -g transfer

Download just the auxiliary HF models (sam2 / grounding-dino / depth-anything /
siglip / siglip2 / Qwen3Guard) used by control-video preprocessing & guardrail:

    uv run --env-file .env python ./scripts/download_checkpoints.py -g auxiliary

Download a single model by name or UUID:

    uv run --env-file .env python ./scripts/download_checkpoints.py \
        -m nvidia/Cosmos-Transfer2.5-2B/general/edge

Download everything (use COSMOS_EXPERIMENTAL_CHECKPOINTS=1 to include EA models):

    uv run --env-file .env python ./scripts/download_checkpoints.py --all
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated

import tyro
from cosmos_oss.checkpoints_transfer2 import register_checkpoints
from loguru import logger

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common.core import (
    GUARDRAIL1_CHECKPOINT,
)
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import (
    _CHECKPOINTS,
    CheckpointConfig,
    CheckpointDirHf,
    download_checkpoint,
)

# Populate the checkpoint registry; otherwise _CHECKPOINTS stays empty
# and `--list` produces no entries.
register_checkpoints()


# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------
# Groups are matched against ``CheckpointConfig.name`` using simple prefix /
# substring rules. The order below also defines the listing order.

# Logical group identifier -> human readable description.
GROUP_DESCRIPTIONS: dict[str, str] = {
    "prerequisites": "Shared assets required by every pipeline (tokenizer / VAE / Reason1 / Guardrail).",
    "predict": "Cosmos-Predict2.5 base / post-trained / multiview / robot world models.",
    "transfer": "Cosmos-Transfer2.5 control models (edge / depth / blur / seg / multiview / distilled).",
    "auxiliary": "Auxiliary HF models for control-video preprocessing & guardrail "
    "(sam2 / grounding-dino / depth-anything / siglip / siglip2 / Qwen3Guard).",
}

# Repository names that count as "prerequisites" (non model-weights deps).
_PREREQUISITE_REPOS: tuple[str, ...] = (
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Wan2.1/vae",
    "nvidia/Cosmos-Reason1.1-7B",
)

# Name prefixes that identify each model family.
_PREDICT_PREFIX = "nvidia/Cosmos-Predict2.5"
_TRANSFER_PREFIX = "nvidia/Cosmos-Transfer2.5"


# ---------------------------------------------------------------------------
# Auxiliary HF models (sam2 / grounding-dino / depth-anything / siglip / etc.)
# ---------------------------------------------------------------------------
# These third-party HuggingFace repositories are pulled at runtime by various
# preprocessing & guardrail components, but are NOT registered in `_CHECKPOINTS`.
# We expose them here so users can fetch them up-front (offline-friendly).
#
# Each entry is (repo_id, revision). Revisions default to "main" because these
# upstream repos do not pin a specific commit in the codebase. Pin to a SHA if
# strict reproducibility is required.
AUXILIARY_HF_REPOS: tuple[tuple[str, str], ...] = (
    # SAM2 video segmentation (used by `seg` control & on-the-fly mask gen).
    ("facebook/sam2-hiera-large", "main"),
    # GroundingDINO text-prompted detector (paired with SAM2).
    # ("IDEA-Research/grounding-dino-base", "main"),
    # Video-Depth-Anything-Small for `depth` control preprocessing.
    ("depth-anything/Video-Depth-Anything-Small", "main"),
    # SigLIP vision encoder used by the video-content-safety guardrail.
    # ("google/siglip-so400m-patch14-384", "main"),
    # SigLIP2 image-context encoder used by the conditioner.
    ("google/siglip2-so400m-patch16-naflex", "main"),
    # Qwen3Guard text guardrail.
    ("Qwen/Qwen3Guard-Gen-0.6B", "main"),
)


# ---------------------------------------------------------------------------
# Minimum one-click bundle
# ---------------------------------------------------------------------------
# Names below MUST match `CheckpointConfig.name` values registered by
# `register_checkpoints()`. They define the smallest set required to run the
# documented single-view edge & depth transfer demos out of the box.
MINIMUM_MODEL_NAMES: tuple[str, ...] = (
    "nvidia/Cosmos-Predict2.5-2B/base/pre-trained",
    "nvidia/Cosmos-Transfer2.5-2B/general/edge",
    "nvidia/Cosmos-Transfer2.5-2B/general/depth",
)

# Auxiliary repos required by the minimum set (depth + seg preprocessing,
# guardrail vision encoder, conditioner).
MINIMUM_AUX_REPOS: tuple[str, ...] = (
    "depth-anything/Video-Depth-Anything-Small",
    "facebook/sam2-hiera-large",
    "IDEA-Research/grounding-dino-base",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-so400m-patch16-naflex",
    "Qwen/Qwen3Guard-Gen-0.6B",
)


def _group_of(cfg: CheckpointConfig) -> str:
    """Classify a checkpoint into one of GROUP_DESCRIPTIONS."""
    if cfg.name in _PREREQUISITE_REPOS:
        return "prerequisites"
    if cfg.name.startswith(_PREDICT_PREFIX):
        return "predict"
    if cfg.name.startswith(_TRANSFER_PREFIX):
        return "transfer"
    # Anything unknown is treated as a prerequisite so it still gets downloaded
    # with the default `prerequisites` group.
    return "prerequisites"


@dataclass
class _Entry:
    """A single resolved checkpoint to be downloaded."""

    label: str  # human friendly identifier (variant name or UUID)
    config: CheckpointConfig


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------
def _all_registered() -> list[CheckpointConfig]:
    """Return unique registered checkpoint configs (de-duplicated by UUID)."""
    seen: set[str] = set()
    out: list[CheckpointConfig] = []
    for cfg in _CHECKPOINTS.values():
        if cfg.uuid in seen:
            continue
        seen.add(cfg.uuid)
        out.append(cfg)
    return out


def _grouped_registered() -> dict[str, list[CheckpointConfig]]:
    """Return registered configs bucketed by group, preserving declaration order.

    Note: the 'auxiliary' group only contains plain HF repos (not registered in
    `_CHECKPOINTS`), so its bucket is always returned empty here and handled
    separately by `_collect_auxiliary` / the listing helper.
    """
    buckets: dict[str, list[CheckpointConfig]] = {g: [] for g in GROUP_DESCRIPTIONS}
    for cfg in _all_registered():
        buckets[_group_of(cfg)].append(cfg)
    return buckets


def _resolve_uri(uri: str) -> CheckpointConfig | None:
    """Resolve a UUID / S3 URI / registered alias to a config."""
    return CheckpointConfig.maybe_from_uri(uri)


def _resolve_name(name: str) -> CheckpointConfig | None:
    """Resolve a checkpoint by its registered ``name`` (e.g. variant alias)."""
    for cfg in _all_registered():
        if cfg.name == name:
            return cfg
    # Fall back to URI resolution so users can pass UUID / S3 alike.
    return _resolve_uri(name)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def _print_listing() -> None:
    logger.info("Registered checkpoints (grouped):")
    grouped = _grouped_registered()
    # Compute a global column width so output stays aligned across groups
    # (incl. the unregistered auxiliary entries below).
    aux_names = [repo for repo, _ in AUXILIARY_HF_REPOS]
    width = max(
        max((len(cfg.name) for cfgs in grouped.values() for cfg in cfgs), default=1),
        max((len(n) for n in aux_names), default=1),
        len("nvidia/Cosmos-Guardrail1"),
    )
    for group, cfgs in grouped.items():
        if group == "auxiliary" or not cfgs:
            continue
        logger.info("")
        logger.info(f"[{group}] {GROUP_DESCRIPTIONS[group]}")
        for cfg in cfgs:
            logger.info(f"  {cfg.name:<{width}}  {cfg.uuid}")
    # Auxiliary HF repos (sam2 / depth-anything / siglip / etc.).
    logger.info("")
    logger.info(f"[auxiliary] {GROUP_DESCRIPTIONS['auxiliary']}")
    for repo, revision in AUXILIARY_HF_REPOS:
        logger.info(f"  {repo:<{width}}  @{revision}")
    # Guardrail bundle does not live in _CHECKPOINTS but is always required.
    logger.info("")
    logger.info("[prerequisites] (extra, always required)")
    logger.info(f"  {'nvidia/Cosmos-Guardrail1':<{width}}  {GUARDRAIL1_CHECKPOINT.revision}")
    # Minimum bundle hint.
    logger.info("")
    logger.info("[minimum] One-click ready-to-run bundle (use --minimum):")
    for name in MINIMUM_MODEL_NAMES:
        logger.info(f"  {name:<{width}}  (model)")
    for repo in MINIMUM_AUX_REPOS:
        logger.info(f"  {repo:<{width}}  (auxiliary)")
    logger.info(f"  {'nvidia/Cosmos-Guardrail1':<{width}}  (guardrail)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Args:
    """Download Cosmos-Transfer2.5 checkpoints into the local HF cache.

    See module docstring for usage examples.
    """

    list: Annotated[bool, tyro.conf.arg(aliases=("-l",))] = False
    """List registered checkpoints grouped by category, then exit."""

    groups: Annotated[tuple[str, ...], tyro.conf.arg(aliases=("-g",))] = ()
    """Logical groups to download. Choices: prerequisites / predict / transfer / auxiliary."""

    models: Annotated[tuple[str, ...], tyro.conf.arg(aliases=("-m",))] = ()
    """Specific checkpoint names (e.g. 'nvidia/Cosmos-Transfer2.5-2B/general/edge')."""

    uris: Annotated[tuple[str, ...], tyro.conf.arg(aliases=("-u",))] = ()
    """Checkpoint UUID or S3 URI. Same identifiers accepted by inference."""

    all: bool = False
    """Download every registered checkpoint.

    Combine with ``COSMOS_EXPERIMENTAL_CHECKPOINTS=1`` (in env or .env) to
    include experimental checkpoints.
    """

    minimum: Annotated[bool, tyro.conf.arg(aliases=("-M",))] = False
    """One-click minimum, ready-to-run bundle.

    Equivalent to: prerequisites + base predict + edge/depth transfer +
    auxiliary HF models (sam2 / grounding-dino / depth-anything /
    siglip / siglip2 / Qwen3Guard) + Cosmos-Guardrail1.
    """

    skip_prerequisites: bool = False
    """Skip auto-downloading the prerequisites bundle (incl. Guardrail) when a
    model group is requested."""

    dry_run: bool = False
    """Only print what would be downloaded."""


def _validate_groups(groups: tuple[str, ...]) -> None:
    bad = [g for g in groups if g not in GROUP_DESCRIPTIONS]
    if bad:
        raise SystemExit(f"Unknown group(s): {bad}. Available: {list(GROUP_DESCRIPTIONS)}")


def _wants_prerequisites(args: Args) -> bool:
    """Return True if prerequisites (incl. guardrail) should be auto-added."""
    if args.skip_prerequisites:
        return False
    if args.all or args.minimum:
        return True
    # Auto-add prerequisites whenever the user asks for an actual model family.
    requested_families = {g for g in args.groups if g != "prerequisites"}
    return bool(requested_families or args.models or args.uris) or ("prerequisites" in args.groups)


def _collect(args: Args) -> list[_Entry]:
    """Resolve CLI inputs into a deduplicated list of checkpoints to fetch."""
    entries: list[_Entry] = []
    used_uuids: set[str] = set()

    def _add(label: str, cfg: CheckpointConfig | None) -> None:
        if cfg is None:
            logger.warning(f"'{label}' is not registered, skipped")
            return
        if cfg.uuid in used_uuids:
            return
        used_uuids.add(cfg.uuid)
        entries.append(_Entry(label=label, config=cfg))

    if args.all:
        for cfg in _all_registered():
            _add(cfg.name, cfg)
        return entries

    grouped = _grouped_registered()

    # Prerequisites first so they appear at the top of the plan.
    if _wants_prerequisites(args):
        for cfg in grouped["prerequisites"]:
            _add(cfg.name, cfg)

    # Minimum bundle: pre-resolved registered checkpoints required to run the
    # documented edge/depth transfer demos out of the box.
    if args.minimum:
        for name in MINIMUM_MODEL_NAMES:
            _add(name, _resolve_name(name))

    for group in args.groups:
        if group in ("prerequisites", "auxiliary"):
            # 'prerequisites' is handled above; 'auxiliary' is not in
            # `_CHECKPOINTS` and is downloaded via `_collect_auxiliary`.
            continue
        for cfg in grouped.get(group, ()):
            _add(cfg.name, cfg)

    for name in args.models:
        _add(name, _resolve_name(name))

    for uri in args.uris:
        _add(uri, _resolve_uri(uri))

    return entries


def _collect_auxiliary(args: Args) -> list[tuple[str, str]]:
    """Resolve auxiliary HF repos to download (repo_id, revision)."""
    if args.all:
        return list(AUXILIARY_HF_REPOS)
    selected: list[tuple[str, str]] = []
    repo_to_rev = dict(AUXILIARY_HF_REPOS)
    if "auxiliary" in args.groups:
        selected.extend(AUXILIARY_HF_REPOS)
    if args.minimum:
        for repo in MINIMUM_AUX_REPOS:
            if repo in repo_to_rev and (repo, repo_to_rev[repo]) not in selected:
                selected.append((repo, repo_to_rev[repo]))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for repo, rev in selected:
        if repo in seen:
            continue
        seen.add(repo)
        out.append((repo, rev))
    return out


def _download_auxiliary(repo: str, revision: str, dry_run: bool) -> bool:
    """Download a single auxiliary HF repository. Returns True on success."""
    logger.info(f"==> auxiliary:{repo}  (revision={revision})")
    logger.info(f"    repo: {repo}@{revision}")
    if dry_run:
        return True
    try:
        path = CheckpointDirHf(repository=repo, revision=revision).download()
        logger.info(f"    -> {path}")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"    {exc}")
        return False


def _download_guardrail(dry_run: bool) -> bool:
    """Download the Cosmos-Guardrail1 bundle. Returns True on success."""
    label = f"guardrail:{GUARDRAIL1_CHECKPOINT.repository}"
    logger.info(f"==> {label}  (revision={GUARDRAIL1_CHECKPOINT.revision})")
    logger.info(f"    repo: {GUARDRAIL1_CHECKPOINT.repository}@{GUARDRAIL1_CHECKPOINT.revision}")
    if dry_run:
        return True
    try:
        path = GUARDRAIL1_CHECKPOINT.download()
        logger.info(f"    -> {path}")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"    {exc}")
        return False


def main(args: Args) -> int:
    if args.list:
        _print_listing()
        return 0

    _validate_groups(args.groups)

    if not (args.groups or args.models or args.uris or args.all or args.minimum):
        logger.error("Nothing to download. Pass --list, --groups, --models, --uris, --minimum or --all.")
        return 2

    entries = _collect(args)
    aux_entries = _collect_auxiliary(args)
    download_guardrail = _wants_prerequisites(args) or args.all

    if not entries and not aux_entries and not download_guardrail:
        logger.error("No matching checkpoints resolved.")
        return 1

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    logger.info(f"HF cache         : {hf_home}")
    total = len(entries) + len(aux_entries) + (1 if download_guardrail else 0)
    logger.info(f"Resolved {total} checkpoint(s) to download.")

    failed = 0
    for entry in entries:
        cfg = entry.config
        logger.info(f"==> {entry.label}  ({cfg.uuid})")
        logger.info(f"    repo: {cfg.hf.repository}@{cfg.hf.revision}")
        if args.dry_run:
            continue
        try:
            path = download_checkpoint(entry.config.s3.uri)
            logger.info(f"    -> {path}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error(f"    {exc}")

    for repo, revision in aux_entries:
        if not _download_auxiliary(repo, revision, args.dry_run):
            failed += 1

    if download_guardrail:
        if not _download_guardrail(args.dry_run):
            failed += 1

    if failed:
        logger.error(f"[DONE] {failed} checkpoint(s) failed.")
        return 1
    logger.info("[DONE] All checkpoints are ready locally.")
    logger.info("Tip: export HF_HUB_OFFLINE=1 to run inference fully offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
