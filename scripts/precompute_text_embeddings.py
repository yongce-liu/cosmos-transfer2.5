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

import copy
import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import pydantic
import torch
import tyro
from cosmos_oss.init import cleanup_environment, init_environment

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2.config import (
    DEFAULT_MODEL_KEY,
    MODEL_CHECKPOINTS,
    MODEL_KEYS,
    get_model_literal,
    handle_tyro_exception,
    is_rank0,
)

EncoderClass = Literal["auto", "T5", "reason1_2B", "reason1_7B", "reason1p1_7B"]
TensorDType = Literal["bfloat16", "float32"]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Directory used to save exported embeddings."""

    prompt: str | None = None
    """Positive prompt text."""
    prompt_path: Path | None = None
    """Path to a text file containing the positive prompt."""
    prompt_output_name: str = "prompt_embedding.pt"
    """Output filename for the positive prompt embedding."""

    negative_prompt: str | None = None
    """Negative prompt text."""
    negative_prompt_path: Path | None = None
    """Path to a text file containing the negative prompt."""
    negative_output_name: str = "negative_prompt_embedding.pt"
    """Output filename for the negative prompt embedding."""

    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal() = DEFAULT_MODEL_KEY.name
    """Model whose text encoder config should be mirrored when `text_encoder_class=auto`."""
    experiment: str | None = None
    """Optional experiment override used to resolve the text encoder config."""
    config_file: str = ""
    """Optional config file override used to resolve the text encoder config."""
    text_encoder_class: EncoderClass = "auto"
    """Text encoder to export with. `auto` mirrors the selected model config."""

    dtype: TensorDType = "bfloat16"
    """Output tensor dtype."""
    device: str = "cuda"
    """Device used to compute embeddings."""
    cache_dir: str | None = None
    """Optional cache directory used by the T5 text encoder."""
    overwrite: bool = False
    """Overwrite existing output files."""

    @pydantic.model_validator(mode="after")
    def validate_inputs(self):
        if self.prompt is not None and self.prompt_path is not None:
            raise ValueError("Only one of prompt or prompt_path can be provided.")
        if self.negative_prompt is not None and self.negative_prompt_path is not None:
            raise ValueError("Only one of negative_prompt or negative_prompt_path can be provided.")
        if (
            self.prompt is None
            and self.prompt_path is None
            and self.negative_prompt is None
            and self.negative_prompt_path is None
        ):
            raise ValueError("At least one of prompt/prompt_path or negative_prompt/negative_prompt_path is required.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("device=cuda was requested, but CUDA is not available.")
        return self


def _load_prompt_text(text: str | None, path: Path | None) -> str | None:
    if text is not None:
        return text
    if path is None:
        return None
    return path.read_text().strip()


def _resolve_output_dtype(dtype_name: TensorDType) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _resolve_model_config(
    model_name: str, experiment_override: str | None, config_file_override: str
) -> tuple[str, str, str, list[str]]:
    model_key = MODEL_KEYS[model_name]
    checkpoint = MODEL_CHECKPOINTS[model_key]
    resolved_experiment = experiment_override or checkpoint.experiment
    resolved_config_file = config_file_override or (
        "cosmos_transfer2/_src/interactive/configs/registry_transfer2p5.py"
        if model_key.distilled
        else "cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py"
    )

    if model_key.distilled:
        registered_exp_name = resolved_experiment
        exp_override_opts: list[str] = []
    elif experiment_override is not None:
        registered_exp_name = resolved_experiment
        exp_override_opts = []
    else:
        registered_exp_name = EXPERIMENTS[resolved_experiment].registered_exp_name
        exp_override_opts = EXPERIMENTS[resolved_experiment].command_args.copy()

    return resolved_experiment, resolved_config_file, registered_exp_name, exp_override_opts


def _load_model_text_encoder_config(
    resolved_config_file: str,
    registered_exp_name: str,
    exp_override_opts: list[str],
) -> tuple[str, object | None]:
    config_module = get_config_module(resolved_config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={registered_exp_name}"] + exp_override_opts)
    model_config = config.model.config
    return model_config.text_encoder_class, model_config.text_encoder_config


def _compute_embeddings(
    prompt_text: str,
    encoder_class: str,
    encoder_config: object | None,
    device: str,
    cache_dir: str | None,
) -> torch.Tensor:
    if encoder_class == "T5":
        from cosmos_transfer2._src.predict2.inference.get_t5_emb import get_text_embedding

        return get_text_embedding(
            prompt_text,
            device=device,
            cache_dir=cache_dir,
            text_encoder_class="T5",
        )

    if device != "cuda":
        raise ValueError(f"{encoder_class} offline export currently requires device=cuda.")

    if encoder_config is None:
        raise ValueError(f"text_encoder_config is required to compute embeddings for {encoder_class}.")

    from cosmos_transfer2._src.predict2.text_encoders.text_encoder import TextEncoder

    encoder_config = copy.deepcopy(encoder_config)
    encoder_config.compute_online = True
    text_encoder = TextEncoder(encoder_config, device=device)
    try:
        return text_encoder.compute_text_embeddings_online({"prompt": [prompt_text]}, "prompt")
    finally:
        if hasattr(text_encoder, "model") and text_encoder.model is not None:
            text_encoder.model = text_encoder.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_embedding(
    output_path: Path,
    embedding: torch.Tensor,
    role: str,
    prompt_text: str,
    encoder_class: str,
    model_name: str,
    experiment_name: str,
    config_file: str,
    output_dtype: torch.dtype,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Re-run with --overwrite to replace it.")

    payload = {
        "t5_text_embeddings": embedding.detach().to(device="cpu", dtype=output_dtype).contiguous(),
        "metadata": {
            "role": role,
            "prompt": prompt_text,
            "text_encoder_class": encoder_class,
            "model": model_name,
            "experiment": experiment_name,
            "config_file": config_file,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "dtype": str(output_dtype),
        },
    }
    torch.save(payload, output_path)
    log.info(f"Saved {role} embedding to {output_path} with shape {tuple(payload['t5_text_embeddings'].shape)}")


def main(args: Args) -> None:
    prompt_text = _load_prompt_text(args.prompt, args.prompt_path)
    negative_prompt_text = _load_prompt_text(args.negative_prompt, args.negative_prompt_path)
    output_dtype = _resolve_output_dtype(args.dtype)

    resolved_experiment, resolved_config_file, registered_exp_name, exp_override_opts = _resolve_model_config(
        model_name=args.model,
        experiment_override=args.experiment,
        config_file_override=args.config_file,
    )

    model_encoder_class: str | None = None
    encoder_config: object | None = None
    needs_model_text_encoder_config = args.text_encoder_class == "auto" or args.text_encoder_class != "T5"

    if needs_model_text_encoder_config:
        model_encoder_class, encoder_config = _load_model_text_encoder_config(
            resolved_config_file=resolved_config_file,
            registered_exp_name=registered_exp_name,
            exp_override_opts=exp_override_opts,
        )

    resolved_encoder_class = model_encoder_class if args.text_encoder_class == "auto" else args.text_encoder_class
    if (
        args.text_encoder_class != "auto"
        and model_encoder_class is not None
        and args.text_encoder_class != model_encoder_class
    ):
        log.warning(
            f"Requested text_encoder_class={args.text_encoder_class}, while model {args.model} resolves to {model_encoder_class}."
        )

    log.info(
        "Resolved encoder config: "
        f"model={args.model}, experiment={resolved_experiment}, text_encoder_class={resolved_encoder_class}"
    )

    if prompt_text is not None:
        prompt_embedding = _compute_embeddings(
            prompt_text=prompt_text,
            encoder_class=resolved_encoder_class,
            encoder_config=encoder_config,
            device=args.device,
            cache_dir=args.cache_dir,
        )
        _save_embedding(
            output_path=args.output_dir / args.prompt_output_name,
            embedding=prompt_embedding,
            role="positive",
            prompt_text=prompt_text,
            encoder_class=resolved_encoder_class,
            model_name=args.model,
            experiment_name=resolved_experiment,
            config_file=resolved_config_file,
            output_dtype=output_dtype,
            overwrite=args.overwrite,
        )

    if negative_prompt_text is not None:
        negative_embedding = _compute_embeddings(
            prompt_text=negative_prompt_text,
            encoder_class=resolved_encoder_class,
            encoder_config=encoder_config,
            device=args.device,
            cache_dir=args.cache_dir,
        )
        _save_embedding(
            output_path=args.output_dir / args.negative_output_name,
            embedding=negative_embedding,
            role="negative",
            prompt_text=negative_prompt_text,
            encoder_class=resolved_encoder_class,
            model_name=args.model,
            experiment_name=resolved_experiment,
            config_file=resolved_config_file,
            output_dtype=output_dtype,
            overwrite=args.overwrite,
        )

    if resolved_encoder_class == "T5":
        from cosmos_transfer2._src.predict2.inference.get_t5_emb import offload_text_encoder

        offload_text_encoder(device="cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    init_environment()
    try:
        cli_args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as exc:
        handle_tyro_exception(exc)

    try:
        main(cli_args)
    finally:
        cleanup_environment()
