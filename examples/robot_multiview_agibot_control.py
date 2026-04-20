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

"""Transfer2.5 Agibot control-conditioned multiview inference entry point."""

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2.config import handle_tyro_exception, is_rank0
from cosmos_transfer2.robot_multiview_control_agibot_config import (
    RobotMultiviewControlAgibotInferenceArguments,
    RobotMultiviewControlAgibotSetupArguments,
)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s).
    If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.
    """
    setup: RobotMultiviewControlAgibotSetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: RobotMultiviewControlAgibotInferenceArguments | None = None
    """Inference parameter overrides. These can either be provided in the input json file or via CLI."""


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _get_first_input_file(argv: list[str]) -> Path | None:
    for i, token in enumerate(argv):
        if token in {"-i", "--input-files"} and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser().resolve()
        if token.startswith("-i=") or token.startswith("--input-files="):
            return Path(token.split("=", 1)[1]).expanduser().resolve()
    return None


def _autofill_setup_args(argv: list[str]) -> list[str]:
    if _has_flag(argv, "--input-root") and _has_flag(argv, "--control-type"):
        return argv

    input_file = _get_first_input_file(argv)
    if input_file is None:
        return argv

    argv = list(argv)
    if not _has_flag(argv, "--input-root"):
        argv.extend(["--input-root", str(input_file.parent)])
    if not _has_flag(argv, "--control-type"):
        control_type = None
        if input_file.suffix == ".json":
            data = json.loads(input_file.read_text())
            if isinstance(data, dict):
                control_type = data.get("control_type")
        if control_type is None:
            for control_type in ("depth", "edge", "vis", "seg"):
                if control_type in input_file.stem.lower():
                    break
            else:
                control_type = None
        if control_type is not None:
            argv.extend(["--control-type", control_type])
    return argv


def _strip_setup_only_fields(data: object) -> object:
    if not isinstance(data, dict):
        return data

    sanitized = dict(data)
    # Allow setup-only fields in json inputs for CLI autofill without breaking inference validation.
    sanitized.pop("control_type", None)
    return sanitized


def _load_inference_args(
    input_files: list[Path],
    overrides: RobotMultiviewControlAgibotInferenceArguments | None = None,
) -> list[RobotMultiviewControlAgibotInferenceArguments]:
    override_data = {} if overrides is None else overrides.model_dump(exclude_none=True)
    inference_args: list[RobotMultiviewControlAgibotInferenceArguments] = []

    for path in input_files:
        if path.suffix == ".json":
            data_list = [json.loads(path.read_text())]
        elif path.suffix == ".jsonl":
            data_list = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        cwd = os.getcwd()
        os.chdir(path.parent)
        try:
            for i, data in enumerate(data_list):
                try:
                    sanitized_data = _strip_setup_only_fields(data)
                    if isinstance(sanitized_data, dict):
                        sanitized_data = sanitized_data | override_data
                    inference_args.append(RobotMultiviewControlAgibotInferenceArguments.model_validate(sanitized_data))
                except pydantic.ValidationError as e:
                    if is_rank0():
                        print(f"Error validating parameters from '{path}' at line {i}\n{e}", file=sys.stderr)
                    sys.exit(1)
        finally:
            os.chdir(cwd)

    return inference_args


def main(args: Args) -> None:
    inference_args = _load_inference_args(args.input_files, overrides=args.overrides)
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    # Use class-based inference API (consistent with multiview.py)
    from cosmos_transfer2.robot_multiview import RobotMultiviewControlAgibotInference

    inference_pipeline = RobotMultiviewControlAgibotInference(args.setup)
    inference_pipeline.generate(inference_args, args.setup.output_dir)
    inference_pipeline.cleanup()


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            args=_autofill_setup_args(sys.argv[1:]),
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
