#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers.transformer_dit_moe import DiTMoETransformer2DModel
from diffusers.schedulers.scheduling_flow_match_dit_moe import DiTMoEFlowMatchScheduler


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "DiT-S/2": {"depth": 12, "hidden_size": 384, "patch_size": 2, "num_heads": 6},
    "DiT-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    "DiT-L/2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
    "DiT-G/2": {"depth": 40, "hidden_size": 1408, "patch_size": 2, "num_heads": 16},
}


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module", "ema"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "transformer.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _save_config(output_dir: Path, config: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(state_dict, str(output_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path, scheduler_class: str, vae: str | None):
    model_index = {
        "_class_name": "DiTMoEPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", scheduler_class],
        "transformer": ["diffusers", "DiTMoETransformer2DModel"],
    }
    if vae is not None:
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert original DiT-MoE checkpoints to a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to an original DiT-MoE .pt/.bin/.safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--pretraining-tp", type=int, default=2)
    parser.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rectified-flow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--copy-vae", default=None, help="Optional local VAE directory to copy into output/vae.")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    latent_size = args.image_size // 8
    state_dict = _load_state_dict(args.checkpoint)
    config = {
        "input_size": latent_size,
        "in_channels": 4,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "pretraining_tp": args.pretraining_tp,
        "use_flash_attn": args.use_flash_attn,
        "learn_sigma": args.learn_sigma,
        **MODEL_PRESETS[args.model],
    }

    if args.check_load:
        model = DiTMoETransformer2DModel(**config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            raise SystemExit(1)

    _save_config(transformer_dir, {"_class_name": "DiTMoETransformer2DModel", **config})
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    if args.rectified_flow:
        scheduler_config = {
            "_class_name": "DiTMoEFlowMatchScheduler",
            "num_train_timesteps": 1000,
            "mode": "ode",
            "path_type": "linear",
        }
        scheduler_class = "DiTMoEFlowMatchScheduler"
    else:
        scheduler_config = {
            "_class_name": "DDIMScheduler",
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "clip_sample": False,
            "set_alpha_to_one": True,
            "steps_offset": 0,
            "prediction_type": "epsilon",
        }
        scheduler_class = "DDIMScheduler"

    _save_config(scheduler_dir, scheduler_config)

    if args.copy_vae is not None:
        target_vae_dir = output_dir / "vae"
        if target_vae_dir.exists():
            shutil.rmtree(target_vae_dir)
        shutil.copytree(args.copy_vae, target_vae_dir)
    elif args.vae:
        with open(output_dir / "vae_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as handle:
            handle.write(args.vae + os.linesep)

    _write_model_index(output_dir, scheduler_class, args.vae)
    print(f"Saved Diffusers-style DiT-MoE pipeline to {output_dir}")


if __name__ == "__main__":
    main()
