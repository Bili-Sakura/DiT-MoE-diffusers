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

from _bootstrap import bootstrap_repo_src

bootstrap_repo_src()

import torch

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers.transformer_dit_moe import DiTMoETransformer2DModel
from diffusers.schedulers.scheduling_flow_match_dit_moe import DiTMoEFlowMatchScheduler


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = REPO_ROOT.parents[1]

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "DiT-S/2": {"depth": 12, "hidden_size": 384, "patch_size": 2, "num_heads": 6},
    "DiT-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    "DiT-L/2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
    "DiT-G/2": {"depth": 40, "hidden_size": 1408, "patch_size": 2, "num_heads": 16},
}

DDIM_SCHEDULER_CONFIG = {
    "_class_name": "DDIMScheduler",
    "_diffusers_version": "0.36.0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "clip_sample": False,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "set_alpha_to_one": True,
    "steps_offset": 0,
    "trained_betas": None,
}

RF_SCHEDULER_CONFIG = {
    "_class_name": "DiTMoEFlowMatchScheduler",
    "_diffusers_version": "0.36.0",
    "mode": "ode",
    "num_train_timesteps": 1000,
    "path_type": "linear",
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


def infer_learn_sigma(state_dict: Dict[str, torch.Tensor], patch_size: int, in_channels: int = 4) -> bool:
    weight = state_dict.get("final_layer.linear.weight")
    if weight is None:
        return True
    base = patch_size * patch_size * in_channels
    return int(weight.shape[0]) == base * 2


def load_imagenet_id2label() -> Dict[int, str]:
    reference_paths = [
        COLLECTION_ROOT / "models/BiliSakura/DiT-diffusers/DiT-XL-2-256/model_index.json",
        COLLECTION_ROOT / "models/BiliSakura/DiT-diffusers/DiT-XL-2-512/model_index.json",
        COLLECTION_ROOT / "models/BiliSakura/NiT-diffusers/NiT-S/model_index.json",
    ]
    for path in reference_paths:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            id2label = raw.get("id2label")
            if isinstance(id2label, dict):
                return {int(key): value for key, value in id2label.items()}
    raise FileNotFoundError("Could not find a reference model_index.json with ImageNet id2label.")


def _save_config(output_dir: Path, config: Dict[str, Any], filename: str = "config.json"):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, "w", encoding="utf-8") as handle:
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


def _write_model_index(
    output_dir: Path,
    *,
    rectified_flow: bool,
):
    if rectified_flow:
        scheduler_entry = ["scheduling_flow_match_dit_moe", "DiTMoEFlowMatchScheduler"]
    else:
        scheduler_entry = ["diffusers", "DDIMScheduler"]

    id2label_int = load_imagenet_id2label()
    model_index = {
        "_class_name": ["pipeline", "DiTMoEPipeline"],
        "_diffusers_version": "0.36.0",
        "scheduler": scheduler_entry,
        "transformer": ["transformer_dit_moe", "DiTMoETransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "id2label": {str(class_id): id2label_int[class_id] for class_id in range(1000)},
    }
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2)
        handle.write("\n")


def _write_readme(output_dir: Path, *, variant_name: str, model: str, rectified_flow: bool, image_size: int):
    sampler = "rectified-flow (DiTMoEFlowMatchScheduler)" if rectified_flow else "DDIM"
    content = f"""---
license: apache-2.0
library_name: diffusers
pipeline_tag: unconditional-image-generation
tags:
  - diffusers
  - dit-moe
  - image-generation
  - class-conditional
  - imagenet
inference: true
---

# {variant_name}

Self-contained Diffusers checkpoint for **{model}** with MoE routing, converted from [`feizhengcong/DiT-MoE`](https://huggingface.co/feizhengcong/DiT-MoE).

Each subfolder is a self-contained Diffusers model repo with:

- `model_index.json` (includes ImageNet `id2label`)
- `pipeline.py` (custom `DiTMoEPipeline`)
- `transformer/transformer_dit_moe.py` and weights
- `vae/diffusion_pytorch_model.safetensors`
- `scheduler/scheduler_config.json`

## ImageNet class labels

Each variant keeps an English `id2label` map in `model_index.json` (DiT-style).

- `pipe.id2label[207]` — `"golden retriever"`
- `pipe.get_label_ids("golden retriever")` — `[207]`
- `pipe(class_labels="golden retriever", ...)` — string labels resolved automatically

## Recommended inference ({image_size}×{image_size})

| Setting | Value |
| --- | --- |
| Resolution | {image_size}×{image_size} |
| Sampler | {sampler} |
| Steps | 50 |
| CFG scale | 4.0 |
| VAE | `stabilityai/sd-vae-ft-mse` (bundled under `vae/`) |

```python
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

model_dir = Path("./{variant_name}").resolve()
pipe = DiffusionPipeline.from_pretrained(
    str(model_dir),
    local_files_only=True,
    custom_pipeline=str(model_dir / "pipeline.py"),
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
pipe.to("cuda")

print(pipe.id2label[207])
print(pipe.get_label_ids("golden retriever"))

generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(
    class_labels="golden retriever",
    height={image_size},
    width={image_size},
    num_inference_steps=50,
    guidance_scale=4.0,
    generator=generator,
).images[0]
image.save("demo.png")
```
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def make_self_contained_repo(output_dir: Path, *, rectified_flow: bool, variant_name: str, model: str, image_size: int):
    shutil.copy2(REPO_ROOT / "templates/pipeline.py", output_dir / "pipeline.py")
    shutil.copy2(
        REPO_ROOT / "src/diffusers/models/transformers/transformer_dit_moe.py",
        output_dir / "transformer/transformer_dit_moe.py",
    )

    scheduler_dir = output_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    legacy_config = scheduler_dir / "config.json"
    if legacy_config.exists():
        legacy_config.unlink()

    if rectified_flow:
        shutil.copy2(
            REPO_ROOT / "src/diffusers/schedulers/scheduling_flow_match_dit_moe.py",
            scheduler_dir / "scheduling_flow_match_dit_moe.py",
        )
        scheduler_config = RF_SCHEDULER_CONFIG
    else:
        scheduler_config = DDIM_SCHEDULER_CONFIG

    _save_config(scheduler_dir, scheduler_config, filename="scheduler_config.json")

    id2label = load_imagenet_id2label()
    _write_model_index(output_dir, rectified_flow=rectified_flow)
    _write_readme(
        output_dir,
        variant_name=variant_name,
        model=model,
        rectified_flow=rectified_flow,
        image_size=image_size,
    )


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
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=None)
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
    preset = MODEL_PRESETS[args.model]
    learn_sigma = infer_learn_sigma(state_dict, patch_size=preset["patch_size"]) if args.learn_sigma is None else args.learn_sigma
    config = {
        "input_size": latent_size,
        "in_channels": 4,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "pretraining_tp": args.pretraining_tp,
        "use_flash_attn": args.use_flash_attn,
        "learn_sigma": learn_sigma,
        **preset,
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

    scheduler_config = RF_SCHEDULER_CONFIG if args.rectified_flow else DDIM_SCHEDULER_CONFIG
    _save_config(scheduler_dir, scheduler_config, filename="scheduler_config.json")

    if args.copy_vae is not None:
        _copy_vae(Path(args.copy_vae), output_dir / "vae")
    elif args.vae:
        with open(output_dir / "vae_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as handle:
            handle.write(args.vae + os.linesep)

    variant_name = output_dir.name
    make_self_contained_repo(
        output_dir,
        rectified_flow=args.rectified_flow,
        variant_name=variant_name,
        model=args.model,
        image_size=args.image_size,
    )
    print(f"Saved Diffusers-style DiT-MoE pipeline to {output_dir}")


def _copy_vae(source_vae_dir: Path, target_vae_dir: Path):
    if target_vae_dir.exists():
        shutil.rmtree(target_vae_dir)
    shutil.copytree(source_vae_dir, target_vae_dir)

    safetensors_path = target_vae_dir / "diffusion_pytorch_model.safetensors"
    bin_path = target_vae_dir / "diffusion_pytorch_model.bin"
    if not safetensors_path.exists() and bin_path.exists():
        if safe_save_file is None:
            raise ImportError("Install safetensors to convert bundled VAE weights.")
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        safe_save_file(state_dict, str(safetensors_path), metadata={"format": "pt"})
        bin_path.unlink()


if __name__ == "__main__":
    main()
