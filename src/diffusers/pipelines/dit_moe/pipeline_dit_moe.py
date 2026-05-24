# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from ..._hf_diffusers import get_hf_attr
from ...models.transformers.transformer_dit_moe import DiTMoETransformer2DModel

VaeImageProcessor = get_hf_attr("diffusers.image_processor.VaeImageProcessor")
DiffusionPipeline = get_hf_attr("diffusers.pipelines.pipeline_utils.DiffusionPipeline")
ImagePipelineOutput = get_hf_attr("diffusers.pipelines.pipeline_utils.ImagePipelineOutput")
DDIMScheduler = get_hf_attr("diffusers.schedulers.DDIMScheduler")
randn_tensor = get_hf_attr("diffusers.utils.torch_utils.randn_tensor")
from ...schedulers.scheduling_flow_match_dit_moe import DiTMoEFlowMatchScheduler


class DiTMoEPipeline(DiffusionPipeline):
    r"""
    Pipeline for class-conditional image generation with DiT-MoE.

    Supports DDIM diffusion sampling and rectified-flow (RF) sampling.
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer: DiTMoETransformer2DModel,
        scheduler: Union[DDIMScheduler, DiTMoEFlowMatchScheduler],
        vae=None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)
        self._use_rectified_flow = isinstance(scheduler, DiTMoEFlowMatchScheduler)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        vae_subfolder = model_kwargs.pop("vae_subfolder", None)
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"
        if vae_subfolder is None and (base_path / "vae").exists():
            vae_subfolder = "vae"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            transformer_path = (
                str(base_path / transformer_subfolder) if transformer_subfolder else pretrained_model_name_or_path
            )
            transformer = DiTMoETransformer2DModel.from_pretrained(transformer_path, **model_kwargs)

            scheduler = None
            scheduler_config_path = base_path / (scheduler_subfolder or "scheduler") / "scheduler_config.json"
            if scheduler_config_path.exists():
                scheduler_config = json.loads(scheduler_config_path.read_text(encoding="utf-8"))
                if scheduler_config.get("_class_name") == "DiTMoEFlowMatchScheduler":
                    scheduler = DiTMoEFlowMatchScheduler.from_pretrained(
                        pretrained_model_name_or_path,
                        subfolder=scheduler_subfolder or "scheduler",
                    )
                else:
                    scheduler = DDIMScheduler.from_pretrained(
                        pretrained_model_name_or_path,
                        subfolder=scheduler_subfolder or "scheduler",
                    )
            if scheduler is None:
                scheduler = DDIMScheduler(num_train_timesteps=1000)

            vae = None
            if vae_subfolder is not None:
                try:
                    autoencoder_kl = get_hf_attr("diffusers.AutoencoderKL")
                    vae = autoencoder_kl.from_pretrained(str(base_path / vae_subfolder), **model_kwargs)
                except Exception:
                    vae = None

            return cls(transformer=transformer, scheduler=scheduler, vae=vae)

    def _get_null_labels(self, batch_size: int, device: torch.device) -> torch.LongTensor:
        return torch.full((batch_size,), self.transformer.config.num_classes, device=device, dtype=torch.long)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> torch.Tensor:
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        return randn_tensor(
            (batch_size, num_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _apply_cfg(self, model_output: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        if guidance_scale <= 1.0:
            return model_output
        cond, uncond = model_output.chunk(2)
        if self.transformer.learn_sigma:
            eps_cond, rest_cond = cond[:, : self.transformer.in_channels], cond[:, self.transformer.in_channels :]
            eps_uncond, rest_uncond = uncond[:, : self.transformer.in_channels], uncond[:, self.transformer.in_channels :]
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            guided = torch.cat([eps, rest_cond], dim=1)
        else:
            guided = uncond + guidance_scale * (cond - uncond)
        return guided

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if self.vae is None:
            if output_type == "latent":
                return latents
            raise ValueError("Cannot decode latents without a VAE.")
        latents = latents / 0.18215
        if output_type == "latent":
            return latents
        image = self.vae.decode(latents).sample
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        device = self._execution_device
        model_dtype = next(self.transformer.parameters()).dtype

        if torch.is_tensor(class_labels):
            class_labels_tensor = class_labels.to(device=device, dtype=torch.long).reshape(-1)
        else:
            if isinstance(class_labels, int):
                class_labels = [class_labels]
            class_labels_tensor = torch.tensor(class_labels, device=device, dtype=torch.long).reshape(-1)

        batch_size = class_labels_tensor.shape[0]
        latents = self.prepare_latents(
            batch_size,
            self.transformer.config.in_channels,
            height,
            width,
            model_dtype,
            device,
            generator,
        )

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            latents = torch.cat([latents, latents], dim=0)
            labels = torch.cat([class_labels_tensor, self._get_null_labels(batch_size, device)], dim=0)
        else:
            labels = class_labels_tensor

        if self._use_rectified_flow:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            for index, timestep in enumerate(self.progress_bar(timesteps)):
                next_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else torch.tensor(
                    0.0, device=device
                )
                timestep_batch = torch.full((labels.shape[0],), float(timestep), device=device, dtype=model_dtype)
                model_output = self.transformer(latents, timestep_batch, labels, return_dict=True).sample
                model_output = self.transformer.split_velocity(model_output)
                if do_cfg:
                    model_output = self._apply_cfg(model_output, guidance_scale)
                    latents_cfg = latents[:batch_size]
                else:
                    latents_cfg = latents
                step_output = self.scheduler.step(
                    model_output[:batch_size] if do_cfg else model_output,
                    timestep_batch[:batch_size] if do_cfg else timestep_batch,
                    latents_cfg,
                    next_timestep=next_timestep,
                ).prev_sample
                latents = step_output if not do_cfg else torch.cat([step_output, step_output], dim=0)
            latents = latents[:batch_size]
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            for timestep in self.progress_bar(self.scheduler.timesteps):
                timestep_batch = torch.full((labels.shape[0],), timestep, device=device, dtype=torch.long)
                model_output = self.transformer(latents, timestep_batch, labels, return_dict=True).sample
                if do_cfg:
                    model_output = self._apply_cfg(model_output, guidance_scale)
                    latents_input = latents[:batch_size]
                else:
                    latents_input = latents
                    model_output = model_output
                latents = self.scheduler.step(
                    model_output[:batch_size] if do_cfg else model_output,
                    timestep,
                    latents_input,
                ).prev_sample
                if do_cfg:
                    latents = torch.cat([latents, latents], dim=0)

        image = self.decode_latents(latents, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


DiTMoEPipelineOutput = ImagePipelineOutput
