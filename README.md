## Scaling Diffusion Transformers with Mixture of Experts

[![arXiv](https://img.shields.io/badge/arXiv-2407.11633-b31b1b.svg)](https://arxiv.org/abs/2407.11633)

Diffusers-style implementation of DiT-MoE for class-conditional image generation. The legacy standalone training, diffusion, and sampling codepaths have been removed so the tree mirrors the package boundaries used by [huggingface/diffusers](https://github.com/huggingface/diffusers), following the layout in [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers).

![DiT-MoE framework](visuals/framework.png)

### Package layout

- `src/diffusers/models/transformers/transformer_dit_moe.py`: `DiTMoETransformer2DModel`, a `ModelMixin`/`ConfigMixin` sparse DiT transformer.
- `src/diffusers/schedulers/scheduling_flow_match_dit_moe.py`: `DiTMoEFlowMatchScheduler` for rectified-flow checkpoints.
- `src/diffusers/pipelines/dit_moe/pipeline_dit_moe.py`: `DiTMoEPipeline` with classifier-free guidance and VAE decoding.
- `scripts/convert_dit_moe_to_diffusers.py`: converts original `.pt` checkpoints to a Diffusers pipeline directory.
- `scripts/sample_dit_moe.py`: samples from a converted pipeline.

### Install

```bash
pip install -e ".[dev]"
```

### Convert a checkpoint

```bash
python scripts/convert_dit_moe_to_diffusers.py \
  --checkpoint /path/to/dit_moe_xl_8E2A.pt \
  --output dit-moe-xl-diffusers \
  --model DiT-XL/2 \
  --num-experts 8 \
  --num-experts-per-tok 2 \
  --rectified-flow
```

Use `--copy-vae /path/to/sd-vae-ft-mse` to vendor a local VAE into `output/vae`, or rely on `vae_pretrained_model_name_or_path.txt` for Hub download at sample time.

Use `--torch-dtype float32` for full-precision debugging; `bfloat16` is the default in `sample_dit_moe.py`.

### Sample

```bash
python scripts/sample_dit_moe.py \
  --model dit-moe-xl-diffusers \
  --class-label 207 360 387 974 \
  --image-size 256 \
  --num-inference-steps 50 \
  --guidance-scale 4.0
```

### Pre-trained weights

| Model | Resolution | Weights | Sampler |
|-------|------------|---------|---------|
| DiT-MoE-S/2-8E2A | 256 | [HF](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_s_8E2A.pt) | DDIM |
| DiT-MoE-B/2-8E2A | 256 | [HF](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_b_8E2A.pt) | DDIM |
| DiT-MoE-XL/2-8E2A | 256 | [HF](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_xl_8E2A.pt) | RF |
| DiT-MoE-G/2-16E2A | 256 | [HF](https://huggingface.co/feizhengcong/DiT-MoE/blob/main/dit_moe_g_16E2A.pt) | RF |

### Upstreaming to Diffusers

Copy the files under `src/diffusers` into the matching locations in the `huggingface/diffusers` repository and register the classes in Diffusers' lazy import tables.

### Citation

```bibtex
@article{FeiDiTMoE2024,
  title={Scaling Diffusion Transformers to 16 Billion Parameters},
  author={Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Jusnshi Huang},
  year={2024},
  journal={arXiv preprint},
}
```
