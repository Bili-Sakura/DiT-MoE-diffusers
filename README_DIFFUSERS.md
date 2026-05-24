# DiT-MoE Diffusers integration

The core implementation lives under `src/diffusers`:

- `models/transformers/transformer_dit_moe.py` provides `DiTMoETransformer2DModel`.
- `schedulers/scheduling_flow_match_dit_moe.py` provides `DiTMoEFlowMatchScheduler`.
- `pipelines/dit_moe/pipeline_dit_moe.py` provides `DiTMoEPipeline`.

See [README.md](README.md) for convert and sample commands.
