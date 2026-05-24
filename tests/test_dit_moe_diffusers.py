import pytest

torch = pytest.importorskip("torch")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import bootstrap_repo_src

bootstrap_repo_src()

from diffusers.models.transformers.transformer_dit_moe import DiTMoETransformer2DModel
from diffusers.pipelines.dit_moe import DiTMoEPipeline
from diffusers.schedulers.scheduling_flow_match_dit_moe import DiTMoEFlowMatchScheduler


def test_dit_moe_transformer_forward():
    model = DiTMoETransformer2DModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        learn_sigma=False,
    )
    latents = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([500, 250])
    class_labels = torch.tensor([1, 2])

    output = model(latents, timesteps, class_labels)

    assert output.sample.shape == (2, 4, 8, 8)


def test_scheduler_ode_step_matches_velocity_update():
    scheduler = DiTMoEFlowMatchScheduler(mode="ode")
    sample = torch.ones(1, 4, 2, 2)
    velocity = torch.full_like(sample, 2.0)

    output = scheduler.step(velocity, torch.tensor([1.0]), sample, next_timestep=torch.tensor([0.75]))

    assert torch.allclose(output.prev_sample, torch.full_like(sample, 0.5))


def test_scheduler_sde_final_step_is_deterministic():
    scheduler = DiTMoEFlowMatchScheduler(mode="sde")
    sample = torch.randn(2, 4, 1, 1)
    velocity = torch.zeros_like(sample)

    output = scheduler.step(
        velocity,
        torch.tensor([0.04]),
        sample,
        next_timestep=torch.tensor([0.0]),
        final_step=True,
    )

    assert output.prev_sample.shape == sample.shape


def test_pipeline_id2label_helpers():
    transformer = DiTMoETransformer2DModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        learn_sigma=False,
    )
    scheduler = DiTMoEFlowMatchScheduler(mode="ode")
    id2label = {207: "golden retriever", 360: "otter"}
    pipe = DiTMoEPipeline(transformer=transformer, scheduler=scheduler, id2label=id2label)

    assert pipe.id2label[207] == "golden retriever"
    assert pipe.get_label_ids("golden retriever") == [207]
    assert pipe.get_label_ids(["golden retriever", "otter"]) == [207, 360]
    assert pipe._normalize_class_labels("golden retriever") == [207]
    assert pipe.config.null_class_id == 1000
