from .models.transformers import DiTMoETransformer2DModel
from .schedulers import DiTMoEFlowMatchScheduler

__all__ = [
    "DiTMoETransformer2DModel",
    "DiTMoEFlowMatchScheduler",
]


def __getattr__(name: str):
    if name in {"DiTMoEPipeline", "DiTMoEPipelineOutput"}:
        from .pipelines.dit_moe import DiTMoEPipeline, DiTMoEPipelineOutput

        return DiTMoEPipeline if name == "DiTMoEPipeline" else DiTMoEPipelineOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
