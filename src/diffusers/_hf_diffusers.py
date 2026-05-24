"""Load utilities from the installed Hugging Face diffusers package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _src_root() -> Path:
    return Path(__file__).resolve().parents[1]


_LOCAL_PRESERVE_PREFIXES = (
    "diffusers.pipelines.dit_moe",
    "diffusers.models.transformers.transformer_dit_moe",
    "diffusers.schedulers.scheduling_flow_match_dit_moe",
)


def _should_preserve_local_module(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in _LOCAL_PRESERVE_PREFIXES
    )


def import_hf_diffusers_submodule(module_name: str) -> ModuleType:
    src_root = str(_src_root())
    removed_paths = []
    for path in list(sys.path):
        if path == src_root:
            sys.path.remove(path)
            removed_paths.append(path)

    cached = sys.modules.get("diffusers")
    cached_paths = getattr(cached, "__path__", None) if cached is not None else None
    local_shadow = cached_paths is not None and any(src_root in str(item) for item in cached_paths)

    preserved: dict[str, ModuleType] = {}
    if local_shadow:
        for name in list(sys.modules):
            if name == "diffusers" or name.startswith("diffusers."):
                if _should_preserve_local_module(name):
                    preserved[name] = sys.modules[name]
                del sys.modules[name]

    try:
        module = importlib.import_module(module_name)
    finally:
        sys.modules.update(preserved)
        for path in reversed(removed_paths):
            sys.path.insert(0, path)
    return module


def get_hf_attr(dotted_path: str) -> Any:
    module_name, _, attr_name = dotted_path.rpartition(".")
    module = import_hf_diffusers_submodule(module_name)
    return getattr(module, attr_name)
