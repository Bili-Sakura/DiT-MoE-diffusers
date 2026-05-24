"""Ensure local DiT-MoE diffusers extensions win over the installed package."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_src() -> Path:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    repo_src = repo_src.resolve()
    repo_src_str = str(repo_src)

    if repo_src_str in sys.path:
        sys.path.remove(repo_src_str)
    sys.path.insert(0, repo_src_str)

    cached = sys.modules.get("diffusers")
    cached_paths = getattr(cached, "__path__", None) if cached is not None else None
    local_shadow = cached_paths is not None and any(repo_src_str in str(item) for item in cached_paths)
    if cached is not None and not local_shadow:
        for name in list(sys.modules):
            if name == "diffusers" or name.startswith("diffusers."):
                del sys.modules[name]

    return repo_src
