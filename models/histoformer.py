"""Load a user-supplied pinned Histoformer implementation.

The upstream Histoformer repository does not currently contain a repository-
level license file. The public release therefore must not silently redistribute
its architecture source under this project's license. Users fetch the pinned
upstream commit separately and provide the architecture file through
``HISTOFORMER_ARCH_PATH``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Sequence


PINNED_HISTOFORMER_COMMIT = "1f045f06c03551c31504d8042dbe6ff9b9569108"
PINNED_HISTOFORMER_ARCH_SHA256 = (
    "2480609a85c1d1c02144743f992bf4f8c9e979ac5cb7126f1c2e02be956670b0"
)
DEFAULT_RELATIVE_ARCH_PATH = Path(
    "third_party/Histoformer/basicsr/models/archs/histoformer_arch.py"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_architecture_path() -> Path:
    configured = os.environ.get("HISTOFORMER_ARCH_PATH")
    if configured:
        path = Path(configured).expanduser().resolve(strict=True)
    else:
        repository_root = Path(__file__).resolve().parents[1]
        path = (repository_root / DEFAULT_RELATIVE_ARCH_PATH).resolve(
            strict=True
        )
    actual = _sha256_file(path)
    if actual != PINNED_HISTOFORMER_ARCH_SHA256:
        raise RuntimeError(
            "Pinned Histoformer architecture SHA-256 mismatch: "
            f"expected={PINNED_HISTOFORMER_ARCH_SHA256}, actual={actual}, "
            f"path={path}"
        )
    return path


def _load_upstream_histoformer():
    path = _resolve_architecture_path()
    specification = importlib.util.spec_from_file_location(
        "_smoe_pinned_histoformer_arch",
        path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import pinned Histoformer source: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    upstream = getattr(module, "Histoformer", None)
    if upstream is None:
        raise ImportError(f"Histoformer class is missing from {path}")
    return upstream


_UpstreamHistoformer = _load_upstream_histoformer()


class Histoformer(_UpstreamHistoformer):
    """Pinned upstream Histoformer with the project's locked defaults."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 36,
        num_blocks: Sequence[int] = (4, 4, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.667,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        dual_pixel_task: bool = False,
    ):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=list(num_blocks),
            num_refinement_blocks=num_refinement_blocks,
            heads=list(heads),
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
        )


__all__ = [
    "DEFAULT_RELATIVE_ARCH_PATH",
    "Histoformer",
    "PINNED_HISTOFORMER_ARCH_SHA256",
    "PINNED_HISTOFORMER_COMMIT",
]
