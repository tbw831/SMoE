#!/usr/bin/env python3
"""Precompute DA-CLIP clean/degradation embeddings with auditable metadata."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def trusted_checkpoint_load(checkpoint_path: Path):
    """Allow legacy pickle only for one already SHA-verified checkpoint.

    PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default.
    Historical DA-CLIP training checkpoints also store optimizer metadata
    containing NumPy scalars, so OpenCLIP's unmodified loader cannot read them.
    Keep the compatibility exception local to the exact trusted file and
    restore ``torch.load`` immediately after model construction.
    """

    trusted_path = checkpoint_path.expanduser().resolve()
    original_torch_load = torch.load

    def scoped_torch_load(file, *args, **kwargs):
        requested_path = None
        if isinstance(file, (str, os.PathLike)):
            requested_path = Path(file).expanduser().resolve()
        if requested_path == trusted_path:
            kwargs["weights_only"] = False
        return original_torch_load(file, *args, **kwargs)

    torch.load = scoped_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            # PIL's explicit conversion avoids the historical OpenCV BGR/RGB swap.
            tensor = self.preprocess(image.convert("RGB"))
        return path.stem, tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--daclip-source", required=True)
    parser.add_argument("--model", default="daclip_ViT-B-32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--meta-info-file")
    parser.add_argument("--expected-meta-info-sha256")
    parser.add_argument(
        "--split-role",
        choices=("train5400", "official600", "other"),
        default="other",
    )
    parser.add_argument("--forbidden-meta-info-file")
    parser.add_argument("--expected-forbidden-meta-info-sha256")
    parser.add_argument(
        "--precision",
        choices=("float32", "autocast", "autocast_bf16"),
        default="float32",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def metadata_stems(path: Path) -> list[str]:
    stems = [
        Path(line.strip().split()[0]).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not stems or len(stems) != len(set(stems)):
        raise ValueError(
            f"Metadata must contain unique non-empty image names: {path}"
        )
    return stems


def resolve_meta_image(roots: list[Path], token: str) -> Path:
    token_path = Path(token)
    matches = []
    for root in roots:
        exact = root / token_path
        if exact.is_file():
            # Keep the manifest-facing path instead of resolving symlinks.
            # Materialized mixed datasets intentionally rename source images;
            # the cache key must match that alias, not the source basename.
            matches.append(exact)
            continue
        stem = exact.with_suffix("")
        matches.extend(
            path
            for path in stem.parent.glob(stem.name + ".*")
            if path.is_file()
            and path.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve {token!r} in input roots: {matches}"
        )
    return matches[0]


def collect_images(
    input_dirs: list[str],
    meta_info_file: Path | None = None,
) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    roots = [Path(directory).expanduser().resolve() for directory in input_dirs]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    if meta_info_file is not None:
        tokens = [
            line.strip().split()[0]
            for line in meta_info_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        paths = [resolve_meta_image(roots, token) for token in tokens]
        names = [path.stem for path in paths]
        expected_names = [Path(token).stem for token in tokens]
        if names != expected_names:
            raise ValueError(
                "Resolved image order/names differ from metadata"
            )
        if len(names) != len(set(names)):
            raise ValueError("Metadata resolves duplicate image stems")
        return paths

    paths = []
    for root in roots:
        paths.extend(
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions
        )
    paths = sorted(paths, key=lambda path: (path.stem, str(path)))
    names = [path.stem for path in paths]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate image stems across input dirs: {duplicates[:10]}")
    return paths


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    sidecar_path = output_path.with_suffix(
        output_path.suffix + ".json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing cache: {output_path}")
    if sidecar_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing sidecar: {sidecar_path}"
        )

    source_path = Path(args.daclip_source).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    sys.path.insert(0, str(source_path))
    import open_clip  # pylint: disable=import-error,import-outside-toplevel

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
        raise ValueError(
            "DA-CLIP checkpoint SHA256 mismatch: "
            f"expected {args.expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )

    meta_path = None
    meta_sha256 = None
    if args.meta_info_file:
        meta_path = Path(args.meta_info_file).expanduser().resolve()
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        if not args.expected_meta_info_sha256:
            raise ValueError(
                "--meta-info-file requires --expected-meta-info-sha256"
            )
        meta_sha256 = sha256_file(meta_path)
        if meta_sha256 != args.expected_meta_info_sha256.lower():
            raise ValueError(
                "Metadata SHA mismatch: "
                f"expected={args.expected_meta_info_sha256.lower()}, "
                f"actual={meta_sha256}"
            )
    elif args.expected_meta_info_sha256:
        raise ValueError(
            "--expected-meta-info-sha256 requires --meta-info-file"
        )

    paths = collect_images(args.input_dir, meta_path)
    if args.expected_count is not None and len(paths) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} images, found {len(paths)}")
    role_expected = {
        "train5400": 5400,
        "official600": 600,
    }.get(args.split_role)
    if role_expected is not None and len(paths) != role_expected:
        raise ValueError(
            f"{args.split_role} requires {role_expected} images, "
            f"found {len(paths)}"
        )
    if role_expected is not None and meta_path is None:
        raise ValueError(
            f"{args.split_role} requires a hash-bound --meta-info-file"
        )

    forbidden_record = None
    if args.forbidden_meta_info_file:
        forbidden_path = Path(
            args.forbidden_meta_info_file
        ).expanduser().resolve()
        if not forbidden_path.is_file():
            raise FileNotFoundError(forbidden_path)
        if not args.expected_forbidden_meta_info_sha256:
            raise ValueError(
                "--forbidden-meta-info-file requires its expected SHA"
            )
        forbidden_sha256 = sha256_file(forbidden_path)
        if (
            forbidden_sha256
            != args.expected_forbidden_meta_info_sha256.lower()
        ):
            raise ValueError(
                "Forbidden metadata SHA mismatch: "
                f"expected={args.expected_forbidden_meta_info_sha256.lower()}, "
                f"actual={forbidden_sha256}"
            )
        forbidden_names = set(metadata_stems(forbidden_path))
        overlap = sorted(
            {path.stem for path in paths} & forbidden_names
        )
        if overlap:
            raise RuntimeError(
                "Context export includes forbidden split names: "
                f"{overlap[:5]}"
            )
        forbidden_record = {
            "path": str(forbidden_path),
            "sha256": forbidden_sha256,
            "count": len(forbidden_names),
            "overlap": 0,
        }
    elif args.expected_forbidden_meta_info_sha256:
        raise ValueError(
            "Expected forbidden SHA requires --forbidden-meta-info-file"
        )
    if args.split_role == "train5400" and forbidden_record is None:
        raise ValueError(
            "train5400 export requires the sealed official600 metadata as "
            "--forbidden-meta-info-file"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    # The exception is safe only after the checkpoint passed the expected-SHA
    # check above. It does not relax loading for any other file.
    with trusted_checkpoint_load(checkpoint_path):
        model, preprocess = open_clip.create_model_from_pretrained(
            args.model, pretrained=str(checkpoint_path)
        )
    model = model.to(device).eval()
    model.requires_grad_(False)
    loader = DataLoader(
        ImageDataset(paths, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    all_names: list[str] = []
    clean_contexts = []
    degradation_contexts = []
    autocast_enabled = (
        args.precision in {"autocast", "autocast_bf16"} and device.type == "cuda"
    )
    with torch.inference_mode():
        for batch_index, (names, images) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            autocast_context = (
                torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if args.precision == "autocast_bf16" else torch.float16,
                    enabled=True,
                )
                if autocast_enabled
                else contextlib.nullcontext()
            )
            with autocast_context:
                clean, degradation = model.encode_image(
                    images, control=True, normalize=False
                )
            if clean.ndim != 2 or degradation.ndim != 2:
                raise ValueError(
                    "Unexpected DA-CLIP outputs: "
                    f"clean={tuple(clean.shape)}, degradation={tuple(degradation.shape)}"
                )
            if not torch.isfinite(clean).all() or not torch.isfinite(degradation).all():
                raise FloatingPointError(f"Non-finite context in batch {batch_index}")
            all_names.extend(names)
            clean_contexts.append(clean.float().cpu())
            degradation_contexts.append(degradation.float().cpu())
            print(
                f"[{batch_index:04d}/{len(loader):04d}] cached {len(all_names)}/{len(paths)}",
                flush=True,
            )

    image_context = torch.cat(clean_contexts, dim=0)
    degradation_context = torch.cat(degradation_contexts, dim=0)
    if len(all_names) != len(paths):
        raise RuntimeError("DataLoader output count mismatch")
    if all_names != [path.stem for path in paths]:
        raise RuntimeError(
            "DataLoader changed the hash-bound image order"
        )

    artifact = {
        "schema_version": 1,
        "names": all_names,
        "image_context": image_context,
        "degradation_context": degradation_context,
        "metadata": {
            "model": args.model,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "preprocess": "open_clip official preprocess; PIL RGB",
            "normalize_embeddings": False,
            "input_dirs": [str(Path(path).resolve()) for path in args.input_dir],
            "meta_info_file": str(meta_path) if meta_path else None,
            "meta_info_sha256": meta_sha256,
            "split_role": args.split_role,
            "forbidden_meta_info": forbidden_record,
            "precision": args.precision,
            "tf32": False,
            "checkpoint_load_policy": (
                "weights_only_false_for_sha_verified_checkpoint_only"
            ),
            "count": len(all_names),
            "clean_shape": list(image_context.shape),
            "degradation_shape": list(degradation_context.shape),
            # Cast TorchVersion to a plain string so the artifact remains
            # directly loadable with torch.load(weights_only=True).
            "torch_version": str(torch.__version__),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(artifact, temporary_path)
    os.replace(temporary_path, output_path)
    artifact_sha256 = sha256_file(output_path)
    sidecar = {
        "schema_version": 1,
        "artifact_type": "daclip_context_cache_sidecar",
        "cache": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": artifact_sha256,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "meta_info_file": (
            {
                "path": str(meta_path),
                "bytes": meta_path.stat().st_size,
                "sha256": meta_sha256,
            }
            if meta_path
            else None
        ),
        "split_role": args.split_role,
        "forbidden_meta_info": forbidden_record,
        "count": len(all_names),
        "ordered_names_sha256": hashlib.sha256(
            json.dumps(
                all_names,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
        "clean_shape": list(image_context.shape),
        "degradation_shape": list(degradation_context.shape),
        "precision": args.precision,
        "tf32": False,
        "checkpoint_load_policy": (
            "weights_only_false_for_sha_verified_checkpoint_only"
        ),
    }
    temporary_sidecar = sidecar_path.with_suffix(
        sidecar_path.suffix + ".tmp"
    )
    temporary_sidecar.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_sidecar, sidecar_path)
    print(f"saved: {output_path}")
    print(f"sidecar: {sidecar_path}")
    print(f"artifact_sha256: {artifact_sha256}")
    print(f"checkpoint_sha256: {checkpoint_sha256}")
    print(f"clean mean/std: {image_context.mean():.6f}/{image_context.std():.6f}")
    print(
        "degradation mean/std: "
        f"{degradation_context.mean():.6f}/{degradation_context.std():.6f}"
    )


if __name__ == "__main__":
    main()
