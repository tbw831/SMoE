#!/usr/bin/env python3
"""Evaluate Raindrop Clarity predictions with the official PSNR/SSIM protocol.

The benchmark reads images with OpenCV (BGR), converts them to the limited-range
BT.601 Y channel, and computes PSNR-Y and MATLAB-style SSIM-Y.  SSIM uses an
11x11 Gaussian window (sigma=1.5) and discards a five-pixel border from the
filtered maps, matching the official Raindrop Clarity implementation.

The evaluator is deliberately strict: by default every ground-truth image must
have exactly one prediction and the evaluated set must contain 300 images.
It also records per-image metrics and an aggregate hash of the prediction files
so that a reported number can be tied to a concrete prediction set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_map(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in files:
                raise ValueError(
                    f"Duplicate image stem {path.stem!r}: {files[path.stem]} and {path}"
                )
            files[path.stem] = path
    return files


def bgr_to_y(img: np.ndarray) -> np.ndarray:
    """Official BasicSR/Raindrop Clarity BGR -> Y conversion in [0, 255]."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC BGR image, got {img.shape}")
    # Preserve the slightly unusual dtype path used by the official code:
    # float32 input -> float64 np.dot coefficients -> float32 output.
    img_f = img.astype(np.float32) / 255.0
    y = np.dot(img_f, [24.966, 128.553, 65.481]) + 16.0
    return (y / 255.0).astype(np.float32) * 255.0


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    """MATLAB-style SSIM used by the official Raindrop Clarity evaluator."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    if min(img1.shape[:2]) < 11:
        raise ValueError(f"SSIM requires both dimensions >= 11, got {img1.shape}")

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2)
    if img1.ndim != 3 or img1.shape[2] != 3:
        raise ValueError(f"Expected HW or HWC-3 image, got {img1.shape}")
    return float(np.mean([_ssim_single_channel(img1[..., i], img2[..., i]) for i in range(3)]))


def quadrant_name(value: Any) -> str:
    """Normalize both legacy integer labels and current metadata dictionaries."""
    if isinstance(value, int):
        legacy = {
            0: "Day-BackgroundFocus",
            1: "Day-RaindropFocus",
            2: "Night-BackgroundFocus",
            3: "Night-RaindropFocus",
        }
        if value not in legacy:
            raise ValueError(f"Unknown integer quadrant {value}")
        return legacy[value]
    if isinstance(value, dict):
        time = str(value.get("time", "unknown")).strip().lower()
        focus = str(value.get("focus", "unknown")).strip().lower()
        time_name = {"day": "Day", "night": "Night"}.get(time, time.title())
        focus_name = {
            "background": "BackgroundFocus",
            "raindrop": "RaindropFocus",
        }.get(focus, focus.title().replace(" ", ""))
        return f"{time_name}-{focus_name}"
    return str(value)


def finite_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--quadrants", type=Path)
    parser.add_argument("--method", default="SMoE")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-image-csv", type=Path)
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Also compute standard AlexNet LPIPS on RGB images in [-1, 1].",
    )
    parser.add_argument("--lpips-device", default="cuda:0")
    parser.add_argument("--print-freq", type=int, default=50)
    args = parser.parse_args()

    pred = image_map(args.pred_dir)
    gt = image_map(args.gt_dir)
    common = sorted(pred.keys() & gt.keys())
    missing = sorted(gt.keys() - pred.keys())
    extra = sorted(pred.keys() - gt.keys())

    if not args.allow_subset:
        if missing or extra:
            raise SystemExit(
                f"Prediction/GT mismatch: common={len(common)}, missing={len(missing)}, "
                f"extra={len(extra)}; first_missing={missing[:5]}, first_extra={extra[:5]}"
            )
        if len(common) != args.expected_count:
            raise SystemExit(
                f"Expected {args.expected_count} matched images, found {len(common)}"
            )
    if not common:
        raise SystemExit("No matching prediction/GT stems")

    quadrant_meta: dict[str, Any] = {}
    if args.quadrants:
        quadrant_meta = json.loads(args.quadrants.read_text())

    lpips_model = None
    torch = None
    if args.lpips:
        import lpips as lpips_package
        import torch as torch_package

        torch = torch_package
        # Match the main-result numerical protocol; TF32 also affects LPIPS.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        lpips_model = lpips_package.LPIPS(net="alex", verbose=False).to(
            args.lpips_device
        )
        lpips_model.eval()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)

    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    aggregate_hash = hashlib.sha256()

    for index, stem in enumerate(common, start=1):
        pred_path = pred[stem]
        gt_path = gt[stem]
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if pred_img is None or gt_img is None:
            raise RuntimeError(f"Failed to read {pred_path} or {gt_path}")
        if pred_img.shape != gt_img.shape:
            raise RuntimeError(
                f"Shape mismatch for {stem}: {pred_img.shape} vs {gt_img.shape}"
            )

        pred_y = bgr_to_y(pred_img)
        gt_y = bgr_to_y(gt_img)
        metrics = {
            "psnr_y": calculate_psnr(pred_y, gt_y),
            "ssim_y": calculate_ssim(pred_y, gt_y),
            "psnr_rgb": calculate_psnr(pred_img, gt_img),
            "ssim_rgb": calculate_ssim(pred_img, gt_img),
        }
        if lpips_model is not None and torch is not None:
            def lpips_tensor(image_bgr: np.ndarray):
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(image_rgb.copy()).float() / 127.5 - 1.0
                return tensor.permute(2, 0, 1).unsqueeze(0).to(args.lpips_device)

            with torch.inference_mode():
                metrics["lpips_alex"] = float(
                    lpips_model(lpips_tensor(pred_img), lpips_tensor(gt_img)).item()
                )
        q_name = quadrant_name(quadrant_meta[stem]) if stem in quadrant_meta else "All"
        row = {"image": stem, "quadrant": q_name, **metrics}
        rows.append(row)
        for key, value in metrics.items():
            grouped[q_name][key].append(value)

        # Tie the aggregate result to exact encoded prediction files and order.
        aggregate_hash.update(stem.encode("utf-8"))
        aggregate_hash.update(b"\0")
        aggregate_hash.update(bytes.fromhex(sha256_file(pred_path)))

        if index % args.print_freq == 0 or index == len(common):
            print(
                f"[{args.method}] {index}/{len(common)} "
                f"PSNR-Y={finite_mean([r['psnr_y'] for r in rows]):.6f} "
                f"SSIM-Y={finite_mean([r['ssim_y'] for r in rows]):.6f}",
                flush=True,
            )

    metric_names = ["psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb"]
    if args.lpips:
        metric_names.append("lpips_alex")
    overall = {
        key: finite_mean([float(row[key]) for row in rows]) for key in metric_names
    }
    overall["n"] = len(rows)
    per_quadrant = {
        name: {**{key: finite_mean(values[key]) for key in metric_names}, "n": len(values["psnr_y"])}
        for name, values in sorted(grouped.items())
    }

    result: dict[str, Any] = {
        "method": args.method,
        "protocol": {
            "benchmark": "Raindrop Clarity official",
            "color_order": "OpenCV BGR",
            "y_conversion": "limited-range ITU-R BT.601 (BasicSR bgr2ycbcr)",
            "crop_border": 0,
            "ssim_window": "11x11 Gaussian, sigma=1.5, valid 5-pixel filtered-map crop",
            "lpips": "AlexNet, RGB normalized to [-1, 1]" if args.lpips else None,
            "aggregation": "arithmetic mean over per-image scores",
        },
        "inputs": {
            "pred_dir": str(args.pred_dir.resolve()),
            "gt_dir": str(args.gt_dir.resolve()),
            "quadrants": str(args.quadrants.resolve()) if args.quadrants else None,
            "prediction_set_sha256": aggregate_hash.hexdigest(),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        },
        "coverage": {
            "matched": len(common),
            "missing": missing,
            "extra": extra,
            "expected_count": args.expected_count,
        },
        "overall": overall,
        "per_quadrant": per_quadrant,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    csv_path = args.per_image_csv or args.out.with_suffix(".per_image.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if any(not math.isfinite(float(overall[key])) for key in metric_names):
        raise SystemExit(f"Non-finite aggregate metric: {overall}")
    print(json.dumps(overall, indent=2), flush=True)
    print(f"Wrote {args.out} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
