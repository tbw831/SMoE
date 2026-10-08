"""SMoE main-result inference: FP32, tile320/overlap64, no test-time augmentation."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from runtime_io import load_checkpoint, load_contexts, sha256_file
from models import HistoSMoEScore


def tile_starts(length, size, stride):
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def cosine_axis(length, overlap, left, right, device):
    weight = torch.ones(length, dtype=torch.float32, device=device)
    fade = min(overlap, length // 2)
    if fade:
        ramp = torch.sin(torch.linspace(math.pi / (2 * (fade + 1)), math.pi / 2,
                                       fade, device=device, dtype=torch.float32)).square()
        if left:
            weight[:fade] = ramp
        if right:
            weight[-fade:] = ramp.flip(0)
    return weight


def forward_once(model, image, de, ce):
    height, width = image.shape[-2:]
    ph, pw = (-height) % 8, (-width) % 8
    if ph or pw:
        mode = 'reflect' if height > ph and width > pw else 'replicate'
        image = F.pad(image, (0, pw, 0, ph), mode=mode)
    result, _ = model(image, de, ce)
    return result[..., :height, :width]


@torch.inference_mode()
def forward_tiled(model, image, de, ce, size=320, overlap=64):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require tile_size > overlap >= 0')
    height, width = image.shape[-2:]
    if height <= size and width <= size:
        return forward_once(model, image, de, ce)
    result = torch.zeros_like(image, dtype=torch.float32)
    weights = torch.zeros_like(image[:, :1], dtype=torch.float32)
    for top in tile_starts(height, size, size - overlap):
        for left in tile_starts(width, size, size - overlap):
            tile = image[..., top:top + size, left:left + size]
            th, tw = tile.shape[-2:]
            vertical = cosine_axis(th, overlap, top > 0, top + th < height, image.device)
            horizontal = cosine_axis(tw, overlap, left > 0, left + tw < width, image.device)
            weight = vertical.view(1, 1, -1, 1) * horizontal.view(1, 1, 1, -1)
            result[..., top:top + th, left:left + tw].add_(forward_once(model, tile, de, ce) * weight)
            weights[..., top:top + th, left:left + tw].add_(weight)
    return result / weights.clamp_min(1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--context-cache', type=Path, required=True)
    parser.add_argument('--test-list', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'})
    if args.test_list:
        names = {Path(line.strip().split()[0]).stem for line in args.test_list.read_text().splitlines() if line.strip()}
        paths = [p for p in paths if p.stem in names]
        if {p.stem for p in paths} != names:
            raise ValueError('Test manifest does not match input files')
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Input directory is empty or contains duplicate stems')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = HistoSMoEScore().float()
    parameters = model.assert_parameter_contract()
    checkpoint_hash = load_checkpoint(model, args.checkpoint)
    model.to(args.device).eval()
    contexts = load_contexts(args.context_cache, [p.stem for p in paths])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = torch.from_numpy(np.array(source.convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1)[None].to(args.device)
        de, ce = (value[None].to(args.device) for value in contexts[path.stem])
        output = forward_tiled(model, image, de, ce)
        if not torch.isfinite(output).all():
            raise FloatingPointError('Non-finite restoration output')
        array = output[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(array).save(args.output_dir / (path.stem + '.png'))
        print(f'[{index}/{len(paths)}] {path.name}', flush=True)
    manifest = dict(checkpoint_sha256=checkpoint_hash, context_sha256=sha256_file(args.context_cache),
                    branch='params_ema', count=len(paths), tile_size=320, tile_overlap=64,
                    blend='cosine-feather', tta=False, attention_tlc=False, precision='float32',
                    parameters=parameters)
    (args.output_dir / 'INFERENCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
