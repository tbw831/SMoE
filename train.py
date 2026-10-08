"""Portable main-model training on paired image folders, with optional two-GPU DDP.

This folder-based launcher uses the published model/objectives. It is not a
bit-exact resume of the historical BasicSR/LMDB training jobs.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from pytorch_msssim import ssim
import yaml
from models import HistoSMoEScore
from runtime_io import load_checkpoint, load_contexts


class PairedImages(Dataset):
    def __init__(self, root, names, contexts, labels, crop):
        self.root, self.names, self.contexts = Path(root), names, contexts
        self.labels, self.crop = labels, crop

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        images = []
        for folder in ('input', 'target'):
            with Image.open(self.root / folder / (name + '.png')) as image:
                images.append(torch.from_numpy(np.array(image.convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1))
        lq, gt = images
        if lq.shape != gt.shape:
            raise ValueError(f'Pair shape mismatch: {name}')
        h, w = lq.shape[-2:]
        if min(h, w) < self.crop:
            raise ValueError(f'Image smaller than crop: {name}')
        top, left = random.randrange(h - self.crop + 1), random.randrange(w - self.crop + 1)
        lq, gt = [v[:, top:top + self.crop, left:left + self.crop] for v in images]
        for axis in (-1, -2):
            if random.random() < 0.5:
                lq, gt = lq.flip(axis), gt.flip(axis)
        if random.random() < 0.5:
            lq, gt = lq.transpose(-2, -1), gt.transpose(-2, -1)
        de, ce = self.contexts[name]
        label = self.labels[name]
        if not isinstance(label, int) or not 0 <= label < 4:
            raise ValueError('quadrants.json must map image stems to integer labels 0..3')
        return lq.contiguous(), gt.contiguous(), de, ce, label


def semantic(name):
    return name.startswith(('semantic_router.', 'clean_fusion.', 'cross_modal_stage_gate.')) or '.expert_bank.' in name


def rgb_to_y(image):
    weights = image.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
    return (image * weights).sum(1, keepdim=True) / 255 + 16 / 255


def pearson(prediction, target):
    a, b = prediction.flatten(1), target.flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    denominator = a.square().sum(1).sqrt() * b.square().sum(1).sqrt()
    return ((1 - (a * b).sum(1) / denominator.clamp_min(1e-6)) * 0.5).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--train-list', type=Path, required=True)
    parser.add_argument('--quadrants', type=Path, required=True)
    parser.add_argument('--context-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Optional complete SMoE EMA warm start')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group('nccl')
    seed = config.get('seed', 3407) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = HistoSMoEScore().float()
    model.assert_parameter_contract()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    model.to(device).train()
    ema = deepcopy(model).eval().requires_grad_(False)
    groups = [[p for name, p in model.named_parameters() if semantic(name) == flag] for flag in (False, True)]
    base_lrs = [config['backbone_lr'], config['semantic_lr']]
    optimizer = torch.optim.AdamW([dict(params=g, lr=lr) for g, lr in zip(groups, base_lrs)],
                                 betas=(0.9, 0.999), weight_decay=1e-4)
    net = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    names = [Path(line.split()[0]).stem for line in args.train_list.read_text().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate training names')
    contexts = load_contexts(args.context_cache, names)
    labels = json.loads(args.quadrants.read_text())
    iteration, epoch = 0, 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stage in config['stages']:
        dataset = PairedImages(args.data_root, names, contexts, labels, stage['crop'])
        sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
        loader = DataLoader(dataset, batch_size=stage['batch_per_gpu'], shuffle=sampler is None,
                            sampler=sampler, num_workers=config.get('workers', 4), drop_last=True, pin_memory=True)
        if not len(loader):
            raise ValueError('Dataset is too small for the configured batch')
        completed = 0
        while completed < stage['iterations']:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                iteration += 1
                completed += 1
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = config['eta_min'] + (base_lr - config['eta_min']) * (1 + math.cos(math.pi * (iteration - 1) / config['total_iter'])) / 2
                lq, gt, de, ce, label = [v.to(device, non_blocking=True) for v in batch]
                optimizer.zero_grad(set_to_none=True)
                prediction, auxiliary = net(lq, de, ce)
                loss = F.l1_loss(prediction, gt) + config.get('pearson_weight', 1.0) * pearson(prediction, gt) + auxiliary
                loss = loss + config.get('route_supervision_weight', 0.05) * F.cross_entropy(model.last_router_logits, label)
                py, gy = rgb_to_y(prediction), rgb_to_y(gt)
                loss = loss + config.get('y_l1_weight', 0.0) * F.l1_loss(py, gy)
                loss = loss + config.get('y_mse_weight', 0.0) * F.mse_loss(py, gy)
                if config.get('ssim_weight', 0.0):
                    loss = loss + config['ssim_weight'] * (1 - ssim(prediction, gt, data_range=1.0, size_average=True))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                for group in groups:
                    torch.nn.utils.clip_grad_norm_(group, 1.0)
                optimizer.step()
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()):
                        ep.mul_(config['ema_decay']).add_(p, alpha=1 - config['ema_decay'])
                    for eb, b in zip(ema.buffers(), model.buffers()):
                        eb.copy_(b)
                if rank == 0 and iteration % 100 == 0:
                    print(f'iter={iteration} loss={loss.item():.6f}', flush=True)
                if rank == 0 and (iteration % config.get('save_every', 10000) == 0 or iteration == config['total_iter']):
                    state = dict(params=model.state_dict(), params_ema=ema.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration)
                    torch.save(state, args.output_dir / f'net_g_{iteration}.pth')
                if completed == stage['iterations']:
                    break
            epoch += 1
    if iteration != config['total_iter']:
        raise ValueError('Stage iterations do not sum to total_iter')
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
