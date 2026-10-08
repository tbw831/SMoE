"""Strict, tensor-only restoration checkpoint and context-cache loading."""
import hashlib
from pathlib import Path
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(model, path, branch='params_ema'):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if branch not in artifact:
        raise KeyError(f'Checkpoint has no {branch} branch')
    state = {key.removeprefix('module.'): value for key, value in artifact[branch].items()}
    model.load_state_dict(state, strict=True)
    return sha256_file(path)


def load_contexts(path, names):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    cache_names = artifact['names']
    if len(cache_names) != len(set(cache_names)):
        raise ValueError('Duplicate context names')
    clean, degradation = artifact['image_context'], artifact['degradation_context']
    for tensor in (clean, degradation):
        if tensor.shape != (len(cache_names), 512) or not torch.isfinite(tensor).all():
            raise ValueError('Invalid 512-dimensional context cache')
    cache = {name: (degradation[i].float(), clean[i].float()) for i, name in enumerate(cache_names)}
    missing = set(names) - cache.keys()
    if missing:
        raise ValueError(f'Missing contexts: {sorted(missing)[:10]}')
    return cache
