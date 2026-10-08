"""Score-first Histo-SMoE without a BasicSR runtime dependency.

The shared restoration path is the complete matched Histoformer. A single
DE-only router chooses one of four semantic experts per image, and the same
choice is reused in all 32 non-latent transformer blocks. Only the selected
expert is executed. CE is injected once at the bottleneck through bounded,
low-rank cross-attention. A shared low-rank DE×CE gate also modulates the
selected expert per stage and channel without spatial high-resolution
attention.

Every expert output projection and the CE output projection are initialized to
zero. Therefore, after loading the matched Histoformer weights, the model is an
exact functional identity to Histoformer before optimization.

The default restoration-network parameter contract is:

* complete Histoformer shared path: 16,615,100;
* all four experts, router, CE, and cross-modal gate included: 19,198,364;
* active Top-1 path per image: 17,640,692.

The frozen DA-CLIP conditioner is required at inference but is external to the
restoration network and is not included in these counts.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from functools import lru_cache
from itertools import product
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .histoformer import Histoformer


SCORE_NUM_EXPERTS = 4
SCORE_TOP_K = 1
SCORE_ROUTER_DIM = 64
SCORE_EXPERT_EXPANSION = 0.5
SCORE_EXPERT_RESIDUAL_CAP = 0.05
SCORE_CE_CAP = 0.004
SCORE_FUSION_DIM = 64
SCORE_STAGE_GATE_CAP = 0.25

EXPECTED_SHARED_PARAMETERS = 16_615_100
EXPECTED_TOTAL_PARAMETERS = 19_198_364
EXPECTED_ACTIVE_TOP1_PARAMETERS = 17_640_692
EXPECTED_MOE_BLOCKS = 32


@lru_cache(maxsize=16)
def _balanced_assignment_candidates(
    batch_size: int,
    num_experts: int,
) -> torch.Tensor:
    """Enumerate deterministic near-equal assignments for small batches."""

    batch_size = int(batch_size)
    num_experts = int(num_experts)
    if batch_size <= 0:
        raise ValueError("balanced routing requires a positive batch size")
    if batch_size > 8:
        raise ValueError(
            "exact balanced routing supports per-rank batches up to eight"
        )
    candidates = []
    for assignment in product(range(num_experts), repeat=batch_size):
        counts = [assignment.count(index) for index in range(num_experts)]
        if max(counts) - min(counts) <= 1:
            candidates.append(assignment)
    if not candidates:
        raise RuntimeError("no balanced routing assignment exists")
    return torch.tensor(candidates, dtype=torch.long)


def _balanced_top1_assignment(logits: torch.Tensor) -> torch.Tensor:
    """Maximize router score subject to near-equal per-expert capacity."""

    if logits.ndim != 2:
        raise ValueError("balanced routing logits must have shape [B,E]")
    batch_size, num_experts = logits.shape
    candidates = _balanced_assignment_candidates(
        batch_size,
        num_experts,
    ).to(device=logits.device)
    sample_index = torch.arange(
        batch_size,
        device=logits.device,
    ).view(1, batch_size)
    scores = logits.float()[sample_index, candidates].sum(dim=1)
    return candidates[scores.argmax()].detach()


def _cache_sample_indices_by_expert(
    selected_expert: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, ...]:
    """Build the global Top-1 sample groups once for all sparse blocks.

    Repeating four ``torch.nonzero`` calls in each of 32 blocks creates up to
    128 index-discovery operations per forward. The route vector is tiny, so
    copy it once, build the same ascending groups on the host, and reuse the
    four CUDA index tensors in every sparse block.
    """

    if selected_expert.ndim != 1:
        raise ValueError("selected_expert must be one-dimensional")
    assignments = selected_expert.detach().to(
        device="cpu",
        dtype=torch.long,
    ).tolist()
    grouped = [[] for _ in range(int(num_experts))]
    for sample_index, expert_index in enumerate(assignments):
        expert_index = int(expert_index)
        if not 0 <= expert_index < int(num_experts):
            raise ValueError(
                f"selected expert {expert_index} is outside "
                f"[0,{int(num_experts)})"
            )
        grouped[expert_index].append(sample_index)
    return tuple(
        selected_expert.new_tensor(indices, dtype=torch.long)
        for indices in grouped
    )


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _fp32_guard(value: torch.Tensor):
    if value.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=value.device.type, enabled=False)
    return nullcontext()


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().square().mean().sqrt()


def layer_norm_zero_affine(
    value: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Parameter-free FP32 LayerNorm that preserves the input dtype."""

    if value.ndim not in {2, 3}:
        raise ValueError(
            f"LN0 expects [B,D] or [B,N,D], got {tuple(value.shape)}"
        )
    _require_finite("LN0 input", value)
    with _fp32_guard(value):
        normalized = F.layer_norm(
            value.float(),
            (value.shape[-1],),
            weight=None,
            bias=None,
            eps=float(eps),
        )
    _require_finite("LN0 output", normalized)
    return normalized.to(dtype=value.dtype)


def _soft_cap_spatial_residual(
    features: torch.Tensor,
    raw_update: torch.Tensor,
    cap: float,
    eps: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """Bound residual RMS per sample and channel relative to feature RMS."""

    if features.ndim != 4 or raw_update.shape != features.shape:
        raise ValueError(
            "features/raw_update must share [B,C,H,W], got "
            f"{tuple(features.shape)} and {tuple(raw_update.shape)}"
        )
    if float(cap) <= 0 or float(eps) <= 0:
        raise ValueError("spatial residual cap and eps must be positive")
    _require_finite("cap features", features)
    _require_finite("cap raw update", raw_update)

    with _fp32_guard(features):
        feature_fp32 = features.float()
        raw_fp32 = raw_update.float()
        reference_power = (
            feature_fp32.square()
            .mean(dim=(-2, -1), keepdim=True)
            .detach()
            + float(eps)
        )
        raw_relative_power = (
            raw_fp32.square().mean(dim=(-2, -1), keepdim=True)
            / reference_power
        )
        cap_scale = torch.rsqrt(
            1.0 + raw_relative_power / (float(cap) ** 2)
        )
        protected = raw_fp32 * cap_scale
        protected_relative = (
            protected.square().mean(dim=(-2, -1), keepdim=True)
            / reference_power
        ).sqrt()
        utilization = protected_relative / float(cap)

    _require_finite("protected update", protected)
    telemetry = {
        "raw_update_rms": _rms(raw_fp32),
        "protected_update_rms": _rms(protected),
        "protected_update_relative_rms": (
            protected_relative.detach().mean()
        ),
        "cap_scale_min": cap_scale.detach().min(),
        "cap_scale_mean": cap_scale.detach().mean(),
        "cap_utilization_mean": utilization.detach().mean(),
        "cap_utilization_max": utilization.detach().max(),
        "cap_saturation_fraction": (
            utilization.detach() >= 0.90
        ).float().mean(),
    }
    return (
        protected.to(dtype=features.dtype),
        telemetry,
        reference_power,
    )


def _hidden_width(channels: int, expansion: float) -> int:
    """Return a PixelShuffle-compatible compact expert width."""

    width = int(round(float(channels) * float(expansion)))
    return max(4, ((width + 3) // 4) * 4)


class CompactDGFFBranch(nn.Module):
    """One compact dual-scale gated residual expert."""

    def __init__(
        self,
        channels: int,
        expansion: float,
        bias: bool = False,
        zero_output: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.expansion = float(expansion)
        self.hidden_features = _hidden_width(
            self.channels,
            self.expansion,
        )
        hidden = self.hidden_features
        self.project_in = nn.Conv2d(
            self.channels,
            hidden * 2,
            kernel_size=1,
            bias=bias,
        )
        self.dwconv_5 = nn.Conv2d(
            hidden // 4,
            hidden // 4,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=hidden // 4,
            bias=bias,
        )
        self.dwconv_dilated2_1 = nn.Conv2d(
            hidden // 4,
            hidden // 4,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            groups=hidden // 4,
            bias=bias,
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.project_out = nn.Conv2d(
            hidden,
            self.channels,
            kernel_size=1,
            bias=bias,
        )
        if zero_output:
            nn.init.zeros_(self.project_out.weight)
            if self.project_out.bias is not None:
                nn.init.zeros_(self.project_out.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.pixel_shuffle(self.project_in(features))
        branch_1, branch_2 = features.chunk(2, dim=1)
        branch_1 = self.dwconv_5(branch_1)
        branch_2 = self.dwconv_dilated2_1(branch_2)
        features = F.mish(branch_2) * branch_1
        return self.project_out(self.pixel_unshuffle(features))


class GlobalSemanticTop1Router(nn.Module):
    """Choose one semantic expert per image from only the 512-D DE."""

    def __init__(
        self,
        context_dim: int = 512,
        router_dim: int = SCORE_ROUTER_DIM,
        num_experts: int = SCORE_NUM_EXPERTS,
        balanced_training_dispatch: bool = False,
    ):
        super().__init__()
        if int(context_dim) <= 0 or int(router_dim) <= 0:
            raise ValueError("context_dim and router_dim must be positive")
        if int(num_experts) != SCORE_NUM_EXPERTS:
            raise ValueError("score router requires exactly four experts")
        self.context_dim = int(context_dim)
        self.router_dim = int(router_dim)
        self.num_experts = int(num_experts)
        self.balanced_training_dispatch = bool(
            balanced_training_dispatch
        )
        self.context_projection = nn.Linear(
            self.context_dim,
            self.router_dim,
            bias=False,
        )
        self.classifier = nn.Linear(
            self.router_dim,
            self.num_experts,
            bias=False,
        )

    def forward(
        self,
        degradation_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if degradation_context.ndim != 2:
            raise ValueError(
                "degradation_context must be [B,D], got "
                f"{tuple(degradation_context.shape)}"
            )
        if degradation_context.shape[1] != self.context_dim:
            raise ValueError(
                f"DE dimension must be {self.context_dim}, got "
                f"{degradation_context.shape[1]}"
            )
        _require_finite("degradation_context", degradation_context)
        normalized = layer_norm_zero_affine(degradation_context)
        hidden = F.gelu(self.context_projection(normalized))
        logits = self.classifier(hidden)
        probabilities = F.softmax(logits.float(), dim=1)
        argmax_expert = probabilities.argmax(dim=1)
        balanced_dispatch_applied = bool(
            self.training and self.balanced_training_dispatch
        )
        selected_expert = (
            _balanced_top1_assignment(logits)
            if balanced_dispatch_applied
            else argmax_expert
        )
        selected_probability = probabilities.gather(
            1,
            selected_expert.unsqueeze(1),
        ).squeeze(1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "argmax_expert": argmax_expert,
            "selected_expert": selected_expert,
            "selected_probability": selected_probability,
            "balanced_dispatch_applied": balanced_dispatch_applied,
            "sample_indices_by_expert": _cache_sample_indices_by_expert(
                selected_expert,
                self.num_experts,
            ),
        }


class CrossModalStageGate(nn.Module):
    """Fuse global DE and CE, then gate the selected in-backbone expert."""

    def __init__(
        self,
        stage_channels: Mapping[str, int],
        context_dim: int = 512,
        fusion_dim: int = SCORE_FUSION_DIM,
        gate_cap: float = SCORE_STAGE_GATE_CAP,
    ):
        super().__init__()
        if int(context_dim) <= 0 or int(fusion_dim) <= 0:
            raise ValueError("context_dim and fusion_dim must be positive")
        if not 0 < float(gate_cap) <= 0.50:
            raise ValueError("gate_cap must be in (0,0.50]")
        if not stage_channels or any(
            not name or int(channels) <= 0
            for name, channels in stage_channels.items()
        ):
            raise ValueError("stage_channels must be a non-empty positive map")
        self.context_dim = int(context_dim)
        self.fusion_dim = int(fusion_dim)
        self.gate_cap = float(gate_cap)
        self.stage_channels = {
            str(name): int(channels)
            for name, channels in stage_channels.items()
        }
        self.de_projection = nn.Linear(
            self.context_dim,
            self.fusion_dim,
            bias=False,
        )
        self.ce_projection = nn.Linear(
            self.context_dim,
            self.fusion_dim,
            bias=False,
        )
        self.stage_heads = nn.ModuleDict(
            {
                name: nn.Linear(
                    self.fusion_dim,
                    channels,
                    bias=False,
                )
                for name, channels in self.stage_channels.items()
            }
        )
        for head in self.stage_heads.values():
            nn.init.zeros_(head.weight)

    def forward(
        self,
        degradation_context: torch.Tensor,
        clean_context: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        if degradation_context.ndim != 2 or degradation_context.shape[
            1
        ] != self.context_dim:
            raise ValueError(
                "stage-gate DE must be "
                f"[B,{self.context_dim}], got "
                f"{tuple(degradation_context.shape)}"
            )
        if clean_context.shape != degradation_context.shape:
            raise ValueError(
                "stage-gate CE must match DE, got "
                f"{tuple(clean_context.shape)} and "
                f"{tuple(degradation_context.shape)}"
            )
        _require_finite("stage-gate DE", degradation_context)
        _require_finite("stage-gate CE", clean_context)

        de_code = F.gelu(
            self.de_projection(
                layer_norm_zero_affine(degradation_context)
            )
        )
        ce_code = F.gelu(
            self.ce_projection(
                layer_norm_zero_affine(clean_context)
            )
        )
        fused = F.gelu(de_code + ce_code + de_code * ce_code)
        gates = {
            name: (
                1.0
                + self.gate_cap
                * torch.tanh(self.stage_heads[name](fused))
            ).view(
                degradation_context.shape[0],
                channels,
                1,
                1,
            )
            for name, channels in self.stage_channels.items()
        }
        all_gates = torch.cat(
            [gate.flatten(1) for gate in gates.values()],
            dim=1,
        )
        telemetry = {
            "de_code_rms": _rms(de_code),
            "ce_code_rms": _rms(ce_code),
            "fused_code_rms": _rms(fused),
            "gate_mean": all_gates.detach().float().mean(),
            "gate_min": all_gates.detach().float().min(),
            "gate_max": all_gates.detach().float().max(),
        }
        return gates, telemetry


class SparseResidualExpertBank(nn.Module):
    """Four compact branches with true per-sample Top-1 execution."""

    def __init__(
        self,
        channels: int,
        num_experts: int = SCORE_NUM_EXPERTS,
        expert_expansion: float = SCORE_EXPERT_EXPANSION,
        residual_cap: float = SCORE_EXPERT_RESIDUAL_CAP,
        protection_eps: float = 1.0e-8,
        bias: bool = False,
    ):
        super().__init__()
        if int(num_experts) != SCORE_NUM_EXPERTS:
            raise ValueError("score expert bank requires four experts")
        if float(expert_expansion) <= 0:
            raise ValueError("expert_expansion must be positive")
        if not 0 < float(residual_cap) <= 0.10:
            raise ValueError("residual_cap must be in (0,0.10]")
        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.expert_expansion = float(expert_expansion)
        self.residual_cap = float(residual_cap)
        self.protection_eps = float(protection_eps)
        self.experts = nn.ModuleList(
            [
                CompactDGFFBranch(
                    channels=self.channels,
                    expansion=self.expert_expansion,
                    bias=bias,
                    zero_output=True,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        normalized_features: torch.Tensor,
        reference_features: torch.Tensor,
        route: Mapping[str, torch.Tensor],
        channel_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if normalized_features.shape != reference_features.shape:
            raise ValueError("expert input/reference shapes must match")
        selected_expert = route["selected_expert"]
        selected_probability = route["selected_probability"]
        batch = normalized_features.shape[0]
        if selected_expert.shape != (batch,):
            raise ValueError("selected_expert must have shape [B]")
        if channel_gate is not None and channel_gate.shape != (
            batch,
            self.channels,
            1,
            1,
        ):
            raise ValueError(
                "channel_gate must be "
                f"[{batch},{self.channels},1,1], got "
                f"{tuple(channel_gate.shape)}"
            )

        cached_indices = route.get("sample_indices_by_expert")
        update = torch.zeros_like(reference_features)
        executed = []
        cap_utilization = []
        dispatch_mode = "top1_sparse"
        if cached_indices is not None and (
            not isinstance(cached_indices, (tuple, list))
            or len(cached_indices) != self.num_experts
        ):
            raise ValueError(
                "sample_indices_by_expert must contain one tensor "
                "per expert"
            )
        for expert_index, expert in enumerate(self.experts):
            sample_indices = (
                cached_indices[expert_index]
                if cached_indices is not None
                else torch.nonzero(
                    selected_expert == expert_index,
                    as_tuple=False,
                ).flatten()
            )
            if (
                not torch.is_tensor(sample_indices)
                or sample_indices.dtype != torch.long
                or sample_indices.device != selected_expert.device
                or sample_indices.ndim != 1
            ):
                raise ValueError(
                    "cached sample indices must be 1D long tensors on "
                    "the route device"
                )
            if sample_indices.numel() == 0:
                continue
            selected_input = normalized_features.index_select(
                0,
                sample_indices,
            )
            selected_reference = reference_features.index_select(
                0,
                sample_indices,
            )
            raw_update = expert(selected_input)
            if channel_gate is not None:
                raw_update = raw_update * channel_gate.index_select(
                    0,
                    sample_indices,
                ).to(dtype=raw_update.dtype)
            protected_update, protection, _ = (
                _soft_cap_spatial_residual(
                    selected_reference,
                    raw_update,
                    self.residual_cap,
                    self.protection_eps,
                )
            )
            if self.training:
                probability = selected_probability.index_select(
                    0,
                    sample_indices,
                ).to(dtype=protected_update.dtype)
                # The forward multiplier is exactly one; its gradient
                # supplies restoration supervision to the hard router.
                straight_through = (
                    1.0 + probability - probability.detach()
                ).view(-1, 1, 1, 1)
                protected_update = (
                    protected_update * straight_through
                )
            update = update.index_copy(
                0,
                sample_indices,
                protected_update,
            )
            executed.append(expert_index)
            cap_utilization.append(
                protection["cap_utilization_mean"]
            )

        telemetry: Dict[str, object] = {
            "selected_expert": selected_expert.detach(),
            "expert_counts": torch.bincount(
                selected_expert,
                minlength=self.num_experts,
            ).detach(),
            "executed_experts": tuple(executed),
            "dispatch_mode": dispatch_mode,
            "route_index_cache_used": cached_indices is not None,
            "update_rms": _rms(update),
            "cap_utilization_mean": (
                torch.stack(cap_utilization).mean().detach()
                if cap_utilization
                else update.new_zeros(())
            ),
            "channel_gate_mean": (
                channel_gate.detach().float().mean()
                if channel_gate is not None
                else update.new_ones(())
            ),
        }
        return update, telemetry

    def parameter_report(self) -> Dict[str, int]:
        per_expert = [
            sum(parameter.numel() for parameter in expert.parameters())
            for expert in self.experts
        ]
        if len(set(per_expert)) != 1:
            raise RuntimeError("score experts do not have identical sizes")
        return {
            "one_active_expert": per_expert[0],
            "all_experts": sum(per_expert),
            "inactive_experts": sum(per_expert[1:]),
        }


class ScoreMoETransformerBlock(nn.Module):
    """An unchanged Histoformer HTB plus one sparse residual bank."""

    def __init__(
        self,
        original_block: nn.Module,
        channels: int,
        num_experts: int,
        expert_expansion: float,
        expert_residual_cap: float,
        bias: bool,
    ):
        super().__init__()
        # These names and objects preserve the vanilla Histoformer state dict.
        self.attn_g = original_block.attn_g
        self.norm_g = original_block.norm_g
        self.ffn = original_block.ffn
        self.norm_ff1 = original_block.norm_ff1
        self.expert_bank = SparseResidualExpertBank(
            channels=channels,
            num_experts=num_experts,
            expert_expansion=expert_expansion,
            residual_cap=expert_residual_cap,
            bias=bias,
        )

    def forward(
        self,
        features: torch.Tensor,
        route: Optional[Mapping[str, torch.Tensor]] = None,
        enable_experts: bool = True,
        channel_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, object]]]:
        features = features + self.attn_g(self.norm_g(features))
        normalized = self.norm_ff1(features)
        features = features + self.ffn(normalized)
        if route is None or not enable_experts:
            return features, None
        update, telemetry = self.expert_bank(
            normalized,
            features,
            route,
            channel_gate=channel_gate,
        )
        return features + update, telemetry


class StableLowRankBottleneckCrossAttention(nn.Module):
    """Bias-free rank-128 cosine CE attention with a protected residual."""

    def __init__(
        self,
        channels: int,
        context_dim: int = 512,
        rank: int = 128,
        num_tokens: int = 4,
        num_heads: int = 4,
        residual_cap: float = SCORE_CE_CAP,
        attention_logit_scale: float = 2.0,
        presence_eps: float = 1.0e-8,
        protection_eps: float = 1.0e-8,
    ):
        super().__init__()
        expected = (
            int(context_dim),
            int(rank),
            int(num_tokens),
            int(num_heads),
        )
        if expected != (512, 128, 4, 4):
            raise ValueError(
                "score CE requires context/rank/tokens/heads=512/128/4/4, "
                f"got {expected}"
            )
        if rank % num_heads:
            raise ValueError("CE rank must be divisible by heads")
        if not 0 < float(residual_cap) <= SCORE_CE_CAP:
            raise ValueError("CE cap must be in (0,0.004]")
        if not 0 < float(attention_logit_scale) <= 2.0:
            raise ValueError("CE attention logit scale must be in (0,2]")

        self.channels = int(channels)
        self.context_dim = int(context_dim)
        self.rank = int(rank)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = self.rank // self.num_heads
        self.residual_cap = float(residual_cap)
        self.attention_logit_scale = float(attention_logit_scale)
        self.presence_eps = float(presence_eps)
        self.protection_eps = float(protection_eps)

        self.context_to_tokens = nn.Linear(
            self.context_dim,
            self.num_tokens * self.rank,
            bias=False,
        )
        self.to_q = nn.Linear(self.channels, self.rank, bias=False)
        self.to_k = nn.Linear(self.rank, self.rank, bias=False)
        self.to_v = nn.Linear(self.rank, self.rank, bias=False)
        self.to_out = nn.Linear(self.rank, self.channels, bias=False)
        nn.init.zeros_(self.to_out.weight)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        features: torch.Tensor,
        clean_context: torch.Tensor,
        *,
        warmup_scale: torch.Tensor,
        branch_gate: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"CE features must be [B,{self.channels},H,W], got "
                f"{tuple(features.shape)}"
            )
        if clean_context.shape != (
            features.shape[0],
            self.context_dim,
        ):
            raise ValueError(
                f"CE must be [{features.shape[0]},{self.context_dim}], got "
                f"{tuple(clean_context.shape)}"
            )
        _require_finite("CE features", features)
        _require_finite("CE context", clean_context)

        batch, channels, height, width = features.shape
        feature_tokens = layer_norm_zero_affine(
            features.flatten(2).transpose(1, 2)
        )
        context = layer_norm_zero_affine(clean_context)
        context_tokens = self.context_to_tokens(context).view(
            batch,
            self.num_tokens,
            self.rank,
        )
        query = self._split_heads(self.to_q(feature_tokens))
        key = self._split_heads(self.to_k(context_tokens))
        value = self._split_heads(self.to_v(context_tokens))
        query = F.normalize(query.float(), dim=-1, eps=1.0e-6)
        key = F.normalize(key.float(), dim=-1, eps=1.0e-6)
        attention_logits = self.attention_logit_scale * torch.matmul(
            query,
            key.transpose(-2, -1),
        )
        attention = F.softmax(attention_logits, dim=-1)
        attended = torch.matmul(
            attention.to(dtype=value.dtype),
            value,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch,
            -1,
            self.rank,
        )
        raw_tokens = self.to_out(attended)
        raw_update = raw_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )
        protected_update, protection, reference_power = (
            _soft_cap_spatial_residual(
                features,
                raw_update,
                self.residual_cap,
                self.protection_eps,
            )
        )

        presence = (
            clean_context.detach().float().norm(dim=1)
            > self.presence_eps
        ).to(dtype=protected_update.dtype).view(batch, 1, 1, 1)
        effective_update = (
            protected_update
            * presence
            * warmup_scale.to(dtype=protected_update.dtype)
            * protected_update.new_tensor(float(branch_gate))
        )
        output = features + effective_update
        _require_finite("CE output", output)

        attention_fp32 = attention.detach().float()
        entropy = -(
            attention_fp32
            * attention_fp32.clamp_min(1.0e-12).log()
        ).sum(dim=-1).mean()
        effective_relative_power = (
            effective_update.float()
            .square()
            .mean(dim=(-2, -1), keepdim=True)
            / reference_power
        )
        utilization_live = (
            effective_relative_power / (self.residual_cap**2)
        ).mean()
        telemetry = {
            "attention_mean": attention_fp32.mean(),
            "attention_max": attention_fp32.max(),
            "attention_min": attention_fp32.min(),
            "attention_entropy": entropy,
            "attention_entropy_normalized": entropy
            / torch.log(features.new_tensor(float(self.num_tokens))),
            "attention_logit_abs_max": (
                attention_logits.detach().abs().max()
            ),
            "context_token_rms": _rms(context_tokens),
            "attended_rms": _rms(attended),
            "feature_rms": _rms(features),
            "update_rms": _rms(effective_update),
            "update_relative_rms": (
                effective_relative_power.detach().mean().sqrt()
            ),
            "presence_fraction": presence.detach().float().mean(),
            "warmup_scale": warmup_scale.detach().float(),
            "branch_enabled": features.new_tensor(
                float(branch_gate)
            ).detach(),
            "residual_cap": features.new_tensor(
                self.residual_cap
            ).detach(),
        }
        telemetry.update(protection)
        telemetry["cap_utilization_effective_mean"] = (
            effective_relative_power.detach().sqrt()
            / self.residual_cap
        ).mean()
        telemetry["cap_utilization_effective_max"] = (
            effective_relative_power.detach().sqrt()
            / self.residual_cap
        ).max()
        return output, telemetry, utilization_live


class HistoSMoEScore(Histoformer):
    """Exact Histoformer shared path plus globally routed sparse experts."""

    _ALL_NONLATENT_LAYOUT = OrderedDict(
        (
            ("encoder_level1", tuple(range(4))),
            ("encoder_level2", tuple(range(4))),
            ("encoder_level3", tuple(range(6))),
            ("decoder_level3", tuple(range(6))),
            ("decoder_level2", tuple(range(4))),
            ("decoder_level1", tuple(range(4))),
            ("refinement", tuple(range(4))),
        )
    )

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 36,
        num_blocks=(4, 4, 6, 8),
        num_refinement_blocks: int = 4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor: float = 2.667,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        dual_pixel_task: bool = False,
        context_dim: int = 512,
        use_degra: bool = True,
        use_image: bool = True,
        num_experts: int = SCORE_NUM_EXPERTS,
        top_k: int = SCORE_TOP_K,
        router_dim: int = SCORE_ROUTER_DIM,
        expert_expansion: float = SCORE_EXPERT_EXPANSION,
        expert_residual_cap: float = SCORE_EXPERT_RESIDUAL_CAP,
        router_balance_weight: float = 1.0e-3,
        ce_rank: int = 128,
        ce_tokens: int = 4,
        ce_heads: int = 4,
        ce_cap: float = SCORE_CE_CAP,
        attention_logit_scale: float = 2.0,
        fusion_dim: int = SCORE_FUSION_DIM,
        stage_gate_cap: float = SCORE_STAGE_GATE_CAP,
        balanced_training_dispatch: bool = False,
    ):
        if int(dim) != 36:
            raise ValueError(f"HistoSMoEScore requires dim=36, got {dim}")
        if tuple(num_blocks) != (4, 4, 6, 8):
            raise ValueError(
                "HistoSMoEScore requires num_blocks=(4,4,6,8)"
            )
        if int(num_refinement_blocks) != 4:
            raise ValueError("HistoSMoEScore requires four refinement blocks")
        if int(num_experts) != SCORE_NUM_EXPERTS or int(top_k) != 1:
            raise ValueError("HistoSMoEScore requires four Top-1 experts")
        if not bool(use_degra) or not bool(use_image):
            raise ValueError("HistoSMoEScore requires both DE and CE")
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
        )

        self.context_dim = int(context_dim)
        self.use_degra = True
        self.use_image = True
        self.num_experts = int(num_experts)
        self.top_k = 1
        self.router_balance_weight = float(router_balance_weight)
        self.semantic_router = GlobalSemanticTop1Router(
            context_dim=self.context_dim,
            router_dim=router_dim,
            num_experts=self.num_experts,
            balanced_training_dispatch=balanced_training_dispatch,
        )

        stage_channels = {
            "encoder_level1": dim,
            "encoder_level2": dim * 2,
            "encoder_level3": dim * 4,
            "decoder_level3": dim * 4,
            "decoder_level2": dim * 2,
            "decoder_level1": dim * 2,
            "refinement": dim * 2,
        }
        self.cross_modal_stage_gate = CrossModalStageGate(
            stage_channels=stage_channels,
            context_dim=self.context_dim,
            fusion_dim=fusion_dim,
            gate_cap=stage_gate_cap,
        )
        for stage_name, indices in self._ALL_NONLATENT_LAYOUT.items():
            stage = getattr(self, stage_name)
            for block_index in indices:
                stage[block_index] = ScoreMoETransformerBlock(
                    original_block=stage[block_index],
                    channels=stage_channels[stage_name],
                    num_experts=self.num_experts,
                    expert_expansion=expert_expansion,
                    expert_residual_cap=expert_residual_cap,
                    bias=bias,
                )

        self.clean_fusion = StableLowRankBottleneckCrossAttention(
            dim * 8,
            context_dim=self.context_dim,
            rank=ce_rank,
            num_tokens=ce_tokens,
            num_heads=ce_heads,
            residual_cap=ce_cap,
            attention_logit_scale=attention_logit_scale,
        )
        self.num_moe_blocks = sum(
            len(indices)
            for indices in self._ALL_NONLATENT_LAYOUT.values()
        )
        self.last_routing: Optional[List[Dict[str, object]]] = None
        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_router_probabilities: Optional[torch.Tensor] = None
        self.last_router_argmax_expert: Optional[torch.Tensor] = None
        self.last_router_selected_expert: Optional[torch.Tensor] = None
        self.last_balanced_dispatch_applied = False
        self.last_ce: Optional[Dict[str, torch.Tensor]] = None
        self.last_cross_modal_gate: Optional[
            Dict[str, torch.Tensor]
        ] = None


    @staticmethod
    def _forward_stage(
        stage_name: str,
        stage: nn.Sequential,
        features: torch.Tensor,
        route: Mapping[str, torch.Tensor],
        enable_experts: bool,
        routing: List[Dict[str, object]],
        channel_gate: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block_index, block in enumerate(stage):
            if isinstance(block, ScoreMoETransformerBlock):
                features, telemetry = block(
                    features,
                    route=route,
                    enable_experts=enable_experts,
                    channel_gate=channel_gate,
                )
                if telemetry is not None:
                    record = dict(telemetry)
                    record["stage"] = stage_name
                    record["block_index"] = block_index
                    record["probabilities"] = route[
                        "probabilities"
                    ].detach()
                    record["top_indices"] = route[
                        "selected_expert"
                    ].detach().unsqueeze(1)
                    routing.append(record)
            else:  # Defensive compatibility with a custom partial layout.
                features = block(features)
        return features

    def _prepare_route(
        self,
        inp_img: torch.Tensor,
        degradation_context: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if degradation_context is None:
            raise ValueError("HistoSMoEScore requires degradation_context")
        if degradation_context.shape != (
            inp_img.shape[0],
            self.context_dim,
        ):
            raise ValueError(
                "degradation_context must be "
                f"[{inp_img.shape[0]},{self.context_dim}], got "
                f"{tuple(degradation_context.shape)}"
            )
        return self.semantic_router(degradation_context)

    def forward(
        self,
        inp_img: torch.Tensor,
        degra_context: Optional[torch.Tensor] = None,
        image_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inp_img.ndim != 4 or inp_img.shape[1] != 3:
            raise ValueError("inp_img must be RGB [B,3,H,W]")
        if degra_context is None:
            raise ValueError("HistoSMoEScore requires degradation_context")
        if degra_context.shape != (
            inp_img.shape[0],
            self.context_dim,
        ):
            raise ValueError(
                "degradation_context must be "
                f"[{inp_img.shape[0]},{self.context_dim}], got "
                f"{tuple(degra_context.shape)}"
            )
        effective_de = degra_context
        route = self.semantic_router(effective_de)
        self.last_router_logits = route["logits"]
        self.last_router_probabilities = route["probabilities"]
        self.last_router_argmax_expert = route["argmax_expert"]
        self.last_router_selected_expert = route["selected_expert"]
        self.last_balanced_dispatch_applied = bool(
            route["balanced_dispatch_applied"]
        )
        routing: List[Dict[str, object]] = []

        enable_experts = True
        enable_ce = True
        if image_context is not None and image_context.shape != (
            inp_img.shape[0],
            self.context_dim,
        ):
            raise ValueError(
                f"image_context must be [{inp_img.shape[0]},"
                f"{self.context_dim}], got "
                f"{tuple(image_context.shape)}"
            )
        if enable_ce and image_context is None:
            raise ValueError("CE-enabled mode requires image_context")
        effective_ce = image_context
        stage_gates: Dict[str, torch.Tensor] = {}
        self.last_cross_modal_gate = None
        if enable_experts:
            stage_gates, self.last_cross_modal_gate = (
                self.cross_modal_stage_gate(
                    effective_de,
                    effective_ce,
                )
            )

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self._forward_stage(
            "encoder_level1",
            self.encoder_level1,
            inp_enc_level1,
            route,
            enable_experts,
            routing,
            stage_gates.get("encoder_level1"),
        )

        inp_enc_level2 = self.down1_2(out_enc_level1)
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(
            torch.cat([inp_enc_level2, skip_enc_level1], dim=1)
        )
        out_enc_level2 = self._forward_stage(
            "encoder_level2",
            self.encoder_level2,
            inp_enc_level2,
            route,
            enable_experts,
            routing,
            stage_gates.get("encoder_level2"),
        )

        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(
            torch.cat([inp_enc_level3, skip_enc_level2], dim=1)
        )
        out_enc_level3 = self._forward_stage(
            "encoder_level3",
            self.encoder_level3,
            inp_enc_level3,
            route,
            enable_experts,
            routing,
            stage_gates.get("encoder_level3"),
        )

        inp_enc_level4 = self.down3_4(out_enc_level3)
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(
            torch.cat([inp_enc_level4, skip_enc_level3], dim=1)
        )
        latent = self.latent(inp_enc_level4)

        self.last_ce = None
        if enable_ce:
            latent, self.last_ce, _ = self.clean_fusion(
                latent,
                effective_ce,
                warmup_scale=latent.new_ones(()),
                branch_gate=1.0,
            )

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self.reduce_chan_level3(
            torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        )
        out_dec_level3 = self._forward_stage(
            "decoder_level3",
            self.decoder_level3,
            inp_dec_level3,
            route,
            enable_experts,
            routing,
            stage_gates.get("decoder_level3"),
        )

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.reduce_chan_level2(
            torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        )
        out_dec_level2 = self._forward_stage(
            "decoder_level2",
            self.decoder_level2,
            inp_dec_level2,
            route,
            enable_experts,
            routing,
            stage_gates.get("decoder_level2"),
        )

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat(
            [inp_dec_level1, out_enc_level1],
            dim=1,
        )
        out_dec_level1 = self._forward_stage(
            "decoder_level1",
            self.decoder_level1,
            inp_dec_level1,
            route,
            enable_experts,
            routing,
            stage_gates.get("decoder_level1"),
        )
        out_dec_level1 = self._forward_stage(
            "refinement",
            self.refinement,
            out_dec_level1,
            route,
            enable_experts,
            routing,
            stage_gates.get("refinement"),
        )

        output = self.output(out_dec_level1) + inp_img
        _require_finite("HistoSMoEScore output", output)
        probabilities = route["probabilities"]
        importance = probabilities.mean(dim=0)
        balance_loss = self.router_balance_weight * (
            self.num_experts * importance.square().sum() - 1.0
        )
        self.last_routing = routing
        return output, balance_loss

    def parameter_report(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        banks = [
            module
            for module in self.modules()
            if isinstance(module, SparseResidualExpertBank)
        ]
        reports = [bank.parameter_report() for bank in banks]
        expert_total = sum(
            report["all_experts"] for report in reports
        )
        inactive = sum(
            report["inactive_experts"] for report in reports
        )
        router = sum(
            parameter.numel()
            for parameter in self.semantic_router.parameters()
        )
        ce = sum(
            parameter.numel()
            for parameter in self.clean_fusion.parameters()
        )
        cross_modal_gate = sum(
            parameter.numel()
            for parameter in self.cross_modal_stage_gate.parameters()
        )
        shared_histoformer = (
            total - expert_total - router - ce - cross_modal_gate
        )
        return {
            "total": total,
            "active_top1": total - inactive,
            "shared_histoformer": shared_histoformer,
            "router": router,
            "experts_all": expert_total,
            "experts_one_active_per_block": (
                expert_total // self.num_experts
            ),
            "inactive_expert_parameters": inactive,
            "ce": ce,
            "cross_modal_gate": cross_modal_gate,
            "moe_blocks": len(banks),
        }

    def assert_parameter_contract(self) -> Dict[str, int]:
        """Fail closed if the locked <20M architecture has drifted."""

        report = self.parameter_report()
        expected = {
            "total": EXPECTED_TOTAL_PARAMETERS,
            "active_top1": EXPECTED_ACTIVE_TOP1_PARAMETERS,
            "shared_histoformer": EXPECTED_SHARED_PARAMETERS,
            "moe_blocks": EXPECTED_MOE_BLOCKS,
        }
        mismatches = {
            key: {"expected": expected_value, "actual": report[key]}
            for key, expected_value in expected.items()
            if report[key] != expected_value
        }
        if mismatches:
            raise RuntimeError(
                f"Histo-SMoE parameter contract drift: {mismatches}"
            )
        if report["total"] >= 20_000_000:
            raise RuntimeError("restoration network is not below 20M")
        return report

    def load_histoformer_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, object]:
        """Strictly warm-start the complete shared path from Histoformer.

        The only permitted missing keys are newly introduced router, expert,
        and CE parameters. Unexpected keys and missing shared keys fail closed.
        """

        own_keys = set(self.state_dict())
        incoming_keys = set(state_dict)
        new_prefix_fragments = (
            "semantic_router.",
            ".expert_bank.",
            "clean_fusion.",
            "cross_modal_stage_gate.",
        )
        new_keys = {
            key
            for key in own_keys
            if key.startswith(new_prefix_fragments[0])
            or new_prefix_fragments[1] in key
            or key.startswith(new_prefix_fragments[2])
            or key.startswith(new_prefix_fragments[3])
        }
        expected_shared = own_keys - new_keys
        missing_shared = sorted(expected_shared - incoming_keys)
        unexpected = sorted(incoming_keys - expected_shared)
        if missing_shared or unexpected:
            raise RuntimeError(
                "incompatible Histoformer warm start: "
                f"missing_shared={missing_shared[:8]}, "
                f"unexpected={unexpected[:8]}"
            )
        incompatible = self.load_state_dict(state_dict, strict=False)
        actual_missing = set(incompatible.missing_keys)
        if actual_missing != new_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "warm-start load contract mismatch: "
                f"missing={sorted(actual_missing)[:8]}, "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        return {
            "loaded_shared_keys": len(expected_shared),
            "identity_initialized_keys": len(new_keys),
            "missing_keys": sorted(actual_missing),
            "unexpected_keys": [],
        }


def build_histoformer(**kwargs) -> Histoformer:
    """Build the locked matched Histoformer and verify its parameter count."""

    model = Histoformer(**kwargs)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if not kwargs and parameter_count != EXPECTED_SHARED_PARAMETERS:
        raise RuntimeError(
            "matched Histoformer parameter contract drift: "
            f"expected {EXPECTED_SHARED_PARAMETERS}, got {parameter_count}"
        )
    return model


def build_histo_smoe_score(**kwargs) -> HistoSMoEScore:
    """Build the score-first model and verify the locked default contract."""

    model = HistoSMoEScore(**kwargs)
    if not kwargs:
        model.assert_parameter_contract()
    return model


__all__ = [
    "CompactDGFFBranch",
    "CrossModalStageGate",
    "EXPECTED_ACTIVE_TOP1_PARAMETERS",
    "EXPECTED_MOE_BLOCKS",
    "EXPECTED_SHARED_PARAMETERS",
    "EXPECTED_TOTAL_PARAMETERS",
    "GlobalSemanticTop1Router",
    "HistoSMoEScore",
    "SCORE_CE_CAP",
    "SCORE_EXPERT_EXPANSION",
    "SCORE_EXPERT_RESIDUAL_CAP",
    "SCORE_FUSION_DIM",
    "SCORE_NUM_EXPERTS",
    "SCORE_ROUTER_DIM",
    "SCORE_STAGE_GATE_CAP",
    "SCORE_TOP_K",
    "ScoreMoETransformerBlock",
    "SparseResidualExpertBank",
    "StableLowRankBottleneckCrossAttention",
    "build_histo_smoe_score",
    "build_histoformer",
    "layer_norm_zero_affine",
]
