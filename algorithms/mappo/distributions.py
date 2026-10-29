"""Squashed diagonal Gaussian and masked Bernoulli.

These match the drop-in ``WorkerPolicy`` call shape: continuous ``log_prob``
scores the pre-squash Gaussian sample, and ``MaskedBernoulli`` contributes 0
on illegal fire rows.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

_LOG_2PI = math.log(2.0 * math.pi)


class SquashedDiagonalGaussian:
    """tanh-squashed diagonal normal. ``log_prob`` is the pre-squash density."""

    def __init__(self, mean: Tensor, log_std: Tensor, action_scale: Tensor) -> None:
        if mean.shape != log_std.shape:
            raise ValueError("mean and log_std shapes differ")
        if action_scale.shape[-1] != mean.shape[-1]:
            raise ValueError("action_scale width differs from the Gaussian")
        self.mean = mean
        self.log_std = log_std
        self.action_scale = action_scale

    def _pre_squash_log_prob(self, pre_squash: Tensor) -> Tensor:
        if pre_squash.shape != self.mean.shape:
            raise ValueError("pre_squash shape differs from the Gaussian")
        var = torch.exp(2.0 * self.log_std)
        return -0.5 * (
            (pre_squash - self.mean).pow(2) / var + (2.0 * self.log_std) + _LOG_2PI
        )

    def log_prob(self, pre_squash: Tensor, dims: Sequence[int] | None = None) -> Tensor:
        log_p = self._pre_squash_log_prob(pre_squash)
        if dims is not None:
            if not dims:
                return torch.zeros(pre_squash.shape[:-1], device=pre_squash.device, dtype=pre_squash.dtype)
            log_p = log_p[..., list(dims)]
        return log_p.sum(dim=-1)

    def sample(
        self,
        generator: torch.Generator,
        dims: Sequence[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw only ``dims``. Unlisted channels stay 0 and are not random."""
        pre_squash = torch.zeros_like(self.mean)
        indexes = tuple(range(self.mean.shape[-1])) if dims is None else tuple(dims)
        if indexes:
            noise = torch.randn(
                *self.mean.shape[:-1],
                len(indexes),
                generator=generator,
                device=self.mean.device,
                dtype=self.mean.dtype,
            )
            std = torch.exp(self.log_std)
            for slot, index in enumerate(indexes):
                pre_squash[..., index] = self.mean[..., index] + std[..., index] * noise[..., slot]
        continuous = torch.tanh(pre_squash) * self.action_scale
        return pre_squash, continuous

    def mode(self) -> Tensor:
        return torch.tanh(self.mean) * self.action_scale

    def entropy(self, dims: Sequence[int] | None = None) -> Tensor:
        var = torch.exp(2.0 * self.log_std)
        ent = 0.5 * (1.0 + _LOG_2PI + torch.log(var))
        if dims is not None:
            if not dims:
                return torch.zeros(self.mean.shape[:-1], device=self.mean.device, dtype=self.mean.dtype)
            ent = ent[..., list(dims)]
        return ent.sum(dim=-1)


class MaskedBernoulli:
    """Bernoulli on legal rows. Illegal rows sample 0 and add 0 log-prob."""

    def __init__(self, logits: Tensor, mask: Tensor) -> None:
        if logits.shape != mask.shape:
            raise ValueError("Bernoulli logits and mask shapes differ")
        self.logits = logits
        self.mask = mask.bool()

    def _probs(self) -> Tensor:
        safe = torch.where(self.mask, self.logits, torch.zeros_like(self.logits))
        return torch.sigmoid(safe)

    def sample(self, generator: torch.Generator) -> Tensor:
        probs = self._probs()
        draw = torch.rand(
            probs.shape,
            generator=generator,
            device=probs.device,
            dtype=probs.dtype,
        )
        bit = draw < probs
        return torch.where(self.mask, bit, torch.zeros_like(bit)).to(dtype=probs.dtype)

    def mode(self) -> Tensor:
        bit = self._probs() >= 0.5
        return torch.where(self.mask, bit, torch.zeros_like(bit)).to(dtype=self.logits.dtype)

    def log_prob(self, action: Tensor) -> Tensor:
        if action.shape != self.logits.shape:
            raise ValueError("Bernoulli action shape differs from logits")
        probs = self._probs().clamp(1e-6, 1.0 - 1e-6)
        taken = action.to(dtype=probs.dtype)
        raw = taken * probs.log() + (1.0 - taken) * (1.0 - probs).log()
        return torch.where(self.mask, raw, torch.zeros_like(raw))

    def entropy(self) -> Tensor:
        probs = self._probs().clamp(1e-6, 1.0 - 1e-6)
        ent = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
        return torch.where(self.mask, ent, torch.zeros_like(ent))
