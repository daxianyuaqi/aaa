"""Single fire-mask entry for LaunchGate, NEZ/Rmax, and soft-LAR.

116 keeps the environment mask. This service does not widen it. NEZ and
soft-LAR are recorded on the same result and applied only when the caller
opts into an E4 / A0′ intersection. There is no second mask function.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch
from torch import Tensor


class LaunchGate(Enum):
    BLOCK = "BLOCK"
    ADVISORY = "ADVISORY"
    READY = "READY"


@dataclass(frozen=True, slots=True)
class SoftLarFeatures:
    """113 soft-LAR features. They are inputs to LaunchGate, not a second mask."""

    in_lar: bool | None = None
    lar_bucket: str | None = None


@dataclass(frozen=True, slots=True)
class NezRmaxWindow:
    """E4 weapon window. Absent on the frozen 116 path."""

    in_nez: bool | None = None
    range_m: float | None = None
    rmax_m: float | None = None
    rmin_m: float | None = None


@dataclass(frozen=True, slots=True)
class LaunchGateResult:
    fire_mask: bool
    launch_gate: LaunchGate
    gate_reason: str
    soft_lar_bucket: str | None
    source: str


@dataclass(frozen=True, slots=True)
class LaunchGateBatch:
    """The only object ``compose_hybrid_step`` accepts as a fire mask."""

    results: tuple[LaunchGateResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("LaunchGateBatch requires one result per aircraft")

    @property
    def fire_mask(self) -> tuple[bool, ...]:
        return tuple(item.fire_mask for item in self.results)

    @property
    def launch_gate(self) -> tuple[str, ...]:
        return tuple(item.launch_gate.value for item in self.results)

    @property
    def gate_reason(self) -> tuple[str, ...]:
        return tuple(item.gate_reason for item in self.results)

    @property
    def source(self) -> str:
        return "LaunchGateService"

    def as_tensor(self, *, device: torch.device | None = None) -> Tensor:
        return torch.tensor(self.fire_mask, dtype=torch.bool, device=device)


def _as_tuple(values: Sequence[object] | None, n: int, name: str) -> tuple[object | None, ...]:
    if values is None:
        return tuple(None for _ in range(n))
    if len(values) != n:
        raise ValueError(f"{name} length {len(values)} differs from fire mask length {n}")
    return tuple(values)


def resolve_fire_mask(
    env_fire_mask: Sequence[bool],
    *,
    soft_lar: Sequence[SoftLarFeatures] | None = None,
    nez_rmax: Sequence[NezRmaxWindow] | None = None,
    apply_nez: bool = False,
    apply_soft_lar: bool = False,
) -> LaunchGateBatch:
    """Build ``fire_mask`` from the env mask, optionally intersected with E4.

    ``env_fire_mask`` is the 116 / A0 mask (S5-T advisory inside the
    environment). A True bit is never invented here. ``apply_nez`` and
    ``apply_soft_lar`` default off so this call is a pass-through for 116.
    """
    if isinstance(env_fire_mask, bool) or not isinstance(env_fire_mask, Sequence):
        raise ValueError("env_fire_mask must be a sequence of bools, one per aircraft")
    base = tuple(bool(bit) for bit in env_fire_mask)
    if not base:
        raise ValueError("env_fire_mask must cover at least one aircraft")
    lar_rows = _as_tuple(soft_lar, len(base), "soft_lar")
    nez_rows = _as_tuple(nez_rmax, len(base), "nez_rmax")
    results: list[LaunchGateResult] = []
    for bit, lar, nez in zip(base, lar_rows, nez_rows):
        results.append(
            _one(
                bit,
                lar if isinstance(lar, SoftLarFeatures) or lar is None else _bad("soft_lar"),
                nez if isinstance(nez, NezRmaxWindow) or nez is None else _bad("nez_rmax"),
                apply_nez=apply_nez,
                apply_soft_lar=apply_soft_lar,
            )
        )
    return LaunchGateBatch(tuple(results))


def _bad(name: str) -> None:
    raise TypeError(f"{name} entries must use the LaunchGate dataclasses")


def _one(
    env_bit: bool,
    lar: SoftLarFeatures | None,
    nez: NezRmaxWindow | None,
    *,
    apply_nez: bool,
    apply_soft_lar: bool,
) -> LaunchGateResult:
    bucket = None if lar is None else lar.lar_bucket
    mask = bool(env_bit)
    reasons: list[str] = []
    if not env_bit:
        reasons.append("env_mask_false")
    else:
        reasons.append("s5t_env_mask_pass_through")
    if lar is not None and lar.in_lar is False:
        reasons.append("soft_lar_disagrees_recorded_not_applied" if not apply_soft_lar else "soft_lar_intersection")
        if apply_soft_lar:
            mask = False
    elif lar is not None:
        reasons.append("soft_lar_recorded")
    if apply_nez:
        in_nez = False if nez is None or nez.in_nez is None else bool(nez.in_nez)
        reasons.append("nez_intersection")
        if not in_nez:
            mask = False
    else:
        reasons.append("nez_not_applied")
    if not env_bit:
        mask = False
    if mask and apply_nez and (nez is not None and nez.in_nez):
        gate = LaunchGate.READY
    elif mask:
        gate = LaunchGate.ADVISORY
    else:
        gate = LaunchGate.BLOCK
    source = "env_pass_through" if not apply_nez and not apply_soft_lar else "env_and_e4_intersection"
    return LaunchGateResult(
        fire_mask=mask,
        launch_gate=gate,
        gate_reason="+".join(reasons),
        soft_lar_bucket=bucket,
        source=source,
    )
