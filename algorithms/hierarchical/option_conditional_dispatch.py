"""Hybrid phase sets and ``merge_hybrid_actions``.

``LEARNED_PHASES`` / ``RULE_PHASES`` keep the 116 names. Which components
inside POSITION/FIRE are PPO-owned is decided by ``WorkerContract``.
SEARCH, DEFEND, and ALERT never take a learned component.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from algorithms.hierarchical.worker_contract import (
    WorkerContract,
    default_worker_contract,
)

HYBRID_DISPATCH_ID = "option_conditional_hybrid_v1"
# TODO(2v1new): the drop-in imports this symbol but not its string. When the
# WSL module is mounted, re-export that module's HYBRID_DISPATCH_ID unchanged.

LEARNED_PHASES = frozenset({"POSITION", "FIRE"})
RULE_PHASES = frozenset({"SEARCH", "DEFEND", "ALERT"})


def _phase(name: object) -> str:
    return str(name).strip().upper()


def assert_no_search_leak(phases: Sequence[object], learned_mask: Sequence[bool]) -> None:
    """Raise if a rule phase, including SEARCH, is marked learned."""
    if len(phases) != len(learned_mask):
        raise ValueError("learned mask length differs from phases")
    for phase, learned in zip(phases, learned_mask):
        if not learned:
            continue
        normalized = _phase(phase)
        if normalized in RULE_PHASES or normalized == "SEARCH":
            raise RuntimeError(f"search leak: learned action on {normalized}")


def merge_hybrid_actions(
    *,
    phases: Sequence[object],
    rule_continuous,
    rule_fire,
    rule_scan,
    learned_continuous,
    learned_fire,
    learned_scan,
    contract: WorkerContract | None = None,
):
    """Blend rule and learned commands.

    Returns ``continuous, fire, scan, learned_mask``. Under ``macro_v1`` the
    continuous command and scan bit stay on the rule arrays. Fire is the
    learned bit only on FIRE. POSITION/SEARCH/DEFEND/ALERT force fire to 0
    on this hybrid path. ``legacy_116`` copies all three learned components
    on POSITION and FIRE and leaves rule phases untouched, including rule fire.
    """
    spec = default_worker_contract() if contract is None else contract
    phase_names = [_phase(phase) for phase in phases]
    n = len(phase_names)
    continuous = np.array(rule_continuous, dtype=np.float64, copy=True)
    fire = np.array(rule_fire, dtype=bool, copy=True)
    scan = np.array(rule_scan, dtype=np.int64, copy=True)
    learned_c = np.asarray(learned_continuous, dtype=np.float64)
    learned_f = np.asarray(learned_fire, dtype=bool)
    learned_s = np.asarray(learned_scan)
    if continuous.shape != (n, 3) or learned_c.shape != (n, 3):
        raise ValueError("continuous commands must have shape (aircraft, 3)")
    if fire.shape != (n,) or scan.shape != (n,):
        raise ValueError("fire and scan must have one entry per aircraft")
    if learned_f.shape != (n,) or learned_s.shape != (n,):
        raise ValueError("learned fire and scan must have one entry per aircraft")
    learned_mask = np.zeros(n, dtype=bool)
    for index, phase in enumerate(phase_names):
        if phase not in LEARNED_PHASES and phase not in RULE_PHASES:
            raise ValueError(f"unknown worker phase {phase!r}")
        if phase in RULE_PHASES:
            if not spec.is_legacy:
                fire[index] = False
            continue
        if spec.is_legacy:
            continuous[index] = learned_c[index]
            fire[index] = bool(learned_f[index])
            scan[index] = int(learned_s[index])
            learned_mask[index] = True
            continue
        took = False
        for dim in spec.continuous_dim_index():
            if phase in spec.ppo_phases():
                continuous[index, dim] = learned_c[index, dim]
                took = True
        if spec.scores_fire and phase == "FIRE":
            fire[index] = bool(learned_f[index])
            took = True
        else:
            fire[index] = False
        # scan is intentionally left on the rule value
        learned_mask[index] = took
    return continuous, fire, scan, learned_mask
