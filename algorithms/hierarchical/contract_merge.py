"""Contract overlay on top of the 116 ``merge_hybrid_actions`` signature.

``legacy_116`` returns that merge unchanged, so a WSL dispatcher with the
same keywords keeps its own fire/scan/continuous blend. ``macro_v1`` starts
from the rule command and copies only the learned components. It does not
edit ``formation_hrl.baselines``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from algorithms.hierarchical.option_conditional_dispatch import (
    LEARNED_PHASES,
    RULE_PHASES,
    learned_agent_mask,
    merge_hybrid_actions,
)
from algorithms.hierarchical.worker_contract import WorkerContract


def _phase(name: object) -> str:
    return str(name).strip().upper()


def apply_worker_contract_actions(
    *,
    phases: Sequence[object],
    rule_continuous,
    rule_fire,
    rule_scan,
    learned_continuous,
    learned_fire,
    learned_scan,
    contract: WorkerContract,
):
    """Return ``continuous, fire, scan, learned_mask`` for ``contract``."""
    if contract.is_legacy:
        return merge_hybrid_actions(
            phases=phases,
            rule_continuous=rule_continuous,
            rule_fire=rule_fire,
            rule_scan=rule_scan,
            learned_continuous=learned_continuous,
            learned_fire=learned_fire,
            learned_scan=learned_scan,
        )
    phase_names = [_phase(phase) for phase in phases]
    learned_agent_mask(phase_names)
    n = len(phase_names)
    continuous = np.array(rule_continuous, dtype=np.float64, copy=True)
    fire = np.array(rule_fire, dtype=bool, copy=True)
    scan = np.array(rule_scan, dtype=np.int64, copy=True)
    learned_c = np.asarray(learned_continuous, dtype=np.float64)
    learned_f = np.asarray(learned_fire, dtype=bool)
    if continuous.shape != (n, 3) or learned_c.shape != (n, 3):
        raise ValueError("continuous commands must have shape (aircraft, 3)")
    if fire.shape != (n,) or learned_f.shape != (n,) or scan.shape != (n,):
        raise ValueError("fire, scan, and learned fire need one entry per aircraft")
    learned_mask = np.zeros(n, dtype=bool)
    ppo_phases = contract.ppo_phases()
    for index, phase in enumerate(phase_names):
        if phase in RULE_PHASES:
            fire[index] = False
            continue
        if phase not in LEARNED_PHASES:
            raise ValueError(f"unknown worker phase {phase!r}")
        took = False
        if phase in ppo_phases:
            for dim in contract.continuous_dim_index():
                continuous[index, dim] = learned_c[index, dim]
                took = True
        if contract.scores_fire and phase == "FIRE":
            fire[index] = bool(learned_f[index])
            took = True
        else:
            fire[index] = False
        learned_mask[index] = took
    return continuous, fire, scan, learned_mask
