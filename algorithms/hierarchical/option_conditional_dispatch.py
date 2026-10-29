"""116-shaped hybrid dispatcher stand-in.

Do not copy this file onto WSL ``2v1new``. ``APPLY.md`` in the L1+L3 drop-in
says ``option_conditional_dispatch.py`` stays the copy already on disk, and
that copy was not included in the drop-in, so its ``HYBRID_DISPATCH_ID``
string is not known here.

``merge_hybrid_actions`` matches the smoke keyword signature and does not
take a contract. Macro v1 is applied afterwards by
``contract_merge.apply_worker_contract_actions``. Phase names match the
credit tests: ``POSITION``/``FIRE`` are learned, ``SEARCH``/``DEFEND`` are
rule, and ``ALERT`` is not a phase (it raises).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Placeholder only. On WSL, read the real constant and do not overwrite it.
# Replacement point: this assignment, and the smoke report field ``dispatch_id``.
HYBRID_DISPATCH_ID_SOURCE = "placeholder_drop_in_omitted_wsl_module"
HYBRID_DISPATCH_ID = "option_conditional_hybrid_v1"

LEARNED_PHASES = frozenset({"POSITION", "FIRE"})
RULE_PHASES = frozenset({"SEARCH", "DEFEND"})


def _phase(name: object) -> str:
    return str(name).strip().upper()


def learned_agent_mask(phases: Sequence[object]) -> np.ndarray:
    """Per-aircraft learned bit. Same phase rule as ``is_learned_step``.

    A phase outside the two sets raises. ``ALERT`` is one of those unknown
    names in the 116 credit tests; alert formation is an environment flag,
    not a dispatcher phase.
    """
    mask = np.zeros(len(phases), dtype=bool)
    for index, phase in enumerate(phases):
        normalized = _phase(phase)
        if normalized in LEARNED_PHASES:
            mask[index] = True
        elif normalized in RULE_PHASES:
            mask[index] = False
        else:
            raise ValueError(f"unknown worker phase {normalized!r}")
    return mask


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
):
    """116 blend. Learned phases take the PPO command; rule phases stay rule.

    Returns ``continuous, fire, scan, learned_mask``. This function does not
    know about ``worker_macro_v1``. Callers that need the contract use
    ``apply_worker_contract_actions``, which leaves this path unchanged for
    ``legacy_116``.
    """
    phase_names = [_phase(phase) for phase in phases]
    learned_agent_mask(phase_names)
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
    learned_mask = learned_agent_mask(phase_names)
    for index, phase in enumerate(phase_names):
        if phase not in LEARNED_PHASES:
            continue
        continuous[index] = learned_c[index]
        fire[index] = bool(learned_f[index])
        scan[index] = int(learned_s[index])
    return continuous, fire, scan, learned_mask
