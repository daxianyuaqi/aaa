"""Local stand-in for the WSL rule Worker. Not A0.

``WSL_OVERWRITE_FORBIDDEN`` is true so the apply notes can refuse to copy
this module onto ``/mnt/d/vmShare/2v1new_backup/2v1new/formation_hrl/baselines.py``.
The real ``rule_worker_action`` stays on that tree. Tests inject it through
``formation_hrl.rule_worker_hook``.
"""

from __future__ import annotations

import numpy as np

WSL_OVERWRITE_FORBIDDEN = True

TRACK_MACROS = {
    "SEARCH": "S_LEG",
    "POSITION": "CRANK_HOLD",
    "FIRE": "HOLD_GEOMETRY",
    "DEFEND": "BEAM",
}

MACRO_CONTINUOUS = {
    "NOOP": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    "STRAIGHT": np.array([0.0, 0.0, 5.0], dtype=np.float64),
    "S_LEG": np.array([0.4, 0.0, 0.0], dtype=np.float64),
    "CRANK_HOLD": np.array([0.35, 0.0, 0.0], dtype=np.float64),
    "HOLD_GEOMETRY": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    "BEAM": np.array([1.2, 0.0, 0.0], dtype=np.float64),
    "PUMP": np.array([1.2, 0.1, 15.0], dtype=np.float64),
    "DRAG": np.array([3.0, 0.0, -10.0], dtype=np.float64),
}

SCAN_BY_PHASE = {
    "SEARCH": 1,
    "POSITION": 1,
    "FIRE": 0,
    "DEFEND": 0,
}


def _phases_of(env: object) -> list[str]:
    raw = getattr(env, "phase", None)
    if raw is None:
        snapshot = getattr(env, "snapshot", None)
        raw = None if snapshot is None else getattr(snapshot, "phase", None)
    if raw is None:
        raise ValueError("rule_worker_action requires env.phase or env.snapshot.phase")
    return [str(phase).strip().upper() for phase in raw]


def rule_worker_action(env: object):
    """Macro table for unit tests. ``is_contract_fallback`` keeps it off the A0 path."""
    phases = _phases_of(env)
    continuous = np.zeros((len(phases), 3), dtype=np.float64)
    fire = np.zeros(len(phases), dtype=bool)
    scan = np.zeros(len(phases), dtype=np.int64)
    for index, phase in enumerate(phases):
        macro = TRACK_MACROS.get(phase)
        if macro is None:
            raise ValueError(f"no rule macro for phase {phase!r}")
        continuous[index] = MACRO_CONTINUOUS[macro]
        scan[index] = int(SCAN_BY_PHASE[phase])
    return continuous, fire, scan


rule_worker_action.is_contract_fallback = True  # type: ignore[attr-defined]
