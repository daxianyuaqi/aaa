"""Rule-macro Worker.

WSL ``formation_hrl.baselines.rule_worker_action`` is the A0 implementation
and is not in this checkout. If the caller installs that function (on the
env or via ``set_host_rule_worker``), it is returned unchanged. The built-in
table is a test fallback, not an A0 substitute and not a weaker doctrine.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

TRACK_MACROS = {
    "SEARCH": "S_LEG",
    "POSITION": "CRANK_HOLD",
    "FIRE": "HOLD_GEOMETRY",
    "DEFEND": "BEAM",
    "ALERT": "BEAM",
}

MACRO_CONTINUOUS = {
    "NOOP": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    "STRAIGHT": np.array([0.0, 0.0, 5.0], dtype=np.float64),
    "S_LEG": np.array([0.4, 0.0, 0.0], dtype=np.float64),
    "CRANK_HOLD": np.array([0.35, 0.0, 0.0], dtype=np.float64),
    "CRANK": np.array([0.6, 0.0, 0.0], dtype=np.float64),
    "HOLD_GEOMETRY": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    "BEAM": np.array([1.2, 0.0, 0.0], dtype=np.float64),
    "PUMP": np.array([1.2, 0.1, 15.0], dtype=np.float64),
    "DRAG": np.array([3.0, 0.0, -10.0], dtype=np.float64),
}

# M02 scan FSM stand-in: search and the position geometry keep the sensor on.
# FIRE and DEFEND do not learn a scan bit; the rule value is fixed.
SCAN_BY_PHASE = {
    "SEARCH": 1,
    "POSITION": 1,
    "FIRE": 0,
    "DEFEND": 0,
    "ALERT": 1,
}

HostRule = Callable[[object], tuple]
HOST_RULE_WORKER: HostRule | None = None


def set_host_rule_worker(fn: HostRule | None) -> None:
    """Install the WSL A0 ``rule_worker_action``. ``None`` clears it."""
    global HOST_RULE_WORKER
    HOST_RULE_WORKER = fn


def _phases_of(env: object) -> list[str]:
    raw = getattr(env, "phase", None)
    if raw is None:
        snapshot = getattr(env, "snapshot", None)
        raw = None if snapshot is None else getattr(snapshot, "phase", None)
    if raw is None:
        raise ValueError("rule_worker_action requires env.phase or env.snapshot.phase")
    return [str(phase).strip().upper() for phase in raw]


def _fallback(env: object):
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
    rule_worker_action.last_source = "fallback_not_a0"  # type: ignore[attr-defined]
    return continuous, fire, scan


def rule_worker_action(env: object, *, contract=None):
    """Return ``continuous[n,3], fire[n], scan[n]``.

    ``contract`` is accepted so call sites can pass the active contract. The
    macro table does not read it: SEARCH/DEFEND/scan are always rule, and A0
    fire stays inside the host function when one is installed.
    """
    del contract
    host = getattr(env, "host_rule_worker_action", None)
    if host is None:
        host = HOST_RULE_WORKER
    if host is not None:
        continuous, fire, scan = host(env)
        rule_worker_action.last_source = "host"  # type: ignore[attr-defined]
        return continuous, fire, scan
    return _fallback(env)


rule_worker_action.last_source = "unset"  # type: ignore[attr-defined]
