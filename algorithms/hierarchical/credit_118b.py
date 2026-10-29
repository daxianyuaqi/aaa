"""118b credit rewrite. Offline on ``tilde_r``; ``reward`` stays put.

``none`` is the 116 path. ``launch_step`` moves a hit share onto the
launcher's last FIRE step that actually fired. ``lc_mscr`` does that and
also pays support steps in proportion to ``link_mask``. SEARCH / DEFEND /
ALERT rows are not given a dense term. This module does not step an
environment and does not claim a trained policy.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from algorithms.hierarchical.worker_contract import CREDIT_LAUNCH_STEP, CREDIT_LC_MSCR, CREDIT_NONE

_RULE_PHASES = frozenset({"SEARCH", "DEFEND", "ALERT"})
_SUPPORT_PHASES = frozenset({"FIRE", "POSITION"})


def rewrite_tilde_r(
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    *,
    mode: str = CREDIT_NONE,
    alpha: float = 1.0,
    beta: float = 1.0,
    r_hit: float | None = None,
) -> list[list[dict[str, Any]]]:
    """Return a copy of each trajectory with a ``tilde_r`` field.

    The input ``reward`` values are copied and then left alone. Hit mass is
    moved only in ``tilde_r``.
    """
    if mode not in {CREDIT_NONE, CREDIT_LAUNCH_STEP, CREDIT_LC_MSCR}:
        raise ValueError("credit mode must be none, launch_step, or lc_mscr")
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("credit shares must be non-negative")
    copied: list[list[dict[str, Any]]] = []
    for trajectory in trajectories:
        rows = [dict(step) for step in trajectory]
        for step in rows:
            step["tilde_r"] = float(step["reward"])
        copied.append(rows)
    if mode == CREDIT_NONE:
        return copied
    for trajectory in copied:
        for step in trajectory:
            if not step.get("is_hit"):
                continue
            hit_r = float(step["reward"]) if r_hit is None else float(r_hit)
            _move_launch(copied, step, alpha * hit_r)
            if mode == CREDIT_LC_MSCR:
                _move_support(copied, step, beta * hit_r)
    return copied


def _move_launch(
    trajectories: list[list[dict[str, Any]]],
    hit: dict[str, Any],
    share: float,
) -> None:
    launcher = hit.get("launcher")
    target = None
    for trajectory in trajectories:
        for step in trajectory:
            if step.get("agent") != launcher:
                continue
            if str(step.get("phase", "")).upper() != "FIRE":
                continue
            if not step.get("fire"):
                continue
            target = step
    if target is None or share == 0.0:
        return
    target["tilde_r"] = float(target["tilde_r"]) + share
    hit["tilde_r"] = float(hit["tilde_r"]) - share


def _move_support(
    trajectories: list[list[dict[str, Any]]],
    hit: dict[str, Any],
    unit: float,
) -> None:
    if unit == 0.0:
        return
    for trajectory in trajectories:
        for step in trajectory:
            if step is hit:
                continue
            phase = str(step.get("phase", "")).upper()
            if phase in _RULE_PHASES or phase not in _SUPPORT_PHASES:
                continue
            if not step.get("support_flag"):
                continue
            link = float(step.get("link_mask") or 0.0)
            if link <= 0.0:
                continue
            bonus = unit * link
            step["tilde_r"] = float(step["tilde_r"]) + bonus
            hit["tilde_r"] = float(hit["tilde_r"]) - bonus
