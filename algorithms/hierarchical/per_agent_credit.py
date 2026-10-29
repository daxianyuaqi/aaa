"""Per-aircraft GAE and terminal credit for the hybrid Worker.

The PPO batch only contains POSITION/FIRE steps. SEARCH/DEFEND stay on the rule
Worker and must not sit in the advantage chain. Episode-end reward is written
onto each aircraft's last learned step when that aircraft was in a rule phase
at termination, then GAE runs separately on each aircraft's trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from algorithms.hierarchical.option_conditional_dispatch import LEARNED_PHASES, RULE_PHASES


def _phase_of(step: Mapping[str, Any]) -> str | None:
    phase = step.get("phase")
    if phase is None:
        return None
    return str(phase)


def is_learned_step(step: Mapping[str, Any]) -> bool:
    """True for POSITION/FIRE rows. A missing phase is treated as learned.

    Rule phases return False. Any other name is a dispatch bug and raises,
    matching ``learned_agent_mask``.
    """
    phase = _phase_of(step)
    if phase is None or phase in LEARNED_PHASES:
        return True
    if phase in RULE_PHASES:
        return False
    raise ValueError(f"unknown worker phase {phase!r}")


def generalized_advantage(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """Standard GAE(γ, λ) along one trajectory, oldest step first."""
    if not (len(rewards) == len(values) == len(next_values) == len(dones)):
        raise ValueError("GAE inputs must have the same length")
    advantages: list[float] = [0.0] * len(rewards)
    last = 0.0
    for index in reversed(range(len(rewards))):
        mask = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_values[index] * mask - values[index]
        last = delta + gamma * lam * mask * last
        advantages[index] = last
    returns = [advantages[i] + values[i] for i in range(len(rewards))]
    return advantages, returns


def flush_open_trajectories(
    open_traj: MutableMapping[int, list[dict[str, Any]]],
    *,
    terminal_reward: float | None = None,
    learned_on_terminal: set[int] | None = None,
    bootstrap_only: bool = False,
) -> dict[str, Any]:
    """Close every open per-aircraft trajectory and clear ``open_traj``.

    Real episode end (``bootstrap_only=False``):
    - Aircraft that took a POSITION/FIRE action on the terminal step already
      store that step's reward. The step is marked done and the reward is not
      added again.
    - Aircraft whose terminal step was SEARCH/DEFEND (or otherwise not learned)
      receive ``terminal_reward`` on their previous POSITION/FIRE step, and
      that step is marked done. Credit is never left on a rule step.

    Budget cut or mid-eval reset (``bootstrap_only=True``): trajectories are
    closed without a terminal reward and without forcing ``done``, so GAE
    bootstraps from the stored next value.

    Rule-phase rows are dropped and never returned to the learner.
    """
    if not bootstrap_only and terminal_reward is None:
        raise ValueError("episode end requires terminal_reward")
    learned_now = set() if learned_on_terminal is None else set(learned_on_terminal)
    trajectories: list[list[dict[str, Any]]] = []
    attached_to_prior_learned = 0
    terminal_on_learned_step = 0
    dropped_rule_steps = 0
    for agent, steps in list(open_traj.items()):
        if not steps:
            continue
        learned_steps = [step for step in steps if is_learned_step(step)]
        dropped_rule_steps += len(steps) - len(learned_steps)
        if not learned_steps:
            continue
        if not bootstrap_only:
            last = learned_steps[-1]
            if agent in learned_now:
                last["done"] = True
                terminal_on_learned_step += 1
            else:
                last["reward"] = float(last["reward"]) + float(terminal_reward)
                last["done"] = True
                attached_to_prior_learned += 1
        trajectories.append(learned_steps)
    open_traj.clear()
    return {
        "trajectories": trajectories,
        "attached_to_prior_learned": attached_to_prior_learned,
        "terminal_on_learned_step": terminal_on_learned_step,
        "dropped_rule_steps": dropped_rule_steps,
    }


def gae_on_trajectories(
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    *,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """Run GAE on each aircraft trajectory and concatenate in the same order.

    A rule-phase row is an error: it would put SEARCH/DEFEND back on the
    learned advantage chain.
    """
    advantages: list[float] = []
    returns: list[float] = []
    for steps in trajectories:
        if not steps:
            continue
        for step in steps:
            if not is_learned_step(step):
                phase = _phase_of(step)
                raise ValueError(
                    f"rule phase {phase!r} entered GAE; terminal credit stays on POSITION/FIRE"
                )
        adv, ret = generalized_advantage(
            [float(step["reward"]) for step in steps],
            [float(step["value"]) for step in steps],
            [float(step["next_value"]) for step in steps],
            [bool(step["done"]) for step in steps],
            gamma=gamma,
            lam=lam,
        )
        advantages.extend(adv)
        returns.extend(ret)
    return advantages, returns


def _gate_ok(row: Mapping[str, Any]) -> bool:
    if row.get("error"):
        return False
    for key in ("nan_events", "illegal_fire_events", "search_leak_events"):
        value = row.get(key)
        if value is None:
            return False
        if int(value) != 0:
            return False
    win = row.get("win_rate")
    if win is None:
        return False
    try:
        win_f = float(win)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(win_f):
        return False
    return True


def checkpoint_is_better(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any] | None,
) -> bool:
    """Best-by-win selection. SR-1 is not wired in this pack.

    A candidate must have zero NaN, illegal-fire, and search-leak events.
    Higher win rate wins; equal win rate breaks ties on wins. Exact ties keep
    the incumbent, which is the earlier checkpoint.
    """
    if not _gate_ok(candidate):
        return False
    if incumbent is None or not _gate_ok(incumbent):
        return True
    c_win = float(candidate["win_rate"])
    i_win = float(incumbent["win_rate"])
    if c_win > i_win:
        return True
    if c_win < i_win:
        return False
    return int(candidate.get("wins") or 0) > int(incumbent.get("wins") or 0)


def _fell_below_best(row: Mapping[str, Any], best: Mapping[str, Any]) -> bool:
    """True when ``row`` is a decline from ``best``.

    A gate failure or an eval error counts as a decline. An equal win rate does
    not, so a tie neither resets the streak nor adds to it.
    """
    if not _gate_ok(row):
        return True
    return float(row["win_rate"]) < float(best["win_rate"])


@dataclass(frozen=True, slots=True)
class PeakDecision:
    """What one mid-eval does to the peak tracker.

    ``is_new_best`` still follows win rate, then wins. A win of 0 can be the
    exported fallback. ``qualified`` is true only when that new best has
    ``win_rate >= min_win_for_best``. ``refresh_anchor`` follows ``qualified``.
    ``cool_down`` is the first qualified best only (``qualified_updates == 1``),
    not the first best of any win rate. ``early_stop`` counts declines only
    after a qualified best exists. Patience 0 never stops. ``min_win_for_best``
    of 0 qualifies every gate-clean new best, which is the previous behavior.
    """

    is_new_best: bool
    refresh_anchor: bool
    cool_down: bool
    early_stop: bool
    below_best: bool
    worse_streak: int
    best_updates: int
    qualified: bool
    qualified_updates: int


def observe_mid_eval(
    row: Mapping[str, Any],
    *,
    best: Mapping[str, Any] | None,
    worse_streak: int,
    best_updates: int,
    patience: int,
    min_win_for_best: float = 0.0,
    qualified_updates: int = 0,
) -> PeakDecision:
    """Advance the L3 peak counter and the L1 re-pin signal from one mid-eval.

    ``min_win_for_best`` does not change which row is saved as best. It only
    decides whether that new best may re-pin the anchor, cool lr/entropy, and
    arm the early-stop streak. The default 0 keeps every gate-clean new best
    qualified, including a start eval of win rate 0.
    """
    if int(patience) < 0:
        raise ValueError("early-stop patience must be non-negative")
    floor = float(min_win_for_best)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("min_win_for_best must be finite and >= 0")
    if int(qualified_updates) < 0:
        raise ValueError("qualified_updates must be non-negative")
    updates = int(best_updates)
    streak = int(worse_streak)
    q_updates = int(qualified_updates)
    if checkpoint_is_better(row, best):
        updates += 1
        qualified = float(row["win_rate"]) >= floor
        if qualified:
            q_updates += 1
        return PeakDecision(
            is_new_best=True,
            refresh_anchor=qualified,
            cool_down=qualified and q_updates == 1,
            early_stop=False,
            below_best=False,
            worse_streak=0,
            best_updates=updates,
            qualified=qualified,
            qualified_updates=q_updates,
        )
    # Declines arm early-stop only after some qualified best exists.
    below = q_updates > 0 and best is not None and _fell_below_best(row, best)
    if below:
        streak += 1
    return PeakDecision(
        is_new_best=False,
        refresh_anchor=False,
        cool_down=False,
        early_stop=int(patience) > 0 and below and streak >= int(patience),
        below_best=below,
        worse_streak=streak,
        best_updates=updates,
        qualified=False,
        qualified_updates=q_updates,
    )


def apply_protocol_credit(
    trajectories: Sequence[Sequence[Mapping[str, Any]]],
    *,
    credit_rewrite: str = "none",
    **kwargs: Any,
) -> list[list[dict[str, Any]]]:
    """116 terminal credit stays in ``flush_open_trajectories``.

    ``credit_rewrite="none"`` (the default, including legacy_116) copies
    ``reward`` onto ``tilde_r`` and does not move mass. ``launch_step`` and
    ``lc_mscr`` are the 118b hooks and never rewrite the frozen ``reward``
    field. This function does not run an environment step.
    """
    from algorithms.hierarchical.credit_118b import rewrite_tilde_r

    return rewrite_tilde_r(trajectories, mode=credit_rewrite, **kwargs)
