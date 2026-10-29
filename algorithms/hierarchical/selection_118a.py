"""118a selection hook.

Qualified-best already lives in ``per_agent_credit.observe_mid_eval``.
This module names the 998064 / ``min_win_for_best=0.125`` block and refuses
to launch training or a 256-case confirmatory eval.
"""

from __future__ import annotations

from typing import Any, Mapping

from algorithms.hierarchical.per_agent_credit import PeakDecision, observe_mid_eval
from algorithms.hierarchical.training_guard import refuse_training_launch
from algorithms.hierarchical.worker_contract import (
    PHASE_A_MID_EVAL_FIRST,
    PHASE_A_MIN_WIN_FOR_BEST,
    Selection118aSpec,
    WorkerContract,
    default_worker_contract,
)


def selection_spec(contract: WorkerContract | None = None) -> Selection118aSpec:
    spec = default_worker_contract() if contract is None else contract
    return spec.selection_118a


def selection_kwargs(contract: WorkerContract | None = None) -> dict[str, Any]:
    """Arguments a future 118a runner would pass. Nothing is executed."""
    spec = selection_spec(contract)
    return {
        "enabled": spec.enabled,
        "mid_eval_first": spec.mid_eval_first,
        "mid_eval_every": spec.mid_eval_every,
        "mid_eval_cases": spec.mid_eval_cases,
        "min_win_for_best": spec.min_win_for_best,
        "runs_training": False,
        "runs_eval": False,
        "phase_a_formal_mid_eval_first": PHASE_A_MID_EVAL_FIRST,
        "phase_a_formal_min_win_for_best": PHASE_A_MIN_WIN_FOR_BEST,
        "eligibility": "per_agent_credit.observe_mid_eval",
    }


def decide(
    row: Mapping[str, Any],
    *,
    best: Mapping[str, Any] | None = None,
    worse_streak: int = 0,
    best_updates: int = 0,
    qualified_updates: int = 0,
    patience: int = 2,
    contract: WorkerContract | None = None,
) -> PeakDecision:
    """Score one already-finished mid-eval row. Does not roll out cases."""
    spec = selection_spec(contract)
    return observe_mid_eval(
        row,
        best=best,
        worse_streak=worse_streak,
        best_updates=best_updates,
        patience=patience,
        min_win_for_best=spec.min_win_for_best,
        qualified_updates=qualified_updates,
    )


def launch_118a_training() -> None:
    refuse_training_launch("selection_118a")
