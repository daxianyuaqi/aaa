"""Illegal-fire shaping gate.

116 and 118 suppress illegal fire in the rollout and do not add a reward
term. A small penalty exists only for a future E4 / A0′ split, and only when
that split turns the flag on. This module does not edit ``raw_team_reward``.
"""

from __future__ import annotations

ILLEGAL_FIRE_PENALTY = -0.01

# Protocols that must keep the 116 suppress-only behavior.
_SUPPRESS_ONLY = frozenset({"116", "118a", "118b", "118c", "118d", "legacy_116"})
_SHAPING_SPLITS = frozenset({"E4", "A0_prime"})

REWARD_PHASE_TABLE: dict[str, dict[str, str]] = {
    "116": {
        "raw_team_reward": "frozen",
        "credit": "terminal_on_last_POSITION_or_FIRE",
        "illegal_fire": "suppress_only",
        "env_shaping": "off",
    },
    "118a": {
        "raw_team_reward": "frozen",
        "credit": "unchanged",
        "selection": "qualified_best_hook",
        "env_shaping": "off",
    },
    "118b": {
        "raw_team_reward": "frozen",
        "credit": "launch_step_or_lc_mscr_tilde_r",
        "env_shaping": "off",
    },
    "118c": {
        "raw_team_reward": "frozen",
        "credit": "same_as_118b",
        "worker": "macro_v1_fire_only_or_heading",
        "env_shaping": "off",
    },
    "E4": {
        "raw_team_reward": "terminal_isomorphic",
        "illegal_fire": "small_penalty_on_A0_prime_only",
        "launch_gate": "nez_rmax_intersection",
    },
    "119": {
        "raw_team_reward": "terminal_plus_option_conditional",
        "defend_dense_in_rl": "forbidden",
        "env_shaping": "E5_new_split",
    },
}


def illegal_fire_penalty(
    *,
    fired: bool,
    fire_mask: bool,
    protocol: str,
    enabled: bool,
) -> float:
    """Return the shaping term for one fire bit.

    The value is 0 unless ``protocol`` is E4 or A0′ and ``enabled`` is true.
    Turning the yaml flag on does not by itself penalize a 116 or 118 step.
    """
    if protocol in _SUPPRESS_ONLY or not enabled:
        return 0.0
    if protocol not in _SHAPING_SPLITS:
        return 0.0
    if bool(fired) and not bool(fire_mask):
        return float(ILLEGAL_FIRE_PENALTY)
    return 0.0
