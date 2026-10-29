"""Injectable front door for ``rule_worker_action``.

Default path calls ``formation_hrl.baselines.rule_worker_action`` and returns
that tuple unchanged. ``set_host_rule_worker`` overrides only this front
door. ``a0_rule_worker_action`` ignores the override and refuses the test
fallback, so A0 eval cannot be pointed at the macro table.

Do not replace ``formation_hrl/baselines.py`` on WSL with the copy in this
checkout. That copy is marked ``is_contract_fallback`` and is not A0.
"""

from __future__ import annotations

from typing import Callable

HostRule = Callable[[object], tuple]
_HOST: HostRule | None = None


def set_host_rule_worker(fn: HostRule | None) -> None:
    """Install an override for the hybrid front door. ``None`` restores A0."""
    global _HOST
    _HOST = fn


def a0_rule_worker_action(env: object):
    """Frozen A0 entry. Ignores ``set_host_rule_worker``.

    Raises if this checkout only has the test fallback, instead of running
    that fallback and calling it A0.
    """
    from formation_hrl import baselines

    fn = baselines.rule_worker_action
    if getattr(fn, "is_contract_fallback", False):
        raise RuntimeError(
            "formation_hrl.baselines.rule_worker_action in this checkout is the "
            "test fallback, not WSL A0. Refusing to run it as A0."
        )
    return fn(env)


def rule_worker_action(env: object):
    """Hybrid rule entry. A0's own function is returned as-is when present."""
    if _HOST is not None:
        rule_worker_action.last_source = "host"  # type: ignore[attr-defined]
        return _HOST(env)
    from formation_hrl import baselines

    a0 = baselines.rule_worker_action
    if getattr(a0, "is_contract_fallback", False):
        rule_worker_action.last_source = "fallback_not_a0"  # type: ignore[attr-defined]
        return a0(env)
    rule_worker_action.last_source = "a0"  # type: ignore[attr-defined]
    return a0(env)


rule_worker_action.last_source = "unset"  # type: ignore[attr-defined]
