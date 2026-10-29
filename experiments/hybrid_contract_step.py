"""Action assembly for the WSL hybrid rollout.

``assemble_hybrid_action`` is the body that replaces the ``rule_worker_action``
+ ``policy.act`` + ``merge_hybrid_actions`` block in
``experiments/learned_worker_v0_smoke.py``. It does not call ``env.step``
and it has no ``__main__``. The diff is written in
``docs/protocols/learned_worker/WSL_APPLY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from algorithms.hierarchical.hybrid_rollout import HybridCommand, compose_hybrid_step
from algorithms.hierarchical.launch_gate import resolve_fire_mask
from algorithms.hierarchical.option_conditional_dispatch import HYBRID_DISPATCH_ID
from algorithms.hierarchical.policy import WorkerPolicy
from algorithms.hierarchical.training_guard import refuse_training_launch
from algorithms.hierarchical.worker_contract import WorkerContract, load_worker_contract
from formation_hrl.rule_worker_hook import rule_worker_action


@dataclass(frozen=True, slots=True)
class AssembledHybridStep:
    """What the smoke loop needs before its existing ``env.step`` call."""

    continuous: np.ndarray
    fire: np.ndarray
    scan: np.ndarray
    learned_mask: np.ndarray
    observation: Tensor
    fire_mask: Tensor
    action_log_prob: Tensor
    learned_components: tuple[str, ...]
    contract_name: str
    dispatch_id: str
    dispatch_id_source: str
    command: HybridCommand
    action: object


def dispatch_identity() -> tuple[str, str]:
    """``(value, source)``. Missing source attribute means the WSL module."""
    from algorithms.hierarchical import option_conditional_dispatch as dispatch

    value = str(dispatch.HYBRID_DISPATCH_ID)
    source = getattr(dispatch, "HYBRID_DISPATCH_ID_SOURCE", "wsl_module")
    return value, str(source)


def _env_fire_mask(snap: object) -> list[bool]:
    raw = getattr(snap, "fire_mask")
    return [bool(bit) for bit in list(raw)]


def assemble_hybrid_action(
    env: object,
    snap: object,
    policy: WorkerPolicy,
    *,
    generator: torch.Generator,
    deterministic: bool = False,
    contract: WorkerContract | None = None,
) -> AssembledHybridStep:
    """Build the hybrid command. The caller still owns ``env.step``."""
    spec = load_worker_contract() if contract is None else contract
    # Keep evaluate_actions / hybrid_kl on the same density as this rollout.
    # WorkerPolicy() itself stays on legacy_116 until this assembly runs.
    policy.contract = spec
    rule_continuous, rule_fire, rule_scan = rule_worker_action(env)
    env_mask = _env_fire_mask(snap)
    gate = resolve_fire_mask(env_mask)
    if any(bit and not env_bit for bit, env_bit in zip(gate.fire_mask, env_mask)):
        raise RuntimeError("LaunchGate widened the environment fire mask")
    observation = torch.as_tensor(snap.worker_observation, dtype=torch.float32)
    command = compose_hybrid_step(
        phases=snap.phase,
        gate=gate,
        rule_continuous=rule_continuous,
        rule_fire=rule_fire,
        rule_scan=rule_scan,
        policy=policy,
        observation=observation,
        generator=generator,
        contract=spec,
        deterministic=deterministic,
    )
    value, source = dispatch_identity()
    if value != HYBRID_DISPATCH_ID and source == "placeholder_drop_in_omitted_wsl_module":
        raise RuntimeError("dispatch id placeholder diverged from the stand-in module")
    return AssembledHybridStep(
        continuous=command.continuous,
        fire=command.fire,
        scan=command.scan,
        learned_mask=command.learned_mask,
        observation=observation,
        fire_mask=gate.as_tensor(),
        action_log_prob=command.log_prob,
        learned_components=command.learned_components,
        contract_name=command.contract_name,
        dispatch_id=value,
        dispatch_id_source=source,
        command=command,
        action=command.action,
    )


if __name__ == "__main__":
    refuse_training_launch("experiments.hybrid_contract_step")
