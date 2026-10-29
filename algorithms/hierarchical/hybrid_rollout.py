"""One hybrid step: LaunchGate mask, contract-limited PPO sample, rule merge.

This is the call the WSL ``learned_worker_v0_smoke`` rollout should use.
It does not call ``env.step`` and it does not run a training update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from algorithms.hierarchical.contract_merge import apply_worker_contract_actions
from algorithms.hierarchical.launch_gate import LaunchGateBatch
from algorithms.hierarchical.option_conditional_dispatch import assert_no_search_leak
from algorithms.hierarchical.policy import WorkerAction, WorkerPolicy
from algorithms.hierarchical.worker_contract import WorkerContract, default_worker_contract


@dataclass(frozen=True, slots=True)
class HybridCommand:
    continuous: np.ndarray
    fire: np.ndarray
    scan: np.ndarray
    learned_mask: np.ndarray
    log_prob: Tensor
    learned_components: tuple[str, ...]
    fire_mask: tuple[bool, ...]
    launch_gate: tuple[str, ...]
    gate_reason: tuple[str, ...]
    contract_name: str
    gate_source: str
    action: WorkerAction


def compose_hybrid_step(
    *,
    phases,
    gate: LaunchGateBatch,
    rule_continuous,
    rule_fire,
    rule_scan,
    policy: WorkerPolicy,
    observation: Tensor,
    generator: torch.Generator,
    contract: WorkerContract | None = None,
    deterministic: bool = False,
) -> HybridCommand:
    """Sample the contract density and overwrite the rest from the rule Worker."""
    if not isinstance(gate, LaunchGateBatch):
        raise TypeError(
            "fire_mask must come from LaunchGateService.resolve_fire_mask; "
            "do not pass a raw env mask into the hybrid step"
        )
    spec = default_worker_contract() if contract is None else contract
    if observation.shape[:-1] != (len(gate.results),):
        raise ValueError("observation batch differs from LaunchGateBatch width")
    action: WorkerAction = policy.act(
        observation,
        gate.as_tensor(device=observation.device),
        generator=generator,
        deterministic=deterministic,
        contract=spec,
    )
    if action.learned_components != spec.learned_components():
        raise RuntimeError("policy sampled a density the contract does not own")
    continuous, fire, scan, learned_mask = apply_worker_contract_actions(
        phases=phases,
        rule_continuous=rule_continuous,
        rule_fire=rule_fire,
        rule_scan=rule_scan,
        learned_continuous=action.continuous.detach().cpu().numpy(),
        learned_fire=action.fire.detach().cpu().numpy().astype(bool),
        learned_scan=action.scan.detach().cpu().numpy(),
        contract=spec,
    )
    assert_no_search_leak(phases, learned_mask)
    return HybridCommand(
        continuous=continuous,
        fire=fire,
        scan=scan,
        learned_mask=learned_mask,
        log_prob=action.log_prob,
        learned_components=action.learned_components,
        fire_mask=gate.fire_mask,
        launch_gate=gate.launch_gate,
        gate_reason=gate.gate_reason,
        contract_name=spec.worker_contract,
        gate_source=gate.source,
        action=action,
    )
