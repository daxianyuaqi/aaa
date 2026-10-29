"""Two independent CTDE PPO policies for the 12-way manager and shared worker.

Actors receive only legal observations. Critics alone receive centralized state.

Constructing ``WorkerPolicy`` without a contract keeps the 116 joint density
(3 continuous + fire + scan) so dropping this file onto the WSL tree does not
retarget an existing smoke run. ``compose_hybrid_step`` passes
``worker_macro_v1`` from the yaml, and that path samples masked fire only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from algorithms.hierarchical.worker_contract import WorkerContract
from algorithms.mappo.distributions import MaskedBernoulli, SquashedDiagonalGaussian

MANAGER_POLICY_ID = "bvr_formation_manager_ctde_ppo_v1"
WORKER_POLICY_ID = "bvr_formation_worker_ctde_ppo_v1"

_KL_PROB_EPS = 1e-6


def _kl_squashed_gaussian(
    current: SquashedDiagonalGaussian,
    anchor: SquashedDiagonalGaussian,
    dims: Sequence[int] | None = None,
) -> Tensor:
    """KL(current || anchor) of the pre-squash diagonal Gaussian.

    Both factors share the tanh squash, so the Jacobian terms cancel. ``dims``
    drops channels the contract does not learn.
    """
    log_std_p = current.log_std
    log_std_q = anchor.log_std
    var_p = torch.exp(2.0 * log_std_p)
    var_q = torch.exp(2.0 * log_std_q).clamp_min(1e-12)
    kl = (
        log_std_q
        - log_std_p
        + (var_p + (current.mean - anchor.mean).pow(2)) / (2.0 * var_q)
        - 0.5
    )
    if dims is not None:
        if not dims:
            return torch.zeros(current.mean.shape[:-1], device=current.mean.device, dtype=kl.dtype)
        kl = kl[..., list(dims)]
    return kl.sum(dim=-1)


def _bernoulli_probs(factor: MaskedBernoulli) -> Tensor:
    """Sigmoid probability of the legal Bernoulli. Illegal rows are unused by the caller."""
    safe_logits = torch.where(factor.mask, factor.logits, torch.zeros_like(factor.logits))
    return torch.sigmoid(safe_logits).clamp(_KL_PROB_EPS, 1.0 - _KL_PROB_EPS)


def _kl_masked_bernoulli(current: MaskedBernoulli, anchor: MaskedBernoulli) -> Tensor:
    """KL(current || anchor) for one masked Bernoulli. Illegal rows contribute 0."""
    p_prob = _bernoulli_probs(current)
    q_prob = _bernoulli_probs(anchor)
    kl = p_prob * (p_prob.log() - q_prob.log()) + (1.0 - p_prob) * (
        (1.0 - p_prob).log() - (1.0 - q_prob).log()
    )
    legal = current.mask.bool() & anchor.mask.bool()
    return torch.where(legal, kl, torch.zeros_like(kl))


def _mlp(width: int, hidden: Sequence[int]) -> tuple[nn.Sequential, int]:
    if width <= 0 or not hidden or any(size <= 0 for size in hidden):
        raise ValueError("network widths must be positive")
    layers: list[nn.Module] = []
    previous = width
    for size in hidden:
        layers.extend((nn.Linear(previous, size), nn.Tanh()))
        previous = size
    return nn.Sequential(*layers), previous


def _zeros_like_batch(observation: Tensor, width: int) -> Tensor:
    return torch.zeros(
        *observation.shape[:-1],
        width,
        device=observation.device,
        dtype=observation.dtype,
    )


@dataclass(frozen=True, slots=True)
class WorkerAction:
    pre_squash: Tensor
    continuous: Tensor
    fire: Tensor
    scan: Tensor
    log_prob: Tensor
    learned_components: tuple[str, ...] = ("delta_course", "flight_path", "delta_speed", "fire", "scan")


class WorkerPolicy(nn.Module):
    """One actor shared by all living Blue aircraft, including solo pretraining.

    The network still has the 116 heads (3 continuous, fire, scan) so a
    legacy checkpoint can load. The active ``WorkerContract`` decides which
    of those heads are sampled and scored.
    """

    def __init__(
        self,
        observation_size: int,
        critic_state_size: int,
        action_scale: Tensor,
        hidden_sizes: Sequence[int] = (128, 128),
        initial_log_std: float = -0.5,
        contract: WorkerContract | None = None,
    ) -> None:
        super().__init__()
        if tuple(action_scale.shape) != (3,) or not bool((action_scale > 0).all()):
            raise ValueError("action_scale must have three positive entries")
        if not -5.0 <= initial_log_std <= 2.0:
            raise ValueError("initial log std is outside the supported range")
        self.observation_size = int(observation_size)
        self.critic_state_size = int(critic_state_size)
        # None keeps the 116 joint density. The yaml contract is applied by
        # compose_hybrid_step / assemble_hybrid_action, not by this default.
        self.contract = WorkerContract.legacy_116() if contract is None else contract
        self.actor_trunk, actor_width = _mlp(observation_size, hidden_sizes)
        self.mean_head = nn.Linear(actor_width, 3)
        self.fire_head = nn.Linear(actor_width, 1)
        self.scan_head = nn.Linear(actor_width, 1)
        self.log_std = nn.Parameter(torch.full((3,), float(initial_log_std)))
        self.critic_trunk, critic_width = _mlp(
            critic_state_size + observation_size, hidden_sizes
        )
        self.value_head = nn.Linear(critic_width, 1)
        self.register_buffer("action_scale", action_scale.detach().clone())
        actor_ids = {
            id(parameter)
            for module in (self.actor_trunk, self.mean_head, self.fire_head, self.scan_head)
            for parameter in module.parameters()
        } | {id(self.log_std)}
        critic_ids = {
            id(parameter)
            for module in (self.critic_trunk, self.value_head)
            for parameter in module.parameters()
        }
        if actor_ids & critic_ids:
            raise ValueError("actor and privileged critic parameters must be separate")

    def copy_actor_from(self, source: "WorkerPolicy") -> None:
        """Transfer only legal-observation behavior from solo to team training.

        The padded critic vectors have equal widths but different aircraft slot
        semantics, so copying the solo critic would silently corrupt team value
        estimates. The caller must separately verify the actor layout identity.
        """
        if self.observation_size != source.observation_size:
            raise ValueError("actor observation width differs across transfer")
        if not torch.equal(self.action_scale, source.action_scale):
            raise ValueError("actor action bounds differ across transfer")
        for name in ("actor_trunk", "mean_head", "fire_head", "scan_head"):
            getattr(self, name).load_state_dict(getattr(source, name).state_dict())
        with torch.no_grad():
            self.log_std.copy_(source.log_std)

    def _spec(self, contract: WorkerContract | None) -> WorkerContract:
        return self.contract if contract is None else contract

    def distribution(
        self,
        observation: Tensor,
        fire_mask: Tensor,
        *,
        contract: WorkerContract | None = None,
    ) -> tuple[SquashedDiagonalGaussian, MaskedBernoulli, MaskedBernoulli]:
        """Always return the three 116 factors so existing callers can unpack them.

        ``contract`` decides which factors ``act`` / ``evaluate_actions`` /
        ``hybrid_kl`` actually sample and score. Unscored factors are still
        built so the module signature matches the WSL policy.
        """
        del contract
        if observation.shape[-1] != self.observation_size:
            raise ValueError("worker observation width differs from policy")
        if fire_mask.shape != observation.shape[:-1]:
            raise ValueError("fire mask shape differs from worker batch")
        hidden = self.actor_trunk(observation)
        mean = self.mean_head(hidden)
        continuous = SquashedDiagonalGaussian(
            mean, self.log_std.expand_as(mean), self.action_scale
        )
        fire = MaskedBernoulli(self.fire_head(hidden).squeeze(-1), fire_mask.bool())
        scan = MaskedBernoulli(
            self.scan_head(hidden).squeeze(-1),
            torch.ones_like(fire_mask, dtype=torch.bool),
        )
        return continuous, fire, scan

    def _score(
        self,
        continuous_dist: SquashedDiagonalGaussian | None,
        fire_dist: MaskedBernoulli | None,
        scan_dist: MaskedBernoulli | None,
        pre_squash: Tensor,
        fire: Tensor,
        scan: Tensor,
        spec: WorkerContract,
    ) -> tuple[Tensor, Tensor]:
        log_prob = torch.zeros(fire.shape, device=fire.device, dtype=pre_squash.dtype)
        entropy = torch.zeros_like(log_prob)
        if spec.scores_continuous:
            if continuous_dist is None:
                raise ValueError("continuous density missing for a contract that scores it")
            dims = spec.continuous_dim_index()
            log_prob = log_prob + continuous_dist.log_prob(pre_squash, dims=dims)
            entropy = entropy + continuous_dist.entropy(dims=dims)
        if spec.scores_fire:
            if fire_dist is None:
                raise ValueError("fire density missing for a contract that scores it")
            log_prob = log_prob + fire_dist.log_prob(fire)
            entropy = entropy + fire_dist.entropy()
        if spec.scores_scan:
            if scan_dist is None:
                raise ValueError("scan density missing for a contract that scores it")
            log_prob = log_prob + scan_dist.log_prob(scan)
            entropy = entropy + scan_dist.entropy()
        return log_prob, entropy

    @torch.no_grad()
    def act(
        self,
        observation: Tensor,
        fire_mask: Tensor,
        *,
        generator: torch.Generator,
        deterministic: bool = False,
        contract: WorkerContract | None = None,
    ) -> WorkerAction:
        spec = self._spec(contract)
        continuous_dist, fire_dist, scan_dist = self.distribution(
            observation, fire_mask, contract=spec
        )
        if not spec.scores_continuous:
            pre_squash = _zeros_like_batch(observation, 3)
            continuous = pre_squash
        elif deterministic:
            pre_squash = continuous_dist.mean
            continuous = continuous_dist.mode()
        elif spec.is_legacy:
            # Real WSL SquashedDiagonalGaussian.sample(generator) has no dims.
            pre_squash, continuous = continuous_dist.sample(generator)
        else:
            pre_squash, continuous = continuous_dist.sample(
                generator, dims=spec.continuous_dim_index()
            )
        if not spec.scores_fire:
            fire = torch.zeros(observation.shape[:-1], device=observation.device, dtype=observation.dtype)
        elif deterministic:
            fire = fire_dist.mode()
        else:
            fire = fire_dist.sample(generator)
        if not spec.scores_scan:
            scan = torch.zeros_like(fire)
        elif deterministic:
            scan = scan_dist.mode()
        else:
            scan = scan_dist.sample(generator)
        log_prob, _entropy = self._score(
            continuous_dist, fire_dist, scan_dist, pre_squash, fire, scan, spec
        )
        return WorkerAction(
            pre_squash,
            continuous,
            fire,
            scan,
            log_prob,
            learned_components=spec.learned_components(),
        )

    def evaluate_actions(
        self,
        observation: Tensor,
        fire_mask: Tensor,
        pre_squash: Tensor,
        fire: Tensor,
        scan: Tensor,
        *,
        contract: WorkerContract | None = None,
    ) -> tuple[Tensor, Tensor]:
        spec = self._spec(contract)
        continuous_dist, fire_dist, scan_dist = self.distribution(
            observation, fire_mask, contract=spec
        )
        return self._score(
            continuous_dist, fire_dist, scan_dist, pre_squash, fire, scan, spec
        )

    def hybrid_kl(
        self,
        anchor: "WorkerPolicy",
        observation: Tensor,
        fire_mask: Tensor,
        *,
        contract: WorkerContract | None = None,
    ) -> Tensor:
        """KL(self || anchor) on the factors this contract learns.

        Gradients flow only into ``self``. The caller must pass rows whose
        phase is inside ``contract.ppo_phases()``. ``anchor`` has to be a
        frozen copy, not the live module, and it must use the same contract.
        """
        if anchor is self:
            raise ValueError("anchor must be a frozen copy, not the live policy")
        if anchor.observation_size != self.observation_size:
            raise ValueError("anchor observation width differs from the live policy")
        spec = self._spec(contract)
        if anchor.contract.learned_components() != spec.learned_components():
            raise ValueError("anchor contract density differs from the live policy")
        continuous_p, fire_p, scan_p = self.distribution(observation, fire_mask, contract=spec)
        with torch.no_grad():
            continuous_q, fire_q, scan_q = anchor.distribution(
                observation, fire_mask, contract=spec
            )
        parts: list[Tensor] = []
        if spec.scores_continuous:
            if continuous_p is None or continuous_q is None:
                raise ValueError("continuous factor missing from KL")
            parts.append(
                _kl_squashed_gaussian(
                    continuous_p, continuous_q, dims=spec.continuous_dim_index()
                )
            )
        if spec.scores_fire:
            if fire_p is None or fire_q is None:
                raise ValueError("fire factor missing from KL")
            parts.append(_kl_masked_bernoulli(fire_p, fire_q))
        if spec.scores_scan:
            if scan_p is None or scan_q is None:
                raise ValueError("scan factor missing from KL")
            parts.append(_kl_masked_bernoulli(scan_p, scan_q))
        if not parts:
            raise ValueError("contract has an empty learned density")
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total

    def value(self, critic_state: Tensor, worker_observation: Tensor) -> Tensor:
        if critic_state.shape[-1] != self.critic_state_size:
            raise ValueError("worker critic state width differs from policy")
        if worker_observation.shape[-1] != self.observation_size:
            raise ValueError("worker critic observation width differs from policy")
        if critic_state.shape[:-1] != worker_observation.shape[:-2]:
            raise ValueError("worker critic batch dimensions differ")
        expanded = critic_state.unsqueeze(-2).expand(
            *worker_observation.shape[:-1], self.critic_state_size
        )
        combined = torch.cat((expanded, worker_observation), dim=-1)
        return self.value_head(self.critic_trunk(combined)).squeeze(-1)


@dataclass(frozen=True, slots=True)
class ManagerAction:
    choice: Tensor
    log_prob: Tensor


class ManagerPolicy(nn.Module):
    """Leader-only 12-way policy scaffold.

    Protocol 116 does not train this module. 119+ would, and is not started
    here. DEFEND is not one of the learned options; it stays a rule preempt.
    """

    def __init__(
        self,
        leader_observation_size: int,
        critic_state_size: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.leader_observation_size = int(leader_observation_size)
        self.critic_state_size = int(critic_state_size)
        self.actor_trunk, actor_width = _mlp(leader_observation_size, hidden_sizes)
        self.logit_head = nn.Linear(actor_width, 12)
        self.critic_trunk, critic_width = _mlp(critic_state_size, hidden_sizes)
        self.value_head = nn.Linear(critic_width, 1)

    def _distribution(self, observation: Tensor, mask: Tensor) -> torch.distributions.Categorical:
        if observation.shape[-1] != self.leader_observation_size:
            raise ValueError("manager observation width differs from policy")
        if mask.shape != (*observation.shape[:-1], 12):
            raise ValueError("manager mask must match batch and have width 12")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("manager has no legal action")
        logits = self.logit_head(self.actor_trunk(observation))
        return torch.distributions.Categorical(
            logits=logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        )

    @torch.no_grad()
    def act(
        self,
        observation: Tensor,
        mask: Tensor,
        *,
        generator: torch.Generator,
        deterministic: bool = False,
    ) -> ManagerAction:
        distribution = self._distribution(observation, mask)
        if deterministic:
            choice = distribution.logits.argmax(dim=-1)
        else:
            probabilities = distribution.probs
            shape = probabilities.shape[:-1]
            choice = torch.multinomial(
                probabilities.reshape(-1, 12), 1, generator=generator
            ).reshape(shape)
        return ManagerAction(choice, distribution.log_prob(choice))

    def evaluate_actions(
        self, observation: Tensor, mask: Tensor, choice: Tensor
    ) -> tuple[Tensor, Tensor]:
        distribution = self._distribution(observation, mask)
        if bool((~mask.bool().gather(-1, choice.long().unsqueeze(-1)).squeeze(-1)).any()):
            raise ValueError("stored manager choice is illegal under its stored mask")
        return distribution.log_prob(choice), distribution.entropy()

    def value(self, critic_state: Tensor) -> Tensor:
        if critic_state.shape[-1] != self.critic_state_size:
            raise ValueError("manager critic state width differs from policy")
        return self.value_head(self.critic_trunk(critic_state)).squeeze(-1)
