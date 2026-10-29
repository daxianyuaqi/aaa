"""Smoke for the worker_macro_v1 contract. No environment step and no training."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from algorithms.hierarchical.credit_118b import rewrite_tilde_r
from algorithms.hierarchical.hybrid_rollout import compose_hybrid_step
from algorithms.hierarchical.launch_gate import (
    LaunchGate,
    NezRmaxWindow,
    SoftLarFeatures,
    resolve_fire_mask,
)
from algorithms.hierarchical.option_conditional_dispatch import (
    LEARNED_PHASES,
    RULE_PHASES,
    assert_no_search_leak,
    merge_hybrid_actions,
)
from algorithms.hierarchical.per_agent_credit import (
    apply_protocol_credit,
    flush_open_trajectories,
)
from algorithms.hierarchical.policy import WorkerPolicy
from algorithms.hierarchical.reward_phases import illegal_fire_penalty
from algorithms.hierarchical.selection_118a import (
    decide,
    launch_118a_training,
    selection_kwargs,
)
from algorithms.hierarchical.training_guard import (
    DECISION_CAP_640K_UNCHANGED,
    PROTOCOL_117_STARTED,
    TRAINING_LAUNCH_FORBIDDEN,
)
from algorithms.hierarchical.worker_contract import (
    WorkerContract,
    load_worker_contract,
    ppo_action_spec,
)
from algorithms.mappo.distributions import MaskedBernoulli, SquashedDiagonalGaussian
from formation_hrl.baselines import WSL_OVERWRITE_FORBIDDEN, rule_worker_action
from formation_hrl.rule_worker_hook import (
    a0_rule_worker_action,
    rule_worker_action as hooked_rule_worker_action,
    set_host_rule_worker,
)

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "protocols" / "learned_worker" / "worker_macro_v1.md"
YAML = ROOT / "configs" / "protocols" / "worker_macro_v1.yaml"


class _View:
    def __init__(self, phases: list[str]) -> None:
        self.phase = phases


def _policy(contract: WorkerContract | None = None) -> WorkerPolicy:
    scale = torch.tensor([1.0, 0.5, 30.0])
    if contract is None:
        contract = WorkerContract.macro_v1()
    return WorkerPolicy(8, 4, scale, hidden_sizes=(16, 16), contract=contract)


def _obs(n: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(n, 8, generator=generator)


def test_yaml_matches_code_defaults_and_doc() -> None:
    loaded = load_worker_contract(YAML)
    defaults = WorkerContract.macro_v1()
    assert loaded == defaults
    assert loaded.worker_contract == "macro_v1"
    assert loaded.learn_fire_only is True
    assert loaded.learn_scan is False
    assert loaded.learn_continuous is False
    assert loaded.continuous_dims == 0
    assert loaded.learn_heading_only is False
    assert loaded.credit_rewrite == "none"
    assert loaded.illegal_fire_shaping is False
    assert loaded.selection_118a.enabled is False
    assert loaded.selection_118a.mid_eval_first == 998064
    assert loaded.selection_118a.min_win_for_best == 0.125
    spec = ppo_action_spec(loaded)
    assert spec["learned_components"] == ("fire",)
    assert spec["continuous_dims"] == 0
    assert spec["fire"] == "MaskedBernoulli(fire_mask)"
    assert spec["scan"] == "rule"
    assert spec["ppo_phases"] == ("FIRE",)
    text = DOC.read_text(encoding="utf-8")
    payload = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    for key in payload:
        assert key in text
    for key in payload["selection_118a"]:
        assert key in text
    for phrase in ("不削弱 A0", "0.941", "−41.6", "998064", "legacy_116", "resolve_fire_mask"):
        assert phrase in text
    assert "worker_contract" in text and "learn_fire_only" in text
    assert "learn_heading_only" in text and "credit_rewrite" in text


def test_legacy_and_heading_switches() -> None:
    legacy = WorkerContract.legacy_116()
    assert legacy.learned_components() == (
        "delta_course",
        "flight_path",
        "delta_speed",
        "fire",
        "scan",
    )
    assert legacy.continuous_dims == 3
    assert legacy.selection_118a.mid_eval_first == 997900
    assert legacy.selection_118a.min_win_for_best == 0.0
    heading = WorkerContract.heading_ablation()
    assert heading.learned_components() == ("delta_course", "fire")
    assert heading.continuous_dims == 1
    assert heading.learn_scan is False
    assert heading.ppo_phases() == frozenset({"POSITION", "FIRE"})
    try:
        WorkerContract(learn_scan=True)
    except ValueError as exc:
        assert "learn_scan" in str(exc)
    else:
        raise AssertionError("macro_v1 must reject learn_scan")


def test_fire_only_does_not_sample_scan_or_continuous() -> None:
    policy = _policy()
    calls = {"gauss": 0, "bern": 0}
    orig_g = SquashedDiagonalGaussian.sample
    orig_b = MaskedBernoulli.sample

    def gsample(self, generator, dims=None):
        calls["gauss"] += 1
        return orig_g(self, generator, dims=dims)

    def bsample(self, generator):
        calls["bern"] += 1
        return orig_b(self, generator)

    SquashedDiagonalGaussian.sample = gsample  # type: ignore[method-assign]
    MaskedBernoulli.sample = bsample  # type: ignore[method-assign]
    try:
        generator = torch.Generator().manual_seed(1)
        action = policy.act(
            _obs(),
            torch.tensor([True, True, False, True]),
            generator=generator,
        )
    finally:
        SquashedDiagonalGaussian.sample = orig_g  # type: ignore[method-assign]
        MaskedBernoulli.sample = orig_b  # type: ignore[method-assign]
    assert calls == {"gauss": 0, "bern": 1}
    assert action.learned_components == ("fire",)
    assert torch.count_nonzero(action.continuous) == 0
    assert torch.count_nonzero(action.scan) == 0
    assert action.fire[2].item() == 0.0


def test_heading_samples_only_course_bin() -> None:
    contract = WorkerContract.heading_ablation()
    policy = _policy(contract)
    generator = torch.Generator().manual_seed(2)
    action = policy.act(
        _obs(),
        torch.ones(4, dtype=torch.bool),
        generator=generator,
        contract=contract,
    )
    assert action.learned_components == ("delta_course", "fire")
    assert torch.count_nonzero(action.pre_squash[:, 1]) == 0
    assert torch.count_nonzero(action.pre_squash[:, 2]) == 0
    assert torch.count_nonzero(action.pre_squash[:, 0]) > 0


def test_fire_only_log_prob_ignores_motion_heads() -> None:
    policy = _policy()
    observation = _obs()
    fire_mask = torch.ones(4, dtype=torch.bool)
    pre = torch.zeros(4, 3)
    fire = torch.ones(4)
    scan = torch.ones(4)
    log_prob, _entropy = policy.evaluate_actions(observation, fire_mask, pre, fire, scan)
    log_prob.sum().backward()
    assert policy.fire_head.weight.grad is not None
    assert float(policy.fire_head.weight.grad.abs().sum()) > 0.0
    assert policy.mean_head.weight.grad is None
    assert policy.scan_head.weight.grad is None


def test_heading_log_prob_uses_only_course_row() -> None:
    contract = WorkerContract.heading_ablation()
    policy = _policy(contract)
    observation = _obs()
    log_prob, _entropy = policy.evaluate_actions(
        observation,
        torch.ones(4, dtype=torch.bool),
        torch.randn(4, 3),
        torch.ones(4),
        torch.ones(4),
        contract=contract,
    )
    log_prob.sum().backward()
    grad = policy.mean_head.weight.grad
    assert grad is not None
    assert float(grad[0].abs().sum()) > 0.0
    assert float(grad[1].abs().sum()) == 0.0
    assert float(grad[2].abs().sum()) == 0.0
    assert policy.scan_head.weight.grad is None


def test_compose_keeps_rule_motion_and_masked_fire() -> None:
    phases = ["SEARCH", "POSITION", "FIRE", "DEFEND"]
    view = _View(phases)
    rule_c, rule_f, rule_s = rule_worker_action(view)
    assert not bool(rule_f.any())
    gate = resolve_fire_mask(
        [False, False, True, False],
        soft_lar=[
            SoftLarFeatures(in_lar=False, lar_bucket="OUTSIDE"),
            SoftLarFeatures(in_lar=True, lar_bucket="INSIDE"),
            SoftLarFeatures(in_lar=True, lar_bucket="INSIDE"),
            SoftLarFeatures(in_lar=None, lar_bucket=None),
        ],
    )
    assert gate.source == "LaunchGateService"
    assert gate.launch_gate == ("BLOCK", "BLOCK", "ADVISORY", "BLOCK")
    assert "soft_lar_disagrees_recorded_not_applied" in gate.gate_reason[0]
    assert gate.fire_mask[0] is False
    policy = _policy()
    with torch.no_grad():
        policy.fire_head.bias.fill_(10.0)
    command = compose_hybrid_step(
        phases=phases,
        gate=gate,
        rule_continuous=rule_c,
        rule_fire=rule_f,
        rule_scan=rule_s,
        policy=policy,
        observation=_obs(),
        generator=torch.Generator().manual_seed(3),
        deterministic=True,
    )
    assert command.learned_components == ("fire",)
    assert command.contract_name == "macro_v1"
    assert list(command.learned_mask) == [False, False, True, False]
    assert command.continuous.tolist() == rule_c.tolist()
    assert command.scan.tolist() == rule_s.tolist()
    assert command.fire.tolist() == [False, False, True, False]
    try:
        compose_hybrid_step(
            phases=phases,
            gate=torch.ones(4, dtype=torch.bool),  # type: ignore[arg-type]
            rule_continuous=rule_c,
            rule_fire=rule_f,
            rule_scan=rule_s,
            policy=policy,
            observation=_obs(),
            generator=torch.Generator().manual_seed(4),
        )
    except TypeError as exc:
        assert "LaunchGateService" in str(exc)
    else:
        raise AssertionError("raw fire mask must be rejected")


def test_launch_gate_never_widens_the_env_mask() -> None:
    gate = resolve_fire_mask(
        [False, True],
        soft_lar=[SoftLarFeatures(in_lar=True, lar_bucket="INSIDE")] * 2,
        nez_rmax=[NezRmaxWindow(in_nez=True, rmax_m=40000.0), NezRmaxWindow(in_nez=False)],
        apply_nez=True,
        apply_soft_lar=True,
    )
    assert gate.fire_mask == (False, False)
    assert gate.launch_gate[0] == LaunchGate.BLOCK.value
    open_gate = resolve_fire_mask(
        [True],
        nez_rmax=[NezRmaxWindow(in_nez=True)],
        apply_nez=True,
    )
    assert open_gate.fire_mask == (True,)
    assert open_gate.launch_gate == (LaunchGate.READY.value,)
    assert open_gate.results[0].source == "env_and_e4_intersection"


def test_legacy_merge_still_learns_motion_and_scan() -> None:
    phases = ["SEARCH", "POSITION", "FIRE"]
    rule_c = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    learned_c = [[9.0, 9.0, 9.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    continuous, fire, scan, learned = merge_hybrid_actions(
        phases=phases,
        rule_continuous=rule_c,
        rule_fire=[True, True, False],
        rule_scan=[1, 0, 0],
        learned_continuous=learned_c,
        learned_fire=[False, True, True],
        learned_scan=[0, 1, 1],
    )
    assert continuous[0].tolist() == [1.0, 0.0, 0.0]
    assert continuous[1].tolist() == [2.0, 3.0, 4.0]
    assert bool(fire[0]) is True
    assert bool(fire[2]) is True
    assert int(scan[1]) == 1
    assert list(learned) == [False, True, True]
    assert_no_search_leak(phases, learned)
    assert LEARNED_PHASES == frozenset({"POSITION", "FIRE"})
    assert RULE_PHASES == frozenset({"SEARCH", "DEFEND"})
    assert "ALERT" not in RULE_PHASES
    try:
        merge_hybrid_actions(
            phases=["ALERT"],
            rule_continuous=[[0.0, 0.0, 0.0]],
            rule_fire=[False],
            rule_scan=[0],
            learned_continuous=[[0.0, 0.0, 0.0]],
            learned_fire=[False],
            learned_scan=[0],
        )
    except ValueError as exc:
        assert "ALERT" in str(exc)
    else:
        raise AssertionError("ALERT is not a 116 phase")


def test_host_rule_worker_is_not_rewritten() -> None:
    seen = {}
    original = rule_worker_action

    def host(env):
        seen["phases"] = list(env.phase)
        n = len(env.phase)
        return (
            [[0.25, 0.0, 1.0]] * n,
            [True] * n,
            [1] * n,
        )

    set_host_rule_worker(host)
    try:
        continuous, fire, scan = hooked_rule_worker_action(_View(["FIRE"]))
        assert hooked_rule_worker_action.last_source == "host"
        assert list(continuous[0]) == [0.25, 0.0, 1.0]
        assert bool(fire[0]) is True
        assert int(scan[0]) == 1
        assert seen["phases"] == ["FIRE"]
        untouched, _, _ = rule_worker_action(_View(["FIRE"]))
        assert list(untouched[0]) != [0.25, 0.0, 1.0]
        assert rule_worker_action is original
    finally:
        set_host_rule_worker(None)


def test_116_terminal_credit_is_unchanged() -> None:
    open_traj = {
        0: [
            {"phase": "FIRE", "reward": 0.0, "value": 0.1, "next_value": 0.2, "done": False},
            {"phase": "SEARCH", "reward": 0.0, "value": 0.0, "next_value": 0.0, "done": False},
        ]
    }
    flushed = flush_open_trajectories(
        open_traj,
        terminal_reward=1.0,
        learned_on_terminal=set(),
    )
    learned = flushed["trajectories"][0]
    assert learned[-1]["phase"] == "FIRE"
    assert learned[-1]["reward"] == 1.0
    assert flushed["dropped_rule_steps"] == 1
    identity = apply_protocol_credit(flushed["trajectories"], credit_rewrite="none")
    assert identity[0][0]["reward"] == 1.0
    assert identity[0][0]["tilde_r"] == 1.0


def test_118b_rewrites_tilde_r_only() -> None:
    trajectories = [
        [
            {"agent": 0, "phase": "FIRE", "fire": True, "reward": 0.0, "t": 1},
            {"agent": 0, "phase": "SEARCH", "fire": False, "reward": 0.0, "t": 2},
            {
                "agent": 0,
                "phase": "FIRE",
                "fire": False,
                "reward": 4.0,
                "is_hit": True,
                "launcher": 0,
                "t": 3,
            },
        ],
        [
            {
                "agent": 1,
                "phase": "POSITION",
                "support_flag": True,
                "link_mask": 1.0,
                "reward": 0.0,
                "t": 2,
            },
            {
                "agent": 1,
                "phase": "DEFEND",
                "support_flag": True,
                "link_mask": 1.0,
                "reward": 0.2,
                "t": 3,
            },
            {
                "agent": 1,
                "phase": "POSITION",
                "support_flag": True,
                "link_mask": 0.0,
                "reward": 0.0,
                "t": 4,
            },
        ],
    ]
    launch = rewrite_tilde_r(trajectories, mode="launch_step", alpha=1.0)
    assert launch[0][0]["tilde_r"] == 4.0
    assert launch[0][0]["reward"] == 0.0
    assert launch[0][1]["tilde_r"] == 0.0
    assert launch[0][2]["tilde_r"] == 0.0
    assert launch[0][2]["reward"] == 4.0
    linked = rewrite_tilde_r(trajectories, mode="lc_mscr", alpha=1.0, beta=0.5)
    assert linked[1][0]["tilde_r"] == 2.0
    assert linked[1][1]["tilde_r"] == 0.2
    assert linked[1][2]["tilde_r"] == 0.0
    assert linked[0][2]["tilde_r"] == 4.0 - 4.0 - 2.0
    assert apply_protocol_credit(trajectories, credit_rewrite="launch_step")[0][0]["tilde_r"] == 4.0


def test_illegal_fire_penalty_stays_off_for_116_and_118() -> None:
    for protocol in ("116", "118a", "118b", "118c"):
        assert illegal_fire_penalty(
            fired=True, fire_mask=False, protocol=protocol, enabled=True
        ) == 0.0
    assert illegal_fire_penalty(
        fired=True, fire_mask=False, protocol="E4", enabled=False
    ) == 0.0
    assert illegal_fire_penalty(
        fired=True, fire_mask=False, protocol="A0_prime", enabled=True
    ) == -0.01
    assert illegal_fire_penalty(
        fired=True, fire_mask=True, protocol="A0_prime", enabled=True
    ) == 0.0


def test_118a_hook_does_not_train() -> None:
    kwargs = selection_kwargs()
    assert kwargs["enabled"] is False
    assert kwargs["runs_training"] is False
    assert kwargs["runs_eval"] is False
    assert kwargs["mid_eval_first"] == 998064
    assert kwargs["min_win_for_best"] == 0.125
    assert kwargs["phase_a_formal_mid_eval_first"] == 997900
    row = {
        "win_rate": 0.25,
        "wins": 4,
        "nan_events": 0,
        "illegal_fire_events": 0,
        "search_leak_events": 0,
    }
    decision = decide(row)
    assert decision.is_new_best is True
    assert decision.qualified is True
    assert decision.refresh_anchor is True
    low = decide({**row, "win_rate": 0.0, "wins": 0})
    assert low.is_new_best is True
    assert low.qualified is False
    assert low.refresh_anchor is False
    assert TRAINING_LAUNCH_FORBIDDEN is True
    assert PROTOCOL_117_STARTED is False
    assert DECISION_CAP_640K_UNCHANGED == 640_000
    try:
        launch_118a_training()
    except RuntimeError as exc:
        assert "640000" in str(exc)
        assert "117" in str(exc)
    else:
        raise AssertionError("118a launch must refuse")


def test_archive_lock_is_in_the_contract_doc() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "不重训" in text
    assert "协议 117" in text or "协议 117" in text
    assert "640000" in text or "640_000" in text or "640000" in text.replace(",", "")
    assert "DEFEND" in text and "规则抢占" in text
    assert "learn_fire_only" in text
    assert "learn_scan" in text
    assert "continuous_dims" in text
    apply = (ROOT / "docs" / "protocols" / "learned_worker" / "WSL_APPLY.md").read_text(encoding="utf-8")
    assert "不要覆盖" in apply
    assert "option_conditional_dispatch.py" in apply
    assert "baselines.py" in apply
    assert "assemble_hybrid_action" in apply
    assert "env.step" in apply
    assert "tests/test_worker_macro_v1.py" in apply


def test_unspecified_policy_keeps_the_116_joint_density() -> None:
    scale = torch.tensor([1.0, 0.5, 30.0])
    policy = WorkerPolicy(8, 4, scale, hidden_sizes=(16, 16))
    assert policy.contract.is_legacy
    calls = {"gauss": 0, "bern": 0}
    orig_g = SquashedDiagonalGaussian.sample
    orig_b = MaskedBernoulli.sample

    def gsample(self, generator, dims=None):
        calls["gauss"] += 1
        return orig_g(self, generator, dims=dims)

    def bsample(self, generator):
        calls["bern"] += 1
        return orig_b(self, generator)

    SquashedDiagonalGaussian.sample = gsample  # type: ignore[method-assign]
    MaskedBernoulli.sample = bsample  # type: ignore[method-assign]
    try:
        action = policy.act(
            _obs(),
            torch.ones(4, dtype=torch.bool),
            generator=torch.Generator().manual_seed(5),
        )
    finally:
        SquashedDiagonalGaussian.sample = orig_g  # type: ignore[method-assign]
        MaskedBernoulli.sample = orig_b  # type: ignore[method-assign]
    assert calls == {"gauss": 1, "bern": 2}
    assert action.learned_components == (
        "delta_course",
        "flight_path",
        "delta_speed",
        "fire",
        "scan",
    )


def test_hook_delegates_to_a0_without_rewriting_it() -> None:
    import formation_hrl.baselines as baselines

    try:
        a0_rule_worker_action(_View(["FIRE"]))
    except RuntimeError as exc:
        assert "not WSL A0" in str(exc)
    else:
        raise AssertionError("the test fallback must not run as A0")
    assert WSL_OVERWRITE_FORBIDDEN is True
    course = [[4.0, 5.0, 6.0]]
    fired = [True]
    scan = [1]

    def real_a0(env):
        return course, fired, scan

    previous = baselines.rule_worker_action
    baselines.rule_worker_action = real_a0
    set_host_rule_worker(lambda env: ("host-override",))
    try:
        got = a0_rule_worker_action(_View(["SEARCH"]))
        assert got[0] is course and got[1] is fired and got[2] is scan
        assert hooked_rule_worker_action(_View(["SEARCH"])) == ("host-override",)
        set_host_rule_worker(None)
        restored = hooked_rule_worker_action(_View(["SEARCH"]))
        assert restored[0] is course and restored[1] is fired and restored[2] is scan
        assert hooked_rule_worker_action.last_source == "a0"
    finally:
        baselines.rule_worker_action = previous
        set_host_rule_worker(None)


def test_alert_phase_stays_unknown_for_credit() -> None:
    from algorithms.hierarchical.per_agent_credit import is_learned_step

    try:
        is_learned_step({"phase": "ALERT"})
    except ValueError as exc:
        assert "ALERT" in str(exc)
    else:
        raise AssertionError("ALERT must stay outside LEARNED_PHASES and RULE_PHASES")


def test_assemble_hybrid_action_does_not_step_the_env() -> None:
    from algorithms.hierarchical import option_conditional_dispatch as dispatch
    from experiments.hybrid_contract_step import assemble_hybrid_action

    class Env:
        def __init__(self) -> None:
            self.phase = ["SEARCH", "FIRE"]
            self.stepped = False

        def step(self, *_args, **_kwargs):
            self.stepped = True
            raise AssertionError("env.step")

    class Snap:
        phase = ["SEARCH", "FIRE"]
        fire_mask = [False, True]
        worker_observation = [[0.0] * 8, [0.0] * 8]

    env = Env()
    bare = WorkerPolicy(8, 4, torch.tensor([1.0, 0.5, 30.0]), hidden_sizes=(16, 16))
    assert bare.contract.is_legacy
    assembled = assemble_hybrid_action(
        env,
        Snap(),
        bare,
        generator=torch.Generator().manual_seed(6),
        deterministic=True,
    )
    assert bare.contract.worker_contract == "macro_v1"
    assert env.stepped is False
    assert assembled.contract_name == "macro_v1"
    assert assembled.learned_components == ("fire",)
    assert list(assembled.learned_mask) == [False, True]
    assert assembled.continuous[0].tolist() == [0.4, 0.0, 0.0]
    assert assembled.continuous[1].tolist() == [0.0, 0.0, 0.0]
    assert bool(assembled.fire[0]) is False
    source = getattr(dispatch, "HYBRID_DISPATCH_ID_SOURCE", "wsl_module")
    assert assembled.dispatch_id == dispatch.HYBRID_DISPATCH_ID
    assert assembled.dispatch_id_source == source
    if source == "placeholder_drop_in_omitted_wsl_module":
        assert assembled.dispatch_id == "option_conditional_hybrid_v1"
    else:
        assert source == "wsl_module"
        assert assembled.dispatch_id

