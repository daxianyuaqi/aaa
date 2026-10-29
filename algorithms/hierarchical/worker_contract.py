"""``worker_macro_v1`` switches.

Default is scheme ②: rule macro for continuous and scan, PPO only on
``MaskedBernoulli(fire_mask)``. ``legacy_116`` restores the frozen
3-continuous + fire + scan joint density. Nothing here launches a rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

HEADING_DIM = 0
CONTINUOUS_NAMES = ("delta_course", "flight_path", "delta_speed")
CONTRACT_MACRO_V1 = "macro_v1"
CONTRACT_LEGACY_116 = "legacy_116"
CREDIT_NONE = "none"
CREDIT_LAUNCH_STEP = "launch_step"
CREDIT_LC_MSCR = "lc_mscr"
CREDIT_MODES = frozenset({CREDIT_NONE, CREDIT_LAUNCH_STEP, CREDIT_LC_MSCR})

# Frozen Phase-A formal knobs. 118a uses a different block; do not overwrite these.
PHASE_A_MID_EVAL_FIRST = 997_900
PHASE_A_MID_EVAL_EVERY = 100_000
PHASE_A_MIN_WIN_FOR_BEST = 0.0

# 118a selection block from the seed-23 gate preset. Not a training launch.
SELECTION_118A_MID_EVAL_FIRST = 998_064
SELECTION_118A_MID_EVAL_EVERY = 20_000
SELECTION_118A_MID_EVAL_CASES = 16
SELECTION_118A_MIN_WIN_FOR_BEST = 0.125

_CONTRACT_KEYS = (
    "worker_contract",
    "learn_fire_only",
    "learn_scan",
    "learn_continuous",
    "continuous_dims",
    "learn_heading_only",
    "credit_rewrite",
    "illegal_fire_shaping",
    "selection_118a",
)
_SELECTION_KEYS = (
    "enabled",
    "mid_eval_first",
    "mid_eval_every",
    "mid_eval_cases",
    "min_win_for_best",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_contract_path() -> Path:
    return repo_root() / "configs" / "protocols" / "worker_macro_v1.yaml"


@dataclass(frozen=True, slots=True)
class Selection118aSpec:
    """Qualified-best knobs. ``enabled`` false means the hook is not armed.

    The eligibility line already lives in ``observe_mid_eval`` /
    ``min_win_for_best``. This spec only names the 118a block. It does not
    run mid-eval.
    """

    enabled: bool = False
    mid_eval_first: int = SELECTION_118A_MID_EVAL_FIRST
    mid_eval_every: int = SELECTION_118A_MID_EVAL_EVERY
    mid_eval_cases: int = SELECTION_118A_MID_EVAL_CASES
    min_win_for_best: float = SELECTION_118A_MIN_WIN_FOR_BEST


@dataclass(frozen=True, slots=True)
class WorkerContract:
    """Action-density contract shared by the policy, the merger, and the yaml."""

    worker_contract: str = CONTRACT_MACRO_V1
    learn_fire_only: bool = True
    learn_scan: bool = False
    learn_continuous: bool = False
    continuous_dims: int = 0
    learn_heading_only: bool = False
    credit_rewrite: str = CREDIT_NONE
    illegal_fire_shaping: bool = False
    selection_118a: Selection118aSpec = field(default_factory=Selection118aSpec)

    def __post_init__(self) -> None:
        if self.worker_contract not in {CONTRACT_MACRO_V1, CONTRACT_LEGACY_116}:
            raise ValueError(
                f"worker_contract must be {CONTRACT_MACRO_V1!r} or {CONTRACT_LEGACY_116!r}"
            )
        if self.credit_rewrite not in CREDIT_MODES:
            raise ValueError(
                "credit_rewrite must be one of none, launch_step, lc_mscr"
            )
        if self.continuous_dims not in {0, 1, 3}:
            raise ValueError("continuous_dims must be 0, 1, or 3")
        if self.worker_contract == CONTRACT_LEGACY_116:
            self._require_legacy()
            return
        self._require_macro_v1()

    def _require_legacy(self) -> None:
        if self.learn_fire_only or self.learn_heading_only or self.illegal_fire_shaping:
            raise ValueError(
                "legacy_116 keeps the joint 3+fire+scan density and suppress-only illegal fire"
            )
        if not (self.learn_scan and self.learn_continuous and self.continuous_dims == 3):
            raise ValueError("legacy_116 requires learn_scan, learn_continuous, continuous_dims=3")
        if self.credit_rewrite != CREDIT_NONE:
            raise ValueError("legacy_116 freezes reward credit; use 118b on macro_v1")
        if self.selection_118a.enabled:
            raise ValueError("118a selection stays off on the legacy_116 rollback")

    def _require_macro_v1(self) -> None:
        if self.learn_scan:
            raise ValueError("macro_v1 forces learn_scan=false; scan stays on the rule FSM")
        if self.learn_heading_only:
            if self.continuous_dims != 1 or not self.learn_continuous:
                raise ValueError(
                    "learn_heading_only requires learn_continuous=true and continuous_dims=1"
                )
        elif self.learn_continuous or self.continuous_dims != 0:
            raise ValueError(
                "macro_v1 leaves continuous on the rule macro unless learn_heading_only is set"
            )
        if not self.learn_fire_only and not self.learn_heading_only:
            raise ValueError("macro_v1 needs learn_fire_only and/or learn_heading_only")

    @property
    def is_legacy(self) -> bool:
        return self.worker_contract == CONTRACT_LEGACY_116

    @property
    def scores_fire(self) -> bool:
        return self.is_legacy or self.learn_fire_only

    @property
    def scores_scan(self) -> bool:
        return self.is_legacy or self.learn_scan

    @property
    def scores_continuous(self) -> bool:
        return self.is_legacy or (
            self.learn_heading_only and self.learn_continuous and self.continuous_dims == 1
        )

    def continuous_dim_index(self) -> tuple[int, ...]:
        if self.is_legacy:
            return (0, 1, 2)
        if self.scores_continuous:
            return (HEADING_DIM,)
        return ()

    def learned_components(self) -> tuple[str, ...]:
        names: list[str] = []
        for index in self.continuous_dim_index():
            names.append(CONTINUOUS_NAMES[index])
        if self.scores_fire:
            names.append("fire")
        if self.scores_scan:
            names.append("scan")
        return tuple(names)

    def ppo_phases(self) -> frozenset[str]:
        """Phases that may contribute a PPO row under this contract.

        116 learned phases stay ``POSITION`` and ``FIRE`` on the legacy path.
        Scheme ② puts PPO only on FIRE (the masked fire bit). The heading
        ablation also opens POSITION, because that is where the single
        ``delta_course`` arm is defined.
        """
        if self.is_legacy:
            return frozenset({"POSITION", "FIRE"})
        phases = set()
        if self.scores_fire:
            phases.add("FIRE")
        if self.scores_continuous:
            phases.update({"POSITION", "FIRE"})
        return frozenset(phases)

    @classmethod
    def macro_v1(cls) -> "WorkerContract":
        return cls()

    @classmethod
    def legacy_116(cls) -> "WorkerContract":
        return cls(
            worker_contract=CONTRACT_LEGACY_116,
            learn_fire_only=False,
            learn_scan=True,
            learn_continuous=True,
            continuous_dims=3,
            learn_heading_only=False,
            credit_rewrite=CREDIT_NONE,
            illegal_fire_shaping=False,
            selection_118a=Selection118aSpec(
                enabled=False,
                mid_eval_first=PHASE_A_MID_EVAL_FIRST,
                mid_eval_every=PHASE_A_MID_EVAL_EVERY,
                mid_eval_cases=16,
                min_win_for_best=PHASE_A_MIN_WIN_FOR_BEST,
            ),
        )

    @classmethod
    def heading_ablation(cls) -> "WorkerContract":
        """118c arm: masked fire plus one heading bin. Scan stays on the rule."""
        return cls(
            learn_fire_only=True,
            learn_scan=False,
            learn_continuous=True,
            continuous_dims=1,
            learn_heading_only=True,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WorkerContract":
        unknown = set(payload) - set(_CONTRACT_KEYS)
        if unknown:
            raise ValueError(f"unknown worker contract keys: {sorted(unknown)}")
        missing = [key for key in _CONTRACT_KEYS if key not in payload]
        if missing:
            raise ValueError(f"worker contract yaml missing keys: {missing}")
        selection_payload = payload["selection_118a"]
        if not isinstance(selection_payload, Mapping):
            raise ValueError("selection_118a must be a mapping")
        unknown_sel = set(selection_payload) - set(_SELECTION_KEYS)
        if unknown_sel:
            raise ValueError(f"unknown selection_118a keys: {sorted(unknown_sel)}")
        missing_sel = [key for key in _SELECTION_KEYS if key not in selection_payload]
        if missing_sel:
            raise ValueError(f"selection_118a missing keys: {missing_sel}")
        selection = Selection118aSpec(
            enabled=bool(selection_payload["enabled"]),
            mid_eval_first=int(selection_payload["mid_eval_first"]),
            mid_eval_every=int(selection_payload["mid_eval_every"]),
            mid_eval_cases=int(selection_payload["mid_eval_cases"]),
            min_win_for_best=float(selection_payload["min_win_for_best"]),
        )
        return cls(
            worker_contract=str(payload["worker_contract"]),
            learn_fire_only=bool(payload["learn_fire_only"]),
            learn_scan=bool(payload["learn_scan"]),
            learn_continuous=bool(payload["learn_continuous"]),
            continuous_dims=int(payload["continuous_dims"]),
            learn_heading_only=bool(payload["learn_heading_only"]),
            credit_rewrite=str(payload["credit_rewrite"]),
            illegal_fire_shaping=bool(payload["illegal_fire_shaping"]),
            selection_118a=selection,
        )


_CACHED: WorkerContract | None = None


def reset_contract_cache() -> None:
    global _CACHED
    _CACHED = None


def load_worker_contract(path: Path | None = None) -> WorkerContract:
    """Load the on-disk contract. The default file is scheme ②."""
    source = default_contract_path() if path is None else Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} did not contain a mapping")
    return WorkerContract.from_mapping(payload)


def default_worker_contract() -> WorkerContract:
    global _CACHED
    if _CACHED is None:
        _CACHED = load_worker_contract()
    return _CACHED


def ppo_action_spec(contract: WorkerContract | None = None) -> dict[str, Any]:
    """Learned-action width for smoke tests and for the WSL rollout to log."""
    spec = default_worker_contract() if contract is None else contract
    return {
        "worker_contract": spec.worker_contract,
        "learned_components": spec.learned_components(),
        "continuous_dims": len(spec.continuous_dim_index()),
        "fire": "MaskedBernoulli(fire_mask)" if spec.scores_fire else "rule",
        "scan": "MaskedBernoulli" if spec.scores_scan else "rule",
        "ppo_phases": tuple(sorted(spec.ppo_phases())),
        "credit_rewrite": spec.credit_rewrite,
    }


def assert_same_names(contract: WorkerContract, payload_keys: Sequence[str]) -> None:
    """Test helper: yaml keys and the dataclass fields stay the same set."""
    if tuple(payload_keys) != _CONTRACT_KEYS:
        raise AssertionError(f"contract keys drifted: {payload_keys}")
