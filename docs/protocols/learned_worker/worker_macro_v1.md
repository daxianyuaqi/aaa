# worker_macro_v1 — 上层三任务与最小飞控契约

本文件是 **118 可开关、本 PR 不训练** 的实现契约。代码开关与这里的字段同名，默认文件是 `configs/protocols/worker_macro_v1.yaml`。

116 Phase-A 已闭合，本 PR **不重训、不改写那次结果、不削弱 A0**。数字只作对照归档，见文末。

## 1. 本仓与 WSL `2v1new` 的映射

本 Git 树（`daxianyuaqi/aaa`）没有挂载 WSL `2v1new` 的 `bvr_sim` / `FormationDecisionEnvironment`。drop-in 里有 `policy.py`、`per_agent_credit.py`、`learned_worker_v0_smoke.py` 的调用形状，但 **没有** `option_conditional_dispatch.py` 和 `formation_hrl/baselines.py` 的函数体。

| 本仓路径 | WSL `2v1new` | 本 PR |
|---|---|---|
| `algorithms/hierarchical/policy.py` | 同路径 `WorkerPolicy` | 保留 3+fire+scan 头；密度由契约裁剪 |
| `algorithms/hierarchical/option_conditional_dispatch.py` | 全量树已有；drop-in 未带字符串 | 按 smoke 的 `merge_hybrid_actions` / `assert_no_search_leak` 签名补最小实现 |
| `algorithms/hierarchical/per_agent_credit.py` | 同路径 | 116 挂步逻辑原样；末尾 `apply_protocol_credit` 转 118b |
| `algorithms/hierarchical/launch_gate.py` | 现网无统一 `LaunchGateService` | **唯一** `resolve_fire_mask` 入口 |
| `algorithms/hierarchical/hybrid_rollout.py` | smoke 里 `policy.act` + `merge_hybrid_actions` 那段 | `compose_hybrid_step`，**不**调用 `env.step` |
| `formation_hrl/baselines.py` | `rule_worker_action` = A0 | 有 host 则原样委托；无 host 时测试宏回退，**不是 A0** |
| `experiments/learned_worker_v0_smoke.py` | 训练入口 | **未复制**。依赖仿真，且 `main()` 会开跑 |
| `experiments/learned_worker_v0_phase_a.py` | 640k × 6 seed | **未复制、不启动** |
| `bvr_sim`、yaml 场景、JSBSim/autopilot | 全量树 | **缺口**。运动学只锁文档 |

`HYBRID_DISPATCH_ID = "option_conditional_hybrid_v1"` 是本仓占位。WSL 模块里的原字符串不在 drop-in 中，接回全量树时改为 re-export，不要并存两套 dispatcher。

## 2. 上层三任务

短期是方案丙：Manager 仍是规则 S5-T（`manager_mode="rule"`），对外叙事收成两个任务；中期方案甲（119+ Manager PPO）**本 PR 不训练**。

| 上层叙事 | S5-T 投影 | 谁执行 | 能否进 PPO option |
|---|---|---|---|
| **搜索·占位** | SEARCH + POSITION | 规则宏（S-leg / straight / crank-hold） | 否。119+ 才作为 O0 |
| **发射** | FIRE，且 `launch_gate ≥ ADVISORY` | 几何继续规则宏；默认只有 masked fire 可学 | FIRE 的 fire 位可学；不是 Manager PPO |
| **DEFEND / 警戒** | DEFEND、`alert_formation_hold`、`defend=beam_always` | **规则抢占**。可打断搜索·占位和发射，结束后回搜索·占位 | **不是 PPO option**。不进 `LEARNED_PHASES` 的可学分量 |

`LEARNED_PHASES = {POSITION, FIRE}` 与 `RULE_PHASES = {SEARCH, DEFEND, ALERT}` 仍是 116 的相位名字。契约决定这些相位里哪些分量进 PPO，而不是再造一套相位字符串。

`assert_no_search_leak` 保留：SEARCH / DEFEND / ALERT 上 `learned_mask` 必须为假。

## 3. 动作表（`worker_contract: macro_v1`）

方案②，默认。连续三维的语义不变：`delta_course`、`flight_path`、`delta_speed`（对应 autopilot `command_bounds`）。RL 不接舵面、α、β。

| 任务 / 相位 | 连续 / 宏 | fire | scan | 可学？ |
|---|---|---|---|---|
| 搜索·占位 SEARCH | 规则 `S_LEG` | 强制 0 | 规则 FSM | 否 |
| 搜索·占位 POSITION | 规则 `CRANK_HOLD` | 强制 0 | 规则 FSM | 否（航向消融除外） |
| 发射 FIRE | 规则 `HOLD_GEOMETRY` | `MaskedBernoulli(fire_mask)` | 规则 FSM | **默认仅 fire** |
| DEFEND / ALERT 抢占 | 规则 `BEAM`（`beam_always`） | 强制 0 | 规则 FSM | 否 |

强制 fire=0 只发生在 **混合执行器** `merge_hybrid_actions`。A0 评测仍应直接调用 host 的 `rule_worker_action`，不经过这条混合路径，因此本契约 **不改 A0 裁决**。

### 开关（与 yaml / `WorkerContract` 同名）

| 字段 | 默认 `macro_v1` | `legacy_116` 回退 |
|---|---|---|
| `worker_contract` | `macro_v1` | `legacy_116` |
| `learn_fire_only` | `true` | `false`（火与连续、scan 联合） |
| `learn_scan` | `false`（写 `true` 会拒绝加载） | `true` |
| `learn_continuous` | `false` | `true` |
| `continuous_dims` | `0` | `3` |
| `learn_heading_only` | `false` | `false` |
| `credit_rewrite` | `none` | 必须 `none` |
| `illegal_fire_shaping` | `false` | 必须 `false` |
| `selection_118a.enabled` | `false` | 必须 `false` |

`learn_heading_only: true` 是 118c 消融臂，不是默认。它要求 `learn_continuous: true` 且 `continuous_dims: 1`，可学维只有 `delta_course`（下标 0），相位为 POSITION 与 FIRE。scan 仍然强制规则。默认方案②不开这臂。可用 `WorkerContract.heading_ablation()`。

116 回退不要改 yaml 默认值，在调用处显式使用 `WorkerContract.legacy_116()`。该对象恢复 POSITION/FIRE 上 3 连续 + fire + scan 的联合 log_prob，规则相位仍全规则。

网络头不删除：`mean_head` / `fire_head` / `scan_head` 仍在，便于对照加载 116 权重。契约只决定 **采样和 log_prob 包含哪些头**。`macro_v1` 默认路径不调用连续采样，也不调用 scan 采样。

## 4. LaunchGate / fire_mask / NEZ·Rmax / soft-LAR

唯一入口：`algorithms.hierarchical.launch_gate.resolve_fire_mask`。`compose_hybrid_step` 只接受它返回的 `LaunchGateBatch`，拒绝裸 `Tensor`，避免再写一条 mask。

```
Manager（规则；119+ 才 PPO）
  → phase / option
  → LaunchGateService.resolve_fire_mask(env_fire_mask, soft_lar, nez_rmax)
       → fire_mask[i], launch_gate[i], gate_reason[i]
Worker
  → fire ~ MaskedBernoulli(fire_mask)    # 默认唯一可学离散
  → continuous, scan ← rule_worker_action
Env.step（本仓不调用）
  → 非法 fire：116/118 只抑制；E4/A0′ 才允许小罚
```

| 输出 | 116 默认（`apply_nez=false`，`apply_soft_lar=false`） | E4 / A0′ |
|---|---|---|
| `fire_mask` | **原样传递**环境 mask。环境为假时这里不能改成真 | 与 NEZ/Rmax、可选 soft-LAR **求交**。只收紧，不放宽 |
| `launch_gate` | 环境 mask 真 → `ADVISORY`；假 → `BLOCK` | 交后仍真 → `READY` |
| `gate_reason` | 记录 soft-LAR 是否与环境 mask 不一致，**不改变 mask** | 写上 `nez_intersection` / `soft_lar_intersection` |
| 教令 | S5-T：`fire_range_m=35000`、`quality_min=0.3`、`bearing_max_deg=30`、`defend=beam_always`。这些阈值留在环境 / A0 里 | 新 split，不回写 116 |

113 soft-LAR 是 `SoftLarFeatures` 输入，不是第二套 `fire_mask`。Worker 与文档读的是同一次 `resolve_fire_mask` 的结果。

A0 的 `_launch_advisory` 仍在 WSL 环境内。本服务在默认参数下不重算 35 km 启发式，所以 **不会把 A0 掩码改弱或改强**。

## 5. 运动学（锁文档，不改 plant）

保持现栈：决策周期约 1 s（200 tick × 0.005 s），蓝方指令零阶保持，autopilot 跟踪，飞机 integrate。RL 观测用 xyzhv（或距离/方位）、相对几何、`track_quality`、`launch_gate` / `fire_mask`。不含 α、β、舵面、四元数。

编队与警戒（`alert_formation_hold`、BEAM）留在规则宏，不进动作维。E2 雷达、E3 数据链、E4 武器默认都不改 integrate。IdealBlue D0 的 116 评测不改运动学。

本仓没有 plant 代码，这一节只是契约，没有对应的仿真改动。

## 6. 奖励阶段

`raw_team_reward` 在 116 与 118 **不改**。116 的终局团队奖仍由 `per_agent_credit.flush_open_trajectories` 挂到该机最后一步 POSITION/FIRE；规则步不进 GAE。本 PR 不改这个函数的挂步行为。

| 阶段 | `raw_team_reward` | 信用 / 选模 | 非法 fire | 本 PR |
|---|---|---|---|---|
| **116** | 冻结。终局稀疏 + 挂末步 POSITION/FIRE | 已有 per-agent GAE | 抑制 + 计数，**无塑形** | 只归档，不重训 |
| **118a** | 不改 | 选模资格：`observe_mid_eval` 的 `min_win_for_best`。钩子块 `mid_eval_first=998064`，`mid_eval_every=20000`，`mid_eval_cases=16`，`min_win_for_best=0.125`。`selection_118a.enabled` 默认 **false** | 不改 | 只提供 `selection_118a.decide`，不跑局 |
| **118b** | 不改 | `credit_rewrite: launch_step` 或 `lc_mscr`，改写副本字段 `tilde_r`。`reward` 不动 | 不改 | 骨架可调用；yaml 默认 `none` |
| **118c** | 不改 | 与 118b 相同；Worker 用本契约或航向消融 | 不改 | 开关已在，不训 |
| **E4 / A0′** | 终局同构的新 split | LaunchGate 与 NEZ 交 | **小罚** `illegal_fire_penalty=-0.01`，且仅 `protocol` 为 `E4` 或 `A0_prime` 并显式 `enabled` | 函数在，默认 0 |
| **119+ / E5** | 新 split 才允许 option 条件轻塑形 | O0/O1；DEFEND 稠密奖 **不进** fire-only Worker | 随新 split | 不实现 Manager PPO |

`launch_step`：命中步把 `alpha * R_hit` 加到发射机最后一次 `phase=FIRE` 且 `fire` 为真的步，并从命中步的 `tilde_r` 扣回。`lc_mscr` 再按 `support_flag * link_mask` 给 POSITION/FIRE 支援步加 `beta` 份额，同样从命中步扣回。SEARCH / DEFEND / ALERT 的 `tilde_r` 保持与 `reward` 相同。

118a 不改写 116 正式选模。Phase-A formal 仍是 `mid_eval_first=997900`、`mid_eval_every=100000`、`min_win_for_best=0`（未传参时的默认）。那是对照，不是本契约的默认训练配置。

## 7. 如何回退 116，以及如何避免误开训练

回退混合密度：

```python
from algorithms.hierarchical.worker_contract import WorkerContract
contract = WorkerContract.legacy_116()
```

不要把 `configs/protocols/worker_macro_v1.yaml` 改成 `legacy_116` 当作新的仓库默认。

不要做的事：

- 不要运行 WSL 上的 `experiments/learned_worker_v0_smoke.py` 或 `learned_worker_v0_phase_a.py`
- 不要把 `DECISION_CAP` 抬过 640000，也不要把它当成 118 的默认预算
- 不要启动协议 117
- 不要重训 seed 11/23，不要改 Phase-A 的 `eval_a0.json` / `eval_a1.json`
- `launch_118a_training()` 与 `refuse_training_launch` 会直接抛错

`compose_hybrid_step` 只组动作。环境 `step`、PPO update、mid-eval 循环都不在本仓。

## 8. 116 Phase-A 归档（对照，非本 PR 结果）

H-W1 **FAIL**。共享 A0 @256：241/256 = **0.941**（`first=998100`）。六路 A1 均值 **−41.6 pp**，6/6 为负（11: −93.0，23: −94.1，37: −32.8，53: −7.8，71: −11.3，89: −10.5）。11/23 记为 weight-broke / 选模失败，不记为场景无解。

这些数字 **没有在本环境重算**。后续 118 要用新协议号，不能把 116 臂洗成 PASS。

## 9. 残留 TODO

1. 把 WSL `option_conditional_dispatch.HYBRID_DISPATCH_ID` 原字符串接回来。
2. `formation_hrl.baselines.rule_worker_action` 用 `set_host_rule_worker` 接上真 A0；在那之前回退路径的宏表只供单测。
3. smoke 的 rollout 循环改为调用 `compose_hybrid_step`，并继续用原来的 `env.step`。本仓没有该循环。
4. E4 在 `resolve_fire_mask(..., apply_nez=True)` 上填 NEZ/Rmax 表；116 调用保持两个 apply 开关为假。
5. 118a/118b 的 runner 另开协议 PR。本仓钩子默认关闭。
6. 119+ Manager 仍是未训练的 12-way scaffold。DEFEND 不进入 option。
