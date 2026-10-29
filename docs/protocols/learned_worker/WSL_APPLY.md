# 把 worker_macro_v1 拷进 WSL `2v1new`

真树路径：`/mnt/d/vmShare/2v1new_backup/2v1new`。本云仓没有挂载它。下面的拷贝清单按 drop-in `APPLY.md` 的边界写：**不要覆盖**已经在磁盘上的 `option_conditional_dispatch.py` 和 A0 的 `formation_hrl/baselines.py`。

不要运行 smoke 的 `main`，不要跑 Phase-A，不要把 `DECISION_CAP` 抬过 640000，不要启动协议 117。

## 拷贝

在 aaa 仓根目录执行。脚本会先备份要替换的两个旧文件。

```bash
DEST=/mnt/d/vmShare/2v1new_backup/2v1new
SRC="$(pwd)"
stamp=$(date +%Y%m%d%H%M%S)

# 整文件替换：未传 contract 的 WorkerPolicy 仍是 116 联合密度。
for rel in \
  algorithms/hierarchical/policy.py \
  algorithms/hierarchical/per_agent_credit.py
do
  cp -v "$DEST/$rel" "$DEST/$rel.bak_pre_macro_v1_$stamp"
  cp -v "$SRC/$rel" "$DEST/$rel"
done

# 新增。这些路径在真树上原本没有。
for rel in \
  algorithms/hierarchical/worker_contract.py \
  algorithms/hierarchical/launch_gate.py \
  algorithms/hierarchical/contract_merge.py \
  algorithms/hierarchical/hybrid_rollout.py \
  algorithms/hierarchical/credit_118b.py \
  algorithms/hierarchical/selection_118a.py \
  algorithms/hierarchical/reward_phases.py \
  algorithms/hierarchical/training_guard.py \
  algorithms/mappo/distributions.py \
  formation_hrl/rule_worker_hook.py \
  experiments/hybrid_contract_step.py \
  configs/protocols/worker_macro_v1.yaml \
  docs/protocols/learned_worker/worker_macro_v1.md \
  docs/protocols/learned_worker/WSL_APPLY.md \
  tests/test_worker_macro_v1.py
do
  mkdir -p "$DEST/$(dirname "$rel")"
  cp -v "$SRC/$rel" "$DEST/$rel"
done
```

`algorithms/mappo/distributions.py` 只在真树缺少同名实现时拷贝。若真树已有 `MaskedBernoulli` / `SquashedDiagonalGaussian`，保留真树文件，不要覆盖。

明确不要拷贝：

| 文件 | 原因 |
|---|---|
| `algorithms/hierarchical/option_conditional_dispatch.py` | 真树原件。drop-in 未带 `HYBRID_DISPATCH_ID` 原字符串。本仓这份是占位替身 |
| `formation_hrl/baselines.py` | A0 `rule_worker_action`。本仓副本是测试宏表，`WSL_OVERWRITE_FORBIDDEN = True` |
| `experiments/learned_worker_v0_smoke.py` | 不整文件替换。只改下面两处调用 |
| `experiments/learned_worker_v0_phase_a.py` | 640k 训练入口，不动 |

## HYBRID_DISPATCH_ID

占位与替换点：

- 本仓替身：`algorithms/hierarchical/option_conditional_dispatch.py` 里的 `HYBRID_DISPATCH_ID = "option_conditional_hybrid_v1"`，`HYBRID_DISPATCH_ID_SOURCE = "placeholder_drop_in_omitted_wsl_module"`。
- 真树：同路径的原赋值。smoke 报告字段 `dispatch_id` 继续从**真模块**导入，不要改成占位字符串。

拷贝完成后在 WSL 打印原值（不要用本仓替身覆盖后再打印）：

```bash
cd /mnt/d/vmShare/2v1new_backup/2v1new
python -c "from algorithms.hierarchical.option_conditional_dispatch import HYBRID_DISPATCH_ID; print(repr(HYBRID_DISPATCH_ID))"
```

`experiments.hybrid_contract_step.dispatch_identity()` 在真模块上没有 `HYBRID_DISPATCH_ID_SOURCE` 时把来源记为 `wsl_module`。

## 规则 Worker

混合路径走 `formation_hrl.rule_worker_hook.rule_worker_action`。没调用 `set_host_rule_worker` 时，它调用 `formation_hrl.baselines.rule_worker_action` 并原样返回那个元组，不改 fire、不改连续、不改 scan。

`a0_rule_worker_action` 忽略 host 注入。`evaluate_a0_rule` 必须继续直接调用 `baselines.rule_worker_action`，不要改成 hook。

## smoke 最小 diff

`env.step` 仍留在 WSL smoke 里，紧挨在非法 fire 抑制之后。下面替换的是组动作的那一段。`train_ppo_hybrid` 与 `evaluate_hybrid` 各一处；`evaluate_a0_rule` 不动。

`train_ppo_hybrid` 里原来的：

```python
rule_c, rule_f, rule_s = rule_worker_action(env)
obs = torch.as_tensor(snap.worker_observation, dtype=torch.float32)
critic = torch.as_tensor(snap.critic_state, dtype=torch.float32)
fire_mask = torch.as_tensor(snap.fire_mask, dtype=torch.bool)
...
action = policy.act(obs.to(device), fire_mask.to(device), generator=generator)
...
continuous, fire, scan, learned_mask = merge_hybrid_actions(...)
```

改为：

```python
from experiments.hybrid_contract_step import assemble_hybrid_action

assembled = assemble_hybrid_action(
    env, snap, policy, generator=generator, deterministic=False,
)
obs = assembled.observation
fire_mask = assembled.fire_mask
action = assembled.action
continuous = assembled.continuous
fire = assembled.fire
scan = assembled.scan
learned_mask = assembled.learned_mask
critic = torch.as_tensor(snap.critic_state, dtype=torch.float32)
```

`evaluate_hybrid` 同样替换，但 `deterministic=True`。后面的 `assert_no_search_leak`、非法 fire 计数和 `env.step(continuous, fire, scan)` 保持原样。

`assemble_hybrid_action` 会把 `policy.contract` 设成 yaml（默认 `macro_v1`），这样同一次运行里的 `evaluate_actions` / `hybrid_kl` 和采样密度一致。回退 116 时显式传入，不要改 yaml 默认值：

```python
from algorithms.hierarchical.worker_contract import WorkerContract
assembled = assemble_hybrid_action(
    env, snap, policy, generator=generator,
    deterministic=False, contract=WorkerContract.legacy_116(),
)
```

这次拷贝本身不要启动 `python -m experiments.learned_worker_v0_smoke`。

## WSL 验收

```bash
cd /mnt/d/vmShare/2v1new_backup/2v1new
python -m pytest tests/test_worker_macro_v1.py -q
python -m pytest experiments/test_phase_a_credit.py tests/test_per_agent_peak_gate_offline.py -q
python -c "from algorithms.hierarchical.option_conditional_dispatch import HYBRID_DISPATCH_ID; print(repr(HYBRID_DISPATCH_ID))"
python -c "from formation_hrl.baselines import rule_worker_action; assert not getattr(rule_worker_action, 'is_contract_fallback', False)"
```

第一条是契约：macro 只学 fire，`WorkerPolicy()` 默认仍是 116 联合密度，`legacy_116` 可回退，组动作不调用 `env.step`。第二条是原来的 116 信用/选模测试，确认挂步和 `ALERT` 未知相位没被改掉。后两条确认没有用占位 dispatcher 或测试宏表盖住真树。
