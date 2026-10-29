# aaa

Worker 契约 `worker_macro_v1` 的文档和可导入骨架在本仓。WSL `2v1new` 的仿真树没有挂载。开关、回退和缺口见 `docs/protocols/learned_worker/worker_macro_v1.md`。

默认 `configs/protocols/worker_macro_v1.yaml` 只学习 masked fire。`WorkerPolicy()` 不传契约时仍是 116 联合密度，避免盖住 WSL 上现有的 smoke。新 rollout 走 `assemble_hybrid_action`。拷进真树的步骤见 `docs/protocols/learned_worker/WSL_APPLY.md`。本仓没有训练入口。
