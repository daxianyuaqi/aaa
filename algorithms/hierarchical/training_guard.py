"""Hard stop for any training entry point added next to this contract.

The contract PR wires switches. It does not run 116, 118, 640k, or protocol 117.
"""

from __future__ import annotations

TRAINING_LAUNCH_FORBIDDEN = True
PROTOCOL_117_STARTED = False
DECISION_CAP_640K_UNCHANGED = 640_000


def refuse_training_launch(caller: str) -> None:
    raise RuntimeError(
        f"{caller} must not start training from the worker_macro_v1 contract tree. "
        "116 Phase-A stays archived. Do not raise the 640000 cap and do not start "
        "protocol 117."
    )
