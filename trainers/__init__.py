"""Pluggable sim-RL trainer backends for NeoDEM sim_rl jobs (TASK-172.C)."""

from .base import (
    BaseSimRlTrainer,
    CancelledError,
    ProgressCallback,
    ProgressEvent,
    SimRlContext,
    TrainerResult,
)


def pick_trainer(kind: str) -> BaseSimRlTrainer:
    """Resolve a trainer by kind. Heavy backends import lazily so the stub path
    never requires torch/stable-baselines3 at import time."""
    kind = (kind or "ppo").lower()
    if kind == "stub":
        from .stub_rl import StubRlTrainer

        return StubRlTrainer()
    if kind == "ppo":
        from .ppo_nav import PpoNavTrainer

        return PpoNavTrainer()
    if kind == "mjx":
        from .mjx_ppo import MjxPpoTrainer

        return MjxPpoTrainer()
    raise ValueError(f"Unknown trainer kind: {kind!r} (expected 'stub' | 'ppo' | 'mjx')")


__all__ = [
    "BaseSimRlTrainer",
    "CancelledError",
    "ProgressCallback",
    "ProgressEvent",
    "SimRlContext",
    "TrainerResult",
    "pick_trainer",
]
