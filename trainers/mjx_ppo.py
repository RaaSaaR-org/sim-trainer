"""MJX/CUDA gait trainer — PLACEHOLDER (TASK-172.C Phase 4, deferred).

Real G1 *locomotion* (a walking gait) cannot be learned model-free on a single,
non-vectorized MuJoCo env on a Mac: legged RL needs thousands of parallel envs on
CUDA/MJX (or rsl_rl). v1 ships navigation only (see README). This module reserves
the slot behind the same ``BaseSimRlTrainer`` interface so a CUDA host can drop in
a real MJX PPO gait trainer without touching the worker, server, or gate.
"""

from __future__ import annotations

from .base import BaseSimRlTrainer, ProgressCallback, SimRlContext, TrainerResult


class MjxPpoTrainer(BaseSimRlTrainer):
    def train(self, ctx: SimRlContext, on_progress: ProgressCallback) -> TrainerResult:
        raise NotImplementedError(
            "MJX/CUDA gait training is deferred to Phase 4 (needs a CUDA host with "
            "MJX/rsl_rl + thousands of parallel envs). v1 ships navigation via "
            "ppo_nav.py. Set TRAINER=ppo or TRAINER_STUB=true."
        )
