"""Stub sim-RL trainer — proves the whole vertical slice with zero RL solved.

It rolls out the bundled ``g1_empty_scene.xml`` for a few ticks with a zero
policy (so the env + wrappers + progress callback are exercised end-to-end), then
constructs an untrained SB3 PPO model and writes a *loadable* policy.zip +
policy.onnx + manifest.json. Phase 1 uses this to drive a sim_rl job
pending→running→completed with a real artifact before any real RL exists
(TASK-172.C Phase 1).
"""

from __future__ import annotations

import logging
from time import monotonic

import numpy as np

from .base import (
    BaseSimRlTrainer,
    CancelledError,
    ProgressCallback,
    ProgressEvent,
    SimRlContext,
    TrainerResult,
    write_policy_artifacts,
)

logger = logging.getLogger(__name__)

_STUB_TICKS = 20


class StubRlTrainer(BaseSimRlTrainer):
    """Zero-policy rollout + loadable artifact export (no learning)."""

    def train(self, ctx: SimRlContext, on_progress: ProgressCallback) -> TrainerResult:
        # Imports are deferred so the module is importable without the ML stack;
        # sim_evaluator must already be on sys.path (Config.ensure_sim_evaluator_on_path).
        from envs.nav_wrappers import ACTION_DIM, make_nav_env
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        logger.info(
            "[StubRL] job=%s device=%s — zero-policy rollout on bundled empty scene",
            ctx.job_id, ctx.device,
        )

        started = monotonic()

        # 1) Zero-policy rollout (always the bundled empty scene, per design).
        env = make_nav_env(scene_path=None, obs_mode="state", max_steps=_STUB_TICKS,
                           shaped=True, domain_rand=False)
        rewards: list[float] = []
        try:
            env.reset(seed=0)
            for tick in range(1, _STUB_TICKS + 1):
                _obs, reward, term, trunc, _info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
                rewards.append(float(reward))
                if not on_progress(
                    ProgressEvent(
                        step=tick,
                        total_steps=_STUB_TICKS,
                        mean_reward=float(np.mean(rewards)),
                    )
                ):
                    raise CancelledError(f"cancelled at tick {tick}")
                if term or trunc:
                    env.reset()
        finally:
            env.close()

        # 2) Construct an untrained PPO model and export a loadable artifact bundle.
        venv = DummyVecEnv([
            lambda: make_nav_env(scene_path=None, obs_mode="state", max_steps=_STUB_TICKS,
                                 shaped=True, domain_rand=False)
        ])
        try:
            model = PPO("MlpPolicy", venv, device=ctx.device, n_steps=64, verbose=0)
            artifact_dir = ctx.work_dir / "artifacts"
            write_policy_artifacts(model, artifact_dir, ctx, vecnorm=None, trainer_name="stub")
        finally:
            venv.close()

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        logger.info("[StubRL] done — mean_reward=%.3f, artifacts in %s", mean_reward, artifact_dir)
        return TrainerResult(
            artifact_dir=artifact_dir,
            final_metrics={
                "meanReward": round(mean_reward, 4),
                "successRate": 0.0,
                "totalTimesteps": 0,
                "trainingTimeSeconds": round(monotonic() - started, 1),
                "trainer": "stub",
            },
        )
