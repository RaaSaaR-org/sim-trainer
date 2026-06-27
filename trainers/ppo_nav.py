"""Real PPO navigation trainer (TASK-172.C Phase 2).

Trains a G1 *navigation* policy (lean/shuffle toward the goal — NOT walking; see
README) with stable-baselines3 PPO over a ``SubprocVecEnv`` of state-only nav
envs wrapped in ``VecNormalize``, with per-episode domain randomization and the
mandatory alive-bonus reward shaping (``nav_wrappers``). Progress + heartbeat
cancellation are bridged into SB3 via a callback. Output is the same
gate-consumable bundle as the stub.
"""

from __future__ import annotations

import logging
import sys
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


def _make_env_factory(scene_file: str | None, sim_eval_path: str, max_steps: int, seed: int):
    """Return a picklable thunk that builds one nav env in a worker process.

    Re-inserts sim_evaluator on sys.path first so it works under both ``fork``
    and ``spawn`` start methods (spawned workers re-import from scratch).
    """

    def _thunk():
        if sim_eval_path and sim_eval_path not in sys.path:
            sys.path.insert(0, sim_eval_path)
        from envs.nav_wrappers import make_nav_env

        env = make_nav_env(
            scene_path=scene_file,
            obs_mode="state",
            max_steps=max_steps,
            shaped=True,
            domain_rand=True,
        )
        env.reset(seed=seed)
        return env

    return _thunk


class PpoNavTrainer(BaseSimRlTrainer):
    def train(self, ctx: SimRlContext, on_progress: ProgressCallback) -> TrainerResult:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

        hp = ctx.hyperparameters or {}
        total_timesteps = int(hp.get("total_timesteps") or ctx.total_timesteps)
        n_envs = int(hp.get("n_envs") or ctx.n_envs)
        max_steps = int(hp.get("max_steps") or 400)
        lr = float(hp.get("learning_rate") or 3e-4)

        scene_file = self._resolve_scene(ctx)
        used_fallback_scene = ctx.scene_file is not None and scene_file is None
        # The sim_evaluator dir the worker placed on sys.path — passed via the
        # context (not rediscovered) so it re-injects correctly into each
        # (possibly spawned) SubprocVecEnv worker even when SIM_EVALUATOR_PATH is
        # a renamed dir or has a trailing slash. rstrip so the value is stable.
        sim_eval_path = (ctx.sim_evaluator_path or "").rstrip("/")

        logger.info(
            "[PpoNav] job=%s scene=%s n_envs=%d timesteps=%d device=%s",
            ctx.job_id, scene_file or "<bundled empty>", n_envs, total_timesteps, ctx.device,
        )
        started = monotonic()

        env_fns = [
            _make_env_factory(scene_file, sim_eval_path, max_steps, seed=i)
            for i in range(n_envs)
        ]
        venv = SubprocVecEnv(env_fns)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

        outer = self

        class _Sb3ProgressCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.cancelled = False

            def _on_step(self) -> bool:
                return not self.cancelled

            def _on_rollout_end(self) -> None:
                infos = list(self.model.ep_info_buffer or [])
                mean_r = float(np.mean([e["r"] for e in infos])) if infos else 0.0
                cont = on_progress(
                    ProgressEvent(
                        step=int(self.num_timesteps),
                        total_steps=total_timesteps,
                        mean_reward=mean_r,
                        learning_rate=lr,
                    )
                )
                if not cont:
                    logger.info("[PpoNav] cancel requested — stopping learn()")
                    self.cancelled = True

        cb = _Sb3ProgressCallback()
        try:
            model = PPO(
                "MlpPolicy",
                venv,
                device=ctx.device,
                learning_rate=lr,
                n_steps=int(hp.get("n_steps") or 512),
                batch_size=int(hp.get("batch_size") or 256),
                gamma=float(hp.get("gamma") or 0.99),
                verbose=0,
            )
            model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)

            if cb.cancelled:
                raise CancelledError(f"cancelled during learn at {model.num_timesteps} steps")

            artifact_dir = ctx.work_dir / "artifacts"
            write_policy_artifacts(model, artifact_dir, ctx, vecnorm=venv, trainer_name="ppo")
        finally:
            venv.close()

        infos = list(model.ep_info_buffer or [])
        mean_reward = float(np.mean([e["r"] for e in infos])) if infos else 0.0
        logger.info(
            "[PpoNav] done — mean_reward=%.3f (fallback_scene=%s)",
            mean_reward, used_fallback_scene,
        )
        return TrainerResult(
            artifact_dir=artifact_dir,
            final_metrics={
                "meanReward": round(mean_reward, 4),
                "totalTimesteps": int(model.num_timesteps),
                "trainingTimeSeconds": round(monotonic() - started, 1),
                "trainer": "ppo",
                # Surfaced so a twin scene that silently degraded to the bundled
                # empty room (e.g. cross-host meshes absent) is not reported as a
                # twin-derived policy without a trace (TASK-172.C review finding).
                "usedFallbackScene": used_fallback_scene,
            },
        )

    # ----------------------------------------------------------------- helpers
    def _resolve_scene(self, ctx: SimRlContext) -> str | None:
        """Use the twin scene if it loads; otherwise fall back to the bundled
        empty scene (built-ins have no mjcfKey, cross-host meshes may be absent)."""
        if not ctx.scene_file:
            return None
        try:
            from envs.g1_env import G1Env

            probe = G1Env(scene_path=str(ctx.scene_file), max_steps=1, obs_mode="state")
            probe.close()
            return str(ctx.scene_file)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[PpoNav] twin scene %s failed to load (%s) — falling back to bundled scene",
                ctx.scene_file, e,
            )
            return None
