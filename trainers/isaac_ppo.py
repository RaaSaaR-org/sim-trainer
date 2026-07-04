"""Isaac Lab GPU RL locomotion trainer (TASK-172.C — the real Isaac path).

Replaces the deferred ``mjx_ppo.py`` slot with a real trainer that learns a
Unitree G1 *walking gait* with Isaac Lab (Isaac Sim / PhysX) + ``rsl_rl`` PPO over
thousands of parallel GPU envs, then exports the same gate-consumable bundle every
other trainer produces — ``policy.onnx`` + ``policy.pt`` + ``manifest.json`` — with
the manifest carrying the locomotion obs/control contract so the MuJoCo sim-to-sim
gate (``sim_evaluator`` ``locomotion_wrappers`` / ``g1_locomotion_env``) can score it.

Design constraints that shape this file:
  * **Mac-importable, GPU-runnable.** The module top imports only ``.base`` +
    stdlib, so ``import trainers.isaac_ppo`` works on the Mac (no isaac/omni/torch
    at import). Every ``isaaclab``/``omni``/``rsl_rl`` import lives inside a method,
    behind ``_require_isaac`` — which raises an actionable error off a CUDA host.
  * **Isaac-API churn is isolated.** Isaac Lab's module names, task ids, wrapper
    locations and runner hooks move between releases. Each fragile surface is a
    small overridable method (``_launch_app``, ``_load_cfgs``, ``_build_env_and_runner``,
    ``_make_runner``) so a version bump touches one place — and so the Mac unit test
    can replace ``_build_env_and_runner`` with a fake that returns a real tiny torch
    actor, exercising the export + gate-loadability for real without any GPU.
  * **The contract is the single source of truth.** ``LOCO_OBS_DIM`` /
    ``LOCO_OBS_LAYOUT`` / ``DEFAULT_CONTROL`` are imported from the sibling
    ``sim_evaluator`` (on sys.path via ``Config.ensure_sim_evaluator_on_path``); a
    build-time assert (``obs_dim == LOCO_OBS_DIM``) fails loudly if Isaac's task obs
    ever drifts from the gate's env.

Run (Linux/CUDA host with Isaac Sim + Isaac Lab):
    TRAINER=isaac TRAINING_DEVICE=cuda N_ENVS=4096 \
    ISAAC_TASK=Isaac-Velocity-Flat-G1-v0 MAX_ITERATIONS=1500 uv run python worker.py
"""

from __future__ import annotations

import logging
import os
import sys
from math import ceil
from time import monotonic

from .base import (
    BaseSimRlTrainer,
    CancelledError,
    ProgressCallback,
    ProgressEvent,
    SimRlContext,
    TrainerResult,
    write_rsl_rl_artifacts,
)

logger = logging.getLogger(__name__)

# The Isaac Sim + Isaac Lab + rsl-rl-lib triple this trainer's adapters were
# written against. Pinned in the README; a version mismatch logs a warning (the
# adapter methods are where a bump is absorbed). Kept as a string, not enforced.
_TESTED_ISAAC_LAB = "v2.1.0"

# Flat (blind) G1 velocity task — the Rough variant's height-scan obs cannot be
# reproduced in the flat MuJoCo gate. Overridable via ISAAC_TASK / hyperparameters.
_DEFAULT_TASK = "Isaac-Velocity-Flat-G1-v0"
_DEFAULT_STEPS_PER_ENV = 24  # rsl_rl num_steps_per_env (velocity task default)


class _CancelSignal(Exception):
    """Internal: raised inside the runner's progress hook to break out of learn()."""


class IsaacPpoTrainer(BaseSimRlTrainer):
    """Real Isaac Lab + rsl_rl PPO G1 locomotion trainer."""

    def train(self, ctx: SimRlContext, on_progress: ProgressCallback) -> TrainerResult:
        self._require_isaac(ctx)

        # Contract constants live in the sibling sim_evaluator (on sys.path via
        # Config.ensure_sim_evaluator_on_path). Imported here (not at module top)
        # because the path is only guaranteed at run time.
        from envs.locomotion_wrappers import (
            ACTION_DIM,
            LOCO_OBS_DIM,
            LOCO_OBS_LAYOUT,
            build_control_manifest,
        )

        task_id, n_envs, max_iters, total_steps, steps_per_env = self._resolve_params(ctx)
        logger.info(
            "[Isaac] job=%s task=%s n_envs=%d max_iters=%d steps/env=%d device=%s",
            ctx.job_id, task_id, n_envs, max_iters, steps_per_env, ctx.device,
        )
        started = monotonic()

        # Progress/cancel bridge shared by the real runner hook and the unit-test
        # fake: raising _CancelSignal breaks learn(); train() maps it to the clean
        # CancelledError the worker skips /failed for.
        state = {"last_reward": 0.0, "last_step": 0}

        def bridge(step: int, mean_reward: float, lr: float = 0.0) -> None:
            state["last_reward"] = float(mean_reward)
            state["last_step"] = int(step)
            cont = on_progress(
                ProgressEvent(
                    step=int(step), total_steps=total_steps,
                    mean_reward=float(mean_reward), learning_rate=lr,
                )
            )
            if not cont:
                raise _CancelSignal()

        app = self._launch_app()
        try:
            runner, env = self._build_env_and_runner(
                ctx, task_id, n_envs, max_iters, total_steps, steps_per_env, bridge, app
            )

            # Contract check BEFORE learn(): env.observation_space is available the
            # moment the env is built, so an obs drift fails in seconds instead of
            # after a multi-hour GPU run is trained and then discarded.
            obs_dim = self._policy_obs_dim(env)
            if obs_dim != LOCO_OBS_DIM:
                raise RuntimeError(
                    f"Isaac policy obs dim {obs_dim} != gate contract {LOCO_OBS_DIM}. "
                    f"Reconcile _load_cfgs (drop base_lin_vel / actuated joint set) or "
                    f"the joint_order permutation before trusting this policy."
                )

            try:
                runner.learn(num_learning_iterations=max_iters, init_at_random_ep_len=True)
            except _CancelSignal:
                raise CancelledError(
                    f"cancelled during Isaac learn() at step {state['last_step']}"
                )

            actor_critic = runner.alg.actor_critic
            normalizer = getattr(runner, "obs_normalizer", None)

            control = build_control_manifest(**self._control_overrides(ctx))
            self._warn_if_placeholder_control(control)
            artifact_dir = ctx.work_dir / "artifacts"
            write_rsl_rl_artifacts(
                actor_critic, artifact_dir, ctx,
                normalizer=normalizer,
                obs_dim=obs_dim,
                obs_layout=LOCO_OBS_LAYOUT,
                action_dim=ACTION_DIM,
                control=control,
                trainer_name="isaac",
                env_kind="locomotion",
            )
        finally:
            self._shutdown_app(app)

        logger.info(
            "[Isaac] done — mean_reward=%.3f iters=%d", state["last_reward"], max_iters
        )
        return TrainerResult(
            artifact_dir=artifact_dir,
            final_metrics={
                "meanReward": round(state["last_reward"], 4),
                "totalTimesteps": state["last_step"] or total_steps,
                "maxIterations": max_iters,
                "trainingTimeSeconds": round(monotonic() - started, 1),
                "trainer": "isaac",
                "isaacTask": task_id,
            },
            # No SB3 .zip — the framework-agnostic onnx is the gate-consumable
            # primary artifact (worker._upload_artifacts only needs primary in uris).
            primary_artifact="policy.onnx",
        )

    # ----------------------------------------------------------------- guards
    def _require_isaac(self, ctx: SimRlContext) -> None:
        """Fail fast with an actionable message off a CUDA Isaac host.

        Mirrors the old mjx_ppo reserved-slot behaviour but is conditional on real
        availability, so on a GPU host with Isaac Lab installed it proceeds.
        """
        try:
            import isaaclab  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Isaac Lab GPU training requires a Linux CUDA host with Isaac Sim + "
                "Isaac Lab installed. On Mac use the nav path (TRAINER=ppo) or the "
                "vertical slice (TRAINER_STUB=true)."
            ) from e
        if ctx.device != "cuda":
            raise RuntimeError(
                f"TRAINER=isaac requires TRAINING_DEVICE=cuda (got {ctx.device!r})."
            )
        version = getattr(sys.modules.get("isaaclab"), "__version__", None)
        if version and version != _TESTED_ISAAC_LAB:
            logger.warning(
                "Isaac Lab %s != tested %s — the adapter methods may need updating",
                version, _TESTED_ISAAC_LAB,
            )

    # ------------------------------------------------------------- parameters
    def _resolve_params(self, ctx: SimRlContext) -> tuple[str, int, int, int, int]:
        """Read Isaac knobs from job hyperparameters with env fallback (mirrors
        ppo_nav's hp.get pattern). rsl_rl counts *iterations*, so derive
        max_iterations from total_timesteps unless one is given explicitly.

        Returns ``(task_id, n_envs, max_iters, total_steps, steps_per_env)``.
        ``steps_per_env`` is fed into the rsl_rl agent cfg (``_load_cfgs``) so the
        rollout length the runner actually uses matches the budget math here.
        """
        hp = ctx.hyperparameters or {}
        task_id = hp.get("isaac_task") or os.environ.get("ISAAC_TASK", _DEFAULT_TASK)
        n_envs = int(hp.get("n_envs") or ctx.n_envs)
        steps_per_env = int(
            hp.get("num_steps_per_env")
            or os.environ.get("NUM_STEPS_PER_ENV", _DEFAULT_STEPS_PER_ENV)
        )
        total_steps = int(hp.get("total_timesteps") or ctx.total_timesteps)
        max_iters = int(hp.get("max_iterations") or os.environ.get("MAX_ITERATIONS", 0) or 0)
        if max_iters <= 0:
            max_iters = max(1, ceil(total_steps / max(1, steps_per_env * n_envs)))
        return task_id, n_envs, max_iters, total_steps, steps_per_env

    def _control_overrides(self, ctx: SimRlContext) -> dict:
        """Per-job overrides for the manifest control block (None => keep default).

        HOST WIRING (required for a trustworthy score): Isaac's REAL default stance,
        actuator stiffness/damping, and DOF ordering must be sourced from the env
        cfg in ``_load_cfgs`` and fed into ``build_control_manifest`` — otherwise the
        manifest ships the placeholder ``DEFAULT_CONTROL`` and the MuJoCo gate
        rebuilds its env with the wrong offset/gains/order. ``_warn_if_placeholder_control``
        logs a loud warning whenever that reconciliation hasn't happened. v1 forwards
        only the job-level knobs a caller might set; see the README host-correction
        checklist."""
        hp = ctx.hyperparameters or {}
        return {
            "command": hp.get("command"),
            "action_scale": hp.get("action_scale"),
            "joint_order": hp.get("joint_order"),
        }

    def _warn_if_placeholder_control(self, control: dict) -> None:
        """Loudly flag when the manifest control block is still the placeholder
        contract — i.e. the host-side reconciliation (Isaac real stance / gains /
        DOF order) has NOT happened, so the gate's simSuccessRate is not trustworthy.
        """
        from envs.locomotion_wrappers import DEFAULT_JOINT_POS

        issues = []
        if control.get("joint_order") is None:
            issues.append(
                "joint_order is identity (Isaac DOF order likely != canonical JOINT_NAMES)"
            )
        if control.get("pd_gains") is None:
            issues.append("pd_gains fall back to the MJCF kp=150/kv=5 (not Isaac's gains)")
        try:
            import numpy as np

            if np.allclose(np.asarray(control.get("default_joint_pos")), DEFAULT_JOINT_POS):
                issues.append("default_joint_pos is the placeholder stance (not Isaac init_state)")
        except Exception:  # noqa: BLE001
            pass
        if issues:
            logger.warning(
                "[Isaac] manifest.control uses PLACEHOLDER values — the MuJoCo gate "
                "score is NOT trustworthy until reconciled on the host: %s. See the "
                "README 'GPU host deploy' host-correction checklist.",
                "; ".join(issues),
            )

    def _policy_obs_dim(self, env) -> int:
        """Policy-group observation dimension. Read, never hardcoded — the build
        assert in train() compares it to the gate contract."""
        space = env.observation_space
        policy = space["policy"]  # gym Dict and plain dict both index by key
        return int(policy.shape[-1])

    # ------------------------------------- Isaac/omni/rsl_rl seams (host-only) --
    # Everything below imports isaac/omni/rsl_rl and only runs on a CUDA Isaac
    # host. The Mac unit test replaces _build_env_and_runner (and no-ops the app
    # launch) so these are never executed without a GPU. Keep each method tiny so a
    # version bump is a one-method edit.

    def _launch_app(self):
        """Start (once) the headless Isaac Sim app and reuse it across jobs.

        Isaac Sim's ``SimulationApp`` is a per-process singleton that CANNOT be
        re-created after it is closed, but ``worker.main`` builds one trainer and
        reuses it across the poll loop. So the app is launched lazily on the first
        job and memoized on the instance; subsequent jobs reuse it. MUST run before
        importing any ``isaaclab_tasks`` / env-cfg module (those import ``omni.*``
        and need a running app).
        """
        app = getattr(self, "_app", None)
        if app is not None:
            return app
        from isaaclab.app import AppLauncher

        self._app = AppLauncher(headless=True).app
        return self._app

    def _shutdown_app(self, app) -> None:
        # Deliberately a no-op: the Isaac SimulationApp is a per-process singleton
        # that cannot be re-instantiated after close(), so it is kept alive for the
        # next job in the poll loop and torn down only when the worker process exits.
        return

    def _build_env_and_runner(
        self, ctx, task_id, n_envs, max_iters, total_steps, steps_per_env, bridge, app
    ):
        """Build the RL env + rsl_rl runner. The single seam the Mac test replaces."""
        import gymnasium

        import isaaclab_tasks  # noqa: F401  (registers Isaac-* tasks; needs the app)
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        env_cfg, agent_cfg = self._load_cfgs(task_id, n_envs, ctx.device, steps_per_env)
        env = gymnasium.make(task_id, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env)
        runner = self._make_runner(
            OnPolicyRunner, env, agent_cfg, ctx, bridge, total_steps, n_envs
        )
        return runner, env

    def _load_cfgs(self, task_id, n_envs, device, steps_per_env):
        """Resolve the env cfg + rsl_rl agent cfg for the task.

        Isolated because task-id strings, the cfg registry entry-points, and the
        obs-group surgery below all move between Isaac Lab releases. The base_lin_vel
        drop makes the policy obs match the 96-dim gate contract (blind transfer).

        HOST WIRING TODO: this is the place to read Isaac's REAL
        ``init_state.joint_pos`` (default stance), actuator stiffness/damping, and
        the articulation joint-name→index map out of ``env_cfg`` and feed them into
        ``build_control_manifest`` (via ``_control_overrides``) so the gate's control
        block is not the placeholder ``DEFAULT_CONTROL``. Until then
        ``_warn_if_placeholder_control`` flags every run.
        """
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

        env_cfg = parse_env_cfg(task_id, device=device, num_envs=n_envs)
        # Drop base_lin_vel from the *policy* obs group (stays in the critic group)
        # so the exported policy obs is the 96-dim blind-transfer contract.
        try:
            env_cfg.observations.policy.base_lin_vel = None
            env_cfg.observations.policy.enable_corruption = True
        except Exception:  # noqa: BLE001
            logger.warning("Could not drop base_lin_vel from policy obs — verify cfg")
        agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
        # Make the runner's actual rollout length match the budget math in
        # _resolve_params (which derived max_iters from steps_per_env). Without this
        # the registry default silently governs and the timestep budget is wrong.
        try:
            agent_cfg.num_steps_per_env = steps_per_env
        except Exception:  # noqa: BLE001
            logger.warning("Could not set agent_cfg.num_steps_per_env — verify cfg")
        return env_cfg, agent_cfg

    def _make_runner(self, OnPolicyRunner, env, agent_cfg, ctx, bridge, total_steps, n_envs):
        """Subclass OnPolicyRunner to bridge progress/cancel off its per-iteration
        ``log`` hook (the version-stable seam vs. re-implementing the collect loop)."""
        outer_bridge = bridge

        class _ProgressRunner(OnPolicyRunner):
            def log(self, locs, width=80, pad=35):
                try:
                    it = int(locs.get("it", 0))
                    rewbuf = list(locs.get("rewbuffer") or [])
                    mean_r = float(sum(rewbuf) / len(rewbuf)) if rewbuf else 0.0
                    steps_per = int(getattr(self.alg, "num_steps_per_env", 0)) or 1
                    outer_bridge(it * n_envs * steps_per, mean_r)
                except _CancelSignal:
                    raise
                except Exception:  # noqa: BLE001
                    logger.debug("progress log hook raised — ignoring", exc_info=True)
                return super().log(locs, width, pad)

        cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg
        return _ProgressRunner(
            env, cfg, log_dir=str(ctx.work_dir / "rsl_rl"), device=ctx.device
        )
