"""Abstract sim-RL trainer interface + shared policy-artifact export.

The trainer abstraction is the only genuinely new code vs. the supervised
training-worker: a trainer turns a claimed sim_rl job (a SimScene = the RL env)
into a gate-consumable policy bundle. ``write_policy_artifacts`` is shared by
every concrete trainer (stub, ppo) so the artifact layout the deployment gate
reads — policy.zip / policy.onnx / vecnormalize.pkl / manifest.json — is defined
exactly once.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SimRlContext:
    """Everything a sim-RL trainer needs to run one job."""

    job_id: str
    scene_id: str | None
    twin_id: str | None
    embodiment_tag: str
    scene_file: Path | None  # local twin MJCF, or None → bundled empty scene
    hyperparameters: dict[str, Any]
    device: str  # "mps" | "cuda" | "cpu"
    work_dir: Path
    total_timesteps: int
    n_envs: int
    # Absolute sim_evaluator dir the worker put on sys.path — passed explicitly so
    # SubprocVecEnv workers (spawn/forkserver) can re-inject it without scanning.
    sim_evaluator_path: str


@dataclass
class ProgressEvent:
    """One sim-RL training progress snapshot."""

    step: int
    total_steps: int
    mean_reward: float
    learning_rate: float = 0.0
    success_rate: float | None = None


@dataclass
class TrainerResult:
    """What a trainer returns on success."""

    artifact_dir: Path  # dir holding policy.zip / policy.onnx / vecnormalize.pkl / manifest.json
    final_metrics: dict[str, Any]
    primary_artifact: str = "policy.zip"  # the file used as ModelVersion.artifactUri


class CancelledError(Exception):
    """Raised when the server requests a cancel mid-train (do NOT POST /failed)."""


# Returns True to keep training, False to stop early (heartbeat/progress cancel).
ProgressCallback = Callable[[ProgressEvent], bool]


class BaseSimRlTrainer(ABC):
    """All sim-RL trainers implement this interface."""

    @abstractmethod
    def train(self, ctx: SimRlContext, on_progress: ProgressCallback) -> TrainerResult:
        """Train a navigation policy and return the artifact bundle."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared artifact export
# ---------------------------------------------------------------------------
def export_policy_onnx(model, out_path: Path, obs_dim: int) -> None:
    """Export an SB3 policy to a deterministic action-only onnx graph.

    Input  'obs'    float32 [batch, obs_dim]  (already VecNormalize-normalized)
    Output 'action' float32 [batch, action_dim]
    Mirrors what ``policy_backend.PolicyBackend`` runs at gate time, so a fixed
    normalized observation yields the same action at train and eval.
    """
    import torch as th

    class _OnnxablePolicy(th.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):  # deterministic action only
            return self.policy._predict(obs, deterministic=True)

    policy = model.policy
    was_training = policy.training
    policy.to("cpu").eval()
    wrapper = _OnnxablePolicy(policy)
    dummy = th.zeros(1, obs_dim, dtype=th.float32)
    th.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        # Stable TorchScript exporter — the dynamo path needs onnxscript and is
        # overkill for this small MLP policy graph.
        dynamo=False,
    )
    if was_training:
        policy.train()


def write_policy_artifacts(
    model,
    out_dir: Path,
    ctx: SimRlContext,
    *,
    vecnorm=None,
    trainer_name: str = "ppo",
    obs_dim: int | None = None,
) -> dict[str, Path]:
    """Persist policy.zip + policy.onnx + vecnormalize.pkl + manifest.json.

    Returns {name: path}. The manifest's ``obs_norm`` block carries the frozen
    VecNormalize mean/var so the torch-free gate reproduces the normalization.
    """
    # nav_wrappers lives in the sibling sim_evaluator (on sys.path via Config).
    from envs.nav_wrappers import ACTION_DIM, NAV_OBS_DIM, OBS_LAYOUT

    obs_dim = obs_dim or NAV_OBS_DIM
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "policy.zip"
    model.save(str(zip_path))

    onnx_path = out_dir / "policy.onnx"
    export_policy_onnx(model, onnx_path, obs_dim)

    obs_norm = None
    if vecnorm is not None:
        vec_path = out_dir / "vecnormalize.pkl"
        vecnorm.save(str(vec_path))
        rms = vecnorm.obs_rms
        obs_norm = {
            "mean": rms.mean.astype(float).tolist(),
            "var": rms.var.astype(float).tolist(),
            "clip": float(getattr(vecnorm, "clip_obs", 10.0)),
            "epsilon": float(getattr(vecnorm, "epsilon", 1e-8)),
        }

    manifest = {
        "kind": "sim_rl",
        "trainer": trainer_name,
        "embodimentTag": ctx.embodiment_tag,
        "obs_layout": OBS_LAYOUT,
        "action_dim": ACTION_DIM,
        "sceneId": ctx.scene_id,
        "twinId": ctx.twin_id,
        "vecnorm": vecnorm is not None,
        "obs_norm": obs_norm,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info("Wrote policy artifacts to %s (%s)", out_dir, trainer_name)
    return {
        "policy.zip": zip_path,
        "policy.onnx": onnx_path,
        "manifest.json": manifest_path,
        **({"vecnormalize.pkl": out_dir / "vecnormalize.pkl"} if vecnorm is not None else {}),
    }
