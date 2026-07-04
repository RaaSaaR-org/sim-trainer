"""Mac-testable coverage for the Isaac Lab locomotion trainer (no CUDA needed).

Isaac Lab cannot run on the Mac, so the two Isaac/omni/rsl_rl seams
(``_launch_app`` and ``_build_env_and_runner``) are replaced with fakes — the fake
runner exposes a **real tiny torch actor**, so ``export_rsl_rl_onnx`` +
``write_rsl_rl_artifacts`` + gate loadability are exercised for real. What this
locks down:
  * the module imports on the Mac (no isaac/torch at import time);
  * the guard raises an actionable error without Isaac / off a CUDA device;
  * the progress/cancel bridge fires with increasing steps and maps a cancel to
    ``CancelledError``;
  * the exported ``policy.onnx`` loads through the gate's torch-free PolicyBackend
    and the manifest carries the locomotion contract (env / action_dim / obs_norm);
  * the build-time obs-dim assert fires on a contract mismatch.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

# Module import itself is the Mac-importability check (only .base + stdlib at top).
from trainers.base import SimRlContext  # noqa: E402
from trainers.isaac_ppo import IsaacPpoTrainer  # noqa: E402

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

import torch as th  # noqa: E402
import torch.nn as nn  # noqa: E402

from config import _default_sim_evaluator_path  # noqa: E402


# --------------------------------------------------------------------- fakes
class _TinyActor(nn.Module):
    """Stand-in for an rsl_rl ActorCritic: exposes act_inference(obs)->action."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Linear(obs_dim, act_dim)

    def act_inference(self, obs):
        return self.net(obs)


class _FakeAlg:
    def __init__(self, ac):
        self.actor_critic = ac
        self.num_steps_per_env = 24


class _EmpiricalNorm(nn.Module):
    """Stand-in for rsl_rl's EmpiricalNormalization: a real nn.Module that shifts +
    scales the obs, so ``export_rsl_rl_onnx`` genuinely bakes it into the graph."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("running_mean", th.full((dim,), 0.3))
        self.register_buffer("running_var", th.full((dim,), 4.0))

    def forward(self, obs):
        return (obs - self.running_mean) / th.sqrt(self.running_var + 1e-8)


class _FakeRunner:
    def __init__(self, bridge, obs_dim, act_dim, *, n_iters=3, normalizer=None):
        self.alg = _FakeAlg(_TinyActor(obs_dim, act_dim))
        self.obs_normalizer = normalizer  # None => identity path; module => baked in
        self._bridge = bridge
        self._n = n_iters

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        for it in range(1, self._n + 1):
            self._bridge(it * 100, float(it))  # raises _CancelSignal on cancel


class _FakeSpace:
    def __init__(self, dim):
        self.shape = (dim,)


class _FakeEnv:
    def __init__(self, obs_dim):
        self.observation_space = {"policy": _FakeSpace(obs_dim)}


def _obs_dims():
    from envs.locomotion_wrappers import ACTION_DIM, LOCO_OBS_DIM

    return LOCO_OBS_DIM, ACTION_DIM


def _ctx(tmp_path, device="cuda"):
    return SimRlContext(
        job_id="isaac-test", scene_id="scene-1", twin_id="twin-1", embodiment_tag="g1",
        scene_file=None, hyperparameters={"n_envs": 8, "total_timesteps": 4096},
        device=device, work_dir=tmp_path, total_timesteps=4096, n_envs=8,
        sim_evaluator_path=_default_sim_evaluator_path(),
    )


def _inject_isaac(monkeypatch):
    """Make ``import isaaclab`` in _require_isaac succeed with the tested version."""
    import sys

    mod = types.ModuleType("isaaclab")
    mod.__version__ = "v2.1.0"
    monkeypatch.setitem(sys.modules, "isaaclab", mod)


def _patch_seams(monkeypatch, trainer, env_obs_dim=None, normalizer=None):
    loco_dim, act_dim = _obs_dims()
    env_obs_dim = loco_dim if env_obs_dim is None else env_obs_dim

    def fake_build(ctx, task_id, n_envs, max_iters, total_steps, steps_per_env, bridge, app):
        return (
            _FakeRunner(bridge, loco_dim, act_dim, normalizer=normalizer),
            _FakeEnv(env_obs_dim),
        )

    monkeypatch.setattr(trainer, "_launch_app", lambda: object())
    monkeypatch.setattr(trainer, "_shutdown_app", lambda app: None)
    monkeypatch.setattr(trainer, "_build_env_and_runner", fake_build)


# --------------------------------------------------------------------- tests
def test_guard_raises_without_isaac(monkeypatch, tmp_path):
    import sys

    monkeypatch.delitem(sys.modules, "isaaclab", raising=False)
    trainer = IsaacPpoTrainer()
    with pytest.raises(RuntimeError, match="Isaac Lab"):
        trainer.train(_ctx(tmp_path, device="cpu"), lambda ev: True)


def test_guard_requires_cuda_device(monkeypatch, tmp_path):
    _inject_isaac(monkeypatch)
    trainer = IsaacPpoTrainer()
    with pytest.raises(RuntimeError, match="cuda"):
        trainer.train(_ctx(tmp_path, device="cpu"), lambda ev: True)


def test_train_exports_gate_loadable_bundle(monkeypatch, tmp_path):
    from policy_backend import PolicyBackend

    loco_dim, act_dim = _obs_dims()
    _inject_isaac(monkeypatch)
    trainer = IsaacPpoTrainer()
    _patch_seams(monkeypatch, trainer)

    steps: list[int] = []
    result = trainer.train(_ctx(tmp_path), lambda ev: steps.append(ev.step) or True)

    # Progress bridge fired with increasing steps.
    assert steps == sorted(steps) and len(steps) == 3

    ad = result.artifact_dir
    assert result.primary_artifact == "policy.onnx"
    for name in ("policy.onnx", "policy.pt", "manifest.json"):
        assert (ad / name).exists(), f"missing {name}"
    assert result.final_metrics["trainer"] == "isaac"

    # Gate side: the torch-free backend loads the onnx + manifest and predicts a
    # 29-dim action from a raw 96-dim locomotion obs (normalizer baked in => the
    # gate runs the identity path).
    backend = PolicyBackend.from_artifacts(ad / "policy.onnx", ad / "manifest.json")
    action = backend.predict(np.zeros(loco_dim, dtype=np.float32))
    assert action.shape == (act_dim,)

    import json

    manifest = json.loads((ad / "manifest.json").read_text())
    assert manifest["env"] == "locomotion"
    assert manifest["action_dim"] == act_dim
    assert manifest["obs_layout"]["dim"] == loco_dim
    assert manifest["obs_norm"] is None
    assert manifest["control"]["control_hz"] == 50.0


def test_export_rsl_rl_onnx_bakes_normalizer(tmp_path):
    """export_rsl_rl_onnx must compose the normalizer INTO the graph (the headline
    feature): the baked graph on RAW obs equals the un-baked graph on PRE-normalized
    obs, and differs from the un-baked graph on the same raw obs."""
    import onnxruntime as ort

    from trainers.base import export_rsl_rl_onnx

    loco_dim, act_dim = _obs_dims()
    th.manual_seed(0)
    actor = _TinyActor(loco_dim, act_dim).eval()
    norm = _EmpiricalNorm(loco_dim).eval()

    raw_path = tmp_path / "raw.onnx"
    baked_path = tmp_path / "baked.onnx"
    export_rsl_rl_onnx(actor, raw_path, loco_dim, normalizer=None)
    export_rsl_rl_onnx(actor, baked_path, loco_dim, normalizer=norm)

    obs = np.full((1, loco_dim), 1.0, dtype=np.float32)
    s_raw = ort.InferenceSession(str(raw_path), providers=["CPUExecutionProvider"])
    s_baked = ort.InferenceSession(str(baked_path), providers=["CPUExecutionProvider"])
    out_raw = s_raw.run(None, {"obs": obs})[0]
    out_baked = s_baked.run(None, {"obs": obs})[0]
    assert out_baked.shape == (1, act_dim)

    # Baking changed the graph.
    assert not np.allclose(out_raw, out_baked, atol=1e-5)
    # Baked(raw) == Unbaked(normalizer(raw)) — normalization is inside the onnx.
    normed = ((obs - 0.3) / np.sqrt(4.0 + 1e-8)).astype(np.float32)
    np.testing.assert_allclose(
        out_baked, s_raw.run(None, {"obs": normed})[0], rtol=1e-4, atol=1e-4
    )


def test_train_records_obs_norm_debug_when_normalizer_present(monkeypatch, tmp_path):
    """A run whose runner exposes an obs_normalizer keeps obs_norm:null (gate
    identity path, normalizer baked into the onnx) and records the running stats
    under obs_norm_debug; the gate still loads + predicts from raw obs."""
    import json

    from policy_backend import PolicyBackend

    loco_dim, act_dim = _obs_dims()
    _inject_isaac(monkeypatch)
    trainer = IsaacPpoTrainer()
    _patch_seams(monkeypatch, trainer, normalizer=_EmpiricalNorm(loco_dim))

    result = trainer.train(_ctx(tmp_path), lambda ev: True)
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text())
    assert manifest["obs_norm"] is None
    assert manifest["obs_norm_debug"] is not None
    assert manifest["obs_norm_debug"]["mean"][0] == pytest.approx(0.3)

    backend = PolicyBackend.from_artifacts(
        result.artifact_dir / "policy.onnx", result.artifact_dir / "manifest.json"
    )
    assert backend.predict(np.zeros(loco_dim, dtype=np.float32)).shape == (act_dim,)


def test_cancel_maps_to_cancelled_error(monkeypatch, tmp_path):
    from trainers.base import CancelledError

    _inject_isaac(monkeypatch)
    trainer = IsaacPpoTrainer()
    _patch_seams(monkeypatch, trainer)

    with pytest.raises(CancelledError):
        trainer.train(_ctx(tmp_path), lambda ev: False)  # request cancel immediately


def test_obs_dim_mismatch_fails_loudly(monkeypatch, tmp_path):
    loco_dim, _ = _obs_dims()
    _inject_isaac(monkeypatch)
    trainer = IsaacPpoTrainer()
    _patch_seams(monkeypatch, trainer, env_obs_dim=loco_dim + 3)  # drift

    with pytest.raises(RuntimeError, match="gate contract"):
        trainer.train(_ctx(tmp_path), lambda ev: True)
