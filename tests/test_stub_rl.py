"""Phase 1 acceptance: StubRlTrainer drives the vertical slice and writes a
*loadable* SB3 .zip + policy.onnx + manifest.json, and the exported onnx matches
the SB3 policy for a fixed observation (export fidelity / train-eval parity)."""

from __future__ import annotations

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")
pytest.importorskip("onnxruntime")

from config import _default_sim_evaluator_path  # noqa: E402
from trainers.base import SimRlContext  # noqa: E402
from trainers.stub_rl import StubRlTrainer  # noqa: E402


def _ctx(work_dir) -> SimRlContext:
    return SimRlContext(
        job_id="stub-job", scene_id=None, twin_id=None, embodiment_tag="g1",
        scene_file=None, hyperparameters={}, device="cpu", work_dir=work_dir,
        total_timesteps=0, n_envs=1,
        sim_evaluator_path=_default_sim_evaluator_path(),
    )


def test_stub_produces_loadable_artifacts(tmp_path):
    from envs.nav_wrappers import ACTION_DIM, NAV_OBS_DIM
    from stable_baselines3 import PPO
    import onnxruntime as ort

    events = []
    result = StubRlTrainer().train(_ctx(tmp_path), lambda ev: events.append(ev) or True)
    ad = result.artifact_dir

    # Vertical slice exercised the env + progress callback.
    assert len(events) >= 1
    assert (ad / "policy.zip").exists()
    assert (ad / "policy.onnx").exists()
    assert (ad / "manifest.json").exists()

    # Loadable SB3 policy.
    model = PPO.load(str(ad / "policy.zip"), device="cpu")

    # Loadable onnx with the right I/O shape.
    sess = ort.InferenceSession(str(ad / "policy.onnx"), providers=["CPUExecutionProvider"])
    obs = np.zeros((1, NAV_OBS_DIM), dtype=np.float32)
    action = sess.run(None, {"obs": obs})[0]
    assert action.shape == (1, ACTION_DIM)

    # Manifest carries the layout the gate reads.
    man = json.loads((ad / "manifest.json").read_text())
    assert man["kind"] == "sim_rl"
    assert man["trainer"] == "stub"
    assert man["action_dim"] == ACTION_DIM
    assert man["obs_layout"]["dim"] == NAV_OBS_DIM
    assert man["obs_norm"] is None  # stub has no VecNormalize


def test_onnx_matches_sb3_policy(tmp_path):
    """Export fidelity: onnx(obs) == SB3 policy._predict(obs) for a fixed obs.

    This is the train/eval action-parity guarantee the gate relies on."""
    import torch as th
    from envs.nav_wrappers import NAV_OBS_DIM
    from stable_baselines3 import PPO
    import onnxruntime as ort

    result = StubRlTrainer().train(_ctx(tmp_path), lambda ev: True)
    ad = result.artifact_dir

    model = PPO.load(str(ad / "policy.zip"), device="cpu")
    sess = ort.InferenceSession(str(ad / "policy.onnx"), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((1, NAV_OBS_DIM)).astype(np.float32)
    onnx_action = sess.run(None, {"obs": obs})[0]
    with th.no_grad():
        sb3_action = model.policy._predict(th.as_tensor(obs), deterministic=True).cpu().numpy()

    np.testing.assert_allclose(onnx_action, sb3_action, rtol=1e-4, atol=1e-4)
