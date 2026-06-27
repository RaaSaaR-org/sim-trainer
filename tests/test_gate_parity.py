"""Phase 3 acceptance (cross-repo): the deployment gate's PolicyBackend (in
sim_evaluator) reproduces the trainer's action for a fixed observation.

Trains a tiny PPO policy, then loads its policy.onnx + manifest.json through the
*gate's* torch-free PolicyBackend and checks the action equals the SB3 policy run
on the VecNormalize-normalized observation. This is the train↔gate parity the
sim-to-real validation relies on."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")
pytest.importorskip("onnxruntime")

import pickle  # noqa: E402

from config import _default_sim_evaluator_path  # noqa: E402
from policy_backend import PolicyBackend  # sim_evaluator, on path via conftest  # noqa: E402
from trainers.base import SimRlContext  # noqa: E402
from trainers.ppo_nav import PpoNavTrainer  # noqa: E402


def test_gate_action_matches_trainer(tmp_path):
    import torch as th
    from envs.nav_wrappers import NAV_OBS_DIM
    from stable_baselines3 import PPO

    ctx = SimRlContext(
        job_id="parity", scene_id=None, twin_id=None, embodiment_tag="g1",
        scene_file=None,
        hyperparameters={"total_timesteps": 256, "n_envs": 2, "n_steps": 64,
                         "batch_size": 64, "max_steps": 40},
        device="cpu", work_dir=tmp_path, total_timesteps=256, n_envs=2,
        sim_evaluator_path=_default_sim_evaluator_path(),
    )
    result = PpoNavTrainer().train(ctx, lambda ev: True)
    ad = result.artifact_dir

    # Gate side: torch-free backend (onnx + manifest obs_norm).
    backend = PolicyBackend.from_artifacts(ad / "policy.onnx", ad / "manifest.json")
    # Trainer side: the SB3 policy.
    model = PPO.load(str(ad / "policy.zip"), device="cpu")

    # Normalization parity: the gate reads obs_norm from manifest.json, which
    # write_policy_artifacts extracts by hand from VecNormalize.obs_rms. Load the
    # *real* saved VecNormalize and confirm the gate reproduces its transform —
    # this is what catches a manifest extraction regression (mean/var swap,
    # clip/epsilon mismatch) that an onnx-vs-SB3-on-shared-input check cannot.
    # VecNormalize.__getstate__ drops the (unpicklable) venv, so a bare unpickle
    # yields an object whose normalize_obs needs no env.
    with open(ad / "vecnormalize.pkl", "rb") as f:
        vecnorm = pickle.load(f)

    rng = np.random.default_rng(7)
    for _ in range(5):
        raw = rng.standard_normal(NAV_OBS_DIM).astype(np.float32)

        # (a) gate normalize_obs == real VecNormalize.normalize_obs
        np.testing.assert_allclose(
            backend.normalize_obs(raw),
            np.asarray(vecnorm.normalize_obs(raw.reshape(1, -1))).reshape(-1),
            rtol=1e-5, atol=1e-6,
        )

        # (b) gate action == SB3 policy action on the normalized obs
        gate_action = backend.predict(raw)
        normed = backend.normalize_obs(raw)
        with th.no_grad():
            sb3_action = (
                model.policy._predict(th.as_tensor(normed).reshape(1, -1), deterministic=True)
                .cpu()
                .numpy()
                .reshape(-1)
            )
        np.testing.assert_allclose(gate_action, sb3_action, rtol=1e-4, atol=1e-4)
