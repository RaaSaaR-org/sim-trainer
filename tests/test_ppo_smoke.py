"""Phase 2 acceptance: PPO nav trainer runs learn() over a SubprocVecEnv +
VecNormalize, exports the gate bundle (incl. vecnormalize.pkl + obs_norm), aborts
on a heartbeat cancel, and (slow) beats a random policy on a short horizon."""

from __future__ import annotations

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")
pytest.importorskip("onnxruntime")

from config import _default_sim_evaluator_path  # noqa: E402
from trainers.base import CancelledError, SimRlContext  # noqa: E402
from trainers.ppo_nav import PpoNavTrainer  # noqa: E402


def _ctx(work_dir, **hp) -> SimRlContext:
    base_hp = {"total_timesteps": 512, "n_envs": 2, "n_steps": 64,
               "batch_size": 64, "max_steps": 40}
    base_hp.update(hp)
    return SimRlContext(
        job_id="ppo-job", scene_id=None, twin_id=None, embodiment_tag="g1",
        scene_file=None, hyperparameters=base_hp, device="cpu", work_dir=work_dir,
        total_timesteps=base_hp["total_timesteps"], n_envs=base_hp["n_envs"],
        sim_evaluator_path=_default_sim_evaluator_path(),
    )


def test_ppo_smoke_learns_and_exports(tmp_path):
    from envs.nav_wrappers import ACTION_DIM, NAV_OBS_DIM
    import onnxruntime as ort

    steps_seen = []
    result = PpoNavTrainer().train(
        _ctx(tmp_path), lambda ev: steps_seen.append(ev.step) or True
    )
    ad = result.artifact_dir

    # Progress was reported and timesteps advanced.
    assert steps_seen and max(steps_seen) > 0
    assert result.final_metrics["totalTimesteps"] >= 512

    # Full gate bundle, including the VecNormalize stats.
    assert (ad / "policy.onnx").exists()
    assert (ad / "vecnormalize.pkl").exists()
    man = json.loads((ad / "manifest.json").read_text())
    assert man["trainer"] == "ppo"
    assert man["vecnorm"] is True
    assert man["obs_norm"] is not None
    assert len(man["obs_norm"]["mean"]) == NAV_OBS_DIM

    sess = ort.InferenceSession(str(ad / "policy.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"obs": np.zeros((1, NAV_OBS_DIM), dtype=np.float32)})[0]
    assert out.shape == (1, ACTION_DIM)


def test_ppo_heartbeat_cancel_aborts(tmp_path):
    """on_progress returning False (heartbeat 'stop') must abort learn()."""
    calls = {"n": 0}

    def on_progress(ev):
        calls["n"] += 1
        return False  # simulate server-requested cancel on the first rollout

    with pytest.raises(CancelledError):
        PpoNavTrainer().train(_ctx(tmp_path, total_timesteps=10000), on_progress)
    assert calls["n"] >= 1


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason="Honest scope boundary (TASK-172.C): model-free PPO cannot learn G1 "
    "navigation from scratch at v1's CPU/single-env budget. Evaluated correctly "
    "through the gate path (normalized obs), 50k steps does NOT reliably beat "
    "random — real capability needs thousands of parallel envs on MJX/CUDA "
    "(trainers/mjx_ppo.py, deferred). The previously-'passing' version was a "
    "false green: it fed the VecNormalize-trained policy UNNORMALIZED obs, "
    "yielding a degenerate constant action that collected the alive-bonus while "
    "random jiggled and fell. XPASS here would signal the MJX trainer arrived.",
)
def test_ppo_beats_random(tmp_path):
    """50k-step learn() vs. a random policy on shaped reward — evaluated through
    the *gate* path (PolicyBackend on policy.onnx + manifest obs_norm), NOT a bare
    ``PPO.load`` on raw obs: the policy trains inside VecNormalize, so feeding it
    unnormalized observations is a train/serve skew that invalidates the compare.

    Marked xfail (see decorator): v1 is a nav-lifecycle + lean/shuffle policy, NOT
    a walking G1 — gait is deferred to mjx_ppo.py."""
    from envs.nav_wrappers import ACTION_DIM, make_nav_env
    from policy_backend import PolicyBackend

    result = PpoNavTrainer().train(
        _ctx(tmp_path, total_timesteps=50000, n_envs=4, n_steps=512, batch_size=256,
             max_steps=200),
        lambda ev: True,
    )
    # Exactly what the deployment gate runs: onnx + VecNormalize stats.
    backend = PolicyBackend.from_artifacts(
        result.artifact_dir / "policy.onnx", result.artifact_dir / "manifest.json"
    )

    def mean_return(policy_fn, episodes=5):
        env = make_nav_env(obs_mode="state", max_steps=200, shaped=True, domain_rand=False)
        totals = []
        try:
            for ep in range(episodes):
                obs, _ = env.reset(seed=ep)
                total = 0.0
                for _ in range(200):
                    obs, r, term, trunc, _ = env.step(policy_fn(obs))
                    total += r
                    if term or trunc:
                        break
                totals.append(total)
        finally:
            env.close()
        return float(np.mean(totals))

    trained = mean_return(lambda o: backend.predict(o))
    random = mean_return(lambda o: np.random.uniform(-0.1, 0.1, ACTION_DIM).astype(np.float32))
    assert trained > random, f"trained {trained:.2f} !> random {random:.2f}"
