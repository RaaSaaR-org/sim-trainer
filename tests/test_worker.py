"""Phase 1 worker glue: _run_one_job claims a sim_rl job, runs the stub trainer,
uploads the artifact bundle, and POSTs /complete with the policy.zip URI — the
whole pending→running→completed loop, with ServerClient + StorageClient faked
(no live server, no RustFS)."""

from __future__ import annotations

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")

import worker as worker_mod  # noqa: E402
from config import Config, _default_sim_evaluator_path  # noqa: E402
from server_client import ClaimedSimRlJob  # noqa: E402
from trainers.stub_rl import StubRlTrainer  # noqa: E402


class FakeServer:
    def __init__(self):
        self.completed = []
        self.failed_calls = []
        self.progress_calls = 0

    def heartbeat(self, job_id):
        return "continue"

    def progress(self, **kwargs):
        self.progress_calls += 1
        return {}

    def complete(self, job_id, artifact_uri, final_metrics):
        self.completed.append((job_id, artifact_uri, final_metrics))
        return {}

    def failed(self, job_id, msg):
        self.failed_calls.append((job_id, msg))


class FakeStorage:
    def __init__(self):
        self.uploaded = {}

    def ensure_model_bucket(self):
        pass

    def upload_dir(self, job_id, local_dir: Path):
        uris = {f.name: f"s3://model-checkpoints/{job_id}/{f.name}"
                for f in Path(local_dir).iterdir() if f.is_file()}
        self.uploaded[job_id] = uris
        return uris


def _cfg() -> Config:
    return Config(
        server_url="http://server", worker_id="wkr", poll_interval_sec=5,
        rustfs_endpoint="http://rustfs", rustfs_access_key="k", rustfs_secret_key="s",
        rustfs_bucket_twins="digital-twins", rustfs_bucket_models="model-checkpoints",
        device="cpu", trainer_kind="stub", heartbeat_interval_sec=60,
        sim_evaluator_path=_default_sim_evaluator_path(),
        total_timesteps=0, n_envs=1,
    )


def test_run_one_job_completes_with_artifact():
    server, storage = FakeServer(), FakeStorage()
    job = ClaimedSimRlJob(
        id="job-1", scene_id="s1", twin_id=None, scene_mjcf_key=None,
        embodiment_tag="g1", backend="mujoco", bounds=None, hyperparameters={}, status="running",
    )

    worker_mod._run_one_job(_cfg(), server, storage, StubRlTrainer(), job)

    assert len(server.completed) == 1
    job_id, artifact_uri, metrics = server.completed[0]
    assert job_id == "job-1"
    assert artifact_uri.endswith("/policy.zip")
    assert metrics["trainer"] == "stub"
    assert not server.failed_calls
    # The full bundle was uploaded under the job prefix.
    assert {"policy.zip", "policy.onnx", "manifest.json"} <= set(storage.uploaded["job-1"])


def test_run_one_job_posts_failed_on_trainer_error():
    server, storage = FakeServer(), FakeStorage()

    class BoomTrainer(StubRlTrainer):
        def train(self, ctx, on_progress):
            raise RuntimeError("kaboom")

    job = ClaimedSimRlJob(
        id="job-2", scene_id=None, twin_id=None, scene_mjcf_key=None,
        embodiment_tag="g1", backend="mujoco", bounds=None, hyperparameters={}, status="running",
    )
    worker_mod._run_one_job(_cfg(), server, storage, BoomTrainer(), job)

    assert not server.completed
    assert len(server.failed_calls) == 1
    assert "kaboom" in server.failed_calls[0][1]
