"""NeoDEM sim-RL trainer — poll the server, claim sim_rl jobs, train a G1
navigation policy, post callbacks, upload a gate-consumable policy bundle.

Cloned from ../training-worker/worker.py (signal handling, HeartbeatThread, the
poll loop) with the sim_rl-specific bits: it claims only ``kinds:['sim_rl']``,
materializes the twin scene MJCF, runs a sim-RL trainer (stub | ppo | mjx), and
uploads policy.zip / policy.onnx / vecnormalize.pkl / manifest.json.

Run:
    uv run python worker.py                    # uses env / .env (TRAINER=ppo)
    TRAINER_STUB=true uv run python worker.py  # stub vertical slice
"""

from __future__ import annotations

import logging
import signal
import tempfile
import threading
import time
import traceback
from pathlib import Path

from config import Config, require_python_311
from server_client import ClaimedSimRlJob, ServerClient
from storage import StorageClient
from trainers import (
    BaseSimRlTrainer,
    CancelledError,
    ProgressEvent,
    SimRlContext,
    TrainerResult,
    pick_trainer,
)

log = logging.getLogger("sim-trainer")

_shutdown = threading.Event()


# ---------------------------------------------------------------------- signal
def _install_signal_handlers() -> None:
    def handle(signum: int, _frame) -> None:  # noqa: ANN001
        log.info("Received signal %d — requesting shutdown…", signum)
        _shutdown.set()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


# ------------------------------------------------------------------- heartbeat
class HeartbeatThread(threading.Thread):
    """Fires heartbeats every N seconds; flags cancellation in shared state."""

    def __init__(self, server: ServerClient, job_id: str, interval_sec: float,
                 cancel_flag: threading.Event) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{job_id[:8]}")
        self.server = server
        self.job_id = job_id
        self.interval_sec = interval_sec
        self.cancel_flag = cancel_flag
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.server.heartbeat(self.job_id) == "stop":
                    log.info("Heartbeat: server requested cancel for job %s", self.job_id)
                    self.cancel_flag.set()
                    break
            except Exception as e:  # noqa: BLE001
                log.warning("Heartbeat failed: %s", e)
            self._stop.wait(self.interval_sec)

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------- scene helper
def _materialize_scene(cfg: Config, storage: StorageClient | None,
                       job: ClaimedSimRlJob) -> Path | None:
    """Download the twin scene MJCF next to the vendored G1 meshes so its
    relative <include> resolves. Built-in scenes (no mjcfKey) → None (the trainer
    falls back to the bundled empty scene)."""
    if not job.scene_mjcf_key:
        return None
    if storage is None:
        log.warning("Scene has mjcfKey but storage is unavailable — using bundled scene")
        return None
    mjcf_dir = Path(cfg.sim_evaluator_path) / "mjcf"
    dest = mjcf_dir / f".twinscene_{job.id}.xml"
    try:
        storage.download_twin_artifact(job.scene_mjcf_key, dest)
        return dest
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to download scene %s (%s) — using bundled scene",
                    job.scene_mjcf_key, e)
        return None


# -------------------------------------------------------------- single job run
def _run_one_job(cfg: Config, server: ServerClient, storage: StorageClient | None,
                 trainer: BaseSimRlTrainer, job: ClaimedSimRlJob) -> None:
    log.info("▶ Running sim_rl job %s — scene=%s twin=%s embodiment=%s",
             job.id, job.scene_id, job.twin_id, job.embodiment_tag)

    cancel_flag = threading.Event()
    heartbeat = HeartbeatThread(server, job.id, cfg.heartbeat_interval_sec, cancel_flag)
    heartbeat.start()

    # Everything from here is wrapped so the heartbeat thread is always stopped
    # and the scene cleaned up — even if setup (scene download, tempdir creation)
    # throws before the trainer's own try/except. Without this the job would be
    # left stuck in 'running' and (with the main-loop guard) could kill the daemon.
    scene_file: Path | None = None
    try:
        scene_file = _materialize_scene(cfg, storage, job)

        with tempfile.TemporaryDirectory(prefix=f"neodem-simrl-{job.id}-") as tmp:
            ctx = SimRlContext(
                job_id=job.id,
                scene_id=job.scene_id,
                twin_id=job.twin_id,
                embodiment_tag=job.embodiment_tag,
                scene_file=scene_file,
                hyperparameters=job.hyperparameters,
                device=cfg.device,
                work_dir=Path(tmp),
                total_timesteps=cfg.total_timesteps,
                n_envs=cfg.n_envs,
                sim_evaluator_path=cfg.sim_evaluator_path,
            )

            def on_progress(ev: ProgressEvent) -> bool:
                try:
                    result = server.progress(
                        job_id=job.id,
                        step_number=ev.step,
                        total_steps=ev.total_steps,
                        train_loss=-ev.mean_reward,  # lower-is-better for the loss chart
                        learning_rate=ev.learning_rate,
                    )
                    if result.get("status") == "cancel":
                        return False
                except Exception as e:  # noqa: BLE001
                    log.warning("Progress POST failed at step %d: %s", ev.step, e)
                return not cancel_flag.is_set() and not _shutdown.is_set()

            try:
                result: TrainerResult = trainer.train(ctx, on_progress)
            except CancelledError as e:
                log.info("Job %s cancelled by server: %s — skipping /failed", job.id, e)
                return
            except Exception as e:  # noqa: BLE001
                log.error("Job %s failed: %s\n%s", job.id, e, traceback.format_exc())
                _safe_failed(server, job.id, f"{type(e).__name__}: {e}")
                return

            # Upload the artifact bundle (still inside the tempdir so artifact_dir
            # exists); the primary (policy.zip) URI is artifactUri.
            try:
                artifact_uri = _upload_artifacts(cfg, storage, job.id, result)
            except Exception as e:  # noqa: BLE001
                log.error("Artifact upload failed: %s", e)
                _safe_failed(server, job.id, f"artifact upload failed: {e}")
                return

            try:
                server.complete(
                    job.id, artifact_uri=artifact_uri, final_metrics=result.final_metrics
                )
                log.info("✓ Job %s completed — artifact=%s", job.id, artifact_uri)
            except Exception as e:  # noqa: BLE001
                log.error("Failed to POST /complete for job %s: %s", job.id, e)
    except Exception as e:  # noqa: BLE001
        # Setup-time failure outside the trainer's own try (tempdir creation,
        # scene materialization edge cases) — fail the job rather than hang it.
        log.error("Job %s setup failed: %s\n%s", job.id, e, traceback.format_exc())
        _safe_failed(server, job.id, f"{type(e).__name__}: {e}")
    finally:
        heartbeat.stop()
        _cleanup_scene(scene_file)


def _upload_artifacts(cfg: Config, storage: StorageClient | None, job_id: str,
                      result: TrainerResult) -> str:
    if storage is None:
        raise RuntimeError("storage is unavailable — cannot upload policy artifacts")
    storage.ensure_model_bucket()
    uris = storage.upload_dir(job_id, result.artifact_dir)
    primary = result.primary_artifact
    if primary not in uris:
        raise RuntimeError(f"primary artifact {primary} missing from {result.artifact_dir}")
    return uris[primary]


def _cleanup_scene(scene_file: Path | None) -> None:
    if scene_file:
        try:
            scene_file.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def _safe_failed(server: ServerClient, job_id: str, msg: str) -> None:
    try:
        server.failed(job_id, msg)
    except Exception as e:  # noqa: BLE001
        log.error("Failed to POST /failed: %s", e)


def _build_storage(cfg: Config) -> StorageClient | None:
    try:
        return StorageClient(
            endpoint=cfg.rustfs_endpoint,
            access_key=cfg.rustfs_access_key,
            secret_key=cfg.rustfs_secret_key,
            twin_bucket=cfg.rustfs_bucket_twins,
            model_bucket=cfg.rustfs_bucket_models,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Storage unavailable (%s) — scenes/artifacts limited", e)
        return None


# -------------------------------------------------------------------- main loop
def main() -> None:
    require_python_311()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _install_signal_handlers()

    cfg = Config.from_env()
    cfg.ensure_sim_evaluator_on_path()
    log.info("NeoDEM sim-RL trainer starting — %s", cfg.summary())

    trainer = pick_trainer(cfg.trainer_kind)
    server = ServerClient(cfg.server_url, cfg.worker_id, device=cfg.device)
    storage = _build_storage(cfg)

    idle_prints = 0
    try:
        while not _shutdown.is_set():
            try:
                job = server.claim_next_job()
            except Exception as e:  # noqa: BLE001
                log.warning("Claim poll failed: %s — sleeping", e)
                _shutdown.wait(cfg.poll_interval_sec)
                continue

            if job is None:
                if idle_prints % 12 == 0:
                    log.info("No pending sim_rl jobs — polling every %.1fs", cfg.poll_interval_sec)
                idle_prints += 1
                _shutdown.wait(cfg.poll_interval_sec)
                continue

            idle_prints = 0
            try:
                _run_one_job(cfg, server, storage, trainer, job)
            except Exception as e:  # noqa: BLE001
                # Defense in depth: a job must never take the whole daemon down.
                log.error(
                    "Unexpected error running job %s: %s\n%s",
                    job.id, e, traceback.format_exc(),
                )
    finally:
        server.close()
        log.info("Sim-trainer stopped cleanly.")


if __name__ == "__main__":
    main()
