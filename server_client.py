"""HTTP client for the NeoDEM worker callback API (sim-RL flavour).

Cloned from ../training-worker/callbacks.py — identical retry/backoff and the
same POST /api/training/workers/{claim,heartbeat,progress,complete,failed}
endpoints — with one difference: claim sends ``kinds:['sim_rl']`` so this trainer
only ever picks up sim-RL jobs (the supervised training-worker defaults to
``['supervised']`` → no cross-claiming), and the claim response is parsed into a
``ClaimedSimRlJob`` carrying the SimScene (the RL env) instead of a dataset.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

_MAX_RETRIES = 4
_BACKOFF_BASE_SEC = 0.5

# This trainer claims only sim-RL jobs.
CLAIM_KINDS = ["sim_rl"]


@dataclass
class ClaimedSimRlJob:
    """Minimal view of a claimed sim_rl TrainingJob + its SimScene."""

    id: str
    scene_id: str | None
    twin_id: str | None
    scene_mjcf_key: str | None  # twin-artifact key in the digital-twins bucket
    embodiment_tag: str
    backend: str
    bounds: dict[str, float] | None
    hyperparameters: dict[str, Any]
    status: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "ClaimedSimRlJob":
        job = payload.get("job", payload)
        scene = payload.get("scene") or {}
        return cls(
            id=job["id"],
            scene_id=scene.get("id") or job.get("sceneId"),
            twin_id=scene.get("twinId") or job.get("twinId"),
            scene_mjcf_key=scene.get("mjcfKey"),
            embodiment_tag=scene.get("embodimentTag", "g1"),
            backend=scene.get("backend", "mujoco"),
            bounds=scene.get("bounds"),
            hyperparameters=job.get("hyperparameters", {}) or {},
            status=job.get("status", "running"),
            raw=job,
        )


class ServerClient:
    """Thin client around the worker HTTP callback API."""

    def __init__(
        self,
        base_url: str,
        worker_id: str,
        device: str = "cpu",
        timeout_sec: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.worker_id = worker_id
        self.device = device
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout_sec)

    # ------------------------------------------------------------------ claim
    def claim_next_job(self) -> ClaimedSimRlJob | None:
        """POST /api/training/workers/claim with kinds=['sim_rl'] — job or None."""
        resp = self._post(
            "/api/training/workers/claim",
            {"workerId": self.worker_id, "device": self.device, "kinds": CLAIM_KINDS},
        )
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return ClaimedSimRlJob.from_api(resp.json())

    # -------------------------------------------------------------- heartbeat
    def heartbeat(self, job_id: str, gpu_util: float = 0.0, memory_util: float = 0.0) -> str:
        data = self._post_json(
            "/api/training/workers/heartbeat",
            {
                "jobId": job_id,
                "workerId": self.worker_id,
                "device": self.device,
                "gpuUtil": gpu_util,
                "memoryUtil": memory_util,
            },
        )
        return data.get("status", "continue")

    # ---------------------------------------------------------------- progress
    def progress(
        self,
        job_id: str,
        step_number: int,
        total_steps: int,
        train_loss: float,
        learning_rate: float,
        current_epoch: int = 0,
    ) -> dict[str, Any]:
        """POST /api/training/workers/progress.

        The server schema is supervised-shaped {epoch, step, totalSteps,
        trainLoss, learningRate}. RL has no loss, so the trainer reports
        ``train_loss = -mean_episode_reward`` (lower-is-better, so the existing
        loss chart trends *down* as the policy improves).
        """
        return self._post_json(
            "/api/training/workers/progress",
            {
                "jobId": job_id,
                "epoch": current_epoch,
                "step": step_number,
                "totalSteps": total_steps,
                "trainLoss": train_loss,
                "learningRate": learning_rate,
            },
        )

    # ---------------------------------------------------------------- complete
    def complete(
        self,
        job_id: str,
        artifact_uri: str,
        final_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return self._post_json(
            "/api/training/workers/complete",
            {
                "jobId": job_id,
                "artifactUri": artifact_uri,
                "finalMetrics": final_metrics,
            },
        )

    # ------------------------------------------------------------------ failed
    def failed(self, job_id: str, error_message: str) -> None:
        self._post_json(
            "/api/training/workers/failed",
            {"jobId": job_id, "error": error_message},
        )

    # ------------------------------------------------------------------- close
    def close(self) -> None:
        self._http.close()

    # ============================================================ internals
    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._http.post(path, json=body)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_err = e
                delay = _BACKOFF_BASE_SEC * (2**attempt)
                log.warning(
                    "POST %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    path, attempt + 1, _MAX_RETRIES, e, delay,
                )
                time.sleep(delay)
        assert last_err is not None
        raise last_err

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._post(path, body)
        resp.raise_for_status()
        return resp.json() if resp.content else {}
