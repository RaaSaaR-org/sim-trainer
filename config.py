"""Configuration loaded from environment (with .env file support).

Cloned from ../training-worker/config.py — same .env loader and storage/server
fields — plus the sim-RL specifics: which trainer to run, the navigation env
hyperparameters, and where the sibling sim_evaluator package lives.
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    """Minimal .env loader — supports KEY=VALUE lines."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value and value[0] not in ('"', "'"):
            hash_idx = value.find("#")
            if hash_idx != -1:
                value = value[:hash_idx].rstrip()
        value = value.strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _default_sim_evaluator_path() -> str:
    """Default sibling location of the sim_evaluator package.

    Layout: <emai>/sim-trainer/ (this repo) and
            <emai>/robot-management-system/robot-agent/hardware/sim_evaluator/.
    Resolved relative to this file so it works regardless of CWD.
    """
    here = Path(__file__).resolve().parent
    return str(
        here.parent
        / "robot-management-system"
        / "robot-agent"
        / "hardware"
        / "sim_evaluator"
    )


@dataclass
class Config:
    """Sim-trainer configuration."""

    # Server + identity
    server_url: str
    worker_id: str
    poll_interval_sec: float

    # RustFS / S3-compatible storage
    rustfs_endpoint: str
    rustfs_access_key: str
    rustfs_secret_key: str
    rustfs_bucket_twins: str
    rustfs_bucket_models: str

    # Compute
    device: str  # "mps" | "cuda" | "cpu"

    # Behaviour
    trainer_kind: str  # 'stub' | 'ppo' | 'mjx'
    heartbeat_interval_sec: float

    # sim_evaluator (navigation env + shared wrappers) — resolved on sys.path
    sim_evaluator_path: str

    # RL hyperparameters (overridable per-job via job.hyperparameters)
    total_timesteps: int
    n_envs: int

    @classmethod
    def from_env(cls) -> "Config":
        _load_env_file(Path(__file__).parent / ".env")

        return cls(
            server_url=os.environ.get("NEODEM_SERVER_URL", "http://localhost:3001"),
            worker_id=os.environ.get("WORKER_ID", f"sim-trainer-{socket.gethostname()}"),
            poll_interval_sec=float(os.environ.get("POLL_INTERVAL_SEC", "5")),
            rustfs_endpoint=os.environ.get("RUSTFS_ENDPOINT", "http://localhost:9000"),
            rustfs_access_key=os.environ.get("RUSTFS_ACCESS_KEY", "rustfsadmin"),
            rustfs_secret_key=os.environ.get("RUSTFS_SECRET_KEY", "rustfsadmin"),
            rustfs_bucket_twins=os.environ.get("RUSTFS_BUCKET_TWINS", "digital-twins"),
            rustfs_bucket_models=os.environ.get("RUSTFS_BUCKET_MODELS", "model-checkpoints"),
            device=os.environ.get("TRAINING_DEVICE", "cpu"),
            trainer_kind=_resolve_trainer_kind(),
            heartbeat_interval_sec=float(os.environ.get("HEARTBEAT_INTERVAL_SEC", "30")),
            sim_evaluator_path=os.environ.get(
                "SIM_EVALUATOR_PATH", _default_sim_evaluator_path()
            ),
            total_timesteps=int(os.environ.get("TOTAL_TIMESTEPS", "200000")),
            n_envs=int(os.environ.get("N_ENVS", "8")),
        )

    def summary(self) -> str:
        return (
            f"server={self.server_url} worker_id={self.worker_id} "
            f"device={self.device} trainer={self.trainer_kind} "
            f"poll={self.poll_interval_sec}s n_envs={self.n_envs}"
        )

    def ensure_sim_evaluator_on_path(self) -> None:
        """Put the sibling sim_evaluator on sys.path so `import envs.*` resolves."""
        p = self.sim_evaluator_path
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _resolve_trainer_kind() -> str:
    """TRAINER_STUB=true forces the stub; otherwise TRAINER selects the backend."""
    if os.environ.get("TRAINER_STUB", "false").lower() in ("1", "true", "yes"):
        return "stub"
    return os.environ.get("TRAINER", "ppo").lower()


def require_python_311() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11+ is required")
