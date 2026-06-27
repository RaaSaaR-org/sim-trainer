"""S3-compatible (RustFS) storage for sim-RL: twin-scene download + policy upload.

Cloned from ../training-worker/storage.py. The sim-trainer reads the twin MJCF
from the digital-twins bucket (the scene's ``mjcfKey``) and uploads the trained
policy artifacts (policy.zip / policy.onnx / vecnormalize.pkl / manifest.json) to
the model bucket under ``<jobId>/…`` — the prefix the server's
``materializePolicyFiles`` reads back from ``ModelVersion.trainingJobId``.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class StorageClient:
    """Thin wrapper around boto3 for twin-scene fetch + artifact upload."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        twin_bucket: str,
        model_bucket: str,
    ) -> None:
        try:
            import boto3
            from botocore.client import Config as BotoConfig
        except ImportError as e:
            raise RuntimeError("boto3 is required — `uv pip install boto3`") from e

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.twin_bucket = twin_bucket
        self.model_bucket = model_bucket
        self.endpoint = endpoint

    # ---------------------------------------------------------------- twins
    def download_twin_artifact(self, key: str, dest: Path) -> Path:
        """Download one twin artifact (e.g. the scene MJCF) to ``dest``."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading twin artifact %s/%s → %s", self.twin_bucket, key, dest)
        self._client.download_file(self.twin_bucket, key, str(dest))
        return dest

    # ------------------------------------------------------------- artifacts
    def upload_artifact(self, job_id: str, local_path: Path, name: str | None = None) -> str:
        """Upload a file to models/{job_id}/{name} — returns an s3:// URI."""
        name = name or local_path.name
        key = f"{job_id}/{name}"
        log.info(
            "Uploading artifact %s → %s/%s (%d bytes)",
            local_path.name, self.model_bucket, key, local_path.stat().st_size,
        )
        self._client.upload_file(str(local_path), self.model_bucket, key)
        return f"s3://{self.model_bucket}/{key}"

    def upload_dir(self, job_id: str, local_dir: Path) -> dict[str, str]:
        """Upload every file in ``local_dir`` under models/{job_id}/. Returns
        {filename: s3-uri}."""
        uris: dict[str, str] = {}
        for f in sorted(local_dir.iterdir()):
            if f.is_file():
                uris[f.name] = self.upload_artifact(job_id, f, f.name)
        return uris

    # ---------------------------------------------------------------- ensure
    def ensure_model_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.model_bucket)
        except Exception:
            log.info("Creating model bucket: %s", self.model_bucket)
            self._client.create_bucket(Bucket=self.model_bucket)
