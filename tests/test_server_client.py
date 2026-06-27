"""Tests for the sim-RL ServerClient: claim sends kinds=['sim_rl'] and parses the
SimScene; heartbeat/progress/complete/failed hit the right endpoints. httpx is
mocked with a MockTransport — no server, no torch."""

from __future__ import annotations

import json

import httpx
import pytest

from server_client import ClaimedSimRlJob, ServerClient


def _client_with(handler) -> ServerClient:
    sc = ServerClient("http://server", "wkr-1", device="cpu")
    sc._http = httpx.Client(base_url="http://server", transport=httpx.MockTransport(handler))
    return sc


def test_claim_sends_sim_rl_kind_and_parses_scene():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "job": {"id": "job-1", "kind": "sim_rl", "status": "running",
                        "sceneId": "scene-1", "hyperparameters": {"total_timesteps": 1000}},
                "dataset": None,
                "scene": {
                    "id": "scene-1", "mjcfKey": "twin-1/scene.xml", "twinId": "twin-1",
                    "embodimentTag": "g1", "backend": "mujoco",
                    "bounds": {"minX": 0, "minY": 0, "minZ": 0, "maxX": 2, "maxY": 2, "maxZ": 2},
                },
            },
        )

    job = _client_with(handler).claim_next_job()
    assert seen["path"] == "/api/training/workers/claim"
    assert seen["body"]["kinds"] == ["sim_rl"]
    assert isinstance(job, ClaimedSimRlJob)
    assert job.id == "job-1"
    assert job.scene_id == "scene-1"
    assert job.twin_id == "twin-1"
    assert job.scene_mjcf_key == "twin-1/scene.xml"
    assert job.embodiment_tag == "g1"
    assert job.hyperparameters == {"total_timesteps": 1000}


def test_claim_204_returns_none():
    job = _client_with(lambda req: httpx.Response(204)).claim_next_job()
    assert job is None


def test_claim_builtin_scene_without_mjcf_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "job": {"id": "job-2", "kind": "sim_rl"},
            "dataset": None,
            "scene": {"id": "s2", "mjcfKey": None, "twinId": None,
                      "embodimentTag": "g1", "backend": "mujoco", "bounds": None},
        })

    job = _client_with(handler).claim_next_job()
    assert job.scene_mjcf_key is None  # → trainer falls back to bundled scene


def test_heartbeat_stop():
    sc = _client_with(lambda req: httpx.Response(200, json={"status": "stop"}))
    assert sc.heartbeat("job-1") == "stop"


def test_heartbeat_default_continue_when_empty():
    sc = _client_with(lambda req: httpx.Response(200, json={}))
    assert sc.heartbeat("job-1") == "continue"


def test_progress_maps_reward_to_loss():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "continue"})

    sc = _client_with(handler)
    out = sc.progress("job-1", step_number=100, total_steps=1000,
                      train_loss=-5.0, learning_rate=3e-4, current_epoch=0)
    assert seen["path"] == "/api/training/workers/progress"
    assert seen["body"] == {"jobId": "job-1", "epoch": 0, "step": 100,
                            "totalSteps": 1000, "trainLoss": -5.0, "learningRate": 3e-4}
    assert out == {"status": "continue"}


def test_complete_and_failed():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={})

    sc = _client_with(handler)
    sc.complete("job-1", artifact_uri="s3://model-checkpoints/job-1/policy.zip",
                final_metrics={"meanReward": 1.2})
    sc.failed("job-1", "boom")
    paths = [p for p, _ in seen]
    assert "/api/training/workers/complete" in paths
    assert "/api/training/workers/failed" in paths
    complete_body = next(b for p, b in seen if p.endswith("/complete"))
    assert complete_body["artifactUri"].endswith("policy.zip")
    assert complete_body["finalMetrics"] == {"meanReward": 1.2}
