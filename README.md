# NeoDEM sim-trainer

Sim-RL-Trainer für NeoDEM (TASK-172.C). Schwesterrepo zu `../training-worker`:
gleiche Poll-Schleife (claim / progress / complete / fail / heartbeat), aber
statt SmolVLA-LoRA-Feintuning trainiert er eine **Navigations-Policy** für den
Unitree G1 in einer aus einem Digital Twin abgeleiteten MuJoCo-Szene.

> **Sprache:** Deutsch primär, Englisch sekundär.

## ⚠️ Geltungsbereich (zuerst lesen — ändert die Abnahme)

Es gibt **kein Gang-Primitiv (gait) irgendwo im Repo**, und `g1_env.py` exponiert
**29 rohe Gelenk-Positionsziele**. Modellfreies RL kann **G1-Lokomotion nicht von
Grund auf** auf einer einzelnen, nicht vektorisierten MuJoCo-Umgebung auf einem
Mac lernen — Legged-RL braucht tausende paralleler Envs auf CUDA/MJX.

**v1 liefert den vollständigen sim-RL-*Lebenszyklus* + eine Navigations-Policy,
die sich zum Ziel lehnt/schlurft — KEINEN laufenden Roboter.** Wir werben **nicht**
mit sim-to-real-validierter Lokomotion. Echter Gang ist auf einen MJX/`rsl_rl`-
CUDA-Trainer verschoben (`trainers/mjx_ppo.py`, Platzhalter).

## Architektur

```
Server  ──claim(kinds=['sim_rl'])──▶  worker.py
  ▲                                      │  (pick_trainer: stub | ppo | mjx)
  │ progress/heartbeat/complete          ▼
  └──────────────────────────────  trainers/  (G1-Nav-Policy)
                                         │
            policy.zip / policy.onnx /   ▼
            vecnormalize.pkl / manifest.json  ──▶  model-checkpoints/<jobId>/
```

Die Navigationsumgebung und die geteilten Wrapper leben im **Schwester-Paket**
`sim_evaluator` (`../robot-management-system/robot-agent/hardware/sim_evaluator`).
Es hat kein Build-System, daher wird es **zur Laufzeit über `sys.path`** aufgelöst
(`SIM_EVALUATOR_PATH`, Default = Geschwisterpfad in `config.py`) statt als
editierbares Wheel. So funktioniert `import envs.nav_wrappers` ohne Änderung am
Packaging von `sim_evaluator`.

## Trainer

| Trainer | Datei | Zweck |
|---------|-------|-------|
| **stub** | `trainers/stub_rl.py` | Vertikaler Durchstich: Zero-Policy-Rollout auf `g1_empty_scene.xml`, schreibt ladbares `policy.zip` + `policy.onnx`. Beweist die ganze Schleife ohne gelöstes RL. |
| **ppo** (Default) | `trainers/ppo_nav.py` | Echtes PPO (SB3, `MlpPolicy`) über `SubprocVecEnv` + `VecNormalize` + Domain-Randomisierung + **pflicht-Alive-Bonus-Shaping**. |
| **mjx** | `trainers/mjx_ppo.py` | Platzhalter — CUDA/MJX-Gang, Phase 4, verschoben. |

## Beobachtung & Belohnung (`sim_evaluator/envs/nav_wrappers.py`)

- **Obs (61-dim):** `[29 qpos | 29 qvel | goal_dx | goal_dy | |goal|]`. Identisch
  in Training (`obs_mode='state'`, kein GL) und Gate (`obs_mode='rgb_state'`,
  Frames für die UI) — diese Parität macht das `policy.onnx` für das
  Sim-to-Real-Gate konsumierbar.
- **Reward (Shaping, pflicht):** potenzialbasierter Fortschritt + Alive-Bonus +
  Keepout + Energie + terminales Shaping. Ohne Alive-Bonus ist das
  From-Scratch-Problem schlecht gestellt (zufällige Policy fällt nur um).

## Artefakte → `s3://model-checkpoints/<jobId>/`

- `policy.zip` — SB3-Modell (= `artifactUri` der `ModelVersion`)
- `policy.onnx` — deterministische Action-Policy (vom Gate via onnxruntime gefahren)
- `vecnormalize.pkl` — eingefrorene `VecNormalize`-Statistiken
- `manifest.json` — `{kind:'sim_rl', embodimentTag, obs_layout, action_dim:29,
  sceneId, twinId, obs_norm:{mean,var,clip,epsilon}}`

Das Gate (`sim_evaluator/policy_backend.py`) liest `obs_norm` aus dem Manifest und
reproduziert die `VecNormalize`-Transformation **torch-/SB3-frei** (nur
onnxruntime).

## Setup

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"          # torch + stable-baselines3 + mujoco + onnx
cp .env.example .env                # NEODEM_SERVER_URL, RUSTFS_*, TRAINER, …
```

## Ausführen

```bash
uv run python worker.py                    # TRAINER=ppo (Default)
TRAINER_STUB=true uv run python worker.py  # Stub-Durchstich
TRAINER=mjx uv run python worker.py        # → NotImplementedError (Phase 4)
```

Der Server muss einen `sim_rl`-Job offen haben (TrainingJob `kind='sim_rl'` mit
`sceneId`). Der Trainer beansprucht nur `kinds:['sim_rl']`; der
SmolVLA-`training-worker` (Default `['supervised']`) wird nie quergeclaimt.

## Tests

```bash
uv run python -m pytest tests/ -m "not slow"   # schnell (Stub, PPO-Smoke, Parität, Worker)
uv run python -m pytest tests/ -m slow         # PPO „schlägt Zufall" (Minuten, CPU)
```

| Test | Deckt ab |
|------|----------|
| `test_server_client.py` | claim sendet `kinds:['sim_rl']`, parst SimScene; heartbeat/progress/complete/failed |
| `test_stub_rl.py` | Stub schreibt **ladbares** `.zip`+`.onnx`; onnx == SB3-Policy (Export-Treue) |
| `test_ppo_smoke.py` | `learn()` über SubprocVecEnv+VecNormalize; Heartbeat-Cancel bricht ab; (slow) schlägt Zufall |
| `test_gate_parity.py` | **Cross-Repo:** `PolicyBackend` (Gate) == Trainer-Action für feste Obs |
| `test_worker.py` | `_run_one_job`: claim→train→upload→complete (Server/Storage gemockt) |

## Phasen (TASK-172.C)

- **Phase 0** — Server-Schema + Wiring (`kind`, nullable Spalten, `sceneId`,
  `ModelVersion.modelType`, nullable Gap, kind-aware claim/`completeJob`/`POST
  /jobs`, `DeploymentService` sim-only-Gate) — **erledigt im Server-Repo.**
- **Phase 1** — Stub-Durchstich — **dieses Repo.**
- **Phase 2** — echtes PPO-Nav — **dieses Repo.**
- **Phase 3** — Gate-Konsumierbarkeit (`policy_backend.py`, `evaluate_policy.py`,
  `SimulationService`-`modelType`-Zweig) — **erledigt im sim_evaluator + Server.**
- **Phase 4** — Synthetic-Traj-Export, `mjx_ppo.py` CUDA/MJX-Gang, rl_policy-
  Serving — **verschoben (braucht CUDA-Host).**
