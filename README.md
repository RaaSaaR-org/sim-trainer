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

**Auf dem Mac** liefert v1 den vollständigen sim-RL-*Lebenszyklus* + eine
Navigations-Policy (`TRAINER=ppo`), die sich zum Ziel lehnt/schlurft — KEINEN
laufenden Roboter. Echter **Gang (Lokomotion)** wird jetzt vom **realen**
`TRAINER=isaac`-Pfad trainiert (Isaac Lab + `rsl_rl` PPO, `trainers/isaac_ppo.py`) —
**real-ready, aber der Live-GPU-Lauf ist verschoben**: Isaac Lab läuft nicht auf dem
Mac. Alle Nähte sind auf dem Mac getestet; der echte Lauf startet unverändert,
sobald ein Linux/CUDA-Host angebunden ist (siehe **GPU host deploy** unten). Bis die
Host-Korrektur-Checkliste dort abgearbeitet ist, ist der Lokomotions-Score **nicht**
vertrauenswürdig. `trainers/mjx_ppo.py` bleibt ein reservierter MJX-Platzhalter.

## Architektur

```
Server  ──claim(kinds=['sim_rl'])──▶  worker.py
  ▲                                      │  (pick_trainer: stub | ppo | mjx | isaac)
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
| **mjx** | `trainers/mjx_ppo.py` | Platzhalter — CUDA/MJX-Gang, reserviert. |
| **isaac** | `trainers/isaac_ppo.py` | **Echtes** Isaac Lab + `rsl_rl` PPO — G1-**Gang (Lokomotion)** über tausende paralleler CUDA-Envs (`Isaac-Velocity-Flat-G1-v0`). Exportiert ein gate-konsumierbares `policy.onnx` (Lokomotions-Contract, 96-dim Obs). **Nur Linux/CUDA** — auf dem Mac klare Fehlermeldung. Siehe **GPU host deploy**. |

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
TRAINER=mjx uv run python worker.py        # → NotImplementedError (reserviert)
TRAINER=isaac uv run python worker.py      # G1-Gang — NUR Linux/CUDA (siehe GPU host deploy)
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

## GPU host deploy (`TRAINER=isaac`)

Echter G1-Gang wird mit **Isaac Lab** (Isaac Sim/PhysX) + `rsl_rl` PPO über tausende
paralleler CUDA-Envs trainiert. Das läuft **nicht** auf dem Mac; dort wirft
`TRAINER=isaac` eine klare Fehlermeldung (`_require_isaac`) — nutze `TRAINER=ppo` oder
`TRAINER_STUB=true`. Der resultierende Isaac-Bundle unterscheidet sich vom SB3-Bundle:
`policy.onnx` (Normalizer **in den Graph eingebacken** → `obs_norm:null`, Gate fährt
rohe Obs), `policy.pt` (rsl_rl-Checkpoint) und `manifest.json` mit `env:'locomotion'`
+ `control`-Block (statt `policy.zip`/`vecnormalize.pkl`). Primär-Artefakt ist
`policy.onnx`.

**Gepinnt** (in `isaac_ppo._TESTED_ISAAC_LAB` + dem `[isaac]`-Extra): Isaac Lab
`v2.1.0`, `rsl-rl-lib==2.1.0`.

1. Linux + NVIDIA-GPU; Isaac Sim + Isaac Lab **out-of-band** via NVIDIA-Installer
   (liefert gebündeltes Python + CUDA-Torch). EGL für `headless=True` sicherstellen.
2. In Isaacs Python: `uv pip install -e ".[isaac]"` (nur `rsl-rl-lib`; `isaaclab`/
   `isaacsim` sind KEINE PyPI-Wheels). `numpy/gymnasium/onnx/onnxruntime` müssen
   vorhanden sein; `mujoco` wird auf dem Trainer-Host **nicht** gebraucht (nur die
   Layout-Konstanten aus `sim_evaluator` werden importiert).
3. Env: `TRAINER=isaac TRAINING_DEVICE=cuda N_ENVS=4096 ISAAC_TASK=Isaac-Velocity-Flat-G1-v0
   MAX_ITERATIONS=1500 SIM_EVALUATOR_PATH=<checkout>` + `NEODEM_SERVER_URL`/`RUSTFS_*`.
4. `uv run python worker.py` → claim `sim_rl` → `AppLauncher(headless=True)` →
   `OnPolicyRunner.learn` → Export → Upload nach `model-checkpoints/<jobId>/`.

> **Ein-Job-pro-Prozess:** Isaacs `SimulationApp` ist ein Prozess-Singleton; die App
> wird **einmal** gestartet und über die Poll-Schleife wiederverwendet
> (`_launch_app` memoisiert, `_shutdown_app` ist ein No-op).

### Host-Korrektur-Checkliste (PFLICHT vor dem ersten vertrauenswürdigen Score)

`locomotion_wrappers.DEFAULT_CONTROL` liefert **Platzhalter**-Physik. Bis diese durch
Isaacs echte Werte ersetzt sind, ist `simSuccessRate` **nicht** vertrauenswürdig — der
Trainer loggt bei jedem Lauf eine Warnung (`_warn_if_placeholder_control`).

1. **`joint_order`** — Isaac ordnet die G1-DOFs per Baum-Traversierung (links/rechts
   verschränkt), was NICHT der MJCF-Aktuator-Reihenfolge (`JOINT_NAMES`) entspricht.
   Identität ist fast sicher falsch → die echte Permutation setzen. Der
   Build-Zeit-Assert `obs_dim==96` fängt nur die Dimension, **nicht** die Permutation.
2. **`default_joint_pos`** — die geratene Hocke durch `G1_CFG.init_state.joint_pos` ersetzen.
3. **`pd_gains`** (aktuell `None` = MJCF `kp=150/kv=5`) — durch Isaacs Stiffness/Damping ersetzen.
4. **`obs_scales`** — jeden Term gegen Isaacs `ObservationsCfg` prüfen.
5. **`sim_evaluator/tests/test_locomotion_parity.py`** auf dem Host laufen lassen
   (aktuell `pytest.skip`): beweist, dass die Isaac-Policy-Obs der vom Gate
   reproduzierten Obs entspricht.

Der saubere Ort für 1–4 ist `isaac_ppo._load_cfgs` (liest `env_cfg`) →
`_control_overrides` → `build_control_manifest`.

## Phasen (TASK-172.C)

- **Phase 0** — Server-Schema + Wiring (`kind`, nullable Spalten, `sceneId`,
  `ModelVersion.modelType`, nullable Gap, kind-aware claim/`completeJob`/`POST
  /jobs`, `DeploymentService` sim-only-Gate) — **erledigt im Server-Repo.**
- **Phase 1** — Stub-Durchstich — **dieses Repo.**
- **Phase 2** — echtes PPO-Nav — **dieses Repo.**
- **Phase 3** — Gate-Konsumierbarkeit (`policy_backend.py`, `evaluate_policy.py`,
  `SimulationService`-`modelType`-Zweig) — **erledigt im sim_evaluator + Server.**
- **Phase 4** — **Isaac Lab GPU-Gang (`TRAINER=isaac`) — real-ready in diesem Repo,
  Live-GPU-Lauf verschoben** (braucht Linux/CUDA-Host; siehe *GPU host deploy* +
  Host-Korrektur-Checkliste). MJX (`mjx_ppo.py`) bleibt reservierter Platzhalter;
  Synthetic-Traj-Export + rl_policy-Serving weiterhin offen.
