# CLAUDE.md — sim-trainer

NeoDEM sim-RL-Trainer (TASK-172.C). Klon der Poll-Schleife von
`../training-worker`; trainiert eine G1-**Navigations**-Policy (kein Gang) in
einer Twin-MuJoCo-Szene und lädt ein gate-konsumierbares `policy.onnx` hoch.

**Sprache:** Deutsch primär, Englisch sekundär.

## Goldene Regeln

1. **Navigation, nicht Lokomotion.** Niemals sim-to-real-validierten Gang
   bewerben. Echter Gang = `trainers/mjx_ppo.py` (Platzhalter, CUDA/MJX, Phase 4).
2. **Physik bleibt echt.** Den Pelvis-Freejoint NICHT teleportieren — das
   bekämpft den Solver und verfälscht `collision_count`/`fallen`, die das Gate liest.
3. **Eine Obs-Quelle.** Obs-Layout + Reward-Shaping leben ausschließlich in
   `sim_evaluator/envs/nav_wrappers.py` (`make_nav_env`). Trainer **und** Gate
   bauen ihre Env darüber → Train/Eval-Parität. Layout nie hier duplizieren.
4. **Alive-Bonus ist pflicht.** Ohne ihn ist das From-Scratch-Problem schlecht
   gestellt (zufällige Policy fällt nur um).

## Layout

```
worker.py            Poll-Schleife (claim sim_rl → train → upload → complete)
server_client.py     HTTP-Callbacks; claim sendet kinds=['sim_rl']
storage.py           Twin-Szene laden (digital-twins) + Policy hochladen (model-checkpoints)
config.py            Env-Config + SIM_EVALUATOR_PATH-Auflösung
trainers/
  base.py            Interface + write_policy_artifacts (zip/onnx/vecnorm/manifest)
  stub_rl.py         Zero-Policy-Durchstich
  ppo_nav.py         SB3 PPO + SubprocVecEnv + VecNormalize + DR
  mjx_ppo.py         Platzhalter (Phase 4)
tests/               server_client, stub_rl, ppo_smoke, gate_parity, worker
```

## sim_evaluator-Abhängigkeit

`sim_evaluator` (in `robot-management-system/robot-agent/hardware/`) hat **kein
Build-System** → wird zur **Laufzeit über `sys.path`** aufgelöst
(`config.ensure_sim_evaluator_on_path`, `SIM_EVALUATOR_PATH`). Nicht als Wheel
installieren. `import envs.nav_wrappers` / `import policy_backend` setzen voraus,
dass dieser Pfad gesetzt ist (Worker + `tests/conftest.py` tun das).

## Server-Vertrag (nicht hier ändern)

- Claim-Antwort: `{job, dataset:null, scene:{id, mjcfKey, twinId, embodimentTag,
  backend, bounds}}` → `ClaimedSimRlJob`.
- Progress: Server-Schema ist supervised-förmig; RL meldet `trainLoss =
  -mean_reward` (kleiner = besser, damit die Loss-Kurve sinkt).
- `complete` setzt `ModelVersion(modelType='rl_policy', artifactUri=policy.zip)`;
  das Gate liest `policy.onnx`/`manifest.json` unter `<trainingJobId>/`.

## Tests

```bash
uv run python -m pytest tests/ -m "not slow"
```
Voraussetzungen: `mujoco`, `stable-baselines3`, `onnxruntime` installiert; das
Schwester-`sim_evaluator` am erwarteten Pfad (sonst `SIM_EVALUATOR_PATH` setzen).
ONNX-Export nutzt den Legacy-TorchScript-Exporter (`dynamo=False`) — der neue
Pfad bräuchte `onnxscript`.
