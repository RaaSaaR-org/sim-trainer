"""Test fixtures: put the repo root + the sibling sim_evaluator on sys.path so
`from config import …`, `from trainers …`, and `from envs.nav_wrappers …` all
resolve (sim_evaluator is a runtime path-dep, not an installed wheel)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import _default_sim_evaluator_path  # noqa: E402

_SIM_EVAL = _default_sim_evaluator_path()
if _SIM_EVAL not in sys.path:
    sys.path.insert(0, _SIM_EVAL)
