# Repository Guidelines

## Project Structure & Module Organization
- Core Python modules live at the repo root: `card_game.py`, `card_game_dynamic.py`, `web_game.py`, `web_wrangler.py`, `precomp.py`, `precomp_joint.py`, `sim_viz.py`, `sim_viz_2.py`, `fns.py`.
- UI assets: `stage_actions.html` and `assets/` (fonts). Streamlit config: `.streamlit/config.toml`.
- Batch/HPC helpers: `vc_sweep.sh`, `vc_sweep_dyn.sh` (PBS array jobs).
- Generated data lives in `output/` and `output_joint/` (created by scripts; not tracked).

## Build, Run, and Test
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Web game (local HTTP UI): `python web_game.py --signal-mode median --signal-cost 5`
- Streamlit viz: `streamlit run sim_viz.py` (or `sim_viz_2.py`)
- Precompute posteriors: `python precomp.py --seed 123 --rounds 200000 --out output/post_mc.npz --procs 8`
- Dynamic joint posteriors: `python precomp_joint.py --seed 123 --rounds 200000 --out output_joint/post_joint.npz --procs 8`
- Simulation (v7, single run): `python card_game.py --seed 123 --rounds 100000 --max_signals 9 --procs 8`

## Coding Style & Naming Conventions
- Python 3.10+; 4‑space indentation; keep lines ~100 chars.
- Naming: functions/vars `snake_case`; constants `UPPER_SNAKE` (e.g., `ACE_RANK`); files `snake_case.py`.
- Determinism: derive RNGs with `round_seed(base, r)`; avoid hidden randomness in pure functions.
- Formatting/lint: no enforced tool; keep consistent. If available, prefer `black` and `ruff` locally.

## Testing Guidelines
- Scripted tests (no pytest):
  - `python testing.py --test all`
  - `python testing_dynamic.py --test all`
- Each script header documents focused runs (e.g., posterior precompute, single‑round diagnostics). Ensure NPZ inputs (e.g., `output/post_mc.npz`) exist when required.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped titles (e.g., `web: gate stage‑2 invests`, `sim: add joint posterior check`).
- PRs: include what/why, sample commands, and screenshots/GIFs for UI changes. Link issues and note parameter defaults changed.

## Security & Configuration Tips
- Local server binds `127.0.0.1` and auto‑falls back to a free port. Avoid exposing externally.
- Keep artifacts out of Git: large `.npz` files in `output*/`. Respect paths and don’t write outside the repo.
