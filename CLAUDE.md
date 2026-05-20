# CLAUDE.md — Working notes for AI assistants in this repo

This file is read on every Claude Code / Claude API session that opens this
repository. Keep it concise; link out to longer docs.

## What this project is

A Windows-focused fork of *Time-Travel Rephotography* (Luo et al.,
SIGGRAPH Asia 2021). The research method projects an aligned face crop into
StyleGAN2 latent space to reconstruct a "modern photo" of a historical
subject. This fork wraps the research CLI with a PySide6 desktop GUI and
PowerShell pipeline scripts so non-researchers can run it end-to-end on a
Windows + NVIDIA workstation.

## High-level layout

| Path | Role |
| --- | --- |
| `projector.py` | Per-image research CLI (StyleGAN2 latent projection) |
| `projector_batch.py` | Batch version that reuses loaded models across faces |
| `gui/app.py` | PySide6 GUI (single 8.7k-LOC file — see "Architectural debt") |
| `run_rephoto_with_facecrop.ps1` | Windows wrapper: detect → GFPGAN → projector → blend |
| `run_rephoto_with_facecrop_batch.ps1` | Batch variant of the wrapper |
| `bootstrap_local_assets.ps1` | Downloads / copies model checkpoints |
| `build_gui_exe.ps1` | PyInstaller pack → `dist/TimeTravelRephotoGUI.exe` |
| `losses/` | Perceptual, contextual, color, reconstruction, noise reg |
| `optim/` | RAdam optimizer |
| `op/` | StyleGAN2 CUDA kernels (Windows falls back to pure-PyTorch) |
| `models/` | StyleGAN2 generator, encoder4editing, VGGFace, face detector |
| `tools/` | Initializer, face parsing, histogram matching, etc. |
| `preprocess/` | Face crops + GFPGAN runs (working dirs) |
| `tests/` | One file (`test_color_blend.py`) — see "Tests" |

Reference docs:
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) — verified install + run recipes.
- [docs/REPHOTO_PARAMETER_GUIDE.md](docs/REPHOTO_PARAMETER_GUIDE.md) — projector
  parameter tuning.
- [PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md) — March 2026 perf audit.
- [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) — fixes applied so far.

## Environment

- Python 3.8 + PyTorch 2.4.1 + CUDA 11.8 on Windows.
- Canonical env: `rephoto_cuda11_portable.yml` (conda).
- `requirements.txt` mirrors the same pins for pip-only installs; torch wheels
  must come from the PyTorch CUDA 11.8 index.
- No `.python-version`, no `pyproject.toml`. Conda is the source of truth.

## Architectural debt — read before refactoring

`gui/app.py` is currently 8,749 lines and contains one god-object
`MainWindow` with 142+ attributes and 98+ signal/slot connections. **Avoid
small refactors that touch many call sites here** — there are 83 references
to `self.advanced_dialog.<widget>` alone, and there is no GUI test coverage.
The audit roadmap calls for a module split (`widgets/`, `dialogs/`,
`controllers/`, `workers/`, `pipeline/`, etc.); until that lands, surgical
fixes are safer than design changes.

State-mutation patterns to be aware of:
- Pause/Cancel use **filesystem flag files** (not signals) to the subprocess.
  See `current_stop_flag_path` and `current_pause_flag_path`.
- Subprocess progress comes from **regex stdout parsing** in the GUI. Wrapper
  script changes can silently break GUI status updates.
- `closeEvent` (post-audit) terminates the main rephoto subprocess on window
  close — do not regress this.

## Performance constraints

- Generator checkpoint **must** be loaded with `map_location="cpu"` then
  `.to(device)` — loading directly to CUDA + `.to(device)` issues a
  redundant device-to-device copy (~1 s + ~1 GB peak VRAM).
- Contextual loss has a per-instance sampling-indices cache. To force the
  original per-forward resampling behavior, set
  `REPHOTO_DISABLE_CTX_CACHE=1`.
- Face-parsing masks are content-hash cached under `<mask_dir>/_content_cache`.
  To force a recompute (e.g., after debugging the face-parsing model itself),
  set `REPHOTO_PARSE_CACHE_DISABLE=1`.
- Per-face VRAM cleanup in `projector_batch.process_single_face` is in a
  `try/finally`; do not collapse it back into the success path or batches
  will fragment after ~5–10 faces.

## Tests & CI

- Only `tests/test_color_blend.py` exists. CI (`.github/workflows/ci.yml`)
  runs that one file on push. Heavy deps (torch, PySide6, dlib, tensorflow)
  are intentionally NOT installed in CI to keep it fast; future projector /
  GUI tests should live behind a self-hosted GPU runner.
- Locally: `python -m pytest tests/`.

## When you're asked to "fix" something

1. Check `PERFORMANCE_AUDIT.md` + `OPTIMIZATION_REPORT.md` first — many
   "obvious" fixes have already been applied or explicitly deferred.
2. For UI/GUI work, prefer narrow patches over restructuring `gui/app.py`.
3. For ML/numerical changes, verify SSIM is unchanged on a fixed input before
   declaring done. There is no automated regression harness yet.
4. Never `git rm --cached checkpoint/*` or otherwise touch the large model
   files in tracked storage without explicit user confirmation — users may
   depend on the shipped binaries.

## Commit style

Recent commits follow the form:
```
<imperative summary in 1 line, no period>

<optional body paragraph>
```
e.g. `Fix wide filmstrip header sizing and card clipping`.
