# FastVAR Agent Guide

This repo contains FastVAR, a cached-token-pruning acceleration method for
Visual Autoregressive (VAR) models. Two model backbones are included:
- Infinity (in `Infinity/`)
- HART (in `HART/`)

No Cursor or Copilot rule files are present (`.cursor/rules/`, `.cursorrules`,
`.github/copilot-instructions.md`).

-------------------------------------------------------------------------------
Repo Layout (what lives where)
-------------------------------------------------------------------------------
Root
- `README.md`: paper overview, high-level usage, and links to eval docs.
- `assets/`: figures used in the README.
- `Infinity/`: FastVAR + Infinity backbone implementation and eval scripts.
- `HART/`: FastVAR + HART backbone implementation, plus CUDA kernels.

Infinity subproject
- `Infinity/inference.py`: simple inference entrypoint (single image).
- `Infinity/tools/run_infinity.py`: main CLI for Infinity inference.
- `Infinity/infinity/models/fastvar_utils.py`: FastVAR cached-token pruning
  implementation (merge/unmerge/prune logic).
- `Infinity/infinity/models/fastvar_basic.py`: FastVAR-aware model blocks.
- `Infinity/infinity/models/infinity.py`: core Infinity architecture.
- `Infinity/evaluation/README.md`: eval benchmark instructions.
- `Infinity/scripts/`: shell scripts for inference/evaluation.

HART subproject
- `HART/inference.py`: inference entrypoint for HART.
- `HART/hart/modules/networks/fastvar_utils.py`: FastVAR pruning logic
  adapted for HART.
- `HART/hart/modules/networks/fastvar_basic.py`: FastVAR-aware network blocks.
- `HART/hart/modules/models/transformer/hart_transformer_t2i.py`: HART T2I
  transformer implementation.
- `HART/hart/kernels/`: custom CUDA extensions (fused kernels).
- `HART/evaluation/README.md`: eval benchmark instructions.

-------------------------------------------------------------------------------
Build / Install
-------------------------------------------------------------------------------
Python dependencies
- Infinity: `pip install -r Infinity/requirements.txt`
- HART: `pip install -e HART/`

HART CUDA kernels
- Build in-place: `python HART/hart/kernels/setup.py build_ext --inplace`
  (requires CUDA 11.0+; see `HART/hart/kernels/setup.py` for arch rules).

-------------------------------------------------------------------------------
Run / Test (evaluation is the main "test" surface)
-------------------------------------------------------------------------------
Quick smoke runs
- Infinity: `python Infinity/inference.py`
- HART: `python HART/inference.py --model_path /path/to/model \
  --text_model_path /path/to/Qwen2 --prompt "YOUR_PROMPT" \
  --sample_folder_dir /path/to/save_dir`

Infinity scripts
- Inference: `bash Infinity/scripts/infer.sh`
- Evaluation (includes GenEval/FID/HPS/ImageReward):
  `bash Infinity/scripts/eval.sh`
- GenEval only: `bash Infinity/scripts/geneval.sh`

Benchmark helpers (see `Infinity/evaluation/README.md`)
- ImageReward: `infer_eval_image_reward` (function in `Infinity/scripts/eval.sh`)
- HPSv2.1: `infer_eval_hpsv21` (function in `Infinity/scripts/eval.sh`)
- GenEval: `test_gen_eval` (function in `Infinity/scripts/eval.sh`)
- MJHQ30K: `python mjhq30k_fid_clip.py`

HART evaluation
- Same flow as Infinity per `HART/evaluation/README.md`.

Single test / targeted run
- There is no `pytest`-style unit test suite. Use the smoke runs above or a
  single benchmark command from the eval scripts.

-------------------------------------------------------------------------------
Lint / Format
-------------------------------------------------------------------------------
Formatting tools are defined in `HART/pyproject.toml`:
- Black: `black .`
- Isort (Black profile): `isort .`
- Pre-commit available (if installed): `pre-commit run --all-files`

-------------------------------------------------------------------------------
Code Style and Conventions
-------------------------------------------------------------------------------
Imports
- Standard library first, then third-party (torch/transformers/etc.), then
  local modules. This is mostly followed but not enforced everywhere.

Formatting
- 4-space indentation.
- Black style is expected in the HART subproject; keep lines reasonably short.
- Use double quotes or single quotes as already present in the file you edit.

Types
- Type hints are used in many files but not everywhere.
- Add hints for new public functions or complex return types when practical.

Naming
- Functions and variables: `snake_case`.
- Classes: `CamelCase`.
- Constants: `UPPER_SNAKE_CASE`.

Error handling
- Prefer explicit `ValueError`/`RuntimeError` for invalid inputs.
- Use `assert` for internal invariants when consistent with surrounding code.
- Avoid silent failure; print or raise with actionable messages.

Torch usage
- Many code paths assume CUDA; guard with `torch.cuda.is_available()` only
  when adding new behavior that might run on CPU.
- Inference paths frequently use `torch.inference_mode()` / `torch.no_grad()`
  and `torch.autocast` for bf16/float16. Follow existing patterns when editing.

Performance
- Be careful with memory; utilities frequently clear cache or measure peak
  memory. Avoid extra copies unless required.

Paths and assets
- Scripts often assume local weight paths (see `Infinity/scripts/*.sh` and
  `HART/inference.py`). Update paths via CLI args rather than hardcoding.

-------------------------------------------------------------------------------
When editing FastVAR logic
-------------------------------------------------------------------------------
- Core pruning logic lives in:
  - `Infinity/infinity/models/fastvar_utils.py`
  - `HART/hart/modules/networks/fastvar_utils.py`
- Model block changes typically go into `fastvar_basic.py` or model-specific
  transformer code. Keep pruning merge/unmerge signatures consistent with
  existing call sites.

-------------------------------------------------------------------------------
Notes for agentic edits
-------------------------------------------------------------------------------
- Avoid large refactors while fixing bugs; keep changes localized.
- Respect model-specific differences between Infinity and HART.
- Keep GPU-heavy evaluation commands out of default workflows unless asked.
