# MIDA — ML workspace

This directory holds everything that runs **only on the dev box** (training,
experiments, notebooks, model export). The production server never imports
this code — it consumes the `model.onnx` artefact that lands in `weights/`.

## Layout

| Path          | Purpose                                                    |
| ------------- | ---------------------------------------------------------- |
| `training/`   | Training / eval scripts (Python modules, CLI-invocable).   |
| `notebooks/`  | Exploratory Jupyter notebooks (not used by tests).         |
| `configs/`    | Hydra / YAML configs for experiments.                      |
| `weights/`    | Checkpoints and exported ONNX models. **Gitignored.**      |
| `data/`       | Local dataset cache (RSNA Pneumonia, etc.). **Gitignored.**|

## One-time setup (dev box, Windows + RTX 3060)

The virtualenv at `.venv/` already has **torch 2.11.0+cu128** installed.
If you're provisioning from scratch:

```bash
python -m venv .venv
.venv\Scripts\activate
# CUDA 12.8 wheels (matches cu128 driver on the dev box)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Everything else
pip install -e ".[dev]"
pre-commit install
```

Sanity-check the stack:

```bash
python backend/ml/sanity_check.py
```

Expected: torch prints `cuda available: True` and a CUDA matmul runs on the
3060.

## Kaggle credentials

We use Kaggle's **new API Token** format (tokens with a `KGAT_` prefix) —
supported by `kaggle` CLI ≥ 1.8. Authentication is done via an environment
variable, not a `kaggle.json` file:

```bash
# Windows (persistent, user-level — takes effect in new shells):
setx KAGGLE_API_TOKEN "KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# POSIX:
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Generate a token at https://www.kaggle.com/settings → **API → Create New
API Token** (the dialog shows it **once** — copy it immediately). **Never
paste the token into any file in this repo.** `.gitignore` blocks
`kaggle.json`, `kaggle_api_token`, and `.kaggle/` as a safety net, but the
token belongs in the OS env, not on disk.

Smoke-test auth in a new shell:

```bash
kaggle competitions list -s pneumonia
```

Download the RSNA dataset:

```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge -p backend/ml/data/rsna
```

## Commands (wired up so far)

```bash
python backend/ml/sanity_check.py      # verify the ML stack
pytest                                  # run backend tests
ruff check .                            # lint
black .                                 # format
```

Everything else (training loop, ONNX export, INT8 quantization) lands in
Phase 1+ — see [development-plan.md](../../development-plan.md).
