# MIDA — Medical Imaging Recognition Assistant

Headless, API-first platform for medical imaging analysis: DICOM study → ML inference → classification + Grad-CAM explanation.

> **⚠️ Disclaimer.** MIDA is a research/educational prototype. It is **not** a medical device, is not certified as diagnostic software, and must not be used in clinical practice. All results are an auxiliary signal for learning and research only.

---

## Features

- Accept DICOM/PNG/NIfTI studies via REST API.
- Anonymize PHI and extract metadata.
- Run ML inference asynchronously through Celery + Redis with WebSocket status updates.
- Return predictions (label + probability) and a Grad-CAM heatmap.
- Persist study history in PostgreSQL and files in MinIO (S3-compatible storage).
- Display everything in a React viewer (Cornerstone.js) with slice scrolling, heatmap overlay and metadata browsing.

The first ML tool targets **RSNA Pneumonia Detection** (binary classification on chest X-rays). The architecture is a 5-fold DenseNet-121 (384×384) ensemble + TTA. Best result on the 4,003-patient held-out test set: **AUC 0.8927**, sensitivity 0.788, specificity 0.826 (Youden threshold 0.4396).

---

## Architecture

```
React SPA (Vite + TS + Cornerstone.js)
        │ HTTP/REST + WebSocket
        ▼
FastAPI (Pydantic v2 + OpenAPI 3.1)
        │
        ├── Core modules (DICOM / Preprocess / Postprocess / Explainability)
        ├── ML Tools Registry  ── PneumoniaTool
        ├── Inference ABC
        │     ├── PyTorchInference (dev, GPU)
        │     └── OnnxInference   (prod, CPU + INT8)
        │
        ├── Celery worker (--concurrency=1 --pool=solo in prod)
        ├── PostgreSQL (metadata)
        ├── Redis (queue + cache)
        └── MinIO (DICOM files and model weights)
```

The full architecture and phase plan live in [development-plan.md](development-plan.md).

### Headless core

Business logic lives only in `backend/app/modules/`. The API layer only orchestrates modules. Every module is invocable from the CLI without FastAPI running.

### ML tool registry

Adding a new clinical task takes three steps:
1. Create an `MLTool` subclass in `backend/app/modules/ml_tools/<tool>/tool.py`.
2. Implement `info`, `load`, `predict`, `get_preprocessing_config`.
3. Register it in `registry.build_registry()`.

The API, DICOM parsing, and pre-/post-processing layers stay untouched.

### Two-environment topology

| Environment | Hardware | Role |
|---|---|---|
| **Desktop (dev)** | i7-10700 + RTX 3060 12GB | Model training (PyTorch + CUDA), local full-stack |
| **Server (prod)** | i3-4030U, 16 GB RAM | Inference via ONNX Runtime, CPU-only |

Artifacts flow one way: train → export ONNX → INT8 quantize → ship to the server. No training happens on the server.

---

## Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic v2, SQLModel + Alembic, Celery + Redis.
- **ML:** PyTorch 2.x + CUDA 12.8 (dev only), MONAI, pydicom, SimpleITK, Captum, ONNX + ONNX Runtime + INT8 quantization.
- **Frontend:** React 18 + TypeScript, Vite, TanStack Query, Zustand, Cornerstone.js, TailwindCSS + shadcn/ui.
- **Storage:** PostgreSQL, MinIO, Redis.
- **DevOps:** Docker + docker-compose (separate dev/prod), Caddy (reverse proxy + SSL), GitHub Actions.

---

## Quick start (dev)

```bash
# 1. Install torch separately (CUDA-specific wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install the rest of the dependencies
pip install -e ".[dev,train]"
pre-commit install

# 3. Copy the env template and fill it in
cp .env.example .env

# 4. Bring up infrastructure (Postgres + Redis + MinIO)
docker compose -f infra/docker-compose.dev.yml up -d

# 5. Apply migrations
alembic upgrade head

# 6. Run the API + worker + frontend
uvicorn backend.app.main:app --reload                    # API on :8000
celery -A backend.app.workers.celery_app worker -l info  # Celery
cd frontend && npm install && npm run dev                # Vite on :5173
```

Swagger UI is available at `http://localhost:8000/docs`.

### Tests and linting

```bash
ruff check .
black .
mypy backend/
pytest                                       # full suite
pytest --cov=backend --cov-report=term-missing
```

### ML training

```bash
# Baseline ResNet50
python -m backend.ml.training.train_baseline --config backend/ml/configs/baseline_resnet50.yaml

# NIH pretrain (DenseNet-121 multi-label)
python -m backend.ml.training.pretrain_nih --config backend/ml/configs/pretrain_nih_densenet121.yaml

# 5-fold CV on RSNA
python -m backend.ml.training.cv_train --config backend/ml/configs/cv_rsna_densenet121_384.yaml

# Ensemble evaluation
python -m backend.ml.training.ensemble_eval --checkpoints fold1.pt fold2.pt fold3.pt fold4.pt fold5.pt --tta
```

Datasets, splits and MinIO artifacts are documented in [CLAUDE.md](CLAUDE.md).

---

## API contract

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/studies` | Upload a study |
| GET | `/api/v1/studies/{id}` | Study metadata |
| GET | `/api/v1/studies/{id}/image` | Image preview |
| POST | `/api/v1/studies/{id}/inference` | Start inference (async) |
| GET | `/api/v1/tasks/{task_id}` | Task status |
| GET | `/api/v1/inference/{id}/result` | Inference result |
| GET | `/api/v1/inference/{id}/explanation` | Grad-CAM PNG |
| GET | `/api/v1/models` | List available models |
| GET / PATCH | `/api/v1/tools/{tool_id}/config` | Tool parameters (threshold etc.) |
| WS | `/ws/tasks/{task_id}` | Task status stream |

OpenAPI 3.1 is auto-generated by FastAPI and served at `/docs` and `/openapi.json`. The frontend TS client is generated from that schema (`openapi-typescript-codegen`).

---

## Repository layout

```
backend/
  app/
    api/v1/         # FastAPI routers (studies, inference, tasks, models, tool_settings)
    api/ws/         # WebSocket channels
    core/           # Settings, DB, security
    models/         # SQLModel (study, inference_result, tool_setting)
    modules/        # Headless core (see below)
    services/       # Module orchestration
    workers/        # Celery
    main.py
  modules/
    ml_tools/       # Registry + PneumoniaTool
    dicom/          # Loading / anonymization
    preprocessing/  # CLAHE → resize → normalize (torch-free)
    postprocessing/ # Interpretation + confidence band
    explainability/ # Grad-CAM
    inference/      # ABC + PyTorch + ONNX backends
  ml/
    training/       # Training and evaluation scripts
    configs/        # YAML + splits_rsna_v1.json
    notebooks/      # EDA
  tests/            # pytest, 25+ files
  alembic/          # DB migrations
frontend/
  src/
    pages/          # Upload, Study, History, Settings
    components/     # StudyViewer (Cornerstone), InferencePanel, MetadataPanel, ResultCard
    api/            # Generated TS client
infra/
  docker-compose.dev.yml      # GPU + full stack
  docker-compose.prod.yml     # CPU-only, memory limits
  docker-compose.minio.yml    # MinIO only
  caddy/                      # Caddyfile
  docker/                     # Backend + frontend Dockerfiles
.github/workflows/  # ci.yml, build.yml, deploy.yml, deploy-model.yml
```

---

## Deployment (prod)

CI/CD pipeline: `ci.yml` → `build.yml` → `deploy.yml`.

1. **CI** — on every push/PR to `main`: ruff → mypy → pytest. Torch is installed CPU-only.
2. **Build** — on push to `main` after CI passes: multi-stage Docker images (`mida-backend`, `mida-frontend`) pushed to GHCR.
3. **Deploy** — after a successful build: SSH to the prod server, `docker compose pull` + `docker compose up -d` in `/opt/MIDA/infra/`.
4. **Deploy model** — on `git tag v*`: pull `.pt` from MinIO → export ONNX → INT8-quantize → `scp` `model.onnx` to prod → restart the Celery worker.

Caddy on prod handles automatic SSL and proxies the API plus the frontend's static files. The target domain is `mida.akent.site`.

Required GitHub Actions secrets: `SSH_PRIVATE_KEY`, `PROD_HOST`, `PROD_SSH_PORT`, `PROD_USER`, `PROD_DEPLOY_PATH`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`.

---

## Current status

| Phase | Description | Status |
|---|---|---|
| 0 | Bootstrap (repo, linters, environment) | ✅ |
| 1 | ML model (RSNA, AUC 0.8927) | ✅ |
| 1b | CheXpert (14 pathologies) | ⏳ in progress |
| 2 | Backend core (modules/) | ✅ |
| 3 | REST API + Celery + WebSocket | ✅ |
| 4 | Frontend (React + Cornerstone) | ✅ |
| 5 | DevOps + CI/CD + Caddy | ✅ |
| 6 | Report, presentation, demo | ⏳ in progress |

ML quality (plan §8): target AUC ≥ 0.90 — currently 0.8927. Core test coverage ≥ 70% — met.

---

## License

MIT (see `pyproject.toml`).
