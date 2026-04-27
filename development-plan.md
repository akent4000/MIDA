# План разработки: MIDA — Medical Imaging Recognition Assistant

**Архитектура:** Headless, модульная, API-first
**Предмет:** Получение, обработка и анализ медицинских изображений

## Инфраструктура

| Узел | Железо | Роль |
|------|--------|------|
| Десктоп (dev) | Intel i7-10700 + RTX 3060 12GB VRAM | Обучение модели, разработка, локальный запуск стека |
| Сервер (prod) | Intel i3-4030U, 16 GB RAM, 44 GB свободно | Продакшен деплой, инференс через ONNX на CPU |

**Схема workflow:**
```
Десктоп                          Сервер
───────────────────────          ──────────────────────────
PyTorch + CUDA обучение  ──►    ONNX Runtime (CPU)
Jupyter EDA              ──►    FastAPI + фронтенд
docker-compose.dev.yml   ──►    docker-compose.prod.yml
git push / scp model.onnx──►    Caddy (уже установлен ✅)
```

---

## 1. Концепция архитектуры

### 1.0. Расширяемость: реестр ML-инструментов

MIDA спроектирована как **платформа** с подключаемыми ML-инструментами, а не как приложение под одну задачу. Первый инструмент — пневмония по рентгенограммам (RSNA). В будущем планируется добавить:

| Инструмент | Модальность | Задача |
|---|---|---|
| Пневмония (RSNA) ✅ | Chest X-Ray | Бинарная классификация |
| Опухоли мозга (BraTS) | МРТ | Сегментация |
| Дерматоскопия (ISIC) | Дерматоскопия | Классификация |
| Диабетическая ретинопатия | Фундус | Классификация |
| Патология лёгких (NIH) | Chest X-Ray | Многоклассовая классификация |

Для добавления нового инструмента достаточно **трёх шагов**:
1. Создать подкласс `MLTool` в `backend/app/modules/ml_tools/<tool_name>/tool.py`.
2. Реализовать `info`, `load`, `predict`, `get_preprocessing_config`.
3. Вызвать `registry.register_class(MyTool.TOOL_ID, MyTool)` в `build_registry()`.

API, DICOM-парсинг, пре- и постпроцессинг при этом **не изменяются**.

```
backend/app/modules/ml_tools/
├── base.py              # MLTool ABC + ToolResult иерархия (Classification/Segmentation/Detection)
├── registry.py          # ToolRegistry — регистрация, загрузка весов, диспетчеризация
├── __init__.py
├── pneumonia/           # Инструмент 1: RSNA Pneumonia Classifier ✅
│   └── tool.py
├── brain_tumor/         # Инструмент 2 (будущий): BraTS Segmentation
│   └── tool.py
└── dermoscopy/          # Инструмент 3 (будущий): ISIC Classification
    └── tool.py
```

### 1.1. Принципы

**Headless** означает, что ядро системы (бэкенд + ML) полностью отделено от представления. Взаимодействие только через HTTP/REST API. Это даёт:

- Возможность подключить любой клиент: веб, мобильное приложение, скрипт, PACS-систему
- Независимую разработку фронтенда и бэкенда
- Простое тестирование — каждый модуль проверяется изолированно
- Возможность замены любого компонента без переписывания остальных

**Модульность** означает, что система состоит из независимых слабосвязанных модулей, общающихся через чёткие контракты (интерфейсы). Каждый модуль имеет одну ответственность.

### 1.2. Общая схема

```
┌─────────────────────────────────────────────────────────────┐
│                   CLIENT LAYER (Frontend)                    │
│  React SPA + Cornerstone.js (DICOM Viewer) + TailwindCSS    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST + WebSocket
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      API GATEWAY                             │
│              FastAPI + Pydantic + OpenAPI                    │
│        Auth │ Rate Limit │ Validation │ Routing              │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌───────▼────────┐ ┌──────▼─────────┐
│  CORE MODULES  │ │  ML MODULES    │ │ STORAGE MODULES│
│                │ │                │ │                │
│ • DICOM Parser │ │ • Model Registry│ │ • File Store  │
│ • Preprocessor │ │ • Inference    │ │ • Metadata DB  │
│ • Postprocessor│ │ • Explainability│ │ • Cache       │
│ • Validator    │ │ • Batch Queue  │ │                │
└────────────────┘ └────────────────┘ └────────────────┘
                           │
                  ┌────────▼────────┐
                  │  WORKER LAYER   │
                  │ Celery + Redis  │
                  │ (async tasks)   │
                  └─────────────────┘
```

### 1.3. Клинический контекст и ограничения

Приложение разрабатывается как **исследовательский прототип для учебных целей**. Оно не является медицинским изделием и не может использоваться для постановки диагноза. Это должно быть явно указано в интерфейсе и документации.

---

## 2. Выбор задачи

**Рекомендуемая задача:** сегментация опухолей мозга на МРТ-снимках (датасет BraTS) или классификация пневмонии на рентгенограммах (датасет RSNA).

Далее план составлен универсально — применим к обеим задачам. Конкретный выбор фиксируется на этапе 1.

---

## 3. Технологический стек

### 3.1. Machine Learning (десктоп, RTX 3060)
- Python 3.11+
- PyTorch 2.x + CUDA 12.8 (cu128 wheels; план изначально указывал 12.1, но wheels под Python 3.13 есть только с cu124+)
- MONAI (специализированный фреймворк для медицинских изображений)
- pydicom, SimpleITK, nibabel (работа с медицинскими форматами)
- NumPy, OpenCV, scikit-image
- Captum (интерпретируемость моделей)
- Weights & Biases (трекинг экспериментов)
- ONNX + ONNX Runtime (экспорт для продакшена)
- onnxruntime-tools (квантизация INT8 перед деплоем)

### 3.2. Backend
- FastAPI (REST API + OpenAPI/Swagger)
- Pydantic v2 (валидация, сериализация)
- Uvicorn (ASGI-сервер)
- Celery + Redis (фоновые задачи, 1 воркер на сервере из-за i3-4030U)
- SQLModel + Alembic (ORM и миграции; объединяет Pydantic-схемы и SQLAlchemy-модели в одном классе)
- pytest (тестирование)

### 3.3. Frontend
- React 18 + TypeScript
- Vite (сборка)
- TanStack Query (работа с API)
- Zustand (state management)
- Cornerstone.js + OHIF-компоненты (DICOM-вьюер)
- TailwindCSS + shadcn/ui (стилизация)
- Vitest + Testing Library (тесты)

### 3.4. Storage
- PostgreSQL (метаданные)
- MinIO (S3-совместимое хранилище файлов)
- Redis (кэш и очередь задач)

### 3.5. DevOps
- Docker + docker-compose (раздельные конфиги dev/prod)
- GitHub Actions (CI/CD)
- Caddy (reverse proxy + автоматический SSL — уже установлен на сервере ✅)
- Ruff + Black + mypy (линтеры Python)
- ESLint + Prettier (линтеры JS/TS)
- scp / GitHub Actions для деплоя model.onnx на сервер

---

## 4. Структура проекта

```
medical-imaging-app/
├── backend/
│   ├── app/
│   │   ├── api/              # REST-эндпоинты (routers)
│   │   │   ├── v1/
│   │   │   │   ├── studies.py
│   │   │   │   ├── inference.py
│   │   │   │   └── models.py
│   │   ├── core/             # Конфиг, безопасность, зависимости
│   │   ├── modules/          # Бизнес-модули (headless core)
│   │   │   ├── dicom/        # Парсинг и валидация DICOM
│   │   │   ├── preprocessing/# Нормализация, ресайз, аугментации
│   │   │   ├── inference/    # Обёртка над моделью
│   │   │   ├── postprocessing/# Обработка масок, метрики
│   │   │   └── explainability/# Grad-CAM и другие методы
│   │   ├── models/           # SQLModel-модели (таблицы БД + Pydantic-схемы в одном классе)
│   │   ├── services/         # Оркестрация модулей
│   │   ├── workers/          # Celery-задачи
│   │   └── main.py
│   ├── ml/
│   │   ├── training/         # Скрипты обучения
│   │   ├── notebooks/        # Jupyter для экспериментов
│   │   ├── weights/          # Сохранённые веса
│   │   └── configs/          # YAML-конфиги обучения
│   ├── tests/
│   ├── alembic/              # Миграции БД
│   ├── pyproject.toml
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── api/              # Клиент API (генерируется из OpenAPI)
│   │   ├── components/
│   │   │   ├── viewer/       # DICOM-вьюер на Cornerstone
│   │   │   ├── upload/
│   │   │   └── ui/           # shadcn-компоненты
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── stores/           # Zustand stores
│   │   └── main.tsx
│   ├── package.json
│   └── Dockerfile
│
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml   # десктоп, GPU
│   ├── docker-compose.prod.yml  # сервер, CPU
│   ├── caddy/                   # Caddyfile конфиг
│   └── .env.example
│
├── docs/
│   ├── architecture.md
│   ├── api-contract.md
│   └── ml-model-card.md
│
├── .github/workflows/        # CI/CD
└── README.md
```

---

## 5. Модули ядра (headless core)

Каждый модуль имеет чёткий интерфейс и может быть заменён без изменения остальных.

### 5.1. DICOM Module
**Ответственность:** чтение, валидация и сериализация медицинских изображений.

Интерфейс:
```python
class DicomService:
    def load(self, file: bytes) -> Study
    def extract_metadata(self, study: Study) -> Metadata
    def to_numpy(self, study: Study) -> np.ndarray
    def anonymize(self, study: Study) -> Study
```

### 5.2. Preprocessing Module
**Ответственность:** подготовка изображения для подачи в модель.

Интерфейс:
```python
class PreprocessingPipeline:
    def apply(self, image: np.ndarray) -> torch.Tensor
```

Реализуется через паттерн Strategy — пайплайн конфигурируется из YAML.

### 5.3. Inference Module
**Ответственность:** загрузка модели и предсказание.

Интерфейс (абстрактный):
```python
class ModelInference(ABC):
    @abstractmethod
    def predict(self, tensor: torch.Tensor) -> Prediction
```

Реализации: `SegmentationInference`, `ClassificationInference` — легко добавляются новые.

Два бэкенда за одним интерфейсом:
```python
class PyTorchInference(ModelInference):
    # Десктоп — быстро, используется при разработке
    def predict(self, tensor: torch.Tensor) -> Prediction: ...

class OnnxInference(ModelInference):
    # Сервер — ONNX Runtime CPU, квантизация INT8
    def predict(self, tensor: torch.Tensor) -> Prediction: ...
```
Выбор бэкенда через переменную окружения `INFERENCE_BACKEND=onnx|pytorch`.

### 5.4. Postprocessing Module
**Ответственность:** превращение выхода модели в понятный результат (маска, bounding box, метки с вероятностями).

### 5.5. Explainability Module
**Ответственность:** генерация тепловых карт (Grad-CAM, attention maps) для интерпретации решений модели.

### 5.6. Storage Module
**Ответственность:** сохранение и извлечение файлов и метаданных. Абстракция поверх MinIO/файловой системы.

---

## 6. Контракт API (основные эндпоинты)

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/v1/studies` | Загрузка нового исследования (DICOM/NIfTI/PNG) |
| GET | `/api/v1/studies/{id}` | Получение метаданных исследования |
| GET | `/api/v1/studies/{id}/image` | Получение изображения (срез/вся серия) |
| POST | `/api/v1/studies/{id}/inference` | Запуск предсказания моделью |
| GET | `/api/v1/tasks/{task_id}` | Статус фоновой задачи |
| GET | `/api/v1/inference/{id}/result` | Результат предсказания (маска/метки) |
| GET | `/api/v1/inference/{id}/explanation` | Grad-CAM тепловая карта |
| GET | `/api/v1/models` | Список доступных моделей |
| WS | `/ws/tasks/{task_id}` | WebSocket для обновлений в реальном времени |

Полная спецификация автоматически генерируется FastAPI в формате OpenAPI 3.1 и доступна по `/docs`.

---

## 7. План разработки по фазам

### Фаза 0: Подготовка (3–5 дней)

- Выбор конкретной задачи и датасета (BraTS / RSNA / ISIC)
- Изучение датасета, написание EDA-ноутбука
- Создание Git-репозитория, настройка структуры, линтеров, pre-commit hooks
- Написание README с постановкой задачи
- Настройка окружения на десктопе: CUDA, PyTorch, проверка `nvidia-smi`
- Настройка `docker-compose.dev.yml` на десктопе (с GPU) и `docker-compose.prod.yml` для сервера (CPU)
- Очистка Docker на сервере (`docker builder prune`, `docker image prune -a`) — освобождает ~20 GB

**Результат:** пустой, но рабочий каркас проекта с поднятой инфраструктурой.

### Фаза 1: ML-модель (2–3 недели) — выполняется на десктопе (RTX 3060)

- Написание пайплайна загрузки данных на MONAI
- Baseline-модель (например, Light U-Net для сегментации)
- Обучение на RTX 3060 с mixed precision (torch.cuda.amp) — ускоряет в 1.5–2x
- Цикл обучения с логированием в W&B
- Валидация на отложенной выборке (метрики: Dice, IoU, Sensitivity, Specificity)
- Эксперименты: 2–3 архитектуры, сравнение, выбор лучшей
- Экспорт в ONNX: `torch.onnx.export(model, dummy, 'model.onnx')`
- Квантизация INT8 для ускорения CPU-инференса на сервере
- Написание model card (описание модели, ограничения, метрики)

**Результат:** обученная модель с документированными метриками и артефакт с весами.

### Фаза 2: Backend-ядро (2 недели) — **В РАБОТЕ**

Архитектура ядра строится вокруг **реестра ML-инструментов** (см. §1.0), что позволяет добавлять новые клинические задачи без изменения остальных слоёв.

Реализованные модули (`backend/app/modules/`):

| Модуль | Путь | Статус |
|---|---|---|
| ML Tools Registry | `ml_tools/` | ✅ готов |
| DICOM Service | `dicom/` | ✅ готов |
| Preprocessing Pipeline | `preprocessing/` | ✅ готов |
| Postprocessing Pipeline | `postprocessing/` | ✅ готов |
| Explainability (Grad-CAM) | `explainability/` | ✅ готов |
| Inference ABC + backends | `inference/` | ✅ (Phase 1) |

Пендинг:
- Интеграционный тест полного пайплайна (DICOM → preprocess → predict → postprocess → explain)
- CLI-утилита `python -m backend.app.cli` для headless smoke-теста
- Настройка structlog

**Результат:** работающее ядро, тестируемое без FastAPI.

### Фаза 3: REST API (1–2 недели)

- Настройка FastAPI, Pydantic-схем, зависимостей
- Реализация эндпоинтов согласно контракту
- Настройка SQLModel + Alembic, миграции
- Интеграция Celery + Redis для асинхронного инференса
- WebSocket-канал для статуса задач
- Автоматическая генерация OpenAPI-клиента для фронтенда
- API-тесты через httpx

**Результат:** задокументированный рабочий API, проверяемый через Swagger UI.

### Фаза 4: Frontend (2–3 недели)

- Настройка Vite + React + TypeScript
- Генерация API-клиента из OpenAPI-схемы (`openapi-typescript-codegen`)
- Страница загрузки исследования (drag-and-drop)
- Интеграция Cornerstone.js для просмотра DICOM (прокрутка срезов, window/level, зум)
- Визуализация результата: наложение маски с регулируемой прозрачностью
- Отображение Grad-CAM поверх изображения
- Панель метаданных пациента и параметров модели
- Индикатор статуса фоновой задачи через WebSocket
- Страница истории исследований

**Результат:** полноценный клиент, работающий поверх headless-ядра.

### Фаза 5: DevOps и развёртывание (1 неделя)

- Dockerfile для backend и frontend (multi-stage builds)
- `docker-compose.dev.yml` — десктоп, с секцией `deploy.resources` для GPU
- `docker-compose.prod.yml` — сервер, CPU-only, лимиты памяти на контейнеры
- Настройка Caddy на сервере для проксирования FastAPI и отдачи фронтенда (Caddy уже установлен ✅)
- GitHub Actions — три workflow:
  - **`ci.yml`** ✅ (готово) — на каждый push/PR в main: `ruff` → `mypy` → `pytest --cov`; torch устанавливается CPU-only (`whl/cpu`) отдельно от `pyproject.toml`
  - **`build.yml`** — на push в main после CI: сборка multi-stage Docker-образов backend + frontend, push в GHCR (`ghcr.io/<org>/mida-backend`, `mida-frontend`)
  - **`deploy-model.yml`** — на `git tag v*`: скачать `.pt` из MinIO → экспорт ONNX → INT8-квантизация → `scp model.onnx` на prod-сервер → SSH-перезапуск Celery-воркера
- Секреты GitHub Actions (Settings → Secrets → Actions): `SSH_PRIVATE_KEY`, `PROD_HOST`, `PROD_USER`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- Настройка Celery с `--concurrency=1 --pool=solo` под i3-4030U

**Результат:** приложение разворачивается одной командой `docker-compose up`.

### Фаза 6: Документация и защита (1 неделя)

- Написание итогового отчёта (pdf): постановка задачи, обзор литературы, архитектура, результаты экспериментов, метрики, выводы
- Диаграммы архитектуры (C4-модель или UML)
- Презентация для защиты
- Демонстрационное видео работы приложения (2–3 минуты)
- README с инструкцией запуска и скриншотами

**Результат:** готовый к защите проект.

---

## 8. Контроль качества

### 8.1. Тестирование
- **Unit-тесты:** каждый модуль ядра, покрытие ≥ 70%
- **Интеграционные тесты:** связки модулей
- **API-тесты:** все эндпоинты
- **E2E-тесты (опционально):** Playwright — сценарий «загрузил → получил результат»

### 8.2. Code Quality
- Обязательный code review по pull request
- Автоматические линтеры в CI: ruff, mypy, eslint
- Pre-commit hooks для локальных проверок
- Conventional Commits для истории изменений

### 8.3. Метрики ML-модели

Минимальные требования (корректируются в зависимости от задачи):
- Сегментация: Dice ≥ 0.80 на тестовой выборке
- Классификация: AUC ROC ≥ 0.90, Sensitivity ≥ 0.85

Модель должна оцениваться не только по средним метрикам, но и по стратифицированным — по возрасту, полу, типу патологии, если данные это позволяют.

---

## 9. Критерии готовности

Проект считается готовым, когда:

- Модель обучена, метрики задокументированы в model card
- Ядро работает независимо от API (проверяется через CLI)
- API соответствует OpenAPI-контракту и задокументирован
- Фронтенд позволяет загрузить исследование и получить результат с визуализацией
- Вся система разворачивается командой `docker-compose up`
- Покрытие тестами ядра не менее 70%
- Написан отчёт и подготовлена презентация

---

## 10. Риски и митигация

| Риск | Митигация |
|------|-----------|
| Медленный инференс на CPU сервера (i3-4030U) | ONNX Runtime + квантизация INT8 + асинхронный Celery — пользователь не ждёт блокировки |
| Большой объём DICOM-файлов | Чанковая загрузка, сжатие, ограничение количества срезов |
| Долгий инференс блокирует API | Асинхронные задачи через Celery + WebSocket-уведомления |
| Несогласованность API и фронтенда | Автогенерация TypeScript-клиента из OpenAPI |
| Переобучение модели | Кросс-валидация, регуляризация, аугментации, ранняя остановка |

---

## 11. Распределение ролей (если командный проект)

- **ML-инженер:** обучение модели, эксперименты, экспорт
- **Backend-разработчик:** ядро, API, инфраструктура
- **Frontend-разработчик:** клиент, DICOM-вьюер, UI/UX
- **DevOps/Tech Lead:** интеграция, CI/CD, развёртывание, документация

Если проект индивидуальный — фазы выполняются последовательно, при необходимости сокращаются (например, заменой React-фронтенда на Gradio).

---

## 12. Ориентировочный график (12 недель)

```
Недели 1      2      3      4      5      6      7      8      9     10     11     12
       ├──────┤
       Фаза 0
              ├────────────────────┤
              Фаза 1 (ML)
                                   ├─────────────┤
                                   Фаза 2 (Core)
                                                 ├─────────────┤
                                                 Фаза 3 (API)
                                                                ├────────────────────┤
                                                                Фаза 4 (Frontend)
                                                                              ├──────┤
                                                                              Фаза 5 (DevOps)
                                                                                     ├──────┤
                                                                                     Фаза 6
```

График адаптируется под реальный срок сдачи — фазы 2–4 можно запараллелить при командной работе.
