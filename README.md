# Dinov2-BigEarthS2

**Multi-label satellite image classification** with a DINOv2-B backbone (86M params) on **BigEarthNet-S2** (Sentinel-2, 43 classes). Three components:

1. **Kaggle training notebook** — phased fine-tuning, checkpointing, curves.
2. **FastAPI backend** — inference server with comprehensive logging.
3. **React + TypeScript frontend** — upload an image, get predictions.

Multi-label only (no segmentation). RGB bands (B04/B03/B02). Local deployment.

---

## Repository layout

```
Dinov2-BigEarthS2/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point (lifespan, CORS, router)
│   │   ├── config.py            # Settings + shared CLASSES_43 list
│   │   ├── api/routes.py        # /predict, /health, /classes
│   │   ├── models/dinov2.py     # MultiLabelDinoV2 + checkpoint loader
│   │   ├── services/predictor.py
│   │   ├── services/logger.py
│   │   └── utils/image_processing.py
│   ├── checkpoints/             # <- place model_best.pth here
│   ├── logs/                    # server log files written here
│   ├── requirements.txt
│   └── Dockerfile
├── kaggle/
│   ├── train.py                # single-file training script — just run it
│   └── train_requirements.txt
├── src/                         # React frontend (repo root)
│   ├── App.tsx
│   ├── components/              # ImageUploader, PredictionResults, ConfidenceBar
│   ├── services/api.ts
│   └── types/index.ts           # shared types + CLASS_NAMES
├── package.json, vite.config.ts, tsconfig*.json
└── README.md
```

> The 43-class nomenclature is the **single source of truth** and is duplicated in three places that must stay in sync: `backend/app/config.py` (`CLASSES_43`), `kaggle/train.py` (`CLASSES_43`), and `src/types/index.ts` (`CLASS_NAMES`).

---

## Part 1 — Training

A single Python script — just run it.

```bash
cd kaggle
pip install -r train_requirements.txt
python train.py
```

Works on Kaggle (GPU P100 or T4 ×2, Internet ON) or any machine with CUDA. The script:

- Streams BigEarthNet-S2 via HuggingFace `datasets` (no 66 GB download).
- Extracts RGB bands (B04, B03, B02) only — preserves DINOv2's ImageNet pretraining.
- 3-phase fine-tuning: **head** (5 ep, lr 1e-3) → **last 2 blocks** (10 ep, lr 1e-4) → **full** (≤15 ep, lr 1e-5).
- `BCEWithLogitsLoss`, AdamW (wd 0.01), `ReduceLROnPlateau` (factor 0.5, patience 3), early stopping (patience 7, min_delta 0.001).
- FP16 mixed precision, `DataParallel` on T4×2, gradient accumulation.
- Saves `model_epoch_{n}.pth` every epoch + `model_best.pth` on val F1-micro improvement.
- Training curves (2×2 grid) every 5 epochs, TensorBoard logs, and `training_history.json`.

Outputs land in `./checkpoints/`, `./training_curves/`, and `./logs/` (on Kaggle these are under `/kaggle/working/`). **Copy `model_best.pth`** into `backend/checkpoints/`.

---

## Part 2 — Backend (FastAPI)

### Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# Place the trained model:
#   model_best.pth  ->  backend/checkpoints/model_best.pth

python -m app.main              # or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at **http://localhost:8000**, Swagger UI at **http://localhost:8000/docs**.

### Endpoints

| Method | Path                       | Description                          |
|--------|----------------------------|--------------------------------------|
| POST   | `/api/v1/predict`          | Upload an image → multi-label preds  |
| GET    | `/api/v1/health`           | Server/model status                  |
| GET    | `/api/v1/classes`          | The 43 class names                   |
| GET    | `/`                        | Info + doc links                     |

### Logging

Every operation is logged to both the console (INFO) and a timestamped file under `backend/logs/` (DEBUG): server startup, model load (param count, device, arch), each request (id, filename, size, content-type), every preprocessing step (original size/mode/pixel range → tensor shape/normalized range), inference details (raw logits, sigmoid probs, threshold, detected classes, time), and full stack traces on errors.

### Docker (optional)

```bash
docker build -t dinov2-bigearths2-backend ./backend
# CPU-only:
docker run -p 8000:8000 -v $(pwd)/backend/checkpoints:/app/checkpoints dinov2-bigearths2-backend
# With GPU:
docker run --gpus all -p 8000:8000 -v $(pwd)/backend/checkpoints:/app/checkpoints dinov2-bigearths2-backend
```

---

## Part 3 — Frontend (React + TS + Vite)

```bash
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies `/api` → `http://localhost:8000` (see `vite.config.ts`). For a production build pointed elsewhere, set `VITE_API_BASE` (see `.env.example`).

UI: drag-and-drop or browse an image → preview → **Classify** → detected classes with green confidence bars (sorted, above 0.5 threshold), a metadata strip (model, inference ms, threshold, request id), and an expandable view of **all 43 class probabilities**. Dark theme, responsive.

```bash
npm run build      # typecheck + production build -> dist/
npm run preview    # serve the production build
```

---

## End-to-end test

1. Start the backend (with `model_best.pth` in place).
2. Start the frontend (`npm run dev`).
3. Open http://localhost:5173, upload a Sentinel-2 RGB image, click **Classify**.
4. Verify predictions render with confidence bars.
5. Watch the backend terminal — every preprocessing and inference step is logged.
6. Try the `/predict` endpoint directly via the Swagger UI at `/docs` or:

```bash
curl -X POST http://localhost:8000/api/v1/predict -F "file=@test.jpg"
```

---

## Model architecture

```
Input (1, 3, 224, 224)
   │
   ├── DINOv2-B backbone  (timm: vit_base_patch14_dinov2.lvd142m, 768-d, ~86M params)
   │       └── num_classes=0  → pooled 768-d feature
   │
   └── Head:
           Linear(768 → 512) → ReLU → Dropout(0.3) → Linear(512 → 43)
   │
   └── raw logits (1, 43)   # sigmoid applied at inference / BCEWithLogitsLoss at training
```

No sigmoid inside the model — probabilities come from `torch.sigmoid` at inference, and `BCEWithLogitsLoss` applies it numerically-stably during training. Identical architecture across training, backend loader, and checkpoint format.

---

## Configuration overrides

Backend settings can be overridden via environment variables (see `backend/app/config.py`):

| Var                   | Default                                   | Notes                          |
|-----------------------|-------------------------------------------|--------------------------------|
| `MODEL_CHECKPOINT`    | `backend/checkpoints/model_best.pth`      | Path to trained weights        |
| `DEVICE`              | `auto` (cuda if available, else cpu)      | Force a device                 |
| `INFERENCE_THRESHOLD` | `0.5`                                     | Class-detection cutoff         |
| `CORS_ORIGINS`        | `["*"]`                                   | Restrict in production         |
