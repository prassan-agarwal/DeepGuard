# 🔍 Deepfake Detector

A deep learning-based video deepfake detection system using a **Hybrid Multi-Branch Architecture** (Spatial + Temporal + Frequency analysis) with a FastAPI backend and Next.js frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14+-000000?logo=next.js)

---

## 🏗️ Architecture

The model uses a **three-branch hybrid architecture** that fuses features from multiple analysis domains:

```
Video Input → Frame Extraction → Face Detection (MTCNN)
                                        │
                    ┌───────────────────┬┴──────────────────┐
                    ▼                   ▼                   ▼
            Spatial Branch      Temporal Branch      Frequency Branch
           (EfficientNet)         (BiLSTM)              (FFT)
                    │                   │                   │
                    └───────────────────┴───────────────────┘
                                        │
                                  Feature Fusion
                                        │
                                  Classification
                                  (Real / Fake)
```

- **Spatial Branch**: EfficientNet-B0 backbone for per-frame visual artifact detection
- **Temporal Branch**: Bidirectional LSTM for inter-frame consistency analysis
- **Frequency Branch**: FFT-based analysis for detecting spectral anomalies

---

## 📁 Project Structure

```
Deepfake_Detector-Anti/
├── models/                 # Model architecture definitions
│   ├── hybrid_model.py     # Main hybrid model (Spatial + Temporal + Frequency)
│   ├── spatial_model.py    # EfficientNet-based spatial branch
│   ├── temporal_model.py   # BiLSTM temporal branch
│   └── frequency_model.py  # FFT-based frequency branch
├── preprocessing/          # Data preprocessing pipeline
│   ├── extract_frames.py   # Video → frame extraction
│   ├── face_detection.py   # MTCNN face detection & cropping
│   └── process_dataset.py  # Full dataset processing
├── training/
│   └── train.py            # Model training script
├── inference/              # Inference & evaluation
│   ├── detect_single_video.py  # Single video detection
│   ├── predict_video.py    # Video prediction pipeline
│   ├── evaluate.py         # Model evaluation & metrics
│   ├── export_onnx.py      # ONNX model export
│   └── gradcam.py          # GradCAM visualization
├── utils/
│   └── dataset_loader.py   # PyTorch dataset & dataloader
├── backend/                # FastAPI REST API
│   ├── main.py             # API server & endpoints
│   └── inference.py        # Backend inference logic
├── frontend/               # Next.js web interface
│   └── src/                # React components & pages
├── app/
│   └── app.py              # Streamlit/Gradio demo app
├── batch_predict.py        # Batch video prediction
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-compatible GPU (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Deepfake_Detector-Anti.git
cd Deepfake_Detector-Anti
```

### 2. Set Up Python Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. Train the Model

Place your dataset videos in `dataset/raw/real/` and `dataset/raw/fake/`, then run:

```bash
python -m preprocessing.process_dataset
python -m training.train
```

### 4. Run the Backend (FastAPI)

```bash
python -m backend.main
```

The API will be available at `http://localhost:8000`.

### 5. Run the Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## 📊 Evaluation Results

The model generates the following evaluation artifacts in `inference_results/`:

- **Confusion Matrix** — classification performance breakdown
- **ROC Curve** — receiver operating characteristic analysis
- **GradCAM Visualizations** — model attention heatmaps on real vs fake frames

---

## 🔌 API Reference

### `POST /api/detect`

Upload a video for deepfake detection.

| Parameter | Type       | Description                      |
|-----------|------------|----------------------------------|
| `video`   | `UploadFile` | Video file (`.mp4`, `.avi`, `.mov`) |

**Response:**
```json
{
  "success": true,
  "filename": "video.mp4",
  "is_fake": true,
  "fake_probability": 0.92,
  "confidence_percentage": "92.0%"
}
```

---

## 🛠️ Tech Stack

| Component       | Technology                     |
|-----------------|--------------------------------|
| Deep Learning   | PyTorch, EfficientNet, BiLSTM  |
| Face Detection  | MTCNN                          |
| Backend API     | FastAPI, Uvicorn               |
| Frontend        | Next.js, TypeScript, Tailwind  |
| Model Export    | ONNX                           |
| Visualization   | GradCAM, Matplotlib            |

---

## 📄 License

This project is for educational and research purposes.
