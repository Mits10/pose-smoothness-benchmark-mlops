# pose-smoothness-benchmark ( Vicon, Xsens, and GoPro/video-based )
Benchmarking human pose smoothness across **pipelines** with reproducible ML workflows, experiment tracking, and deployable inference APIs.

This project compares motion trajectories from:

- **Vicon** (optical motion capture – reference system)
- **Xsens** (IMU-based motion capture)
- **GoPro pose estimation** (vision-based keypoints)

## Overview

This project compares pose quality across multiple motion-capture and pose-estimation systems using **Vicon as the reference system**. It standardizes multimodal motion data, extracts smoothness and kinematic consistency features, trains ML models to estimate pose reliability, and exposes predictions through a production-style API.

---
## Project Goals :

This repository demonstrates how to build a **clean ML pipeline for multimodal motion data**:

- Parse raw motion capture formats
- Standardize data across sensors
- Validate data with **Pydantic schemas**
- Align and preprocess trajectories
- Extract motion smoothness features
- Compare systems against a reference (Vicon)
- Produce a benchmark dataset for machine learning
- reproducible data pipelines with **DVC**
- experiment tracking with **MLflow**
- feature engineering for multimodal time-series
- testable Python package structure
- model serving with **FastAPI**
- Docker + GitHub Actions for deployment readiness

---

## Core workflow

```text
Vicon / Xsens / GoPro pose data
  -> ingestion
  -> synchronization
  -> resampling
  -> coordinate normalization
  -> filtering
  -> feature extraction
  -> model training
  -> evaluation
  -> API inference
  ```

---

# Pipeline Architecture
```text
Raw Motion Data
   │
   ├── Xsens (.mvnx XML)
   ├── Vicon (.csv export)
   └── GoPro pose estimation (.csv)
   │
   ▼
Data Loaders (src/io)
   │
   ▼
Canonical PoseSequence Schema (Pydantic validation)
   │
   ▼
Preprocessing (src/preprocessing)
   ├── resampling
   ├── filtering
   └── temporal alignment
   │
   ▼
Feature Engineering (src/features)
   ├── jerk metrics
   ├── spectral arc length
   └── trajectory smoothness
   │
   ▼
Source Comparison Pipeline (src/pipelines/compare_sources.py)
   │
   ▼
Benchmark Dataset (data/processed/source_comparison.csv)
```
---

# Motion Features

### Jerk Metrics

Jerk is the third derivative of position.

Lower jerk → smoother motion.

jerk = d³x/dt³

Extracted features:

- mean jerk magnitude
- RMS jerk
- normalized jerk

### Spectral Smoothness

Spectral arc length (SPARC) measures smoothness in the frequency domain.

Advantages:

- robust to noise
- invariant to motion duration

### Reference Error Metrics

When comparing against Vicon:

- mean absolute error
- RMSE
- trajectory correlation
- mean spatial offset

---

# Repository Structure
```text
pose-smoothness-benchmark-mlops/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── mkdocs.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── train.yml
│       └── docker.yml
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── train.yaml
│   └── api.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   └── external/.gitkeep
├── docs/
│   ├── index.md
│   ├── architecture.md
│   ├── data-contract.md
│   ├── experiments.md
│   ├── api.md
│   └── results.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_signal_alignment.ipynb
│   └── 03_model_debug.ipynb
├── src/
│   ├── __init__.py
│   ├── io/
│   │   ├── load_vicon.py
│   │   ├── load_xsens.py
│   │   ├── load_gopro_pose.py
│   │   └── schemas.py
│   ├── preprocessing/
│   │   ├── sync.py
│   │   ├── resample.py
│   │   ├── normalize.py
│   │   ├── filters.py
│   │   └── qc.py
│   ├── features/
│   │   ├── smoothness.py
│   │   ├── kinematics.py
│   │   ├── segmentation.py
│   │   └── feature_builder.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── registry.py
│   │   └── inference.py
│   ├── pipelines/
│   │   ├── prepare_dataset.py
│   │   ├── build_features.py
│   │   ├── train_model.py
│   │   ├── batch_score.py
│   │   └── generate_report.py
│   ├── api/
│   │   ├── main.py
│   │   ├── models.py
│   │   └── service.py
│   └── utils/
│       ├── logger.py
│       └── paths.py
├── tests/
│   ├── test_sync.py
│   ├── test_resample.py
│   ├── test_smoothness.py
│   ├── test_feature_builder.py
│   └── test_api.py
├── reports/
│   ├── figures/.gitkeep
│   └── metrics/.gitkeep
└── models/
    └── .gitkeep
```
---

# Example Workflow

### Parse Xsens MVNX

from src.io.load_xsens import load_xsens_from_mvnx

seq = load_xsens_from_mvnx(
    "data/raw/trial01.mvnx",
    sequence_id="xsens_trial_01",
)

### Build feature dataset

python -m src.pipelines.build_features

### Compare motion sources

python -m src.pipelines.compare_sources

---

# Installation

python -m venv .venv
source .venv/bin/activate

pip install -e .[dev]

---

# Run Tests

pytest

---

# Tech Stack

Core:

- Python
- NumPy
- pandas
- SciPy

ML:

- scikit-learn
- PyTorch (future)

Engineering:

- FastAPI
- pytest
- Docker
- GitHub Actions

---

# Fast API Response Example
```text
{
  "sequence_id": "sub01_walk_001",
  "predicted_reliability": 0.92,
  "predicted_class": "reliable",
  "smoothness": {
    "jerk_mean": 0.19,
    "sparc": -1.72,
    "bone_length_var": 0.004
  },
  "alerts": []
}
```

# Author

Machine Learning / Motion Analysis project focused on building **reproducible pipelines for multimodal time-series data**.
