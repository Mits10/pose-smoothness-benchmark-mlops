# pose-smoothness-benchmark-mlops
Benchmarking human pose smoothness across **Vicon, Xsens, and GoPro/video-based pipelines** with reproducible ML workflows, experiment tracking, and deployable inference APIs.

## Overview

This project compares pose quality across multiple motion-capture and pose-estimation systems using **Vicon as the reference system**. It standardizes multimodal motion data, extracts smoothness and kinematic consistency features, trains ML models to estimate pose reliability, and exposes predictions through a production-style API.

## Project outlines :

- reproducible data pipelines with **DVC**
- experiment tracking with **MLflow**
- feature engineering for multimodal time-series
- testable Python package structure
- model serving with **FastAPI**
- Docker + GitHub Actions for deployment readiness

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

