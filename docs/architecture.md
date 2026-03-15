# Architecture

## Overview

This project benchmarks pose smoothness across Vicon, Xsens, and GoPro/video-based pose pipelines using a reproducible ML workflow.

## System flow

```text
Raw pose data
  -> source-specific loaders
  -> canonical schema validation
  -> synchronization
  -> resampling
  -> coordinate normalization
  -> filtering
  -> smoothness feature extraction
  -> model training and evaluation
  -> FastAPI inference service