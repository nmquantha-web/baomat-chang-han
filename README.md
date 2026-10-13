# NetGuard IDS

A Flask-based intrusion detection prototype that combines Isolation Forest and LSTM for network traffic classification.

## Overview

This project loads a CICIDS-like CSV dataset, performs preprocessing, trains an anomaly detection module, and exposes a web dashboard and API for prediction.

## Structure

- `src/app.py` — Flask web application and REST API
- `src/predict.py` — inference pipeline and result summarization
- `src/preprocessing.py` — data cleaning, feature engineering, and sequence generation
- `src/lstm_model.py` — LSTM model definition and training logic
- `src/isolation_forest.py` — Isolation Forest model training and evaluation
- `src/templates/` — HTML UI templates

## Quick start

1. Create a virtual environment
2. Install dependencies
   ```bash
   pip install -r src/requirements.txt
   ```
3. Prepare the dataset in the expected folder structure
4. Run preprocessing and model training if needed
5. Start the app
   ```bash
   cd src
   python app.py
   ```

## Notes

- The project depends on a CICIDS-style dataset and trained model artifacts.
- The current repository intentionally excludes large runtime artifacts such as data files, logs, and trained models via `.gitignore`.
- For production deployment, set environment variables such as `SECRET_KEY` and `UPLOAD_FOLDER` explicitly.
