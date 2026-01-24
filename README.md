# Insider Threat Detection System - Phase 1

Session-level anomaly detection using LSTM and Transformer autoencoders on CERT R4.2 dataset.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download CERT R4.2 dataset
# Download from: https://doi.org/10.1184/R1/12841247.v1
# Extract to: data/raw/

# 3. Ingest data
python scripts/ingest_data.py

# 4. Train model
python scripts/train.py --model lstm --epochs 50

# 5. Evaluate
python scripts/evaluate_model.py

# 6. Run inference
python scripts/inference.py --session-id S12345
```

## Project Structure

```
├── config/config.yaml        # Hyperparameters
├── data/
│   ├── raw/                  # CERT R4.2 CSVs
│   ├── processed/            # Parquet + DuckDB
│   └── outputs/              # Session embeddings
├── src/
│   ├── data/                 # Data processing
│   ├── models/               # LSTM & Transformer
│   ├── training/             # Train & evaluate
│   └── utils/                # Helpers
├── models/                   # Saved checkpoints
├── results/                  # Evaluation reports
└── scripts/                  # CLI scripts
```

## Outputs

| File | Description |
|------|-------------|
| `models/lstm_autoencoder.pt` | Trained LSTM model |
| `models/transformer_autoencoder.pt` | Trained Transformer model |
| `data/outputs/session_outputs.jsonl` | Embeddings for all sessions |
| `results/evaluation_report.json` | AUC-ROC, Precision metrics |
