#!/usr/bin/env python
"""
Explain Model Predictions using SHAP (Shapley Additive exPlanations)
--------------------------------------------------------------------
This script explains WHY a specific session was flagged as anomalous.
It uses SHAP to attribute the reconstruction error (Anomaly Score) back to 
specific input features (e.g., "High number of USB connections").

Methodology:
    We use GradientExplainer (or KernelExplainer) to approximate Shapley values.
    Target Function: Reconstruction Error (MSE) of the session.
    
    We want to answer: "Which features contributed to increasing the error?"
"""

import argparse
import sys
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shap

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import FeatureExtractor, get_session_dataframe, prepare_training_data
from src.models import LSTMAutoencoder, TransformerAutoencoder, create_lstm_autoencoder, create_transformer_autoencoder
from src.utils import setup_logging, load_config

logger = logging.getLogger(__name__)

class ModelWrapper(nn.Module):
    """
    Wraps Autoencoder to output a single scalar (MSE) per sample 
    so SHAP can explain it.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        # output: [batch] (MSE loss)
        
        # We assume no mask for SHAP (treat all as valid or use fixed length)
        # GradientExplainer passes a tensor directly
        seq_len = x.size(1)
        mask = torch.ones(x.size(0), seq_len).to(x.device)
        
        reconstructed, _ = self.model(x, mask)
        
        # Compute MSE per sample (sum over time and features)
        # We explain "Total Error" or "Mean Error"
        # Using Mean Error so scale is consistent
        loss = ((x - reconstructed) ** 2).mean(dim=(1, 2))
        
        # Returns [batch, 1] for SHAP
        return loss.unsqueeze(1)

def main():
    parser = argparse.ArgumentParser(description="Explain model predictions with SHAP")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm", help="Model type")
    parser.add_argument("--model-path", type=Path, help="Path to model checkpoint")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("results/shap_summary.png"))
    parser.add_argument("--num-background", type=int, default=100, help="Number of background samples (normal)")
    parser.add_argument("--num-test", type=int, default=50, help="Number of anomalous samples to explain")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--answers-dir", type=Path, default=Path("data/raw/answers"), help="Directory with ground truth labels")
    args = parser.parse_args()
    
    setup_logging()
    
    # 1. Load Data
    logger.info("Loading session data...")
    if not args.db_path.exists():
        logger.error(f"Database not found: {args.db_path}")
        return
        
    df = get_session_dataframe(args.db_path)
    
    # Load ground truth if available
    insider_users = []
    insider_incidents = []
    if args.answers_dir.exists():
        logger.info(f"Loading ground truth from {args.answers_dir}")
        # Need to import locally or ensure it's imported at top
        from src.utils import load_ground_truth
        ground_truth = load_ground_truth(args.answers_dir, dataset="4.2")
        insider_users = ground_truth.get('insider_users', [])
        insider_incidents = ground_truth.get('insider_incidents', [])
        logger.info(f"Found {len(insider_users)} insider users")
    
    # 2. Prepare Data
    # We need a set of normal sessions (reference) and anomalous sessions (to explain)
    # We'll rely on our standard prep function for simplicity, even though it splits by time
    data = prepare_training_data(
        df, 
        sequence_length=100, 
        train_ratio=0.7, 
        val_ratio=0.1, 
        test_ratio=0.2,
        insider_users=insider_users,
        insider_incidents=insider_incidents,
    )
    
    feature_extractor = data['feature_extractor']
    feature_names = feature_extractor.get_feature_names()
    
    # 3. Load Model
    device = torch.device(args.device)
    feature_dim = data['feature_dim']
    
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = Path("models") / f"{args.model}_autoencoder_best.pt"
        
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading model from {model_path}...")
    if args.model == "lstm":
        model = create_lstm_autoencoder({"feature_dim": feature_dim})
    else:
        model = create_transformer_autoencoder({"feature_dim": feature_dim})
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 4. Select Samples
    # Background: Random Normal Samples from Train Set
    train_seqs = data['train_sequences']
    background_indices = np.random.choice(len(train_seqs), args.num_background, replace=False)
    background_data = torch.FloatTensor(train_seqs[background_indices]).to(device)
    
    # Test: Anomalous Samples from Test Set (where label=1)
    test_seqs = data['test_sequences']
    test_labels = data['test_labels']
    
    # Filter for known insiders (label=1)
    anomaly_indices = np.where(test_labels == 1)[0]
    
    if len(anomaly_indices) == 0:
        logger.warning("No anomalies found in test set labels. Using random test samples instead.")
        anomaly_indices = np.random.choice(len(test_seqs), args.num_test, replace=False)
    elif len(anomaly_indices) > args.num_test:
        anomaly_indices = np.random.choice(anomaly_indices, args.num_test, replace=False)
        
    test_data = torch.FloatTensor(test_seqs[anomaly_indices]).to(device)
    
    logger.info(f"Explaining {len(test_data)} anomalous sessions using {len(background_data)} background samples.")
    
    # 5. Run SHAP
    # We wrap the model to output a single scalar (Reconstruction Error) per sample
    model_wrapper = ModelWrapper(model).to(device)
    
    # GradientExplainer is suitable for PyTorch models
    # It explains "What input features change the output (Error) the most?"
    explainer = shap.GradientExplainer(model_wrapper, background_data)
    
    # Compute Shapley values
    # shap_values will be list of tensors (one per output). We have 1 output (Loss).
    # Shape of element: [batch, seq_len, features]
    logger.info("Computing SHAP values (this may take a moment)...")
    shap_values = explainer.shap_values(test_data)
    
    # shap_values is a list for GradientExplainer? No, usually array if 1 output, or list of arrays.
    # For PyTorch it typically returns a list of tensors/arrays, one for each output class/dim.
    # Since we unsqueezed(1), we have 1 output dimension.
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0] # Take the explanation for the Error
        
    # SHAP values are [batch, seq_len, features]
    # To get global feature importance, we sum magnitude over time and then mean over batch
    
    # 1. Sum over time dimension (Total contribution of feature F in session)
    # Shape: [batch, features]
    shap_sum_time = np.sum(np.abs(shap_values), axis=1)
    
    # 2. Mean over batch (Global importance for these anomalies)
    # Shape: [features]
    global_importance = np.mean(shap_sum_time, axis=0).flatten()
    
    # 6. Visualization
    # Sort features by importance
    indices = np.argsort(global_importance)[::-1]
    top_n = 20
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), global_importance[top_indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel("mean(|SHAP value|) - Average Impact on Anomaly Score")
    plt.title(f"Top {top_n} Features driving Anomaly Scores ({args.model.upper()})")
    plt.gca().invert_yaxis()  # Best feature at top
    plt.tight_layout()
    
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Saved SHAP summary plot to {output_path}")
    
    # Save text report
    report_path = output_path.with_suffix('.txt')
    with open(report_path, "w") as f:
        f.write(f"SHAP Feature Importance ({args.model.upper()})\n")
        f.write("========================================\n\n")
        for i in top_indices:
            name = feature_names[i]
            score = global_importance[i]
            f.write(f"{name:<30}: {score:.6f}\n")
            
    logger.info(f"Saved importance text report to {report_path}")

if __name__ == "__main__":
    main()
