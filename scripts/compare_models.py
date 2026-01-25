#!/usr/bin/env python
"""
Model Comparison Script
Compare LSTM vs Transformer autoencoders side-by-side.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --output results/comparison.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.data import get_session_dataframe, prepare_training_data
from src.models import LSTMAutoencoder, TransformerAutoencoder
from src.training.metrics import compute_ttd
from src.utils import setup_logging, load_ground_truth, save_json


def load_model(model_path: Path, model_type: str, feature_dim: int):
    """Load saved model checkpoint."""
    if model_type == "lstm":
        model = LSTMAutoencoder(input_dim=feature_dim)
    else:
        model = TransformerAutoencoder(input_dim=feature_dim)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def compute_errors(model, sequences, masks, device):
    """Compute reconstruction errors."""
    model = model.to(device)
    model.eval()
    
    errors = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seq = torch.FloatTensor(sequences[i:i+batch_size]).to(device)
            batch_mask = torch.FloatTensor(masks[i:i+batch_size]).to(device)
            
            reconstructed = model(batch_seq, batch_mask)
            mse = ((batch_seq - reconstructed) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    
    return np.array(errors)


def evaluate_at_threshold(errors, labels, percentile=95):
    """Evaluate at a given threshold percentile."""
    threshold = np.percentile(errors, percentile)
    preds = (errors > threshold).astype(int)
    
    try:
        auc = roc_auc_score(labels, errors)
    except ValueError:
        auc = 0.5
    
    return {
        'auc_roc': float(auc),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1_score': float(f1_score(labels, preds, zero_division=0)),
        'threshold': float(threshold),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare LSTM vs Transformer models")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--answers-dir", type=Path, default=Path("data/raw/answers"))
    parser.add_argument("--output", type=Path, default=Path("results/model_comparison.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    setup_logging()
    
    print("=" * 70)
    print("MODEL COMPARISON: LSTM vs Transformer Autoencoder")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = get_session_dataframe(args.db_path)
    print(f"Sessions: {len(df):,}")
    
    # Load ground truth
    ground_truth = load_ground_truth(args.answers_dir, dataset='4.2')
    insider_users = ground_truth.get('insider_users', [])
    insider_incidents = ground_truth.get('insider_incidents', [])
    print(f"Insiders: {len(insider_users)} users, {len(insider_incidents)} incidents")
    
    # Prepare data
    print("\nPreparing test data...")
    data = prepare_training_data(
        df,
        sequence_length=100,
        insider_users=insider_users,
        insider_incidents=insider_incidents,
    )
    
    test_seq = data['test_sequences']
    test_masks = data['test_masks']
    labels_session = data['test_labels']
    labels_user = data['test_labels_user']
    feature_dim = data['feature_dim']
    
    print(f"Test set: {len(test_seq):,} sequences")
    print(f"  Session-level insiders: {labels_session.sum():,}")
    print(f"  User-level insiders: {labels_user.sum():,}")
    
    # Load models
    lstm_path = args.models_dir / "lstm_autoencoder_best.pt"
    transformer_path = args.models_dir / "transformer_autoencoder_best.pt"
    
    results = {'lstm': None, 'transformer': None}
    
    # Evaluate LSTM
    if lstm_path.exists():
        print("\n" + "-" * 70)
        print("Evaluating LSTM Autoencoder...")
        lstm = load_model(lstm_path, "lstm", feature_dim)
        lstm_errors = compute_errors(lstm, test_seq, test_masks, args.device)
        
        # Calculate TTD metrics
        # Use 95th percentile threshold from session-level metrics
        lstm_session = evaluate_at_threshold(lstm_errors, labels_session)
        lstm_thresh = lstm_session['threshold']
        lstm_preds = (lstm_errors > lstm_thresh).astype(int)
        
        lstm_ttd = compute_ttd(
            predictions=lstm_preds,
            test_labels=labels_session,
            test_session_ids=data['test_session_ids'],
            test_timestamps=data['test_timestamps'],
            insider_incidents=insider_incidents
        )
        
        results['lstm'] = {
            'session_level': lstm_session,
            'user_level': evaluate_at_threshold(lstm_errors, labels_user),
            'ttd': lstm_ttd
        }
    else:
        print(f"\nWarning: LSTM model not found at {lstm_path}")
    
    # Evaluate Transformer
    if transformer_path.exists():
        print("\n" + "-" * 70)
        print("Evaluating Transformer Autoencoder...")
        transformer = load_model(transformer_path, "transformer", feature_dim)
        transformer_errors = compute_errors(transformer, test_seq, test_masks, args.device)
        
        # Calculate TTD metrics
        trans_session = evaluate_at_threshold(transformer_errors, labels_session)
        trans_thresh = trans_session['threshold']
        trans_preds = (transformer_errors > trans_thresh).astype(int)
        
        trans_ttd = compute_ttd(
            predictions=trans_preds,
            test_labels=labels_session,
            test_session_ids=data['test_session_ids'],
            test_timestamps=data['test_timestamps'],
            insider_incidents=insider_incidents
        )
        
        results['transformer'] = {
            'session_level': trans_session,
            'user_level': evaluate_at_threshold(transformer_errors, labels_user),
            'ttd': trans_ttd
        }
    else:
        print(f"\nWarning: Transformer model not found at {transformer_path}")
    
    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    def print_comparison(level_name, label_key):
        print(f"\n{level_name}:")
        print(f"{'Metric':<20} {'LSTM':>15} {'Transformer':>15} {'Winner':>12}")
        print("-" * 70)
        
        metrics = ['auc_roc', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            lstm_val = results['lstm'][label_key][metric] if results['lstm'] else 0
            trans_val = results['transformer'][label_key][metric] if results['transformer'] else 0
            
            if lstm_val > trans_val:
                winner = "LSTM"
            elif trans_val > lstm_val:
                winner = "Transformer"
            else:
                winner = "Tie"
            
            print(f"{metric:<20} {lstm_val:>15.4f} {trans_val:>15.4f} {winner:>12}")
    
    if results['lstm'] or results['transformer']:
        print_comparison("Session-Level (Detecting Malicious Sessions)", "session_level")
        print_comparison("User-Level (Identifying Insider Users)", "user_level")
        
        print("\nTime-to-Detect (TTD):")
        print(f"{'Metric':<25} {'LSTM':>15} {'Transformer':>15} {'Winner':>12}")
        print("-" * 70)
        
        ttd_metrics = [
            ('n_incidents_detected', 'Detected Incidents', 'higher'),
            ('ttd_hours_mean', 'Mean TTD (Hours)', 'lower'),
            ('pct_first_session', '% First Session', 'higher'),
            ('ttd_lag_mean', 'Mean Lag (%)', 'lower')
        ]
        
        for key, name, direction in ttd_metrics:
            lstm_val = results['lstm']['ttd'][key] if results['lstm'] else 0
            trans_val = results['transformer']['ttd'][key] if results['transformer'] else 0
            
            if direction == 'higher':
                if lstm_val > trans_val: winner = "LSTM"
                elif trans_val > lstm_val: winner = "Transf"
                else: winner = "Tie"
            else:
                # Lower is better, but handle 0 (no detection case)
                if lstm_val == 0 and trans_val == 0: winner = "Tie"
                elif lstm_val == 0: winner = "Transf"  # LSTM detected nothing
                elif trans_val == 0: winner = "LSTM"   # Transf detected nothing
                elif lstm_val < trans_val: winner = "LSTM"
                elif trans_val < lstm_val: winner = "Transf"
                else: winner = "Tie"

            print(f"{name:<25} {lstm_val:>15.1f} {trans_val:>15.1f} {winner:>12}")
            
    # Overall winner
    print("\n" + "=" * 70)
    if results['lstm'] and results['transformer']:
        lstm_session_auc = results['lstm']['session_level']['auc_roc']
        trans_session_auc = results['transformer']['session_level']['auc_roc']
        
        if lstm_session_auc > trans_session_auc:
            print("OVERALL WINNER (Session-Level AUC): LSTM")
        elif trans_session_auc > lstm_session_auc:
            print("OVERALL WINNER (Session-Level AUC): Transformer")
        else:
            print("OVERALL: TIE")
    print("=" * 70)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, args.output)
    print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
