#!/usr/bin/env python
"""
Model Evaluation Script
Evaluate trained model with dual-level metrics (session + user level).

Usage:
    python scripts/evaluate_model.py --model lstm
    python scripts/evaluate_model.py --model transformer --output results/evaluation.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, precision_recall_curve,
    roc_curve, average_precision_score
)

from src.data import get_session_dataframe, prepare_training_data
from src.models import LSTMAutoencoder, TransformerAutoencoder
from src.training import ModelEvaluator
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


def compute_reconstruction_errors(model, sequences, masks, device="cuda"):
    """Compute reconstruction errors for all sequences."""
    model = model.to(device)
    model.eval()
    
    errors = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seq = torch.FloatTensor(sequences[i:i+batch_size]).to(device)
            batch_mask = torch.FloatTensor(masks[i:i+batch_size]).to(device)
            
            # Model returns (reconstructed, embedding)
            output = model(batch_seq, batch_mask)
            if isinstance(output, tuple):
                reconstructed, embedding = output
            else:
                reconstructed = output
                embedding = None
            
            # MSE per sequence
            mse = ((batch_seq - reconstructed) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy())
            
            if embedding is not None:
                embeddings.append(embedding.cpu().numpy())
    
    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
    else:
        embeddings = np.array([])
        
    return np.array(errors), embeddings


def compute_metrics(errors, labels, threshold_percentile=95):
    """Compute classification metrics at given threshold."""
    threshold = np.percentile(errors, threshold_percentile)
    predictions = (errors > threshold).astype(int)
    
    metrics = {
        'threshold': float(threshold),
        'threshold_percentile': threshold_percentile,
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1_score': float(f1_score(labels, predictions, zero_division=0)),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    # AUC-ROC
    try:
        metrics['auc_roc'] = float(roc_auc_score(labels, errors))
    except ValueError:
        metrics['auc_roc'] = 0.5
    
    return metrics


def compute_precision_at_k(errors, labels, k=100):
    """Compute Precision at top K highest anomaly scores."""
    # Sort by error (descending)
    indices = np.argsort(errors)[::-1]
    top_k_indices = indices[:k]
    
    # Get labels for top k
    top_k_labels = labels[top_k_indices]
    
    # Precision = fraction of actual positives in top k
    precision = top_k_labels.sum() / k
    return precision


def compute_recall_at_fpr(errors, labels, target_fpr=0.01):
    """Compute Recall at a given False Positive Rate."""
    fpr, tpr, thresholds = roc_curve(labels, errors)
    
    # Find the threshold closest to target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))
    recall_at_fpr = tpr[idx]
    actual_fpr = fpr[idx]
    threshold_at_fpr = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    
    return {
        'recall': float(recall_at_fpr),
        'actual_fpr': float(actual_fpr),
        'threshold': float(threshold_at_fpr)
    }


def compute_auc_pr(errors, labels):
    """Compute AUC for Precision-Recall curve (better for imbalanced data)."""
    try:
        auc_pr = average_precision_score(labels, errors)
        return float(auc_pr)
    except ValueError:
        return 0.0


def export_session_details(
    output_path: Path,
    session_ids,
    user_ids,
    timestamps,
    errors,
    labels_session,
    labels_user,
    threshold,
):
    """Export per-session details to CSV for analysis."""
    import pandas as pd
    
    # Compute percentiles
    percentiles = np.array([
        (errors <= e).sum() / len(errors) * 100 for e in errors
    ])
    
    # Determine if detected
    detected = (errors > threshold).astype(int)
    
    df = pd.DataFrame({
        'session_id': session_ids,
        'user_id': user_ids,
        'timestamp': timestamps,
        'anomaly_score': errors,
        'percentile': percentiles,
        'is_insider_user': labels_user.astype(bool),
        'is_insider_session': labels_session.astype(bool),
        'detected': detected.astype(bool),
    })
    
    df.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model with dual-level metrics")
    parser.add_argument("--model", choices=["lstm", "transformer", "both"], default="both",
                        help="Model(s) to evaluate")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"),
                        help="Path to DuckDB database")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Directory with saved model checkpoints")
    parser.add_argument("--answers-dir", type=Path, default=Path("data/raw/answers"),
                        help="Directory with ground truth labels")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_report.json"),
                        help="Output path for evaluation report")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    args = parser.parse_args()
    
    setup_logging()
    
    # Check paths
    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}")
        return 1
    
    print("=" * 60)
    print("Model Evaluation (Dual-Level)")
    print("=" * 60)
    
    # Load data
    print("\nLoading session data...")
    df = get_session_dataframe(args.db_path)
    print(f"Loaded {len(df):,} sessions")
    
    # Load ground truth with proper R4.2 filtering
    print(f"\nLoading ground truth from {args.answers_dir}")
    if args.answers_dir.exists():
        ground_truth = load_ground_truth(args.answers_dir, dataset='4.2')
        insider_users = ground_truth.get('insider_users', [])
        insider_incidents = ground_truth.get('insider_incidents', [])
        print(f"Found {len(insider_users)} insider users, {len(insider_incidents)} incidents")
    else:
        print(f"Warning: Answers directory not found: {args.answers_dir}")
        insider_users = []
        insider_incidents = []
    
    # Prepare data with dual-level labels
    print("\nPreparing evaluation data with dual-level labels...")
    eval_data = prepare_training_data(
        df,
        sequence_length=100,
        insider_users=insider_users,
        insider_incidents=insider_incidents,
    )
    
    # Get test set with dual labels
    test_sequences = eval_data['test_sequences']
    test_masks = eval_data['test_masks']
    test_labels_session = eval_data['test_labels']         # Session-level
    test_labels_user = eval_data['test_labels_user']       # User-level
    
    print(f"\nTest set: {len(test_sequences):,} sequences")
    print(f"  Session-level: {(test_labels_session == 0).sum():,} normal, {(test_labels_session == 1).sum():,} insider")
    print(f"  User-level:    {(test_labels_user == 0).sum():,} normal, {(test_labels_user == 1).sum():,} insider")
    
    feature_dim = eval_data['feature_dim']
    
    # Load and evaluate models
    models_to_eval = []
    if args.model in ["lstm", "both"]:
        models_to_eval.append("lstm")
    if args.model in ["transformer", "both"]:
        models_to_eval.append("transformer")
    
    results = {}
    
    for model_type in models_to_eval:
        model_path = args.models_dir / f"{model_type}_autoencoder_best.pt"
        
        if not model_path.exists():
            print(f"\nWarning: Model not found: {model_path}")
            print(f"Skipping {model_type} evaluation")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_type.upper()} model...")
        print(f"{'=' * 60}")
        
        model = load_model(model_path, model_type, feature_dim)
        
        # Compute reconstruction errors and embeddings
        print("Computing errors and embeddings...")
        errors, embeddings = compute_reconstruction_errors(model, test_sequences, test_masks, args.device)
        
        # Save embeddings for Phase 2 (User Baselines)
        embedding_path = args.output.parent / f"embeddings_{model_type}.npy"
        np.save(embedding_path, embeddings)
        print(f"  Embeddings saved: {embedding_path} shape={embeddings.shape}")
        
        # Compute metrics at session-level
        print("\nSession-Level Metrics (detecting specific malicious sessions):")
        session_metrics = compute_metrics(errors, test_labels_session)
        print(f"  AUC-ROC:   {session_metrics['auc_roc']:.4f}")
        print(f"  Precision: {session_metrics['precision']:.4f}")
        print(f"  Recall:    {session_metrics['recall']:.4f}")
        print(f"  F1 Score:  {session_metrics['f1_score']:.4f}")
        print(f"  TP: {session_metrics['tp']}, FP: {session_metrics['fp']}, FN: {session_metrics['fn']}, TN: {session_metrics['tn']}")
        
        # Compute Precision@K for Session
        print("\n  Precision@K (Session):")
        for k in [100, 500, 1000]:
            pk = compute_precision_at_k(errors, test_labels_session, k=k)
            print(f"    P@{k:<4}: {pk:.4f}")
        
        # AUC-PR (better for imbalanced data)
        auc_pr_session = compute_auc_pr(errors, test_labels_session)
        print(f"\n  AUC-PR (Session): {auc_pr_session:.4f}")
        
        # Recall@FPR (operational metric)
        print("\n  Recall@FPR (Session):")
        for target_fpr in [0.01, 0.05, 0.10]:
            r_at_fpr = compute_recall_at_fpr(errors, test_labels_session, target_fpr)
            print(f"    R@FPR={target_fpr*100:.0f}%: {r_at_fpr['recall']:.4f} (actual FPR={r_at_fpr['actual_fpr']*100:.2f}%)")

        # Compute TTD metrics (based on session-level predictions)
        # Use simple threshold-based predictions from compute_metrics
        threshold = session_metrics['threshold']
        binary_preds = (errors > threshold).astype(int)
        
        ttd_metrics = compute_ttd(
            predictions=binary_preds,
            test_labels=test_labels_session,
            test_session_ids=eval_data['test_session_ids'],
            test_timestamps=eval_data['test_timestamps'],
            insider_incidents=insider_incidents,
            test_user_ids=eval_data.get('test_user_ids'),
        )
        
        print("\nTime-to-Detect (TTD):")
        if ttd_metrics.get('n_incidents_detected', 0) > 0:
            print(f"  Detected:      {ttd_metrics['n_incidents_detected']}/{ttd_metrics['n_incidents_total']} incidents")
            print(f"  First Session: {ttd_metrics['pct_first_session']:.1f}% caught immediately")
            print(f"  Mean Time:     {ttd_metrics['ttd_hours_mean']:.1f} hours")
            print(f"  Median Time:   {ttd_metrics['ttd_hours_median']:.1f} hours")
            print(f"  Mean Sessions: {ttd_metrics['ttd_sessions_mean']:.1f}")
            print(f"  Mean Lag:      {ttd_metrics['ttd_lag_mean']:.1f}%")
        else:
            print("  No incidents detected at this threshold.")
        
        # Compute metrics at user-level
        print("\nUser-Level Metrics (identifying insider users):")
        user_metrics = compute_metrics(errors, test_labels_user)
        print(f"  AUC-ROC:   {user_metrics['auc_roc']:.4f}")
        print(f"  Precision: {user_metrics['precision']:.4f}")
        print(f"  Recall:    {user_metrics['recall']:.4f}")
        print(f"  F1 Score:  {user_metrics['f1_score']:.4f}")
        print(f"  TP: {user_metrics['tp']}, FP: {user_metrics['fp']}, FN: {user_metrics['fn']}, TN: {user_metrics['tn']}")

        # Compute Precision@K for User
        print("\n  Precision@K (User):")
        for k in [100, 500, 1000]:
            pk = compute_precision_at_k(errors, test_labels_user, k=k)
            print(f"    P@{k:<4}: {pk:.4f}")
        
        # Compute additional metrics for export
        precision_at_k_session = {f'p_at_{k}': compute_precision_at_k(errors, test_labels_session, k) for k in [100, 500, 1000]}
        precision_at_k_user = {f'p_at_{k}': compute_precision_at_k(errors, test_labels_user, k) for k in [100, 500, 1000]}
        
        recall_at_fpr_session = {f'r_at_fpr_{int(fpr*100)}': compute_recall_at_fpr(errors, test_labels_session, fpr)['recall'] for fpr in [0.01, 0.05, 0.10]}
        recall_at_fpr_user = {f'r_at_fpr_{int(fpr*100)}': compute_recall_at_fpr(errors, test_labels_user, fpr)['recall'] for fpr in [0.01, 0.05, 0.10]}
        
        auc_pr_user = compute_auc_pr(errors, test_labels_user)
        
        results[model_type] = {
            'session_level': {**session_metrics, **precision_at_k_session, **recall_at_fpr_session, 'auc_pr': auc_pr_session},
            'user_level': {**user_metrics, **precision_at_k_user, **recall_at_fpr_user, 'auc_pr': auc_pr_user},
            'ttd': ttd_metrics,
            'n_test_samples': len(test_sequences),
            'n_session_insiders': int(test_labels_session.sum()),
            'n_user_insiders': int(test_labels_user.sum()),
            # Store for CSV export
            '_errors': errors,
            '_threshold': threshold,
        }
    
    if not results:
        print("\nNo models found to evaluate!")
        return 1
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\nSession-Level (Detecting Malicious Sessions):")
    print(f"{'Model':<15} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for name, r in results.items():
        m = r['session_level']
        print(f"{name:<15} {m['auc_roc']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}")
    
    print("\nUser-Level (Identifying Insider Users):")
    print(f"{'Model':<15} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for name, r in results.items():
        m = r['user_level']
        print(f"{name:<15} {m['auc_roc']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, args.output)
    
    print(f"\nEvaluation report saved to: {args.output}")
    
    # Export session details CSV for each model
    print("\n" + "=" * 60)
    print("EXPORTING DETAILED DATA")
    print("=" * 60)
    
    for model_type, model_results in results.items():
        # Export session details CSV
        csv_path = args.output.parent / f"session_details_{model_type}.csv"
        
        if '_errors' in model_results:
            export_session_details(
                output_path=csv_path,
                session_ids=eval_data['test_session_ids'],
                user_ids=eval_data['test_user_ids'],
                timestamps=eval_data['test_timestamps'],
                errors=model_results['_errors'],
                labels_session=test_labels_session,
                labels_user=test_labels_user,
                threshold=model_results['_threshold'],
            )
            print(f"  Session details: {csv_path}")
            
            # Remove temp keys from results before JSON export
            del model_results['_errors']
            del model_results['_threshold']
        
        # Export per-incident TTD
        ttd_path = args.output.parent / f"ttd_per_incident_{model_type}.json"
        if 'ttd' in model_results and 'per_incident_ttd' in model_results['ttd']:
            save_json(model_results['ttd']['per_incident_ttd'], ttd_path)
            print(f"  Per-incident TTD: {ttd_path}")
    
    print(f"\nAll exports complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
