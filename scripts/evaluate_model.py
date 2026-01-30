#!/usr/bin/env python
"""
Model Evaluation Script
Evaluate trained model with dual-level metrics (session + user level).

Usage:
    python scripts/evaluate_model.py --model lstm
    python scripts/evaluate_model.py --model transformer --output results/evaluation.json
"""

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, precision_recall_curve,
    roc_curve, average_precision_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import get_session_dataframe, prepare_training_data
from src.models import LSTMAutoencoder, PaperLSTMAutoencoder, TransformerAutoencoder
from src.training import ModelEvaluator
from src.training.metrics import compute_ttd
from src.utils import setup_logging, load_ground_truth, save_json, load_json, load_config


def load_model(model_path: Path, model_type: str, feature_dim: int, cfg: dict):
    """Load saved model checkpoint."""
    if model_type == "lstm":
        model = LSTMAutoencoder(
            input_dim=feature_dim,
            hidden_dim=cfg['model']['hidden_dim'],
            embedding_dim=cfg['model']['embedding_dim'],
            num_layers=cfg['model']['num_layers'],
            dropout=cfg['model']['dropout'],
        )
    elif model_type == "paper_lstm":
        model = PaperLSTMAutoencoder(
            input_dim=feature_dim,
            bottleneck_dim=cfg['model'].get('paper_lstm_bottleneck', 16),
            enc_hidden=cfg['model'].get('paper_lstm_enc_hidden'),
            dropout=cfg['model'].get('dropout', 0.0),
        )
    else:
        model = TransformerAutoencoder(
            input_dim=feature_dim,
            d_model=cfg['model']['hidden_dim'],
            nhead=cfg['model']['num_heads'],
            num_layers=cfg['model']['num_layers'],
            dim_feedforward=cfg['model']['ff_dim'],
            embedding_dim=cfg['model']['embedding_dim'],
            dropout=cfg['model']['dropout'],
        )
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def compute_reconstruction_errors(model, sequences, masks, device="cuda"):
    """Compute reconstruction errors for all sequences."""
    model = model.to(device)
    model.eval()
    
    errors = []
    embeddings = []
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


def compute_metrics(errors, labels, threshold_percentile=95, threshold_method: str = "percentile"):
    """Compute classification metrics at a chosen threshold.
    
    threshold_method:
      - 'percentile': use fixed percentile of error distribution
      - 'f1_optimal': choose threshold that maximizes F1 on this set
    """
    if threshold_method == "f1_optimal":
        try:
            precision_arr, recall_arr, pr_thresholds = precision_recall_curve(labels, errors)
            # Last precision/recall entry corresponds to +inf threshold; ignore it for F1
            if pr_thresholds.size > 0:
                f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (
                    precision_arr[:-1] + recall_arr[:-1] + 1e-8
                )
                best_idx = int(np.argmax(f1_scores))
                threshold = pr_thresholds[best_idx]
            else:
                threshold = np.percentile(errors, threshold_percentile)
        except Exception:
            threshold = np.percentile(errors, threshold_percentile)
    else:
        threshold = np.percentile(errors, threshold_percentile)
    predictions = (errors > threshold).astype(int)
    
    metrics = {
        'threshold': float(threshold),
        'threshold_percentile': threshold_percentile,
        'threshold_method': threshold_method,
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
    # Load config
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Evaluate trained model with dual-level metrics")
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer", "paper_lstm", "both"],
        default="both",
        help="Model(s) to evaluate",
    )
    parser.add_argument("--db-path", type=Path, default=Path(cfg['data']['database']),
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
        oversample=args.oversample,
        oversample_target_positive_rate=float(cfg['data'].get('oversample_target_positive_rate', 0.1)),
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
    if args.model == "paper_lstm":
        models_to_eval.append("paper_lstm")
    
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
        
        model = load_model(model_path, model_type, feature_dim, cfg)
        
        # Compute reconstruction errors and embeddings
        print("Computing errors and embeddings...")
        errors, embeddings = compute_reconstruction_errors(model, test_sequences, test_masks, args.device)
        
        # Save embeddings for Phase 2 (User Baselines)
        embedding_path = args.output.parent / f"embeddings_{model_type}.npy"
        np.save(embedding_path, embeddings)
        label_path = args.output.parent / f"labels_{model_type}.npy"
        np.save(label_path, test_labels_session)
        print(f"  Embeddings saved: {embedding_path}")
        print(f"  Labels saved: {label_path}")
        
        # Compute metrics at session-level
        print("\nSession-Level Metrics (detecting specific malicious sessions):")
        session_metrics = compute_metrics(
            errors,
            test_labels_session,
            threshold_percentile=cfg['evaluation']['percentile'],
            threshold_method=cfg['evaluation'].get('threshold_method', 'percentile'),
        )
        print(f"  AUC-ROC:   {session_metrics['auc_roc']:.4f}")
        print(f"  Precision: {session_metrics['precision']:.4f}")
        print(f"  Recall:    {session_metrics['recall']:.4f}")
        print(f"  F1 Score:  {session_metrics['f1_score']:.4f}")
        print(f"  TP: {session_metrics['tp']}, FP: {session_metrics['fp']}, FN: {session_metrics['fn']}, TN: {session_metrics['tn']}")
        print(
            f"  Threshold method: {session_metrics.get('threshold_method', 'percentile')} "
            f"(p={session_metrics.get('threshold_percentile', 0):.1f})"
        )
        print(f"  Threshold value:  {session_metrics['threshold']:.6f}")
        print(f"  FPR at threshold: {session_metrics['fpr']:.4f}")
        
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
            print(f"  Median Lag:    {ttd_metrics['ttd_lag_median']:.1f}%")
        else:
            print("  No incidents detected at this threshold.")
        
        # Compute metrics at user-level
        print("\nUser-Level Metrics (identifying insider users):")
        user_metrics = compute_metrics(
            errors,
            test_labels_user,
            threshold_percentile=cfg['evaluation']['percentile'],
            threshold_method=cfg['evaluation'].get('threshold_method', 'percentile'),
        )
        print(f"  AUC-ROC:   {user_metrics['auc_roc']:.4f}")
        print(f"  Precision: {user_metrics['precision']:.4f}")
        print(f"  Recall:    {user_metrics['recall']:.4f}")
        print(f"  F1 Score:  {user_metrics['f1_score']:.4f}")
        print(f"  TP: {user_metrics['tp']}, FP: {user_metrics['fp']}, FN: {user_metrics['fn']}, TN: {user_metrics['tn']}")
        print(
            f"  Threshold method: {user_metrics.get('threshold_method', 'percentile')} "
            f"(p={user_metrics.get('threshold_percentile', 0):.1f})"
        )
        print(f"  Threshold value:  {user_metrics['threshold']:.6f}")
        print(f"  FPR at threshold: {user_metrics['fpr']:.4f}")

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
    
    # Create model-specific output directory
    model_dir = args.output.parent / args.model if args.model != "both" else args.output.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results (merge with existing)
    output_path = model_dir / args.output.name if args.model != "both" else args.output
    
    final_results = results
    if output_path.exists():
        try:
            print(f"Merging with existing report at {output_path}")
            old_results = load_json(output_path)
            old_results.update(results)
            final_results = old_results
        except Exception as e:
            print(f"Warning: Could not merge with existing results: {e}")
            
    save_json(final_results, output_path)
    
    print(f"\nEvaluation report saved to: {output_path}")
    
    # Export session details CSV for each model
    print("\n" + "=" * 60)
    print("EXPORTING DETAILED DATA")
    print("=" * 60)
    
    for model_type, model_results in results.items():
        # Create model-specific directory for exports
        export_dir = args.output.parent / model_type
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export session details CSV
        csv_path = export_dir / f"session_details_{model_type}.csv"
        
        if '_errors' in model_results:
            errors = model_results['_errors']
            export_session_details(
                output_path=csv_path,
                session_ids=eval_data['test_session_ids'],
                user_ids=eval_data['test_user_ids'],
                timestamps=eval_data['test_timestamps'],
                errors=errors,
                labels_session=test_labels_session,
                labels_user=test_labels_user,
                threshold=model_results['_threshold'],
            )
            print(f"  Session details: {csv_path}")

            # Session-level ROC and PR curves
            try:
                fpr, tpr, _ = roc_curve(test_labels_session, errors)
                precision_arr, recall_arr, _ = precision_recall_curve(test_labels_session, errors)

                # ROC curve (Session)
                plt.figure()
                plt.plot(fpr, tpr, label=f"{model_type} (AUC={model_results['session_level']['auc_roc']:.3f})")
                plt.plot([0, 1], [0, 1], "k--", label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve (Session)")
                plt.legend()
                plt.tight_layout()
                roc_path = export_dir / f"roc_session_{model_type}.png"
                plt.savefig(roc_path)
                plt.close()
                print(f"  ROC curve (session): {roc_path}")

                # PR curve (Session)
                plt.figure()
                plt.plot(recall_arr, precision_arr, label=model_type)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve (Session)")
                plt.tight_layout()
                pr_path = export_dir / f"pr_session_{model_type}.png"
                plt.savefig(pr_path)
                plt.close()
                print(f"  PR curve (session):  {pr_path}")

                # Reconstruction error distribution (Session) - histogram
                normal_errors = errors[test_labels_session == 0]
                insider_errors = errors[test_labels_session == 1]
                plt.figure()
                bins = 50
                plt.hist(normal_errors, bins=bins, alpha=0.5, label="Normal", density=True)
                plt.hist(insider_errors, bins=bins, alpha=0.5, label="Insider", density=True)
                plt.yscale("log")
                plt.xlabel("Reconstruction error")
                plt.ylabel("Density (log scale)")
                plt.title("Reconstruction Error Distribution (Session)")
                plt.legend()
                plt.tight_layout()
                hist_path = export_dir / f"error_hist_session_{model_type}.png"
                plt.savefig(hist_path)
                plt.close()
                print(f"  Error histogram:      {hist_path}")

                # Reconstruction error scatter plot with threshold line
                plt.figure(figsize=(10, 4))
                indices = np.arange(len(errors))
                normal_idx = np.where(test_labels_session == 0)[0]
                insider_idx = np.where(test_labels_session == 1)[0]
                plt.scatter(indices[normal_idx], errors[normal_idx], s=1, alpha=0.3, label="Normal")
                plt.scatter(indices[insider_idx], errors[insider_idx], s=4, alpha=0.6, label="Insider")
                plt.axhline(model_results['_threshold'], color="red", linestyle="--", label="Threshold")
                plt.xlabel("Sample index (test set order)")
                plt.ylabel("Reconstruction error")
                plt.title("Reconstruction Error (per sample) with Threshold (Session)")
                plt.legend(loc="upper right")
                plt.tight_layout()
                scatter_path = export_dir / f"error_scatter_session_{model_type}.png"
                plt.savefig(scatter_path)
                plt.close()
                print(f"  Error scatter plot:   {scatter_path}")
            except Exception as e:
                print(f"  Warning: Failed to generate curves/histogram: {e}")

            # Remove temp keys from results before JSON export
            del model_results['_errors']
            del model_results['_threshold']
        
        # Export per-incident TTD
        ttd_path = export_dir / f"ttd_per_incident_{model_type}.json"
        if 'ttd' in model_results and 'per_incident_ttd' in model_results['ttd']:
            save_json(model_results['ttd']['per_incident_ttd'], ttd_path)
            print(f"  Per-incident TTD: {ttd_path}")
    
    print(f"\nAll exports complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
