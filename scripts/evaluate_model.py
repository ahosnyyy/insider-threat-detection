#!/usr/bin/env python
"""
Model Evaluation Script
Evaluate trained model and generate metrics report.

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

from src.data import get_session_dataframe, prepare_training_data
from src.models import LSTMAutoencoder, TransformerAutoencoder
from src.training import ModelEvaluator, compare_models
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
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
    args = parser.parse_args()
    
    setup_logging()
    
    # Check paths
    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}")
        return 1
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load data
    print("\nLoading session data...")
    df = get_session_dataframe(args.db_path)
    print(f"Loaded {len(df):,} sessions")
    
    # Load ground truth
    if not args.answers_dir.exists():
        print(f"Warning: Answers directory not found: {args.answers_dir}")
        print("Creating synthetic labels for testing...")
        # Create synthetic labels for testing (5% anomaly rate)
        np.random.seed(42)
        labels = np.random.binomial(1, 0.05, len(df))
        insider_users = []
    else:
        ground_truth = load_ground_truth(args.answers_dir)
        insider_users = ground_truth.get('insider_users', [])
        print(f"Found {len(insider_users)} insider users")
        
        # Create labels based on insider users
        labels = df['user_id'].isin(insider_users).astype(int).values
    
    print(f"Total anomalies: {labels.sum()} ({100*labels.mean():.2f}%)")
    
    # Prepare data (include all users for evaluation)
    print("\nPreparing evaluation data...")
    eval_data = prepare_training_data(
        df,
        sequence_length=100,
        train_ratio=1.0,  # Use all data
        normal_only=False,
        insider_users=[],  # Don't filter
    )
    
    # Get all sequences and their labels
    all_sequences = np.concatenate([
        eval_data['train_sequences'],
        eval_data.get('val_sequences', np.array([])).reshape(-1, *eval_data['train_sequences'].shape[1:])
    ]) if 'val_sequences' in eval_data else eval_data['train_sequences']
    
    all_masks = np.concatenate([
        eval_data['train_masks'],
        eval_data.get('val_masks', np.array([])).reshape(-1, eval_data['train_masks'].shape[1])
    ]) if 'val_masks' in eval_data else eval_data['train_masks']
    
    # Match labels to sequences
    all_session_ids = eval_data['train_session_ids'] + eval_data.get('val_session_ids', [])
    session_to_label = dict(zip(df['session_id'], labels))
    eval_labels = np.array([session_to_label.get(sid, 0) for sid in all_session_ids])
    
    print(f"Evaluation set: {len(all_sequences):,} sequences")
    print(f"Anomalies in eval set: {eval_labels.sum()} ({100*eval_labels.mean():.2f}%)")
    
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
        
        print(f"\nEvaluating {model_type.upper()} model...")
        model = load_model(model_path, model_type, feature_dim)
        
        evaluator = ModelEvaluator(model)
        eval_results = evaluator.evaluate(
            all_sequences,
            all_masks,
            eval_labels,
            model_name=model_type,
        )
        
        results[model_type] = eval_results.to_dict()
    
    if not results:
        print("\nNo models found to evaluate!")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'AUC-ROC':>10} {'P@100':>10} {'R@FPR5%':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<15} {r['auc_roc']:>10.4f} {r['precision_at_100']:>10.4f} {r['recall_at_fpr_5']:>10.4f}")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, args.output)
    
    print(f"\nEvaluation report saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
