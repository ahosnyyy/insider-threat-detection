#!/usr/bin/env python
"""
Model Training Script
Train LSTM or Transformer autoencoder on session data.

Usage:
    python scripts/train.py --model lstm --epochs 50
    python scripts/train.py --model transformer --epochs 50 --batch-size 32
    python scripts/train.py --model lstm --temporal-split  # Production mode
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
import torch

from src.data import get_session_dataframe, prepare_training_data, prepare_training_data_temporal
from src.models import LSTMAutoencoder, TransformerAutoencoder
from src.training import Trainer, TrainConfig
from src.utils import load_config, setup_logging, load_ground_truth, save_json


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder model")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm",
                        help="Model type to train")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate on test set every N epochs")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--temporal-split", action="store_true",
                        help="Use temporal split (time-based, no leakage) instead of random")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"),
                        help="Path to DuckDB database")
    parser.add_argument("--answers-dir", type=Path, default=Path("data/raw/answers"),
                        help="Directory with ground truth labels")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                        help="Directory to save model checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with minimal epochs for testing")
    args = parser.parse_args()
    
    setup_logging()
    
    # Dry run settings
    if args.dry_run:
        args.epochs = 2
        args.eval_every = 1
        print("DRY RUN MODE: Training for 2 epochs only")
    
    # Check database exists
    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}")
        print("Run scripts/ingest_data.py first")
        return 1
    
    print("=" * 60)
    print(f"Training {args.model.upper()} Autoencoder")
    print("=" * 60)
    
    # Load session data
    print("\nLoading session data...")
    df = get_session_dataframe(args.db_path)
    print(f"Loaded {len(df):,} sessions")
    
    # Load ground truth if available
    insider_users = []
    insider_incidents = []
    if args.answers_dir.exists():
        print(f"\nLoading ground truth from {args.answers_dir}")
        ground_truth = load_ground_truth(args.answers_dir, dataset="4.2")
        insider_users = ground_truth.get('insider_users', [])
        insider_incidents = ground_truth.get('insider_incidents', [])
        n_events = len(ground_truth.get('malicious_events', []))
        print(f"Found {len(insider_users)} insider users, {len(insider_incidents)} incidents, {n_events} events")
    
    # Prepare training data
    split_type = "TEMPORAL" if args.temporal_split else "RANDOM"
    print(f"\nPreparing training data ({split_type} split, 70/10/20)...")
    
    if args.temporal_split:
        train_data = prepare_training_data_temporal(
            df,
            sequence_length=100,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            insider_users=insider_users,
            insider_incidents=insider_incidents,
        )
    else:
        train_data = prepare_training_data(
            df,
            sequence_length=100,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            normal_only_train=len(insider_users) > 0,
            insider_users=insider_users,
            insider_incidents=insider_incidents,
        )
    
    print(f"Train sequences: {train_data['train_sequences'].shape}")
    print(f"Val sequences: {train_data['val_sequences'].shape}")
    print(f"Test sequences: {train_data['test_sequences'].shape}")
    
    # Show both label types
    test_labels = train_data['test_labels']
    test_labels_user = train_data.get('test_labels_user', test_labels)
    print(f"  Session-level: {(test_labels == 0).sum():,} normal, {(test_labels == 1).sum():,} insider")
    print(f"  User-level:    {(test_labels_user == 0).sum():,} normal, {(test_labels_user == 1).sum():,} insider")
    print(f"Feature dimension: {train_data['feature_dim']}")
    
    # Create model
    feature_dim = train_data['feature_dim']
    
    if args.model == "lstm":
        model = LSTMAutoencoder(
            input_dim=feature_dim,
            hidden_dim=256,
            embedding_dim=128,
            num_layers=2,
            dropout=0.2,
        )
    else:
        model = TransformerAutoencoder(
            input_dim=feature_dim,
            d_model=256,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            embedding_dim=128,
            dropout=0.2,
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model.upper()}")
    print(f"Parameters: {total_params:,}")
    
    # Training config
    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=10,
        checkpoint_dir=args.output_dir,
        eval_every=args.eval_every,
        accumulation_steps=args.accumulation_steps,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Eval every: {config.eval_every} epochs")
    
    # Train with TTD data
    print("\n" + "-" * 60)
    trainer = Trainer(model, config, model_name=f"{args.model}_autoencoder")
    history = trainer.train(
        train_sequences=train_data['train_sequences'],
        train_masks=train_data['train_masks'],
        val_sequences=train_data['val_sequences'],
        val_masks=train_data['val_masks'],
        test_sequences=train_data['test_sequences'],
        test_masks=train_data['test_masks'],
        test_labels=train_data['test_labels'],
        test_labels_user=train_data.get('test_labels_user'),
        test_session_ids=train_data.get('test_session_ids'),
        test_timestamps=train_data.get('test_timestamps'),
        test_user_ids=train_data.get('test_user_ids'),
        insider_incidents=insider_incidents,
    )
    
    # Save training history
    history_path = args.output_dir / f"{args.model}_history.json"
    save_json(history, history_path)
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best val loss: {min(history['val_loss']):.6f}")
    
    # Print final metrics
    if history['f1_score']:
        print(f"\nFinal Test Metrics (last evaluation):")
        print(f"  Training Time: {history['total_training_time'] / 60:.1f} minutes")
        
        print("\n  Session-Level:")
        print(f"    Accuracy:  {history['accuracy'][-1]:.4f}")
        print(f"    Precision: {history['precision'][-1]:.4f}")
        print(f"    Recall:    {history['recall'][-1]:.4f}")
        print(f"    F1 Score:  {history['f1_score'][-1]:.4f}")
        print(f"    AUC-ROC:   {history['auc_roc'][-1]:.4f}")
        print(f"    FPR:       {history['fpr'][-1]:.4f}")
        
        if history['user_f1_score']:
            print("\n  User-Level:")
            print(f"    Accuracy:  {history['user_accuracy'][-1]:.4f}")
            print(f"    Precision: {history['user_precision'][-1]:.4f}")
            print(f"    Recall:    {history['user_recall'][-1]:.4f}")
            print(f"    F1 Score:  {history['user_f1_score'][-1]:.4f}")
            print(f"    AUC-ROC:   {history['user_auc_roc'][-1]:.4f}")
            print(f"    FPR:       {history['user_fpr'][-1]:.4f}")

        print("\n  Time-to-Detect:")
        print(f"    Mean Time:     {history['ttd_hours_mean'][-1]:.1f} hours")
        print(f"    % First Sess:  {history['pct_first_session'][-1]:.1f}%")
        print(f"    Detected:      {history['n_incidents_detected'][-1]} incidents")

        print(f"\nConfusion Matrix (Session):")
        print(f"                Predicted")
        print(f"              Normal  Anomaly")
        print(f"  Actual Normal  {history['tn'][-1]:>6}   {history['fp'][-1]:>6}")
        print(f"        Anomaly  {history['fn'][-1]:>6}   {history['tp'][-1]:>6}")
    
    print(f"\nModel saved to: {args.output_dir / f'{args.model}_autoencoder_best.pt'}")
    
    # Save feature extractor
    extractor_path = args.output_dir / "feature_extractor.pkl"
    train_data['feature_extractor'].save(extractor_path)
    print(f"Feature Extractor saved to: {extractor_path}")
    
    print(f"History saved to: {history_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

