#!/usr/bin/env python
"""
Model Training Script
Train LSTM or Transformer autoencoder on session data.

Usage:
    python scripts/train.py --model lstm --epochs 50
    python scripts/train.py --model transformer --epochs 50 --batch-size 32
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

from src.data import get_session_dataframe, prepare_training_data
from src.models import LSTMAutoencoder, TransformerAutoencoder
from src.training import train_autoencoder, TrainConfig
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
    if args.answers_dir.exists():
        print(f"\nLoading ground truth from {args.answers_dir}")
        ground_truth = load_ground_truth(args.answers_dir)
        insider_users = ground_truth.get('insider_users', [])
        print(f"Found {len(insider_users)} insider users")
    
    # Prepare training data (normal sessions only)
    print("\nPreparing training data...")
    train_data = prepare_training_data(
        df,
        sequence_length=100,
        train_ratio=0.8,
        normal_only=len(insider_users) > 0,
        insider_users=insider_users,
    )
    
    print(f"Train sequences: {train_data['train_sequences'].shape}")
    print(f"Val sequences: {train_data['val_sequences'].shape}")
    print(f"Feature dimension: {train_data['feature_dim']}")
    
    # Create model
    feature_dim = train_data['feature_dim']
    
    if args.model == "lstm":
        model = LSTMAutoencoder(
            input_dim=feature_dim,
            hidden_dim=256,
            embedding_dim=128,
            num_layers=2,
            dropout=0.3,
        )
    else:
        model = TransformerAutoencoder(
            input_dim=feature_dim,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            embedding_dim=128,
            dropout=0.1,
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
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Train
    print("\n" + "-" * 60)
    model, history = train_autoencoder(
        model,
        train_data,
        config,
        model_name=f"{args.model}_autoencoder",
    )
    
    # Save training history
    history_path = args.output_dir / f"{args.model}_history.json"
    save_json(history, history_path)
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best val loss: {min(history['val_loss']):.6f}")
    print(f"Model saved to: {args.output_dir / f'{args.model}_autoencoder_best.pt'}")
    print(f"History saved to: {history_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
