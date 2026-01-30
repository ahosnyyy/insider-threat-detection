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
from src.models import LSTMAutoencoder, PaperLSTMAutoencoder, TransformerAutoencoder
from src.training import Trainer, TrainConfig
from src.utils import load_config, setup_logging, load_ground_truth, save_json


def main():
    # Load config
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Train autoencoder model")
    parser.add_argument("--model", choices=["lstm", "paper_lstm", "transformer"], default=cfg['model']['type'],
                        help="Model type to train")
    parser.add_argument("--epochs", type=int, default=cfg['training']['epochs'],
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=cfg['training']['batch_size'],
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=cfg['training']['learning_rate'],
                        help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate on test set every N epochs")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--temporal-split", action="store_true",
                        help="Use temporal split (time-based, no leakage) instead of random")
    parser.add_argument("--db-path", type=Path, default=Path(cfg['data']['database']),
                        help="Path to DuckDB database")
    parser.add_argument("--answers-dir", type=Path, default=Path("data/raw/answers"),
                        help="Directory with ground truth labels")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                        help="Directory to save model checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with minimal epochs for testing")
    parser.add_argument("--role-features", type=str,
                        choices=["none", "roles", "units"],
                        default=cfg["features"].get("role_features", "none"),
                        help="Role features: none | roles (42 one-hot) | units (6 from role_units.yaml)")
    parser.add_argument("--include-role", action="store_true",
                        help="Alias for --role-features roles (deprecated)")
    parser.add_argument("--role-mapping-file", type=Path,
                        default=Path(cfg["features"].get("role_mapping_file", "config/role_units.yaml")),
                        help="Path to role_units.yaml (used when --role-features units)")
    parser.add_argument("--weighted-loss", action="store_true",
                        help="Use hard mining: focus loss on top-k%% hardest samples per batch (training only)")
    parser.add_argument("--hard-mining-ratio", type=float,
                        default=cfg["training"].get("hard_mining_ratio", 0.1),
                        help="Fraction of hardest samples when --weighted-loss (default: 0.1)")
    parser.add_argument("--oversample", action="store_true",
                        help="Oversample insider sessions in test set to target positive rate (config: oversample_target_positive_rate)")
    args = parser.parse_args()
    
    # Alias: --include-role => --role-features roles
    if args.include_role:
        args.role_features = "roles"
    # Map --weighted-loss flag to config
    if args.weighted_loss:
        args.weighted_loss_mode = "hard_mining"
    else:
        args.weighted_loss_mode = cfg["training"].get("weighted_loss", "none")
    
    setup_logging()
    
    # Dry run settings
    if args.dry_run:
        args.epochs = 5
        args.eval_every = 1
        print("DRY RUN MODE: Training for 5 epochs only")
    
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

    # In dry-run mode, also downsample data to avoid huge allocations.
    # This keeps the run fast and prevents multi-GB sequence tensors.
    if args.dry_run:
        max_sessions = 20_000
        if len(df) > max_sessions:
            df = df.sample(n=max_sessions, random_state=cfg['training']['seed']).reset_index(drop=True)
            print(f"DRY RUN MODE: Downsampled sessions to {len(df):,} for quick iteration (cache disabled)")
    
    # Prepare training data
    split_type = "TEMPORAL" if args.temporal_split else "RANDOM"
    train_ratio = cfg['training']['train_split']
    val_ratio = cfg['training']['val_ratio']
    test_ratio = cfg['training']['test_ratio']
    
    print(f"\nPreparing training data ({split_type} split, {int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)})...")
    
    role_mapping_file = args.role_mapping_file
    if role_mapping_file and not role_mapping_file.is_absolute():
        role_mapping_file = Path(__file__).parent.parent / role_mapping_file
    
    if args.temporal_split:
        train_data = prepare_training_data_temporal(
            df,
            sequence_length=cfg['features']['sequence_length'],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            insider_users=insider_users,
            insider_incidents=insider_incidents,
            use_cache=not args.dry_run,
            role_features=args.role_features,
            role_mapping_file=role_mapping_file if args.role_features == "units" else None,
            oversample=args.oversample,
            oversample_target_positive_rate=float(cfg['data'].get('oversample_target_positive_rate', 0.1)),
        )
    else:
        train_data = prepare_training_data(
            df,
            sequence_length=cfg['features']['sequence_length'],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            normal_only_train=len(insider_users) > 0,
            insider_users=insider_users,
            insider_incidents=insider_incidents,
            use_cache=not args.dry_run,
            role_features=args.role_features,
            role_mapping_file=role_mapping_file if args.role_features == "units" else None,
            oversample=args.oversample,
            oversample_target_positive_rate=float(cfg['data'].get('oversample_target_positive_rate', 0.1)),
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
            hidden_dim=cfg['model']['hidden_dim'],
            embedding_dim=cfg['model']['embedding_dim'],
            num_layers=cfg['model']['num_layers'],
            dropout=cfg['model']['dropout'],
        )
    elif args.model == "paper_lstm":
        model = PaperLSTMAutoencoder(
            input_dim=feature_dim,
            bottleneck_dim=cfg['model'].get('paper_lstm_bottleneck', 16),
            enc_hidden=cfg['model'].get('paper_lstm_enc_hidden'),
            dropout=cfg['model'].get('dropout', 0.0),
        )
    else:
        model = TransformerAutoencoder(
            input_dim=feature_dim,
            d_model=cfg['model']['hidden_dim'],  # Using hidden_dim as d_model base
            nhead=cfg['model']['num_heads'],
            num_layers=cfg['model']['num_layers'],
            dim_feedforward=cfg['model']['ff_dim'],
            embedding_dim=cfg['model']['embedding_dim'],
            dropout=cfg['model']['dropout'],
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model.upper()}")
    print(f"Parameters: {total_params:,}")
    
    # Training config
    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=float(cfg['training'].get('weight_decay', 0)),
        warmup_epochs=int(cfg['training'].get('warmup_epochs', 0)),
        weighted_loss=getattr(args, 'weighted_loss_mode', cfg['training'].get('weighted_loss', 'none')),
        hard_mining_ratio=float(args.hard_mining_ratio),
        patience=cfg['training']['patience'],
        checkpoint_dir=args.output_dir,
        eval_every=args.eval_every,
        accumulation_steps=args.accumulation_steps,
        threshold_method=cfg['evaluation'].get('threshold_method', 'percentile'),
        threshold_percentile=cfg['evaluation'].get('percentile', 95),
    )
    
    # Set seed
    torch.manual_seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])
    
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

        if history.get('user_f1_score') and len(history.get('user_tp', [])) > 0:
            print(f"\nConfusion Matrix (User):")
            print(f"                Predicted")
            print(f"              Normal  Anomaly")
            print(f"  Actual Normal  {history['user_tn'][-1]:>6}   {history['user_fp'][-1]:>6}")
            print(f"        Anomaly  {history['user_fn'][-1]:>6}   {history['user_tp'][-1]:>6}")
    
    print(f"\nModel saved to: {args.output_dir / f'{args.model}_autoencoder_best.pt'}")
    
    # Save feature extractor
    extractor_path = args.output_dir / "feature_extractor.pkl"
    train_data['feature_extractor'].save(extractor_path)
    print(f"Feature Extractor saved to: {extractor_path}")
    
    print(f"History saved to: {history_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

