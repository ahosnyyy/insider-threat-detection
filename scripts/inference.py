#!/usr/bin/env python
"""
Inference Script
Generate embeddings and anomaly scores for sessions.

Usage:
    # Single session
    python scripts/inference.py --session-id S12345
    
    # All sessions (batch)
    python scripts/inference.py --all --output data/outputs/session_outputs.jsonl
    
    # Specific user
    python scripts/inference.py --user-id U0123
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data import get_session_dataframe, FeatureExtractor, SequenceBuilder
from src.models import LSTMAutoencoder, TransformerAutoencoder
from src.utils import setup_logging, load_config


@dataclass
class SessionOutput:
    """Output schema for Phase 1 -> Phase 2/4 integration."""
    session_id: str
    user_id: str
    timestamp: str  # ISO format
    embedding: List[float]  # 128-dim vector
    reconstruction_error: float
    anomaly_percentile: float  # 0-100


class InferenceEngine:
    """Load model and run inference on sessions."""
    
    def __init__(
        self,
        model_path: Path,
        model_type: str = "lstm",
        config: dict = None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.config = config
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Feature extractor (will be fitted on demand)
        self.feature_extractor = None
        self.sequence_builder = SequenceBuilder(sequence_length=100)
        
        # Threshold for percentile calculation (fitted on training data)
        self.error_distribution = None
    
    def _load_model(self, model_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Infer feature dim from state dict
        state_dict = checkpoint['model_state_dict']
        
        # Find input projection layer to get feature dim
        for key, value in state_dict.items():
            if 'input_proj' in key or 'lstm.weight_ih_l0' in key:
                if 'weight' in key:
                    feature_dim = value.shape[1]
                    break
        else:
            feature_dim = 24  # Default
        
        if self.model_type == "lstm":
            model = LSTMAutoencoder(
                input_dim=feature_dim,
                hidden_dim=self.config['model']['hidden_dim'] if self.config else 256,
                embedding_dim=self.config['model']['embedding_dim'] if self.config else 128,
                num_layers=self.config['model']['num_layers'] if self.config else 2,
                dropout=self.config['model']['dropout'] if self.config else 0.2,
            )
        else:
            model = TransformerAutoencoder(
                input_dim=feature_dim,
                d_model=self.config['model']['hidden_dim'] if self.config else 256,
                nhead=self.config['model']['num_heads'] if self.config else 8,
                num_layers=self.config['model']['num_layers'] if self.config else 2,
                dim_feedforward=self.config['model']['ff_dim'] if self.config else 1024,
                embedding_dim=self.config['model']['embedding_dim'] if self.config else 128,
                dropout=self.config['model']['dropout'] if self.config else 0.2,
            )
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model
    
    def fit_extractor(self, df):
        """Fit feature extractor on data."""
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.fit(df)
    
    def set_error_distribution(self, errors: np.ndarray):
        """Set error distribution for percentile calculation."""
        self.error_distribution = np.sort(errors)
    
    def infer_session(
        self,
        session_df,
        user_history_df,
    ) -> SessionOutput:
        """
        Run inference on a single session.
        
        Args:
            session_df: DataFrame row for the target session
            user_history_df: DataFrame with user's historical sessions
            
        Returns:
            SessionOutput with embedding and error
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not fitted. Call fit_extractor first.")
        
        # Combine current and history
        combined_df = user_history_df.copy()
        if session_df['session_id'] not in combined_df['session_id'].values:
            combined_df = pd.concat([combined_df, pd.DataFrame([session_df])], ignore_index=True)
        
        # Extract features
        features = self.feature_extractor.transform(combined_df)
        
        # Build sequence
        sequences, masks, session_ids = self.sequence_builder.build_user_sequences(
            combined_df.reset_index(drop=True),
            features,
        )
        
        # Find target session
        target_idx = session_ids.index(session_df['session_id'])
        
        # Run inference
        seq = torch.FloatTensor(sequences[target_idx:target_idx+1]).to(self.device)
        mask = torch.FloatTensor(masks[target_idx:target_idx+1]).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_embedding(seq, mask).cpu().numpy()[0]
            error = self.model.get_reconstruction_error(seq, mask).cpu().numpy()[0]
        
        # Calculate percentile
        if self.error_distribution is not None:
            percentile = 100 * (self.error_distribution < error).mean()
        else:
            percentile = 50.0  # Default if no distribution
        
        return SessionOutput(
            session_id=str(session_df['session_id']),
            user_id=str(session_df['user_id']),
            timestamp=session_df['start_time'].isoformat() if hasattr(session_df['start_time'], 'isoformat') else str(session_df['start_time']),
            embedding=embedding.tolist(),
            reconstruction_error=float(error),
            anomaly_percentile=float(percentile),
        )
    
    def infer_batch(
        self,
        df,
        batch_size: int = 64,
    ) -> List[SessionOutput]:
        """
        Run inference on all sessions.
        
        Args:
            df: Full sessions DataFrame
            batch_size: Batch size for inference
            
        Returns:
            List of SessionOutput for all sessions
        """
        if self.feature_extractor is None:
            self.fit_extractor(df)
        
        # Extract all features
        features = self.feature_extractor.transform(df)
        
        # Build all sequences
        sequences, masks, session_ids = self.sequence_builder.build_user_sequences(
            df.reset_index(drop=True),
            features,
        )
        
        # Run batch inference
        all_embeddings = []
        all_errors = []
        
        n_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Inference"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(sequences))
            
            seq = torch.FloatTensor(sequences[start:end]).to(self.device)
            mask = torch.FloatTensor(masks[start:end]).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model.get_embedding(seq, mask).cpu().numpy()
                errors = self.model.get_reconstruction_error(seq, mask).cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_errors.append(errors)
        
        all_embeddings = np.concatenate(all_embeddings)
        all_errors = np.concatenate(all_errors)
        
        # Set error distribution for percentile calculation
        self.set_error_distribution(all_errors)
        
        # Create outputs
        outputs = []
        for i, session_id in enumerate(session_ids):
            session_row = df[df['session_id'] == session_id].iloc[0]
            
            percentile = 100 * (all_errors < all_errors[i]).mean()
            
            outputs.append(SessionOutput(
                session_id=str(session_id),
                user_id=str(session_row['user_id']),
                timestamp=session_row['start_time'].isoformat() if hasattr(session_row['start_time'], 'isoformat') else str(session_row['start_time']),
                embedding=all_embeddings[i].tolist(),
                reconstruction_error=float(all_errors[i]),
                anomaly_percentile=float(percentile),
            ))
        
        return outputs


def save_outputs_jsonl(outputs: List[SessionOutput], path: Path):
    """Save outputs as JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        for output in outputs:
            f.write(json.dumps(asdict(output)) + '\n')
    
    print(f"Saved {len(outputs)} session outputs to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on sessions")
    parser.add_argument("--session-id", type=str, help="Specific session to analyze")
    parser.add_argument("--user-id", type=str, help="Analyze all sessions for a user")
    parser.add_argument("--all", action="store_true", help="Process all sessions")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm",
                        help="Model type")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--db-path", type=Path, default=Path("data/processed/cert.duckdb"),
                        help="Path to DuckDB database")
    parser.add_argument("--output", type=Path, default=Path("data/outputs/session_outputs.jsonl"),
                        help="Output path for session outputs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    cfg = load_config()
    
    setup_logging()
    
    # Set default model path
    if args.model_path is None:
        args.model_path = Path(f"models/{args.model}_autoencoder_best.pt")
    
    # Check paths
    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}")
        return 1
    
    if not args.model_path.exists():
        print(f"Error: Model not found: {args.model_path}")
        print("Train a model first: python scripts/train.py")
        return 1
    
    print("=" * 60)
    print("Session Inference")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Database: {args.db_path}")
    
    # Load data
    df = get_session_dataframe(args.db_path)
    print(f"Loaded {len(df):,} sessions")
    
    # Initialize engine
    engine = InferenceEngine(args.model_path, args.model, config=cfg)
    
    if args.session_id:
        # Single session
        session_row = df[df['session_id'] == args.session_id]
        if session_row.empty:
            print(f"Error: Session not found: {args.session_id}")
            return 1
        
        user_id = session_row.iloc[0]['user_id']
        user_df = df[df['user_id'] == user_id]
        
        engine.fit_extractor(df)
        output = engine.infer_session(session_row.iloc[0], user_df)
        
        print(f"\nSession: {output.session_id}")
        print(f"User: {output.user_id}")
        print(f"Reconstruction Error: {output.reconstruction_error:.6f}")
        print(f"Anomaly Percentile: {output.anomaly_percentile:.2f}%")
        print(f"Embedding (first 10): {output.embedding[:10]}")
        
    elif args.user_id:
        # All sessions for a user
        user_df = df[df['user_id'] == args.user_id]
        if user_df.empty:
            print(f"Error: User not found: {args.user_id}")
            return 1
        
        print(f"\nProcessing {len(user_df)} sessions for user {args.user_id}")
        outputs = engine.infer_batch(user_df, args.batch_size)
        save_outputs_jsonl(outputs, args.output)
        
    elif args.all:
        # All sessions
        print(f"\nProcessing all {len(df):,} sessions...")
        outputs = engine.infer_batch(df, args.batch_size)
        save_outputs_jsonl(outputs, args.output)
        
        # Print summary
        errors = [o.reconstruction_error for o in outputs]
        print(f"\nSummary:")
        print(f"  Total sessions: {len(outputs)}")
        print(f"  Error range: {min(errors):.4f} - {max(errors):.4f}")
        print(f"  Error mean: {np.mean(errors):.4f}")
        print(f"  Error std: {np.std(errors):.4f}")
    else:
        print("\nError: Specify --session-id, --user-id, or --all")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
