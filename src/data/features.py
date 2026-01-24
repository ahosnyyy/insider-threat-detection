"""
Feature Engineering Module
Transforms session data into sequences for autoencoder training.

Updated for CERT R4.2 schema with correct feature names.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Features matching session_features table from sessionize.py
NUMERIC_FEATURES = [
    # Session timing
    "duration_minutes",
    "start_hour",
    "day_of_week",
    "is_weekend",
    "is_after_hours",
    # Device (USB) features
    "device_event_count",
    "usb_connect_count",
    "usb_disconnect_count",
    # HTTP features
    "http_request_count",
    "unique_domains",
    # Email features
    "email_count",
    "emails_sent",
    "emails_received",
    "total_attachments",
    "total_email_size",
    "external_email_count",
    # File copy to removable media
    "file_copy_count",
    "exe_copy_count",
    "doc_copy_count",
    "pdf_copy_count",    
    "archive_copy_count",
    # Derived
    "total_actions",
    "action_density",
]

# Optional features (added by enrichment)
OPTIONAL_FEATURES = [
    # LDAP features
    "is_admin",
    "role_changed_this_month",
    "dept_changed_this_month",
    "user_terminated",
    # Psychometric features
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


class FeatureExtractor:
    """Extract and normalize features from session data."""
    
    def __init__(self, feature_names: List[str] = None, include_optional: bool = True):
        self.base_features = feature_names or NUMERIC_FEATURES.copy()
        self.include_optional = include_optional
        self.feature_names = None  # Set during fit
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        """Fit scaler on training data."""
        # Determine available features
        self.feature_names = [f for f in self.base_features if f in df.columns]
        
        if self.include_optional:
            for f in OPTIONAL_FEATURES:
                if f in df.columns:
                    self.feature_names.append(f)
        
        features = self._extract_raw_features(df)
        self.scaler.fit(features)
        self.is_fitted = True
        logger.info(f"Fitted scaler on {len(df):,} sessions, {len(self.feature_names)} features")
        logger.info(f"Features: {self.feature_names}")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform session data to normalized feature matrix."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        features = self._extract_raw_features(df)
        normalized = self.scaler.transform(features)
        return normalized
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def _extract_raw_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract raw feature values from DataFrame."""
        # Extract available features
        available = [f for f in self.feature_names if f in df.columns]
        
        if len(available) < len(self.feature_names):
            missing = set(self.feature_names) - set(available)
            logger.warning(f"Missing features (will be zero-filled): {missing}")
        
        # Start with zeros
        features = np.zeros((len(df), len(self.feature_names)), dtype=np.float32)
        
        # Fill in available features
        for i, fname in enumerate(self.feature_names):
            if fname in df.columns:
                features[:, i] = df[fname].fillna(0).values
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get number of features."""
        return len(self.feature_names) if self.feature_names else len(self.base_features)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names or self.base_features


class SequenceBuilder:
    """Build sequences for autoencoder input."""
    
    def __init__(self, sequence_length: int = 100, feature_dim: int = None):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
    
    def build_user_sequences(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build sequences grouped by user.
        
        Each session becomes a sequence containing t its historical context
        (previous sessions by the same user).
        
        Args:
            df: Session DataFrame with user_id and session_id columns
            features: Normalized feature matrix (N, F)
            
        Returns:
            sequences: (num_sequences, seq_len, feature_dim)
            masks: (num_sequences, seq_len) - 1 for real, 0 for padding
            session_ids: List of session IDs for each sequence
        """
        sequences = []
        masks = []
        session_ids = []
        
        # Feature dimension
        feature_dim = features.shape[1]
        
        # Ensure df has proper index
        df = df.reset_index(drop=True)
        
        # Group by user and process
        for user_id, group in df.groupby("user_id"):
            user_indices = group.index.tolist()
            user_features = features[user_indices]
            user_sessions = group["session_id"].tolist()
            
            # Create one sequence per session (includes history)
            for i in range(len(user_indices)):
                # Get sequence ending at position i
                start = max(0, i - self.sequence_length + 1)
                seq_features = user_features[start:i+1]
                
                # Pad if needed (pad at beginning with zeros)
                seq_len = len(seq_features)
                if seq_len < self.sequence_length:
                    padding = np.zeros((self.sequence_length - seq_len, feature_dim))
                    seq_features = np.vstack([padding, seq_features])
                    mask = np.concatenate([
                        np.zeros(self.sequence_length - seq_len),
                        np.ones(seq_len)
                    ])
                else:
                    mask = np.ones(self.sequence_length)
                
                sequences.append(seq_features)
                masks.append(mask)
                session_ids.append(user_sessions[i])
        
        return (
            np.array(sequences, dtype=np.float32),
            np.array(masks, dtype=np.float32),
            session_ids,
        )


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int = 100,
    train_ratio: float = 0.8,
    normal_only: bool = True,
    insider_users: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, any]:
    """
    Prepare data for autoencoder training.
    
    Args:
        df: Session features DataFrame
        sequence_length: Max sequence length
        train_ratio: Train/val split ratio
        normal_only: If True, train only on normal (non-insider) sessions
        insider_users: List of known insider user IDs (from ground truth)
        seed: Random seed for reproducibility
        
    Returns:
        Dict with train/val sequences, masks, session IDs, and feature extractor
    """
    np.random.seed(seed)
    
    logger.info(f"Preparing training data from {len(df):,} sessions")
    
    # Filter to normal users if specified
    if normal_only and insider_users and len(insider_users) > 0:
        train_df = df[~df["user_id"].isin(insider_users)].copy()
        logger.info(f"Filtered to {len(train_df):,} normal sessions (excluded {len(insider_users)} insider users)")
    else:
        train_df = df.copy()
    
    # Sort by user and time for proper sequence building
    if "start_time" in train_df.columns:
        train_df = train_df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.fit_transform(train_df)
    
    # Build sequences
    builder = SequenceBuilder(sequence_length=sequence_length)
    sequences, masks, session_ids = builder.build_user_sequences(train_df, features)
    
    logger.info(f"Built {len(sequences):,} sequences, shape: {sequences.shape}")
    
    # Split train/val
    n_samples = len(sequences)
    n_train = int(n_samples * train_ratio)
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    return {
        "train_sequences": sequences[train_idx],
        "train_masks": masks[train_idx],
        "train_session_ids": [session_ids[i] for i in train_idx],
        "val_sequences": sequences[val_idx],
        "val_masks": masks[val_idx],
        "val_session_ids": [session_ids[i] for i in val_idx],
        "feature_extractor": extractor,
        "feature_dim": extractor.get_feature_dim(),
        "feature_names": extractor.get_feature_names(),
    }


if __name__ == "__main__":
    import duckdb
    from pathlib import Path
    
    db_path = Path("data/processed/cert.duckdb")
    
    if db_path.exists():
        con = duckdb.connect(str(db_path))
        df = con.execute("SELECT * FROM session_features").fetchdf()
        con.close()
        
        print(f"Loaded {len(df):,} sessions")
        print(f"Columns: {list(df.columns)}")
        
        data = prepare_training_data(df, sequence_length=50)
        
        print(f"\nTraining data prepared:")
        print(f"  Train sequences: {data['train_sequences'].shape}")
        print(f"  Val sequences: {data['val_sequences'].shape}")
        print(f"  Feature dim: {data['feature_dim']}")
        print(f"  Features: {data['feature_names']}")
    else:
        print(f"Database not found: {db_path}")
