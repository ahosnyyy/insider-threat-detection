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
from sklearn.preprocessing import LabelEncoder

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
    # LDAP features (must match column names from sessionize.py)
    "is_admin",
    "role_changed",       # Fixed: was role_changed_this_month
    "dept_changed",       # Fixed: was dept_changed_this_month  
    "terminated",         # Fixed: was user_terminated
    # Psychometric features (Big 5)
    "O",  # Openness
    "C",  # Conscientiousness
    "E",  # Extraversion
    "A",  # Agreeableness
    "N",  # Neuroticism
]

# Categorical features to encode
CATEGORICAL_FEATURES = [
    "role",  # 42 categories - will be one-hot encoded
]


class FeatureExtractor:
    """
    Extract and normalize features from session data.
    
    Includes:
    - Outlier clipping (1st-99th percentile)
    - RobustScaler for normalization (handles outliers better)
    - One-hot encoding for categorical features (role)
    - Missing value imputation with median
    """
    
    def __init__(
        self, 
        feature_names: List[str] = None, 
        include_optional: bool = True,
        include_role: bool = True,
        clip_percentile: float = 99.0,
    ):
        self.base_features = feature_names or NUMERIC_FEATURES.copy()
        self.include_optional = include_optional
        self.include_role = include_role
        self.clip_percentile = clip_percentile
        
        self.feature_names = None  # Set during fit (numeric only)
        self.all_feature_names = None  # Including one-hot encoded
        
        # Use RobustScaler for better outlier handling
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        
        # For one-hot encoding role
        self.role_encoder = None
        self.role_categories = None
        
        # Percentile bounds for clipping
        self.clip_lower = {}
        self.clip_upper = {}
        
        # Medians for imputation
        self.medians = {}
        
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        """Fit scaler and encoders on training data."""
        # Determine available numeric features
        self.feature_names = [f for f in self.base_features if f in df.columns]
        
        if self.include_optional:
            for f in OPTIONAL_FEATURES:
                if f in df.columns:
                    self.feature_names.append(f)
        
        # Compute percentile bounds and medians for each feature
        for fname in self.feature_names:
            if fname in df.columns:
                col = df[fname].dropna()
                if len(col) > 0:
                    self.clip_lower[fname] = np.percentile(col, 100 - self.clip_percentile)
                    self.clip_upper[fname] = np.percentile(col, self.clip_percentile)
                    self.medians[fname] = col.median()
        
        # Fit one-hot encoder for role
        if self.include_role and "role" in df.columns:
            self.role_categories = sorted(df["role"].dropna().unique().tolist())
            logger.info(f"Role categories ({len(self.role_categories)}): {self.role_categories[:5]}...")
        
        # Extract and preprocess features
        features = self._extract_raw_features(df)
        self.scaler.fit(features)
        
        # Build full feature name list
        self.all_feature_names = self.feature_names.copy()
        if self.role_categories:
            for role in self.role_categories:
                self.all_feature_names.append(f"role_{role}")
        
        self.is_fitted = True
        logger.info(f"Fitted on {len(df):,} sessions")
        logger.info(f"  Numeric features: {len(self.feature_names)}")
        logger.info(f"  Role categories: {len(self.role_categories) if self.role_categories else 0}")
        logger.info(f"  Total features: {len(self.all_feature_names)}")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform session data to normalized feature matrix."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        # Extract numeric features
        numeric_features = self._extract_raw_features(df)
        normalized = self.scaler.transform(numeric_features)
        
        # One-hot encode role
        if self.role_categories and "role" in df.columns:
            role_onehot = self._encode_role(df["role"])
            normalized = np.hstack([normalized, role_onehot])
        
        return normalized.astype(np.float32)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def _extract_raw_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract, clip, and impute raw feature values."""
        n_samples = len(df)
        n_features = len(self.feature_names)
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for i, fname in enumerate(self.feature_names):
            if fname in df.columns:
                col = df[fname].values.astype(np.float32)
                
                # Impute missing with median (or 0 if no median computed)
                mask = np.isnan(col) | pd.isna(df[fname])
                median = self.medians.get(fname, 0.0)
                col = np.where(mask, median, col)
                
                # Clip outliers
                if fname in self.clip_lower:
                    col = np.clip(col, self.clip_lower[fname], self.clip_upper[fname])
                
                features[:, i] = col
        
        return features
    
    def _encode_role(self, role_series: pd.Series) -> np.ndarray:
        """One-hot encode role column."""
        n_samples = len(role_series)
        n_roles = len(self.role_categories)
        encoded = np.zeros((n_samples, n_roles), dtype=np.float32)
        
        role_to_idx = {role: i for i, role in enumerate(self.role_categories)}
        
        for i, role in enumerate(role_series):
            if pd.notna(role) and role in role_to_idx:
                encoded[i, role_to_idx[role]] = 1.0
        
        return encoded
    
    def get_feature_dim(self) -> int:
        """Get total number of features (numeric + one-hot)."""
        if self.all_feature_names:
            return len(self.all_feature_names)
        return len(self.base_features)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.all_feature_names or self.base_features

    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted extractor to disk.
        
        Args:
            path: Path to save file (e.g. 'models/feature_extractor.pkl')
        """
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            # Init args
            'feature_names': self.feature_names,
            'include_optional': self.include_optional,
            'include_role': self.include_role,
            'clip_percentile': self.clip_percentile,
            # Fitted state
            'scaler': self.scaler,
            'role_categories': self.role_categories,
            'role_to_idx': self.role_to_idx,
            'all_feature_names': self.all_feature_names,
            'base_features': self.base_features,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
        logging.getLogger(__name__).info(f"FeatureExtractor saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureExtractor':
        """
        Load fitted extractor from disk.
        
        Args:
            path: Path to saved file
            
        Returns:
            Fitted FeatureExtractor instance
        """
        import pickle
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FeatureExtractor file not found: {path}")
            
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        # Create instance with init args
        instance = cls(
            feature_names=state.get('feature_names'),
            include_optional=state.get('include_optional', True),
            include_role=state.get('include_role', True),
            clip_percentile=state.get('clip_percentile', 99.0),
        )
        
        # Restore fitted state
        instance.scaler = state['scaler']
        instance.role_categories = state['role_categories']
        instance.role_to_idx = state['role_to_idx']
        instance.all_feature_names = state['all_feature_names']
        instance.base_features = state['base_features']
        
        return instance


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
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    normal_only_train: bool = True,
    insider_users: Optional[List[str]] = None,
    insider_incidents: Optional[List[Dict]] = None,
    seed: int = 42,
) -> Dict[str, any]:
    """
    Prepare data for autoencoder training with train/val/test split.
    
    Split strategy (per paper):
    - Train (70%): Normal users only - learns normal behavior
    - Val (10%): Normal users only - early stopping based on reconstruction loss
    - Test (20%): ALL users including insiders - for precision/recall/F1 evaluation
    
    Session-level labels:
    - Uses incident time ranges to label ONLY sessions during malicious activity
    - Normal sessions from insider users are labeled as normal
    
    Args:
        df: Session features DataFrame
        sequence_length: Max sequence length
        train_ratio: Training set ratio (default 0.7)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.2)
        normal_only_train: If True, train/val only on normal users
        insider_users: List of known insider user IDs
        insider_incidents: List of dicts with 'user', 'start', 'end' for session-level labels
        seed: Random seed for reproducibility
        
    Returns:
        Dict with train/val/test sequences, masks, session IDs, labels, and extractor
    """
    np.random.seed(seed)
    
    logger.info(f"Preparing training data from {len(df):,} sessions")
    logger.info(f"Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    
    # Sort by user and time
    df = df.copy()
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"])
        df = df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    
    # Identify insider sessions using time ranges (session-level labels)
    insider_user_set = set(insider_users or [])
    
    # Create BOTH labels: user-level and session-level
    def create_labels(row):
        user_id = row["user_id"]
        
        # User-level: 1 if user is in insider list (regardless of timing)
        user_label = 1 if user_id in insider_user_set else 0
        
        # Session-level: 1 only if session falls within incident window
        session_label = 0
        if user_id in insider_user_set and insider_incidents:
            session_time = row.get("start_time")
            if session_time is not None and pd.notna(session_time):
                session_time = pd.to_datetime(session_time)
                
                for incident in insider_incidents:
                    if incident["user"] == user_id:
                        start = incident.get("start")
                        end = incident.get("end")
                        
                        if start is not None and end is not None:
                            if start <= session_time <= end:
                                session_label = 1
                                break
        elif user_id in insider_user_set:
            # Fallback: if no incidents provided, use user-level
            session_label = user_label
        
        return pd.Series({'label_user': user_label, 'label_session': session_label})
    
    labels_df = df.apply(create_labels, axis=1)
    df["label_user"] = labels_df["label_user"]
    df["label_session"] = labels_df["label_session"]
    df["label"] = df["label_session"]  # Default to session-level
    
    n_user_insiders = df["label_user"].sum()
    n_session_insiders = df["label_session"].sum()
    logger.info(f"User-level labels: {n_user_insiders:,} insider sessions ({n_user_insiders/len(df)*100:.2f}%)")
    logger.info(f"Session-level labels: {n_session_insiders:,} insider sessions ({n_session_insiders/len(df)*100:.3f}%)")
    
    # Separate data by USER (not session) - insider users go to test, normal users to train/val/test
    # This ensures proper evaluation: test set has ALL sessions from insider users
    normal_user_df = df[df["label_user"] == 0].copy()  # Sessions from normal users
    insider_user_df = df[df["label_user"] == 1].copy()  # ALL sessions from insider users
    
    logger.info(f"Normal user sessions: {len(normal_user_df):,}, Insider user sessions: {len(insider_user_df):,}")
    
    # Fit feature extractor on normal user data only (no insider users in training)
    extractor = FeatureExtractor()
    extractor.fit(normal_user_df)
    
    # Transform normal user data
    normal_features = extractor.transform(normal_user_df)
    
    # Build sequences for normal users
    builder = SequenceBuilder(sequence_length=sequence_length)
    normal_sequences, normal_masks, normal_session_ids = builder.build_user_sequences(
        normal_user_df.reset_index(drop=True), normal_features
    )
    normal_labels_session = np.zeros(len(normal_sequences), dtype=np.int32)
    normal_labels_user = np.zeros(len(normal_sequences), dtype=np.int32)
    
    logger.info(f"Built {len(normal_sequences):,} normal user sequences")
    
    # Split normal user data into train/val/test_normal
    n_normal = len(normal_sequences)
    n_train = int(n_normal * train_ratio)
    n_val = int(n_normal * val_ratio)
    
    indices = np.random.permutation(n_normal)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_normal_idx = indices[n_train + n_val:]
    
    # Build sequences for ALL insider user sessions (for test set)
    if len(insider_user_df) > 0:
        insider_features = extractor.transform(insider_user_df)
        insider_sequences, insider_masks, insider_session_ids = builder.build_user_sequences(
            insider_user_df.reset_index(drop=True), insider_features
        )
        
        # Map session_id to labels from the DataFrame
        session_to_label_session = dict(zip(insider_user_df["session_id"], insider_user_df["label_session"]))
        session_to_label_user = dict(zip(insider_user_df["session_id"], insider_user_df["label_user"]))
        
        insider_labels_session = np.array([session_to_label_session.get(sid, 0) for sid in insider_session_ids], dtype=np.int32)
        insider_labels_user = np.array([session_to_label_user.get(sid, 1) for sid in insider_session_ids], dtype=np.int32)
        
        logger.info(f"Built {len(insider_sequences):,} insider user sequences")
        logger.info(f"  Session-level insiders: {insider_labels_session.sum():,}")
        logger.info(f"  User-level insiders: {insider_labels_user.sum():,}")
    else:
        insider_sequences = np.array([]).reshape(0, sequence_length, extractor.get_feature_dim())
        insider_masks = np.array([]).reshape(0, sequence_length)
        insider_session_ids = []
        insider_labels_session = np.array([], dtype=np.int32)
        insider_labels_user = np.array([], dtype=np.int32)
    
    # Combine test set: normal user test portion + ALL insider user sessions
    test_sequences = np.concatenate([normal_sequences[test_normal_idx], insider_sequences])
    test_masks = np.concatenate([normal_masks[test_normal_idx], insider_masks])
    test_labels = np.concatenate([normal_labels_session[test_normal_idx], insider_labels_session])  # Session-level
    test_labels_user = np.concatenate([normal_labels_user[test_normal_idx], insider_labels_user])    # User-level
    test_session_ids = [normal_session_ids[i] for i in test_normal_idx] + insider_session_ids
    
    # Build test_timestamps and test_user_ids for TTD calculation
    session_to_time = {}
    session_to_user = {}
    
    # helper to populate lookups
    def populate_lookups(df_source):
        if "start_time" in df_source.columns and "session_id" in df_source.columns:
            session_to_time.update(dict(zip(df_source["session_id"], df_source["start_time"])))
        if "user_id" in df_source.columns and "session_id" in df_source.columns:
            session_to_user.update(dict(zip(df_source["session_id"], df_source["user_id"])))
            
    populate_lookups(normal_user_df)
    populate_lookups(insider_user_df)
    
    test_timestamps = [session_to_time.get(sid) for sid in test_session_ids]
    test_user_ids = [session_to_user.get(sid) for sid in test_session_ids]
    
    logger.info(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_sequences):,}")
    logger.info(f"Test set (session-level): {(test_labels == 0).sum():,} normal, {(test_labels == 1).sum():,} insider")
    logger.info(f"Test set (user-level): {(test_labels_user == 0).sum():,} normal, {(test_labels_user == 1).sum():,} insider")
    
    return {
        # Training data (normal only)
        "train_sequences": normal_sequences[train_idx],
        "train_masks": normal_masks[train_idx],
        "train_session_ids": [normal_session_ids[i] for i in train_idx],
        "train_labels": normal_labels_session[train_idx],
        # Validation data (normal only, for early stopping)
        "val_sequences": normal_sequences[val_idx],
        "val_masks": normal_masks[val_idx],
        "val_session_ids": [normal_session_ids[i] for i in val_idx],
        "val_labels": normal_labels_session[val_idx],
        # Test data (normal + insider, for evaluation metrics)
        "test_sequences": test_sequences,
        "test_masks": test_masks,
        "test_session_ids": test_session_ids,
        "test_timestamps": test_timestamps,     # For TTD calculation
        "test_user_ids": test_user_ids,         # For TTD calculation
        "test_labels": test_labels,             # Session-level (default)
        "test_labels_user": test_labels_user,   # User-level (for comparison)
        # Metadata
        "feature_extractor": extractor,
        "feature_dim": extractor.get_feature_dim(),
        "feature_names": extractor.get_feature_names(),
        "split_type": "random",
        "n_user_level_insiders": int(n_user_insiders),
        "n_session_level_insiders": int(n_session_insiders),
    }


def prepare_training_data_temporal(
    df: pd.DataFrame,
    sequence_length: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    insider_users: Optional[List[str]] = None,
    insider_incidents: Optional[List[Dict]] = None,
) -> Dict[str, any]:
    """
    Prepare data with TEMPORAL split (no data leakage).
    
    Split is based on time, not random:
    - Train: First 70% of timeline (normal users only)
    - Val: Next 10% of timeline (normal users only)
    - Test: Last 20% of timeline (ALL users including insiders)
    
    This prevents the model from "seeing the future" during training.
    """
    logger.info(f"Preparing training data with TEMPORAL split")
    logger.info(f"Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    
    # Ensure we have timestamps
    if "start_time" not in df.columns:
        raise ValueError("Temporal split requires 'start_time' column")
    
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.sort_values("start_time").reset_index(drop=True)
    
    # Find cutoff dates
    min_date = df["start_time"].min()
    max_date = df["start_time"].max()
    total_days = (max_date - min_date).days
    
    train_cutoff = min_date + pd.Timedelta(days=int(total_days * train_ratio))
    val_cutoff = min_date + pd.Timedelta(days=int(total_days * (train_ratio + val_ratio)))
    
    logger.info(f"Date range: {min_date.date()} to {max_date.date()} ({total_days} days)")
    logger.info(f"Train cutoff: {train_cutoff.date()}, Val cutoff: {val_cutoff.date()}")
    
    # Label insider sessions - create BOTH user-level and session-level
    insider_user_set = set(insider_users or [])
    
    def create_labels(row):
        user_id = row["user_id"]
        
        # User-level: 1 if user is in insider list
        user_label = 1 if user_id in insider_user_set else 0
        
        # Session-level: 1 only during incident window
        session_label = 0
        if user_id in insider_user_set and insider_incidents:
            session_time = row["start_time"]
            for incident in insider_incidents:
                if incident["user"] == user_id:
                    start = incident.get("start")
                    end = incident.get("end")
                    if start is not None and end is not None:
                        if start <= session_time <= end:
                            session_label = 1
                            break
        elif user_id in insider_user_set:
            session_label = user_label  # Fallback
        
        return pd.Series({'label_user': user_label, 'label_session': session_label})
    
    labels_df = df.apply(create_labels, axis=1)
    df["label_user"] = labels_df["label_user"]
    df["label_session"] = labels_df["label_session"]
    df["label"] = df["label_session"]  # Default to session-level
    
    n_user_insiders = df["label_user"].sum()
    n_session_insiders = df["label_session"].sum()
    
    # Split by time
    train_df = df[(df["start_time"] < train_cutoff) & (df["label"] == 0)]
    val_df = df[(df["start_time"] >= train_cutoff) & (df["start_time"] < val_cutoff) & (df["label"] == 0)]
    test_df = df[df["start_time"] >= val_cutoff]  # ALL users in test period
    
    logger.info(f"Train period: {len(train_df):,} sessions (normal only)")
    logger.info(f"Val period: {len(val_df):,} sessions (normal only)")
    logger.info(f"Test period: {len(test_df):,} sessions (all users)")
    logger.info(f"  User-level: {(test_df['label_user'] == 1).sum():,} insider, Session-level: {(test_df['label_session'] == 1).sum():,} insider")
    
    # Fit extractor on train data only
    extractor = FeatureExtractor()
    extractor.fit(train_df)
    
    # Build sequences
    builder = SequenceBuilder(sequence_length=sequence_length)
    
    # Transform and build for each split
    def build_split(split_df, name):
        if len(split_df) == 0:
            return (
                np.array([]).reshape(0, sequence_length, extractor.get_feature_dim()),
                np.array([]).reshape(0, sequence_length),
                [],
                np.array([], dtype=np.int32),
            )
        features = extractor.transform(split_df.reset_index(drop=True))
        seqs, masks, sess_ids = builder.build_user_sequences(
            split_df.reset_index(drop=True), features
        )
        
        # Map session IDs to both label types and timestamps
        session_to_label = dict(zip(split_df["session_id"], split_df["label_session"]))
        session_to_label_user = dict(zip(split_df["session_id"], split_df["label_user"]))
        session_to_time = dict(zip(split_df["session_id"], split_df["start_time"]))
        
        labels = np.array([session_to_label.get(sid, 0) for sid in sess_ids], dtype=np.int32)
        labels_user = np.array([session_to_label_user.get(sid, 0) for sid in sess_ids], dtype=np.int32)
        timestamps = [session_to_time.get(sid) for sid in sess_ids]
        
        logger.info(f"Built {len(seqs):,} {name} sequences")
        return seqs, masks, sess_ids, labels, labels_user, timestamps
    
    train_seqs, train_masks, train_ids, train_labels, _, _ = build_split(train_df, "train")
    val_seqs, val_masks, val_ids, val_labels, _, _ = build_split(val_df, "val")
    test_seqs, test_masks, test_ids, test_labels, test_labels_user, test_timestamps = build_split(test_df, "test")
    
    logger.info(f"Test labels (session): {(test_labels == 0).sum():,} normal, {(test_labels == 1).sum():,} insider")
    logger.info(f"Test labels (user): {(test_labels_user == 0).sum():,} normal, {(test_labels_user == 1).sum():,} insider")
    
    return {
        "train_sequences": train_seqs,
        "train_masks": train_masks,
        "train_session_ids": train_ids,
        "train_labels": train_labels,
        "val_sequences": val_seqs,
        "val_masks": val_masks,
        "val_session_ids": val_ids,
        "val_labels": val_labels,
        "test_sequences": test_seqs,
        "test_masks": test_masks,
        "test_session_ids": test_ids,
        "test_timestamps": test_timestamps,     # For TTD calculation
        "test_user_ids": split_data["test"]["user_ids"], # For TTD calculation
        "test_labels": test_labels,             # Session-level (default)
        "test_labels_user": test_labels_user,   # User-level (for comparison)
        "feature_extractor": extractor,
        "feature_dim": extractor.get_feature_dim(),
        "feature_names": extractor.get_feature_names(),
        "split_type": "temporal",
        "train_cutoff": str(train_cutoff.date()),
        "val_cutoff": str(val_cutoff.date()),
        "n_user_level_insiders": int(n_user_insiders),
        "n_session_level_insiders": int(n_session_insiders),
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

