"""
Feature Engineering Module
Transforms session data into sequences for autoencoder training.

Updated for CERT R4.2 schema with correct feature names.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    "role_changed_this_month",       # Fixed: was role_changed
    "dept_changed_this_month",       # Fixed: was dept_changed  
    "user_terminated",               # Fixed: was terminated
    # Psychometric features (Big 5)
    "openness",           # Fixed: was O
    "conscientiousness",  # Fixed: was C
    "extraversion",       # Fixed: was E
    "agreeableness",      # Fixed: was A
    "neuroticism",        # Fixed: was N
]

# Categorical features to encode
CATEGORICAL_FEATURES = [
    "role",  # 42 categories - will be one-hot encoded
]


class FeatureExtractor:
    """
    Extract and normalize features from session data.
    
    Includes:
    - Missing value imputation with median
    - RobustScaler for normalization (handles outliers naturally)
    - Optional outlier clipping AFTER scaling (if clip_percentile < 100)
    - One-hot encoding for categorical features (role)
    
    Processing order:
    1. Impute missing values with median
    2. Apply RobustScaler (handles outliers naturally via IQR)
    3. Optionally clip extreme values after scaling (if clip_percentile < 100)
    4. One-hot encode role
    
    Args:
        feature_names: List of feature names to use (default: NUMERIC_FEATURES)
        include_optional: Include optional features (LDAP, psychometric)
        include_role: Include role as one-hot encoded features
        clip_percentile: Percentile for clipping AFTER scaling (100.0 = no clipping).
                        Example: 99.0 clips at 1st and 99th percentile of scaled features.
    """
    
    def __init__(
        self, 
        feature_names: List[str] = None, 
        include_optional: bool = True,
        include_role: bool = False,  # Default: no role features
        clip_percentile: float = 100.0,  # 100.0 = no clipping, <100 = clip after scaling
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
        self.role_to_idx = None  # Fixed: Initialize attribute
        
        # Percentile bounds for clipping AFTER scaling (if clip_percentile < 100)
        self.clip_lower_scaled = {}
        self.clip_upper_scaled = {}
        
        # Medians for imputation
        self.medians = {}
        
        # Padding value (median of normalized features) for sequence padding
        self.padding_value = None
        
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, role_categories: List[str] = None) -> "FeatureExtractor":
        """Fit scaler and encoders on training data.
        
        Args:
            df: Training DataFrame
            role_categories: Optional list of all role categories to use for one-hot encoding.
                           If provided, ensures consistent dimensions even if some roles
                           only appear in test data.
        """
        # Determine available numeric features
        self.feature_names = [f for f in self.base_features if f in df.columns]
        
        # Warn about missing base features
        missing_base = [f for f in self.base_features if f not in df.columns]
        if missing_base:
            logger.warning(f"Missing {len(missing_base)} base features: {missing_base[:5]}{'...' if len(missing_base) > 5 else ''}")
        
        if self.include_optional:
            for f in OPTIONAL_FEATURES:
                if f in df.columns:
                    self.feature_names.append(f)
        
        # Warn if no features found
        if len(self.feature_names) == 0:
            raise ValueError("No numeric features found in DataFrame. Check column names.")
        
        # Compute medians for imputation
        for fname in self.feature_names:
            if fname in df.columns:
                col = df[fname].dropna().astype(np.float32)
                if len(col) > 0:
                    self.medians[fname] = col.median()
                else:
                    logger.warning(f"Feature '{fname}' has no valid values, using 0.0 for imputation")
                    self.medians[fname] = 0.0
        
        # Fit one-hot encoder for role
        if self.include_role and "role" in df.columns:
            # Use provided role_categories if available, otherwise infer from data
            if role_categories is not None:
                self.role_categories = sorted(role_categories)
            else:
                self.role_categories = sorted(df["role"].dropna().unique().tolist())
            self.role_to_idx = {role: i for i, role in enumerate(self.role_categories)}
            logger.info(f"Role categories ({len(self.role_categories)}): {self.role_categories[:5]}...")
        
        # Extract raw features (impute only, no clipping)
        features = self._extract_raw_features(df)
        
        # Fit RobustScaler (handles outliers naturally)
        self.scaler.fit(features)
        
        # Transform features to get scaled values for clipping bounds and padding
        scaled_features = self.scaler.transform(features)
        
        # If clipping is enabled (clip_percentile < 100), compute bounds AFTER scaling
        if self.clip_percentile < 100.0:
            lower_percentile = 100.0 - self.clip_percentile
            upper_percentile = self.clip_percentile
            
            for i, fname in enumerate(self.feature_names):
                col_scaled = scaled_features[:, i]
                if len(col_scaled) > 0:
                    self.clip_lower_scaled[fname] = np.percentile(col_scaled, lower_percentile)
                    self.clip_upper_scaled[fname] = np.percentile(col_scaled, upper_percentile)
            
            logger.info(f"Clipping enabled: {lower_percentile:.1f}th-{upper_percentile:.1f}th percentile AFTER scaling")
        else:
            logger.info("No clipping: RobustScaler handles outliers naturally")
        
        # Compute padding value (median of normalized features) for sequence padding
        self.padding_value = np.median(scaled_features, axis=0).astype(np.float32)
        
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
        """Extract and impute raw feature values (no clipping - that happens after scaling)."""
        n_samples = len(df)
        n_features = len(self.feature_names)
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for i, fname in enumerate(self.feature_names):
            if fname in df.columns:
                col = df[fname].values.astype(np.float32)
                
                # Impute missing with median (or 0 if no median computed)
                mask = np.isnan(col) | pd.isna(df[fname])
                median = self.medians.get(fname, 0.0)
                if mask.sum() > 0:
                    col = np.where(mask, median, col)
                    if mask.sum() > len(col) * 0.1:  # Warn if >10% missing
                        logger.debug(f"Feature '{fname}': {mask.sum()}/{len(col)} ({mask.sum()/len(col)*100:.1f}%) values imputed")
                
                features[:, i] = col
            else:
                # Feature missing - use median (shouldn't happen if validation works)
                logger.warning(f"Feature '{fname}' missing in DataFrame, using median: {self.medians.get(fname, 0.0)}")
                features[:, i] = self.medians.get(fname, 0.0)
        
        return features
    
    def _encode_role(self, role_series: pd.Series) -> np.ndarray:
        """One-hot encode role column."""
        n_samples = len(role_series)
        n_roles = len(self.role_categories)
        encoded = np.zeros((n_samples, n_roles), dtype=np.float32)
        
        # Use stored mapping if available, otherwise rebuild (fallback)
        if self.role_to_idx:
            mapping = self.role_to_idx
        else:
            mapping = {role: i for i, role in enumerate(self.role_categories)}
        
        for i, role in enumerate(role_series):
            if pd.notna(role) and role in mapping:
                encoded[i, mapping[role]] = 1.0
        
        return encoded
    
    def get_feature_dim(self) -> int:
        """Get total number of features (numeric + one-hot)."""
        if self.all_feature_names:
            return len(self.all_feature_names)
        return len(self.base_features)
    
    def get_padding_value(self) -> Optional[np.ndarray]:
        """Get padding value (median of normalized features) for sequence padding."""
        return self.padding_value
    
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
            # Clipping and imputation state (updated: clipping after scaling)
            'clip_lower_scaled': self.clip_lower_scaled,
            'clip_upper_scaled': self.clip_upper_scaled,
            'medians': self.medians,
            'padding_value': self.padding_value,
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
            clip_percentile=state.get('clip_percentile', 100.0),  # Default: no clipping
        )
        
        # Restore fitted state
        instance.scaler = state['scaler']
        instance.role_categories = state['role_categories']
        instance.role_to_idx = state['role_to_idx']
        instance.all_feature_names = state['all_feature_names']
        instance.base_features = state['base_features']
        # Restore clipping and imputation state (backward compatible)
        instance.clip_lower_scaled = state.get('clip_lower_scaled', {})
        instance.clip_upper_scaled = state.get('clip_upper_scaled', {})
        instance.medians = state.get('medians', {})
        instance.padding_value = state.get('padding_value', None)
        # Mark as fitted
        instance.is_fitted = True
        
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
        padding_value: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build sequences grouped by user.
        
        Each session becomes a sequence containing its historical context
        (previous sessions by the same user).
        
        Args:
            df: Session DataFrame with user_id and session_id columns (must be sorted by time)
            features: Normalized feature matrix (N, F)
            padding_value: Optional array of shape (F,) to use for padding. If None, uses zeros.
                          Should be median of normalized features for better gradient flow.
            
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
        
        # Validate feature dimension matches expected
        if self.feature_dim is not None and feature_dim != self.feature_dim:
            logger.warning(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {feature_dim}. "
                f"Using actual dimension {feature_dim}."
            )
        
        # Ensure df has proper index
        df = df.reset_index(drop=True)
        
        # Validate required columns
        if "user_id" not in df.columns:
            raise ValueError("DataFrame must have 'user_id' column")
        if "session_id" not in df.columns:
            raise ValueError("DataFrame must have 'session_id' column")
        
        # Group by user and process
        for user_id, group in df.groupby("user_id"):
            # Explicitly sort by time to ensure temporal order
            if "start_time" in group.columns:
                group = group.sort_values("start_time").reset_index(drop=True)
            elif "timestamp" in group.columns:
                group = group.sort_values("timestamp").reset_index(drop=True)
            else:
                logger.warning(f"No time column found for user {user_id}, using DataFrame order")
            
            user_indices = group.index.tolist()
            user_features = features[user_indices]
            user_sessions = group["session_id"].tolist()
            
            # Create one sequence per session (includes history)
            for i in range(len(user_indices)):
                # Get sequence ending at position i
                start = max(0, i - self.sequence_length + 1)
                seq_features = user_features[start:i+1]
                
                # Pad if needed (pad at beginning)
                seq_len = len(seq_features)
                if seq_len < self.sequence_length:
                    # Use provided padding value (median of normalized features) or zeros
                    if padding_value is not None:
                        if padding_value.shape[0] != feature_dim:
                            logger.warning(
                                f"Padding value dimension ({padding_value.shape[0]}) doesn't match "
                                f"feature dimension ({feature_dim}), using zeros"
                            )
                            pad_val = np.zeros(feature_dim, dtype=np.float32)
                        else:
                            pad_val = padding_value
                    else:
                        pad_val = np.zeros(feature_dim, dtype=np.float32)
                    
                    padding = np.tile(pad_val, (self.sequence_length - seq_len, 1))
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



if __name__ == "__main__":
    from pathlib import Path
    import duckdb
    db_path = Path("data/processed/cert.duckdb")
    if db_path.exists():
        con = duckdb.connect(str(db_path))
        df = con.execute("SELECT * FROM session_features LIMIT 100").fetchdf()
        con.close()
        print(f"Loaded {len(df)} sessions for testing feature extractor")
        extractor = FeatureExtractor()
        extractor.fit(df)
        print(f"Feature dim: {extractor.get_feature_dim()}")
        print("Features:", extractor.get_feature_names())
