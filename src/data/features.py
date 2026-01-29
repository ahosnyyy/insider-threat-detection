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
import yaml
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Role feature modes
ROLE_FEATURES_NONE = "none"
ROLE_FEATURES_ROLES = "roles"
ROLE_FEATURES_UNITS = "units"


def load_role_units_mapping(
    path: Union[str, Path],
) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    """
    Load role → functional unit mapping from YAML (Option A: by unit).
    
    Args:
        path: Path to role_units.yaml.
        
    Returns:
        unit_order: List of unit names (defines one-hot order).
        role_to_unit: Dict mapping each role name to its unit name.
        default_unit: Unit name for unknown roles, or None.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Role mapping file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not data:
        raise ValueError("Role mapping file is empty")
    
    unit_order = data.get("unit_order")
    if not unit_order:
        raise ValueError("Role mapping must define 'unit_order' (list of unit names)")
    
    default_unit = data.get("default_unit")  # Optional
    
    # Build role → unit from unit → list of roles
    role_to_unit: Dict[str, str] = {}
    for unit in unit_order:
        roles = data.get(unit)
        if roles is None:
            continue
        if isinstance(roles, list):
            for role in roles:
                role_to_unit[str(role).strip()] = unit
        else:
            role_to_unit[str(roles).strip()] = unit
    
    return unit_order, role_to_unit, default_unit


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
    - Role features: none | roles (42 one-hot) | units (6 one-hot from role_units.yaml)
    
    Processing order:
    1. Impute missing values with median
    2. Apply RobustScaler (handles outliers naturally via IQR)
    3. Optionally clip extreme values after scaling (if clip_percentile < 100)
    4. One-hot encode role (if role_features is 'roles' or 'units')
    
    Args:
        feature_names: List of feature names to use (default: NUMERIC_FEATURES)
        include_optional: Include optional features (LDAP, psychometric)
        role_features: "none" | "roles" | "units". roles=42 one-hot, units=functional units from mapping.
        role_mapping_path: Path to role_units.yaml; required when role_features="units".
        clip_percentile: Percentile for clipping AFTER scaling (100.0 = no clipping).
    """
    
    def __init__(
        self, 
        feature_names: List[str] = None, 
        include_optional: bool = True,
        role_features: str = ROLE_FEATURES_NONE,
        role_mapping_path: Optional[Union[str, Path]] = None,
        clip_percentile: float = 100.0,
    ):
        self.base_features = feature_names or NUMERIC_FEATURES.copy()
        self.include_optional = include_optional
        self.role_features = role_features.strip().lower() if role_features else ROLE_FEATURES_NONE
        if self.role_features not in (ROLE_FEATURES_NONE, ROLE_FEATURES_ROLES, ROLE_FEATURES_UNITS):
            raise ValueError(
                f"role_features must be one of '{ROLE_FEATURES_NONE}', '{ROLE_FEATURES_ROLES}', '{ROLE_FEATURES_UNITS}'"
            )
        self.role_mapping_path = Path(role_mapping_path) if role_mapping_path else None
        self.clip_percentile = clip_percentile
        
        self.feature_names = None  # Set during fit (numeric only)
        self.all_feature_names = None  # Including one-hot encoded
        
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        
        # Role encoding: categories = list of role names (roles) or unit names (units)
        self.role_categories = None
        self.role_to_idx = None  # category name -> one-hot index
        # For units only: role -> unit, unit_order, default_unit
        self._role_to_unit: Optional[Dict[str, str]] = None
        self._default_unit: Optional[str] = None
        
        self.clip_lower_scaled = {}
        self.clip_upper_scaled = {}
        self.medians = {}
        self.padding_value = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, role_categories: List[str] = None) -> "FeatureExtractor":
        """Fit scaler and encoders on training data.
        
        Args:
            df: Training DataFrame
            role_categories: Optional list of all role categories (for role_features='roles').
                           Ensures consistent dimensions if some roles only appear in test.
        """
        # Determine available numeric features
        self.feature_names = [f for f in self.base_features if f in df.columns]
        
        missing_base = [f for f in self.base_features if f not in df.columns]
        if missing_base:
            logger.warning(f"Missing {len(missing_base)} base features: {missing_base[:5]}{'...' if len(missing_base) > 5 else ''}")
        
        if self.include_optional:
            for f in OPTIONAL_FEATURES:
                if f in df.columns:
                    self.feature_names.append(f)
        
        if len(self.feature_names) == 0:
            raise ValueError("No numeric features found in DataFrame. Check column names.")
        
        for fname in self.feature_names:
            if fname in df.columns:
                col = df[fname].dropna().astype(np.float32)
                if len(col) > 0:
                    self.medians[fname] = col.median()
                else:
                    logger.warning(f"Feature '{fname}' has no valid values, using 0.0 for imputation")
                    self.medians[fname] = 0.0
        
        # Role encoding: none | roles | units
        if self.role_features == ROLE_FEATURES_NONE:
            self.role_categories = None
            self.role_to_idx = None
            self._role_to_unit = None
            self._default_unit = None
        elif self.role_features == ROLE_FEATURES_ROLES and "role" in df.columns:
            if role_categories is not None:
                self.role_categories = sorted(role_categories)
            else:
                self.role_categories = sorted(df["role"].dropna().unique().tolist())
            self.role_to_idx = {r: i for i, r in enumerate(self.role_categories)}
            self._role_to_unit = None
            self._default_unit = None
            logger.info(f"Role categories ({len(self.role_categories)}): {self.role_categories[:5]}...")
        elif self.role_features == ROLE_FEATURES_UNITS and "role" in df.columns:
            if not self.role_mapping_path or not self.role_mapping_path.exists():
                raise FileNotFoundError(
                    "role_features='units' requires role_mapping_path to a valid role_units.yaml"
                )
            unit_order, role_to_unit, default_unit = load_role_units_mapping(self.role_mapping_path)
            self.role_categories = unit_order
            self.role_to_idx = {u: i for i, u in enumerate(unit_order)}
            self._role_to_unit = role_to_unit
            self._default_unit = default_unit
            # Validate: warn on unknown roles in data
            data_roles = set(df["role"].dropna().astype(str).unique())
            unknown = data_roles - set(role_to_unit.keys())
            if unknown:
                if default_unit and default_unit in self.role_to_idx:
                    logger.warning(
                        f"Unknown roles ({len(unknown)}) mapped to default_unit '{default_unit}': {sorted(unknown)[:5]}{'...' if len(unknown) > 5 else ''}"
                    )
                else:
                    logger.warning(
                        f"Unknown roles in data (no default_unit in mapping): {sorted(unknown)[:10]}{'...' if len(unknown) > 10 else ''}"
                    )
            logger.info(f"Role units ({len(unit_order)}): {unit_order}")
        
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
        """One-hot encode role column (roles = 42 categories, units = 6 functional units)."""
        n_samples = len(role_series)
        n_cats = len(self.role_categories)
        encoded = np.zeros((n_samples, n_cats), dtype=np.float32)
        
        if self.role_features == ROLE_FEATURES_UNITS and self._role_to_unit is not None:
            # Map role -> unit, then one-hot by unit
            for i, role in enumerate(role_series):
                if pd.isna(role):
                    continue
                role_str = str(role).strip()
                unit = self._role_to_unit.get(role_str)
                if unit is None and self._default_unit is not None:
                    unit = self._default_unit
                if unit is not None and unit in self.role_to_idx:
                    encoded[i, self.role_to_idx[unit]] = 1.0
        else:
            # Direct role -> index (roles mode)
            mapping = self.role_to_idx or {r: i for i, r in enumerate(self.role_categories)}
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
            'feature_names': self.feature_names,
            'include_optional': self.include_optional,
            'role_features': self.role_features,
            'role_mapping_path': str(self.role_mapping_path) if self.role_mapping_path else None,
            'clip_percentile': self.clip_percentile,
            'scaler': self.scaler,
            'role_categories': self.role_categories,
            'role_to_idx': self.role_to_idx,
            '_role_to_unit': self._role_to_unit,
            '_default_unit': self._default_unit,
            'all_feature_names': self.all_feature_names,
            'base_features': self.base_features,
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
            
        # Backward compatibility: old pickles had include_role (bool)
        role_features = state.get('role_features')
        if role_features is None:
            role_features = ROLE_FEATURES_ROLES if state.get('include_role', False) else ROLE_FEATURES_NONE
        role_mapping_path = state.get('role_mapping_path')
        if role_mapping_path:
            role_mapping_path = Path(role_mapping_path)
        
        instance = cls(
            feature_names=state.get('feature_names'),
            include_optional=state.get('include_optional', True),
            role_features=role_features,
            role_mapping_path=role_mapping_path,
            clip_percentile=state.get('clip_percentile', 100.0),
        )
        
        instance.scaler = state['scaler']
        instance.role_categories = state.get('role_categories')
        instance.role_to_idx = state.get('role_to_idx')
        instance._role_to_unit = state.get('_role_to_unit')
        instance._default_unit = state.get('_default_unit')
        instance.all_feature_names = state.get('all_feature_names')
        instance.base_features = state.get('base_features')
        instance.clip_lower_scaled = state.get('clip_lower_scaled', {})
        instance.clip_upper_scaled = state.get('clip_upper_scaled', {})
        instance.medians = state.get('medians', {})
        instance.padding_value = state.get('padding_value', None)
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
